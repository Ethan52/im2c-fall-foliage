import os
import sys
import glob
import logging

import numpy as np
import pyproj
import rasterio
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

# forest fraction on the MODIS grid and the MODIS pixel to analysis cell map, step 1 of references/preprocessing_plan.md
# decision D11: a 30 m pixel is stable deciduous or mixed forest if it is NLCD class 41 or 43 in every epoch,
# and a 500 m MODIS pixel is kept if at least half of it is stable forest
NLCD_YEARS = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]  # as downloaded by dataload_nlcd.py
FOREST_CLASSES = (41, 43)  # deciduous forest, mixed forest
REPORT_YEAR = 2021
KEEP_THRESHOLD = 0.5
REPORT_THRESHOLDS = [0.3, 0.5, 0.7]
MASS_BALANCE_TOL = 0.02

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NLCD_DIR = os.path.join(REPO_ROOT, "data", "raw", "nlcd")
MODIS_DIR = os.path.join(REPO_ROOT, "data", "raw", "modis", "MOD09A1.061_2000360_to_2025365")
GRID_NPZ = os.path.join(REPO_ROOT, "data", "processed", "grid", "analysis_grid.npz")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "forest_mask")
OUT_TIF = os.path.join(OUT_DIR, "forest_fraction_500m.tif")
OUT_NPZ = os.path.join(OUT_DIR, "forest_mask.npz")


def to_modis_grid(values: np.ndarray, src_transform, src_crs, modis):
    '''area average of a 30 m array onto the MODIS grid; NaN where the MODIS pixel has no 30 m pixel'''
    out = np.full(modis.shape, np.nan, dtype=np.float32)  # MODIS row, MODIS column
    reproject(
        source=values.astype(np.float32), destination=out,
        src_transform=src_transform, src_crs=src_crs, dst_transform=modis.transform, dst_crs=modis.crs,
        resampling=Resampling.average, dst_nodata=np.nan,
    )
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(OUT_DIR, exist_ok=True)

    # MODIS grid from any MOD09A1 GeoTIFF of the download, all share one grid (V1)
    modis_ref = sorted(glob.glob(os.path.join(MODIS_DIR, "MOD09A1.061_sur_refl_b01_doy*.tif")))[0]
    modis = rasterio.open(modis_ref)
    assert modis.shape == (1099, 1353), f"unexpected MODIS shape {modis.shape}"
    logger.info("MODIS grid from %s: %s, %s", os.path.basename(modis_ref), modis.shape, modis.crs.to_proj4())

    # stable forest at 30 m, accumulated one epoch at a time
    stable = None
    nlcd_profile = None
    for year in NLCD_YEARS:
        with rasterio.open(os.path.join(NLCD_DIR, f"nlcd_{year}_land_cover_maine.tif")) as src:
            if nlcd_profile is None:
                nlcd_profile = (src.shape, src.transform, src.crs)
            assert (src.shape, src.transform, src.crs) == nlcd_profile, f"NLCD {year} grid differs from {NLCD_YEARS[0]}"
            assert src.crs.to_epsg() == 5070 and src.res == (30.0, 30.0), f"NLCD {year} CRS or resolution"
            classes = src.read(1)  # row, column, uint8 class codes, 0 is nodata
            assert classes.dtype == np.uint8
        forest = np.isin(classes, FOREST_CLASSES)
        stable = forest if stable is None else stable & forest
        logger.info("NLCD %d: forest share %.4f, stable so far %.4f", year, forest.mean(), stable.mean())
        if year == REPORT_YEAR:
            report_classes = classes
    nlcd_shape, nlcd_transform, nlcd_crs = nlcd_profile

    frac_stable = to_modis_grid(stable, nlcd_transform, nlcd_crs, modis)
    fracs_report = {c: to_modis_grid(report_classes == c, nlcd_transform, nlcd_crs, modis) for c in (41, 42, 43)}
    finite = np.isfinite(frac_stable)
    assert np.all((frac_stable[finite] >= 0) & (frac_stable[finite] <= 1)), "fraction outside 0 to 1"

    # analysis cell of each MODIS pixel center
    grid = np.load(GRID_NPZ)
    gt = grid["transform"]  # x pixel size, 0, x of left edge, 0, minus y pixel size, y of top edge
    grid_shape = tuple(grid["shape"])
    id_raster = grid["id_raster"]  # grid row, grid column
    rows, cols = np.mgrid[0:modis.height, 0:modis.width]  # MODIS row, MODIS column
    mx = modis.transform.c + (cols + 0.5) * modis.transform.a
    my = modis.transform.f + (rows + 0.5) * modis.transform.e
    ux, uy = pyproj.Transformer.from_crs(modis.crs, f"EPSG:{int(grid['epsg'])}", always_xy=True).transform(mx, my)
    gcol = np.floor((ux - gt[2]) / gt[0]).astype(np.int64)
    grow = np.floor((uy - gt[5]) / gt[4]).astype(np.int64)
    on_grid = (grow >= 0) & (grow < grid_shape[0]) & (gcol >= 0) & (gcol < grid_shape[1])
    pixel_cell = np.full(modis.shape, -1, dtype=np.int32)  # MODIS row, MODIS column
    pixel_cell[on_grid] = id_raster[grow[on_grid], gcol[on_grid]]
    in_maine = pixel_cell >= 0
    assert np.all(finite[in_maine]), "a MODIS pixel in Maine has no NLCD coverage"
    logger.info("%d MODIS pixels have their center in a grid cell", in_maine.sum())

    # mass balance over the MODIS pixels in Maine only, since the NLCD and MODIS rectangles are rotated against each other
    # the Maine mask goes to the 30 m grid by nearest neighbour, so each 30 m pixel takes the MODIS pixel containing its center
    # CONUS Albers and MODIS sinusoidal are both equal-area projections, so the two areas compare directly
    in_maine_30m = np.zeros(nlcd_shape, dtype=np.uint8)  # NLCD row, NLCD column
    reproject(
        source=in_maine.astype(np.uint8), destination=in_maine_30m, src_transform=modis.transform, src_crs=modis.crs,
        dst_transform=nlcd_transform, dst_crs=nlcd_crs, resampling=Resampling.nearest,
    )
    modis_area = abs(modis.transform.a * modis.transform.e)
    area_modis = float(frac_stable[in_maine].sum()) * modis_area
    area_nlcd = float((stable & (in_maine_30m == 1)).sum()) * 900.0
    rel = area_modis / area_nlcd - 1
    logger.info("stable forest area in Maine: %.1f square km from the MODIS fractions, %.1f from NLCD, difference %.3f%%",
                area_modis / 1e6, area_nlcd / 1e6, 100 * rel)
    assert abs(rel) <= MASS_BALANCE_TOL, "forest fraction does not conserve area"

    counts = {}
    for t in REPORT_THRESHOLDS:
        kept_t = in_maine & (frac_stable >= t)
        counts[t] = (int(kept_t.sum()), len(np.unique(pixel_cell[kept_t])))
        logger.info("threshold %.1f: %d kept MODIS pixels, %d cells with at least one", t, *counts[t])
    kept = in_maine & (frac_stable >= KEEP_THRESHOLD)
    forest_cell_id = np.unique(pixel_cell[kept]).astype(np.int32)

    profile = modis.profile.copy()
    profile.update(count=4, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        for band, (name, arr) in enumerate([("frac_stable", frac_stable), ("frac41_2021", fracs_report[41]),
                                            ("frac42_2021", fracs_report[42]), ("frac43_2021", fracs_report[43])], start=1):
            dst.write(arr, band)
            dst.set_band_description(band, name)
    logger.info("wrote %s", OUT_TIF)

    np.savez_compressed(
        OUT_NPZ,
        pixel_cell=pixel_cell, kept=kept, forest_cell_id=forest_cell_id, frac_stable=frac_stable,
        frac41=fracs_report[41], frac42=fracs_report[42], frac43=fracs_report[43],
        thresholds=np.array(REPORT_THRESHOLDS), kept_pixels=np.array([counts[t][0] for t in REPORT_THRESHOLDS]),
        kept_cells=np.array([counts[t][1] for t in REPORT_THRESHOLDS]), keep_threshold=np.array(KEEP_THRESHOLD),
        mass_balance_rel=np.array(rel),
    )
    logger.info("wrote %s with %d forest cells", OUT_NPZ, len(forest_cell_id))


if __name__ == "__main__":
    main()
