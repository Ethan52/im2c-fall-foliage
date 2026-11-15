import os
import sys
import glob
import logging
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

# live-tree basal-area shares of each forest cell from TreeMap, decision D23
# a cell's share is the summed basal area of the group over its TreeMap pixels divided by the summed total basal area,
# over the 30 m TreeMap pixels whose center lies in a kept MODIS pixel of the cell (D11)
EDITIONS = [2016, 2020, 2022, 2023]
YEARS = list(range(2001, 2027))
GROUPS = ["hardwood", "red_maple", "sugar_maple", "birch", "aspen"]
HARDWOOD_MIN_SPCD = 300  # FIA species codes below 300 are softwoods
BA_FACTOR = 0.005454154  # square feet of basal area per tree per squared inch of diameter
TOP_SPECIES = 20

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TREEMAP_DIR = os.path.join(REPO_ROOT, "data", "raw", "treemap")
NLCD_REF = os.path.join(REPO_ROOT, "data", "raw", "nlcd", "nlcd_2021_land_cover_maine.tif")
MASK_NPZ = os.path.join(REPO_ROOT, "data", "processed", "forest_mask", "forest_mask.npz")
MODIS_DIR = os.path.join(REPO_ROOT, "data", "raw", "modis", "MOD09A1.061_2000360_to_2025365")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "static")


def edition_files(edition: int):
    '''raster, raster attribute table and tree table of one TreeMap edition'''
    folder = os.path.join(TREEMAP_DIR, str(edition))
    tif = [p for p in glob.glob(os.path.join(folder, "**", "TreeMap*.tif"), recursive=True) if "SnagHazard" not in p]
    table = glob.glob(os.path.join(folder, "**", "TreeMap*[Tt]ree_[Tt]able.csv"), recursive=True)
    assert len(tif) == 1 and len(table) == 1, f"TreeMap {edition}: rasters {tif}, tree tables {table}"
    return tif[0], tif[0] + ".vat.dbf", table[0]


def main():
    parser = argparse.ArgumentParser(description="Live-tree basal-area shares of each forest cell from TreeMap")
    parser.add_argument("--editions", nargs="+", type=int, default=EDITIONS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(OUT_DIR, exist_ok=True)

    mask = np.load(MASK_NPZ)
    kept = mask["kept"]  # MODIS row, MODIS column
    pixel_cell = mask["pixel_cell"]  # MODIS row, MODIS column
    cell_id = mask["forest_cell_id"]
    n_cells = len(cell_id)
    # one plus the position of each kept MODIS pixel's cell in the forest cell list, 0 for pixels not kept
    slot_modis = np.zeros(kept.shape, dtype=np.int32)  # MODIS row, MODIS column
    slot_modis[kept] = np.searchsorted(cell_id, pixel_cell[kept]) + 1
    assert np.array_equal(cell_id[slot_modis[kept] - 1], pixel_cell[kept]), "kept pixel cell not in the forest cell list"
    modis_ref = sorted(glob.glob(os.path.join(MODIS_DIR, "MOD09A1.061_sur_refl_b01_doy*.tif")))[0]
    with rasterio.open(modis_ref) as m:
        modis_transform, modis_crs = m.transform, m.crs
    with rasterio.open(NLCD_REF) as ref:
        maine_bounds = ref.bounds  # EPSG:5070, the Maine extent of the NLCD download
        assert ref.crs.to_epsg() == 5070

    ba_sum = np.zeros((len(GROUPS) + 1, n_cells, len(args.editions)))  # total then groups, cell, edition
    n_footprint = np.zeros((n_cells, len(args.editions)), dtype=np.int64)
    n_treemap = np.zeros((n_cells, len(args.editions)), dtype=np.int64)
    for e, edition in enumerate(args.editions):
        tif, vat_path, table_path = edition_files(edition)
        with rasterio.open(tif) as src:
            assert src.crs.to_epsg() == 5070 and src.res == (30.0, 30.0), f"TreeMap {edition} CRS or resolution"
            window = from_bounds(*maine_bounds, transform=src.transform).round_offsets().round_lengths()
            plot_value = src.read(1, window=window)  # row, column, raster value of the imputed plot
            win_transform = src.window_transform(window)
            nodata = src.nodata
        assert plot_value.dtype in (np.int32, np.uint32), f"TreeMap {edition} dtype {plot_value.dtype}"  # 2020 is stored unsigned

        # each 30 m pixel takes the kept MODIS pixel containing its center
        slot = np.zeros(plot_value.shape, dtype=np.int32)  # TreeMap row, TreeMap column
        reproject(source=slot_modis, destination=slot, src_transform=modis_transform, src_crs=modis_crs,
                  dst_transform=win_transform, dst_crs="EPSG:5070", resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        slot -= 1  # back to the position in the forest cell list, -1 outside the kept pixels
        in_fp = slot >= 0
        has_plot = in_fp & (plot_value != nodata)
        n_footprint[:, e] = np.bincount(slot[in_fp], minlength=n_cells)
        n_treemap[:, e] = np.bincount(slot[has_plot], minlength=n_cells)
        values = plot_value[has_plot]
        slots = slot[has_plot]
        del plot_value, slot, in_fp, has_plot

        # raster value to TreeMap record number through the raster attribute table
        vat = gpd.read_file(vat_path, ignore_geometry=True)
        vat.columns = [c.lower() for c in vat.columns]
        vat = vat[["value", "tm_id", "balive"]].astype({"value": np.int64, "tm_id": np.int64})
        uniq, inv = np.unique(values, return_inverse=True)
        vat = vat.set_index("value")
        assert np.isin(uniq, vat.index).all(), f"TreeMap {edition}: raster values missing from the attribute table"
        tm_id = vat.loc[uniq, "tm_id"].to_numpy()
        assert len(np.unique(tm_id)) == len(tm_id), f"TreeMap {edition}: two raster values share a record number"
        logger.info("TreeMap %d: %d pixels in the kept MODIS footprint, %d with a plot, %d distinct plots",
                    edition, n_footprint[:, e].sum(), n_treemap[:, e].sum(), len(uniq))

        # live-tree basal area per acre of each plot, total and by group
        trees = pd.read_csv(table_path, usecols=lambda c: c.upper() in ("TM_ID", "STATUSCD", "TPA_UNADJ", "SPCD", "SCIENTIFIC_NAME", "DIA"))
        trees.columns = [c.upper() for c in trees.columns]
        trees = trees[trees["TM_ID"].isin(tm_id) & (trees["STATUSCD"] == 1)]
        assert trees["DIA"].notna().all() and trees["TPA_UNADJ"].notna().all(), f"TreeMap {edition}: missing diameter or trees per acre"
        trees = trees.assign(BA=BA_FACTOR * trees["DIA"] ** 2 * trees["TPA_UNADJ"])
        name = trees["SCIENTIFIC_NAME"].fillna("").str.strip()
        member = {
            "hardwood": trees["SPCD"] >= HARDWOOD_MIN_SPCD,
            "red_maple": name == "Acer rubrum",
            "sugar_maple": name == "Acer saccharum",
            "birch": name.str.startswith("Betula "),
            "aspen": name.isin(["Populus tremuloides", "Populus grandidentata"]),
        }
        plot_ba = np.zeros((len(GROUPS) + 1, len(uniq)))  # total then groups, plot
        order = np.argsort(tm_id)
        pos = order[np.searchsorted(tm_id[order], trees["TM_ID"].to_numpy())]  # plot of each tree
        assert np.array_equal(tm_id[pos], trees["TM_ID"].to_numpy()), "tree plot lookup"
        plot_ba[0] = np.bincount(pos, weights=trees["BA"].to_numpy(), minlength=len(uniq))
        for g, group in enumerate(GROUPS, start=1):
            plot_ba[g] = np.bincount(pos, weights=np.where(member[group], trees["BA"].to_numpy(), 0.0), minlength=len(uniq))

        # check against the attribute table's live basal area, which FIA computes from trees of 1 inch diameter and above
        balive = vat.loc[uniq, "balive"].to_numpy(dtype=np.float64)
        big = trees["DIA"].to_numpy() >= 1.0
        plot_ba_big = np.bincount(pos[big], weights=trees["BA"].to_numpy()[big], minlength=len(uniq))
        ok = balive > 0
        logger.info("TreeMap %d: computed live basal area against the attribute table, median ratio %.4f, 1st %.4f, 99th %.4f over %d plots",
                    edition, np.median(plot_ba_big[ok] / balive[ok]), *np.percentile(plot_ba_big[ok] / balive[ok], [1, 99]), ok.sum())

        # species by basal area summed over all footprint pixels, for the log
        px_per_plot = np.bincount(inv, minlength=len(uniq))
        w = px_per_plot[pos]
        sp = trees.assign(W=trees["BA"] * w).groupby(["SPCD", "SCIENTIFIC_NAME"])["W"].sum().sort_values(ascending=False)
        sp = sp / sp.sum()
        for (spcd, sci), share in sp.head(TOP_SPECIES).items():
            logger.info("TreeMap %d footprint basal-area share: SPCD %d %s %.4f%s", edition, spcd, sci, share,
                        "" if spcd >= HARDWOOD_MIN_SPCD else " (softwood)")

        for g in range(len(GROUPS) + 1):
            ba_sum[g, :, e] = np.bincount(slots, weights=plot_ba[g][inv], minlength=n_cells)

    with np.errstate(divide="ignore", invalid="ignore"):
        share_edition = np.where(ba_sum[0] > 0, ba_sum[1:] / ba_sum[0], np.nan)  # group, cell, edition
    no_ba = ba_sum[0] <= 0
    logger.info("cells without any TreeMap basal area, by edition: %s", dict(zip(args.editions, no_ba.sum(axis=0).tolist())))
    for g, group in enumerate(GROUPS):
        logger.info("%s share: median by edition %s", group,
                    {ed: round(float(np.nanmedian(share_edition[g, :, e])), 4) for e, ed in enumerate(args.editions)})
    # spread between editions, the size of changes that are partly method differences (D23)
    if len(args.editions) > 1:
        for g, group in enumerate(GROUPS):
            spread = np.nanmax(share_edition[g], axis=1) - np.nanmin(share_edition[g], axis=1)
            logger.info("%s share: per-cell range across editions median %.4f, 90th %.4f", group,
                        np.nanmedian(spread), np.nanpercentile(spread, 90))

    # yearly values by D7: the first edition before it, linear between editions, the last edition after it
    eds = np.array(args.editions, dtype=np.float64)
    share_year = np.full((len(GROUPS), n_cells, len(YEARS)), np.nan)  # group, cell, year
    for y, year in enumerate(YEARS):
        if year <= eds[0]:
            share_year[:, :, y] = share_edition[:, :, 0]
        elif year >= eds[-1]:
            share_year[:, :, y] = share_edition[:, :, -1]
        else:
            k = int(np.searchsorted(eds, year, side="right")) - 1
            frac = (year - eds[k]) / (eds[k + 1] - eds[k])
            share_year[:, :, y] = (1 - frac) * share_edition[:, :, k] + frac * share_edition[:, :, k + 1]

    tag = "" if args.editions == EDITIONS else "_test_" + "_".join(map(str, args.editions))
    out = os.path.join(OUT_DIR, f"species{tag}.npz")
    np.savez_compressed(
        out, cell_id=cell_id, groups=np.array(GROUPS), editions=np.array(args.editions), years=np.array(YEARS),
        share_edition=share_edition.astype(np.float32), share_year=share_year.astype(np.float32),
        basal_area_total=ba_sum[0].astype(np.float32), n_footprint_pixels=n_footprint, n_treemap_pixels=n_treemap,
    )
    logger.info("wrote %s; shares are arrays of group, cell, edition or year; basal area total is square feet per acre summed over pixels", out)


if __name__ == "__main__":
    main()
