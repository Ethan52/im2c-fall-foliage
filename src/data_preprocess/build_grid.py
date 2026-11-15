import os
import sys
import logging

import numpy as np
import pandas as pd
import pyproj
import rasterio.features
import geopandas as gpd
from rasterio.transform import from_origin

logger = logging.getLogger(__name__)

# 1 km analysis grid over Maine, step 0 of references/preprocessing_plan.md
GRID_EPSG = 26919  # NAD83 UTM zone 19N, meters
CELL_M = 1000
# Maine land area of the LAND == 'y' polygons, from the build_region log of 2026-11-14 (slurm_output/data_download/region-*.out)
MAINE_LAND_KM2 = 84012.9
AREA_TOL = 0.03

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BOUNDARY_SHP = os.path.join(
    REPO_ROOT, "data", "raw", "maine_boundary_polygon", "Maine_State_Boundary_Polygon_Feature.shp"
)
SITES_CSV = os.path.join(REPO_ROOT, "configs", "sites.csv")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "grid")
OUT_NPZ = os.path.join(OUT_DIR, "analysis_grid.npz")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(OUT_DIR, exist_ok=True)

    boundary = gpd.read_file(BOUNDARY_SHP)
    assert boundary.crs.to_epsg() == GRID_EPSG, f"unexpected boundary CRS {boundary.crs}"
    land = boundary[boundary["LAND"] == "y"]
    logger.info("boundary records %d, land records %d", len(boundary), len(land))

    # grid edges on multiples of 1000 m that enclose all land
    xmin, ymin, xmax, ymax = land.total_bounds
    x0 = np.floor(xmin / CELL_M) * CELL_M
    y1 = np.ceil(ymax / CELL_M) * CELL_M
    ncol = int(np.ceil(xmax / CELL_M) - np.floor(xmin / CELL_M))
    nrow = int(np.ceil(ymax / CELL_M) - np.floor(ymin / CELL_M))
    transform = from_origin(x0, y1, CELL_M, CELL_M)
    logger.info("grid %d rows by %d columns, upper left corner x %.0f y %.0f", nrow, ncol, x0, y1)

    # without all_touched, a cell is marked only when its center lies inside a land polygon
    inside = rasterio.features.rasterize(
        ((geom, 1) for geom in land.geometry), out_shape=(nrow, ncol), transform=transform, fill=0, dtype="uint8"
    ).astype(bool)  # row, column

    rows, cols = np.nonzero(inside)  # row-major order
    n_cells = len(rows)
    id_raster = np.full((nrow, ncol), -1, dtype=np.int32)
    id_raster[rows, cols] = np.arange(n_cells, dtype=np.int32)
    x = x0 + (cols + 0.5) * CELL_M  # cell center easting
    y = y1 - (rows + 0.5) * CELL_M  # cell center northing
    lon, lat = pyproj.Transformer.from_crs(GRID_EPSG, 4326, always_xy=True).transform(x, y)

    rel = n_cells / MAINE_LAND_KM2 - 1
    logger.info("%d cells, %.0f square km, %.2f%% from the Maine land area of %.1f square km", n_cells, n_cells, 100 * rel, MAINE_LAND_KM2)
    assert abs(rel) <= AREA_TOL, "cell count does not match the Maine land area"

    # each site town must fall in a grid cell
    sites = pd.read_csv(SITES_CSV)
    for _, site in sites.iterrows():
        r = int((y1 - site["northing_m"]) // CELL_M)
        c = int((site["easting_m"] - x0) // CELL_M)
        assert 0 <= r < nrow and 0 <= c < ncol and id_raster[r, c] >= 0, f"{site['site_name']} is not in a grid cell"
        logger.info("%s is in cell %d", site["site_name"], id_raster[r, c])

    np.savez_compressed(
        OUT_NPZ,
        cell_id=np.arange(n_cells, dtype=np.int32), x=x, y=y, lon=lon, lat=lat,
        row=rows.astype(np.int32), col=cols.astype(np.int32),
        transform=np.array(tuple(transform)[:6]), shape=np.array([nrow, ncol]), id_raster=id_raster,
        epsg=np.array(GRID_EPSG), cell_m=np.array(CELL_M),
    )
    logger.info("wrote %s", OUT_NPZ)


if __name__ == "__main__":
    main()
