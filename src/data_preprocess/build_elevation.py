import os
import sys
import glob
import logging

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

# elevation of each forest cell from NASADEM, decision D7
# a cell's elevation is the mean of all NASADEM pixels inside the 1 km cell
DEM_VALID = (-48, 1592)  # meters, range of the downloaded tile found by verify_downloads.py (V8)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEM_DIR = os.path.join(REPO_ROOT, "data", "raw", "dem", "NASADEM_NC.001_2000001_to_2026319")
GRID_NPZ = os.path.join(REPO_ROOT, "data", "processed", "grid", "analysis_grid.npz")
MASK_NPZ = os.path.join(REPO_ROOT, "data", "processed", "forest_mask", "forest_mask.npz")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "static")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(OUT_DIR, exist_ok=True)

    grid = np.load(GRID_NPZ)
    cell_id = np.load(MASK_NPZ)["forest_cell_id"]  # sorted forest cell IDs, same order as the MODIS and Daymet files
    gt = grid["transform"]  # x pixel size, 0, x of left edge, 0, minus y pixel size, y of top edge
    grid_shape = tuple(grid["shape"])

    paths = glob.glob(os.path.join(DEM_DIR, "NASADEM_NC.001_NASADEM_HGT_*.tif"))
    assert len(paths) == 1, f"expected one NASADEM elevation file, found {paths}"
    with rasterio.open(paths[0]) as src:
        assert src.crs.to_epsg() == 4326 and src.count == 1, "NASADEM CRS or band count"
        dem = src.read(1)  # row, column, meters
        assert dem.dtype == np.int16 and src.nodata == -32768, f"NASADEM dtype {dem.dtype} or nodata {src.nodata}"
        dem_transform, dem_crs = src.transform, src.crs
    has = dem != -32768
    assert dem[has].min() >= DEM_VALID[0] and dem[has].max() <= DEM_VALID[1], "NASADEM value outside the verified range"
    logger.info("NASADEM %s: %d by %d, %.4f of pixels valid", os.path.basename(paths[0]), *dem.shape, has.mean())

    # area average of the 30 m pixels onto the 1 km analysis grid, nodata pixels are left out of the average
    elev_grid = np.full(grid_shape, np.nan, dtype=np.float32)  # grid row, grid column
    reproject(
        source=dem.astype(np.float32), destination=elev_grid, src_transform=dem_transform, src_crs=dem_crs, src_nodata=-32768.0,
        dst_transform=Affine(*gt[:6]), dst_crs=CRS.from_epsg(int(grid["epsg"])), dst_nodata=np.nan, resampling=Resampling.average,
    )
    elevation = elev_grid[grid["row"][cell_id], grid["col"][cell_id]]  # forest cell
    n_missing = int(np.isnan(elevation).sum())
    assert n_missing == 0, f"{n_missing} forest cells have no NASADEM value"
    assert elevation.min() >= DEM_VALID[0] and elevation.max() <= DEM_VALID[1], "cell elevation outside the verified range"
    logger.info("%d forest cells: elevation median %.0f m, 10th %.0f m, 90th %.0f m, min %.0f m, max %.0f m",
                len(cell_id), np.median(elevation), *np.percentile(elevation, [10, 90]), elevation.min(), elevation.max())

    out = os.path.join(OUT_DIR, "elevation.npz")
    np.savez_compressed(out, cell_id=cell_id, elevation_m=elevation)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
