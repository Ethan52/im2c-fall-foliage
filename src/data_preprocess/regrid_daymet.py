import os
import sys
import time
import logging
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyproj
from scipy.io import netcdf_file

logger = logging.getLogger(__name__)

# daily Daymet on the 1 km forest cells, decision D19: each analysis cell takes the Daymet cell whose center is nearest
VARIABLES = ["tmax", "tmin", "prcp", "srad", "vp", "dayl"]
YEARS = list(range(2001, 2026))
FILL = -9999.0
NUM_DAYS = 365  # Daymet years always have 365 days, leap years drop December 31
DAYMET_RES = 1000.0  # meters
# Daymet Lambert conformal conic projection, as in src/data_download/dataload_daymet.py
DAYMET_CRS = "+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
# January 2025 mean tmax at the Daymet cells nearest the site towns, from src/checks/verify_downloads.py (V8)
SITE_JAN_TMAX_2025 = {"fort_kent": -6.40, "rangeley": -6.82, "acadia_bar_harbor": -1.05}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DAYMET_DIR = os.path.join(REPO_ROOT, "data", "raw", "daymet")
GRID_NPZ = os.path.join(REPO_ROOT, "data", "processed", "grid", "analysis_grid.npz")
MASK_NPZ = os.path.join(REPO_ROOT, "data", "processed", "forest_mask", "forest_mask.npz")
SITES_CSV = os.path.join(REPO_ROOT, "configs", "sites.csv")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "daymet")


def raw_path(var: str, year: int):
    '''path of one downloaded Daymet variable and year'''
    return os.path.join(DAYMET_DIR, f"daymet_v4_daily_maine_{var}_{year}.nc")


def main():
    parser = argparse.ArgumentParser(description="Put daily Daymet on the 1 km forest cells by nearest Daymet cell, one file per year")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(OUT_DIR, exist_ok=True)

    grid = np.load(GRID_NPZ)
    cell_id = np.load(MASK_NPZ)["forest_cell_id"]  # sorted forest cell IDs, same order as the MODIS and brownness files
    n_cells = len(cell_id)

    # Daymet grid coordinates from the subset files, identical in every file
    with netcdf_file(raw_path("tmax", YEARS[0]), "r", mmap=False) as nc:
        dx = nc.variables["x"].data.astype(np.float64)  # increasing, meters
        dy = nc.variables["y"].data.astype(np.float64)  # decreasing, meters
    assert np.allclose(np.diff(dx), DAYMET_RES) and np.allclose(np.diff(dy), -DAYMET_RES), "Daymet grid is not regular 1 km"

    # nearest Daymet cell to each forest cell center
    cx, cy = pyproj.Transformer.from_crs(f"EPSG:{int(grid['epsg'])}", DAYMET_CRS, always_xy=True).transform(grid["x"][cell_id], grid["y"][cell_id])
    ix = np.rint((cx - dx[0]) / DAYMET_RES).astype(np.int64)
    iy = np.rint((dy[0] - cy) / DAYMET_RES).astype(np.int64)
    assert ix.min() >= 0 and ix.max() < len(dx) and iy.min() >= 0 and iy.max() < len(dy), "a forest cell is outside the Daymet subset"
    dist = np.hypot(dx[ix] - cx, dy[iy] - cy)
    assert dist.max() <= DAYMET_RES / np.sqrt(2) + 1e-6, "nearest Daymet center farther than half a cell diagonal"
    logger.info("%d forest cells, nearest Daymet center distance median %.0f m, max %.0f m, %d distinct Daymet cells",
                n_cells, np.median(dist), dist.max(), len(np.unique(iy * len(dx) + ix)))

    for year in args.years:
        start = time.perf_counter()
        out = {}
        dates = None
        for var in VARIABLES:
            with netcdf_file(raw_path(var, year), "r", mmap=False) as nc:
                assert np.array_equal(nc.variables["x"].data, dx) and np.array_equal(nc.variables["y"].data, dy), f"{var} {year} grid differs"
                days = nc.variables["time"].data.copy()  # days since 1950-01-01 00:00, at noon
                yearday = nc.variables["yearday"].data.copy()
                data = nc.variables[var].data  # day, y, x
                assert data.shape == (NUM_DAYS, len(dy), len(dx)), f"{var} {year} shape {data.shape}"
                values = data[:, iy, ix].T.astype(np.float32)  # cell, day
            d = np.array([np.datetime64(date(1950, 1, 1) + timedelta(days=int(np.floor(t)))) for t in days], dtype="datetime64[D]")
            assert np.array_equal(yearday, (d - np.datetime64(f"{year}-01-01")).astype(int) + 1), f"{var} {year} yearday does not match the dates"
            if dates is None:
                dates = d
            assert np.array_equal(d, dates), f"{var} {year} dates differ from {VARIABLES[0]}"
            n_fill = int((values == FILL).sum())
            assert n_fill == 0, f"{var} {year}: {n_fill} cell-days fall on Daymet fill"
            out[var] = values
        np.savez_compressed(os.path.join(OUT_DIR, f"daymet_{year}.npz"), cell_id=cell_id, date=dates, daymet_ix=ix, daymet_iy=iy, **out)
        logger.info("%d: %d days, wrote daymet_%d.npz, %.0f seconds", year, len(dates), year, time.perf_counter() - start)

        # site check against the January 2025 values printed by verify_downloads.py, using the forest cell nearest each town
        if year == 2025:
            sites = pd.read_csv(SITES_CSV)
            for _, s in sites.iterrows():
                j = np.argmin(np.hypot(grid["x"][cell_id] - s["easting_m"], grid["y"][cell_id] - s["northing_m"]))
                jan = out["tmax"][j, :31].mean()
                logger.info("%s: nearest forest cell %d at %.1f km, January 2025 mean tmax %.2f C, town Daymet cell %.2f C",
                            s["site_name"], cell_id[j], np.hypot(grid["x"][cell_id[j]] - s["easting_m"], grid["y"][cell_id[j]] - s["northing_m"]) / 1000,
                            jan, SITE_JAN_TMAX_2025[s["site_id"]])


if __name__ == "__main__":
    main()
