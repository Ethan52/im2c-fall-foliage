import os
import sys
import time
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# public ARCO ERA5 store on Google Cloud, read anonymously
ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# hourly near-surface variables used for the ERA5 climatology (delta method) and the ERA5 to Daymet linear map
VARIABLES = {
    "2m_temperature": "t2m",  # K
    "2m_dewpoint_temperature": "d2m",  # K
    "total_precipitation": "tp",  # m, accumulated over the hour ending at the time stamp
    "mean_surface_downward_short_wave_radiation_flux": "ssrd",  # W/m2, mean over the hour ending at the time stamp
}

# Maine box with a margin of at least one 2.8 degree model cell, longitudes in degrees east
LAT_MIN, LAT_MAX = 40.0, 50.0
LON_MIN, LON_MAX = 285.0, 296.0

# season window with a margin of 16 days on each side for the 31 day climatology window
SEASON_START = "07-15"
SEASON_END = "12-16"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "data", "raw", "era5")


def main():
    parser = argparse.ArgumentParser(description="Download hourly ERA5 near-surface variables over Maine for one year")
    parser.add_argument("--year", type=int, default=None, help="year to download, defaults to 2000 plus SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    year = args.year if args.year is not None else 2000 + int(os.environ["SLURM_ARRAY_TASK_ID"])
    start, end = f"{year}-{SEASON_START}T00", f"{year}-{SEASON_END}T23"
    path = os.path.join(OUT_DIR, f"era5_maine_hourly_{year}.nc")
    os.makedirs(OUT_DIR, exist_ok=True)

    era5 = xr.open_zarr(ERA5_URL, chunks=None, storage_options=dict(token="anon"))
    box = era5[list(VARIABLES)].sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))

    # one month at a time to bound memory and log progress
    t0 = time.perf_counter()
    months = []
    for month_start in np.arange(np.datetime64(start[:7]), np.datetime64(end[:7]) + 1):
        month_end = (month_start + 1).astype("datetime64[h]") - np.timedelta64(1, "h")
        lo = max(np.datetime64(start), month_start.astype("datetime64[h]"))
        hi = min(np.datetime64(end), month_end)
        months.append(box.sel(time=slice(lo, hi)).load())
        logger.info("%d: read %s to %s, %.0f s", year, lo, hi, time.perf_counter() - t0)
    ds = xr.concat(months, dim="time").rename(VARIABLES)

    # checks on the time axis, the grid and the values
    hours = (ds.time.values - ds.time.values[0]) / np.timedelta64(1, "h")
    assert np.array_equal(hours, np.arange(ds.sizes["time"])), "time steps are not consecutive hours"
    assert ds.sizes["latitude"] == 41 and ds.sizes["longitude"] == 45, f"unexpected box shape {dict(ds.sizes)}"
    for name in VARIABLES.values():
        num_nan = int(ds[name].isnull().sum())
        assert num_nan == 0, f"{name} has {num_nan} missing values"
    assert float(ds["t2m"].min()) > 200 and float(ds["t2m"].max()) < 330, "2 m temperature outside 200 to 330 K"
    # ERA5 accumulations and mean fluxes can hold tiny negative values from packing; they are logged and kept unclipped
    for name, tolerance in (("tp", -1e-6), ("ssrd", -1.0)):
        num_negative = int((ds[name] < 0).sum())
        logger.info("%s minimum %.3g, %d negative values of %d", name, float(ds[name].min()), num_negative, ds[name].size)
        assert float(ds[name].min()) >= tolerance, f"{name} has values below {tolerance}"

    ds.attrs["source"] = ERA5_URL
    ds.attrs["downloaded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ds.attrs["slurm_job"] = os.environ.get("SLURM_JOB_ID", "none")
    encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 4} for name in VARIABLES.values()}
    ds.to_netcdf(path, encoding=encoding)
    logger.info("wrote %s, dims %s, %.1f MB", path, dict(ds.sizes), os.path.getsize(path) / 1e6)


if __name__ == "__main__":
    main()
