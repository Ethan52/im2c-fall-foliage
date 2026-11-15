import os
import sys
import time
import logging
import argparse
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests
import geopandas as gpd
from scipy.io import netcdf_file

logger = logging.getLogger(__name__)

# gridMET daily over Maine from the Climatology Lab THREDDS server, see data/README.md
# used to build 2026 weather corrected to Daymet, fitted on the years where both exist
NCSS_URL = "http://thredds.northwestknowledge.net:8080/thredds/ncss/MET"
# file prefix and the name of the data variable inside the file
VARIABLES = {
    "tmmx": "air_temperature",
    "tmmn": "air_temperature",
    "pr": "precipitation_amount",
    "srad": "surface_downwelling_shortwave_flux_in_air",
    "vpd": "mean_vapor_pressure_deficit",
}
YEARS = list(range(2001, 2027))
PAD_DEG = 0.05  # margin around the request region, in degrees
FILL = 32767  # fill of the packed unsigned 16 bit values

# plausible ranges of valid decoded values, units as in gridMET
VALID_RANGE = {
    "tmmx": (200.0, 330.0),  # K
    "tmmn": (200.0, 330.0),  # K
    "pr": (0.0, 500.0),  # mm per day
    "srad": (0.0, 500.0),  # W per square meter, 24 hour mean
    "vpd": (0.0, 10.0),  # kPa
}

MAX_ATTEMPTS = 5

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGION_GEOJSON = os.path.join(REPO_ROOT, "configs", "modis_request_region.geojson")
OUT_DIR = os.path.join(REPO_ROOT, "data", "raw", "gridmet")
RECORD_TSV = os.path.join(OUT_DIR, "download_record.tsv")


def check_file(path: str, prefix: str, year: int):
    '''check one downloaded file for shape, calendar, packing and value range; return a summary dict'''
    # read with scipy directly, since the variable attribute named dimensions breaks the xarray scipy backend
    with netcdf_file(path, "r", mmap=False) as nc:
        var = nc.variables[VARIABLES[prefix]]
        days = nc.variables["day"].data.copy()  # days since 1900-01-01
        lat = nc.variables["lat"].data.copy()
        lon = nc.variables["lon"].data.copy()
        # true dimension order is day, latitude, longitude; the dimensions attribute is not reliable
        assert var.data.shape == (len(days), len(lat), len(lon)), f"shape {var.data.shape} does not match day, lat, lon"
        assert np.all(np.diff(lat) < 0) and np.all(np.diff(lon) > 0), "unexpected latitude or longitude ordering"
        assert np.all(np.diff(days) == 1), "days are not consecutive"
        first = date(1900, 1, 1) + timedelta(days=int(days[0]))
        last = date(1900, 1, 1) + timedelta(days=int(days[-1]))
        assert first == date(year, 1, 1), f"first day {first}"
        num_days_year = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if year < 2026:
            assert len(days) == num_days_year, f"{len(days)} days, expected {num_days_year}"

        scale = float(var.scale_factor)
        offset = float(var.add_offset)
        raw = var.data.view(">u2")  # stored as signed 16 bit with the _Unsigned attribute
        fill = raw == FILL
        valid = raw[~fill].astype(np.float64) * scale + offset
        lo, hi = VALID_RANGE[prefix]
        assert valid.min() >= lo and valid.max() <= hi, f"values {valid.min()} to {valid.max()} outside {lo} to {hi}"
        # 2026 data are partly provisional, the server records which days are final
        slices = {k: str(int(np.ravel(getattr(var, k))[0])) for k in ("last_permanent_slice", "last_provisional_slice", "last_early_slice") if hasattr(var, k)}
        slices.update({k: nc._attributes[k].decode() for k in ("last_permanent_slice", "last_provisional_slice", "last_early_slice") if k in nc._attributes})
        return {
            "num_days": len(days), "num_lat": len(lat), "num_lon": len(lon), "fill_frac": float(fill.mean()),
            "min": float(valid.min()), "max": float(valid.max()), "first": first.isoformat(), "last": last.isoformat(),
            "lon_min": float(lon[0]), "lon_max": float(lon[-1]), "lat_min": float(lat[-1]), "lat_max": float(lat[0]),
            **slices,
        }


def download_one(prefix: str, year: int, bounds: tuple):
    '''download and check one variable and year; return the summary dict with a status field'''
    west, south, east, north = bounds
    path = os.path.join(OUT_DIR, f"gridmet_maine_{prefix}_{year}.nc")
    url = f"{NCSS_URL}/{prefix}/{prefix}_{year}.nc"
    params = {
        "var": VARIABLES[prefix], "north": north, "south": south, "west": west, "east": east, "horizStride": 1,
        "time_start": f"{year}-01-01T00:00:00Z", "time_end": f"{year}-12-31T00:00:00Z", "accept": "netcdf",
    }
    if os.path.exists(path):
        try:
            return {"status": "skipped", "var": prefix, "year": year, "bytes": os.path.getsize(path), "url": url, **check_file(path, prefix, year)}
        except AssertionError as error:
            logger.warning("%s exists but fails the checks (%s), downloading again", path, error)

    part_path = path + ".part"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=900)
            response.raise_for_status()
            with open(part_path, "wb") as f:
                f.write(response.content)
            summary = check_file(part_path, prefix, year)
        except (requests.RequestException, AssertionError, ValueError, OSError, KeyError) as error:
            logger.warning("%s %d attempt %d failed: %s", prefix, year, attempt, error)
            time.sleep(10 * attempt)
            continue
        os.replace(part_path, path)
        logger.info("%s %d done, %d days", prefix, year, summary["num_days"])
        return {"status": "downloaded", "var": prefix, "year": year, "bytes": os.path.getsize(path), "url": url, **summary}

    logger.error("%s %d failed after %d attempts", prefix, year, MAX_ATTEMPTS)
    return {"status": "failed", "var": prefix, "year": year, "bytes": 0, "url": url}


def main():
    parser = argparse.ArgumentParser(description="Download gridMET daily over Maine from the Climatology Lab THREDDS server")
    parser.add_argument("--variables", nargs="+", default=list(VARIABLES), choices=list(VARIABLES))
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    os.makedirs(OUT_DIR, exist_ok=True)
    west, south, east, north = gpd.read_file(REGION_GEOJSON).to_crs(epsg=4326).total_bounds
    bounds = (west - PAD_DEG, south - PAD_DEG, east + PAD_DEG, north + PAD_DEG)
    logger.info("request bounds longitude %.3f to %.3f, latitude %.3f to %.3f", bounds[0], bounds[2], bounds[1], bounds[3])

    results = [download_one(prefix, year, bounds) for prefix in args.variables for year in args.years]

    job = os.environ.get("SLURM_JOB_ID", "none")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    columns = [
        "status", "var", "year", "bytes", "num_days", "num_lat", "num_lon", "fill_frac", "min", "max", "first", "last",
        "lon_min", "lon_max", "lat_min", "lat_max", "last_permanent_slice", "last_provisional_slice", "last_early_slice", "url",
    ]
    with open(RECORD_TSV, "w") as f:
        f.write("\t".join(columns + ["job", "date_utc"]) + "\n")
        for r in results:
            f.write("\t".join([str(r.get(c, "")) for c in columns] + [job, now]) + "\n")
    logger.info("wrote %s", RECORD_TSV)

    failed = [r for r in results if r["status"] == "failed"]
    logger.info(
        "status %s, %d files, %d downloaded, %d already present, %d failed",
        "OK" if not failed else "FAIL", len(results),
        sum(r["status"] == "downloaded" for r in results), sum(r["status"] == "skipped" for r in results), len(failed),
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
