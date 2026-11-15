import os
import sys
import time
import netrc
import logging
import argparse
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import geopandas as gpd
from scipy.io import netcdf_file

logger = logging.getLogger(__name__)

# Daymet V4R1 daily, subset over Maine through NASA Earthdata OPeNDAP, see data/README.md
COLLECTION = "C2532426483-ORNL_CLOUD"
OPENDAP_URL = f"https://opendap.earthdata.nasa.gov/collections/{COLLECTION}/granules"
EARTHDATA_HOST = "urs.earthdata.nasa.gov"
VARIABLES = ["tmax", "tmin", "prcp", "srad", "vp", "dayl"]
YEARS = list(range(2001, 2026))
NUM_DAYS = 365  # Daymet years always have 365 days, leap years drop December 31

# Daymet Lambert conformal conic projection, from the lambert_conformal_conic variable of the granules
DAYMET_CRS = "+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
PAD_M = 2000  # margin around the request region, in meters
FILL = -9999.0

# plausible ranges of valid values, units as in Daymet
VALID_RANGE = {
    "tmax": (-60.0, 50.0),  # degrees C
    "tmin": (-60.0, 50.0),  # degrees C
    "prcp": (0.0, 500.0),  # mm per day
    "srad": (0.0, 1500.0),  # W per square meter, daylight mean
    "vp": (0.0, 10000.0),  # Pa
    "dayl": (0.0, 86400.0),  # seconds per day
}

MAX_ATTEMPTS = 5

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGION_GEOJSON = os.path.join(REPO_ROOT, "configs", "modis_request_region.geojson")
OUT_DIR = os.path.join(REPO_ROOT, "data", "raw", "daymet")
RECORD_TSV = os.path.join(OUT_DIR, "download_record.tsv")


def granule_url(var: str, year: int):
    '''OPeNDAP URL of the North America granule of one variable and year'''
    return f"{OPENDAP_URL}/Daymet_Daily_V4R1.daymet_v4_daily_na_{var}_{year}.nc"


def make_session():
    '''requests session holding the Earthdata credentials in ~/.netrc'''
    credentials = netrc.netrc().authenticators(EARTHDATA_HOST)
    if credentials is None:
        raise RuntimeError(f"no entry for {EARTHDATA_HOST} in ~/.netrc")
    session = requests.Session()
    session.auth = (credentials[0], credentials[2])
    return session


def region_box(session: requests.Session):
    '''index ranges of the Daymet grid covering the request region plus a margin; return y start, y end, x start, x end, inclusive'''
    path = os.path.join(OUT_DIR, "daymet_grid_xy.nc")
    response = session.get(granule_url("tmax", YEARS[-1]) + ".nc?x,y", timeout=300)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    with netcdf_file(path, "r", mmap=False) as nc:
        x = nc.variables["x"].data.copy()  # projection x in meters, increasing
        y = nc.variables["y"].data.copy()  # projection y in meters, decreasing
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) < 0), "unexpected Daymet grid ordering"

    region = gpd.read_file(REGION_GEOJSON).to_crs(DAYMET_CRS)
    xmin, ymin, xmax, ymax = region.total_bounds
    ix = np.where((x >= xmin - PAD_M) & (x <= xmax + PAD_M))[0]
    iy = np.where((y >= ymin - PAD_M) & (y <= ymax + PAD_M))[0]
    logger.info("Daymet box x index %d to %d, y index %d to %d, %d by %d cells", ix[0], ix[-1], iy[0], iy[-1], len(ix), len(iy))
    return int(iy[0]), int(iy[-1]), int(ix[0]), int(ix[-1])


def check_file(path: str, var: str, year: int, shape: tuple):
    '''check one downloaded file against the expected shape, calendar, fill pattern and value range; return a summary dict'''
    with netcdf_file(path, "r", mmap=False) as nc:
        data = nc.variables[var].data  # time, y, x
        days = nc.variables["time"].data.copy()  # days since 1950-01-01 00:00
        yearday = nc.variables["yearday"].data.copy()
        assert data.shape == shape, f"shape {data.shape}, expected {shape}"
        assert data.dtype == np.dtype(">f4"), f"dtype {data.dtype}"
        assert np.all(np.diff(days) == 1), "time steps are not one day"
        assert np.array_equal(yearday, np.arange(1, NUM_DAYS + 1)), "yearday is not 1 to 365"
        first = date(1950, 1, 1) + timedelta(days=int(np.floor(days[0])))
        last = date(1950, 1, 1) + timedelta(days=int(np.floor(days[-1])))
        assert first == date(year, 1, 1), f"first day {first}"

        fill = data == FILL
        # land mask is the same on every day
        assert np.all(fill == fill[0]), "fill pattern changes between days"
        valid = data[~fill]
        lo, hi = VALID_RANGE[var]
        assert np.all(np.isfinite(valid)), "non-finite valid values"
        assert valid.min() >= lo and valid.max() <= hi, f"values {valid.min()} to {valid.max()} outside {lo} to {hi}"
        return {
            "fill_frac": float(fill.mean()), "min": float(valid.min()), "max": float(valid.max()),
            "first": first.isoformat(), "last": last.isoformat(),
        }


def download_one(session: requests.Session, var: str, year: int, box: tuple):
    '''download and check one variable and year; return the summary dict with a status field'''
    iy0, iy1, ix0, ix1 = box
    shape = (NUM_DAYS, iy1 - iy0 + 1, ix1 - ix0 + 1)
    path = os.path.join(OUT_DIR, f"daymet_v4_daily_maine_{var}_{year}.nc")
    url = granule_url(var, year)
    # DAP2 request returning classic NetCDF-3, readable by scipy
    constraint = f"{var}[0:1:{NUM_DAYS - 1}][{iy0}:1:{iy1}][{ix0}:1:{ix1}],x[{ix0}:1:{ix1}],y[{iy0}:1:{iy1}],time,yearday"

    # file from an earlier run that already passes the checks
    if os.path.exists(path):
        try:
            summary = check_file(path, var, year, shape)
            return {"status": "skipped", "var": var, "year": year, "bytes": os.path.getsize(path), "url": url, **summary}
        except AssertionError as error:
            logger.warning("%s exists but fails the checks (%s), downloading again", path, error)

    part_path = path + ".part"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with session.get(url + ".nc?" + constraint, stream=True, timeout=1800) as response:
                response.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            summary = check_file(part_path, var, year, shape)
        except (requests.RequestException, AssertionError, ValueError, OSError) as error:
            logger.warning("%s %d attempt %d failed: %s", var, year, attempt, error)
            time.sleep(15 * attempt)
            continue
        os.replace(part_path, path)
        logger.info("%s %d done, %.0f MB", var, year, os.path.getsize(path) / 1e6)
        return {"status": "downloaded", "var": var, "year": year, "bytes": os.path.getsize(path), "url": url, **summary}

    logger.error("%s %d failed after %d attempts", var, year, MAX_ATTEMPTS)
    return {"status": "failed", "var": var, "year": year, "bytes": 0, "url": url}


def main():
    parser = argparse.ArgumentParser(description="Download Daymet V4R1 daily over Maine through Earthdata OPeNDAP")
    parser.add_argument("--variables", nargs="+", default=VARIABLES, choices=VARIABLES)
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--max_workers", type=int, default=4, help="number of parallel download threads")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    os.makedirs(OUT_DIR, exist_ok=True)
    session = make_session()
    box = region_box(session)
    jobs = [(var, year) for var in args.variables for year in args.years]
    logger.info("%d files to process with %d threads", len(jobs), args.max_workers)

    results = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(download_one, session, var, year, box) for var, year in jobs]
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 10 == 0:
                logger.info("%d of %d files processed, %.0f seconds", i + 1, len(jobs), time.perf_counter() - start)
    assert len(results) == len(jobs), "number of results does not match the number of files"

    job = os.environ.get("SLURM_JOB_ID", "none")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    columns = ["status", "var", "year", "bytes", "fill_frac", "min", "max", "first", "last", "url"]
    # rows from earlier runs for other variables or years are kept
    rows = {}
    if os.path.exists(RECORD_TSV):
        with open(RECORD_TSV) as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                row = dict(zip(header, line.rstrip("\n").split("\t")))
                rows[(row["var"], row["year"])] = row
    for result in results:
        rows[(result["var"], str(result["year"]))] = {**{c: str(result.get(c, "")) for c in columns}, "job": job, "date_utc": now}
    with open(RECORD_TSV, "w") as f:
        f.write("\t".join(columns + ["job", "date_utc"]) + "\n")
        for key in sorted(rows):
            f.write("\t".join(rows[key].get(c, "") for c in columns + ["job", "date_utc"]) + "\n")
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
