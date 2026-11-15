import os
import sys
import time
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import requests
import rasterio
import rasterio.merge
import geopandas as gpd
from rasterio.io import MemoryFile

logger = logging.getLogger(__name__)

# NLCD Land Cover L48 over Maine from the MRLC GeoServer WCS, see data/README.md
WCS_URL = "https://www.mrlc.gov/geoserver/ows"
YEARS = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]
NLCD_EPSG = 5070  # CONUS Albers equal area, meters
RES_M = 30
GRID_OFFSET_M = 15  # NLCD pixel edges lie at 15 m modulo 30 m in both axes
TILE_M = 99990  # tile size in meters, a multiple of 30
PAD_M = 2010  # margin around the request region, a multiple of 30
NODATA = 0
# NLCD legend classes, 41 deciduous forest, 42 evergreen forest, 43 mixed forest
VALID_CLASSES = {11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95}

MAX_ATTEMPTS = 5

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGION_GEOJSON = os.path.join(REPO_ROOT, "configs", "modis_request_region.geojson")
OUT_DIR = os.path.join(REPO_ROOT, "data", "raw", "nlcd")
RECORD_TSV = os.path.join(OUT_DIR, "download_record.tsv")


def snap(value: float, down: bool):
    '''move a coordinate onto the NLCD pixel edges, downward or upward'''
    k = (value - GRID_OFFSET_M) / RES_M
    return GRID_OFFSET_M + RES_M * (np.floor(k) if down else np.ceil(k))


def get_tile(year: int, x0: float, x1: float, y0: float, y1: float):
    '''request one tile of one year from the WCS and return the GeoTIFF bytes'''
    params = [
        ("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
        ("coverageId", f"mrlc_download__NLCD_{year}_Land_Cover_L48"),
        ("subset", f"X({x0},{x1})"), ("subset", f"Y({y0},{y1})"), ("format", "image/geotiff"),
    ]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(WCS_URL, params=params, timeout=600)
            response.raise_for_status()
            if not response.headers.get("content-type", "").startswith("image/tiff") and not response.headers.get("content-type", "").startswith("image/geotiff"):
                raise ValueError(f"unexpected content type {response.headers.get('content-type')}: {response.text[:300]}")
            return response.content
        except (requests.RequestException, ValueError) as error:
            logger.warning("%d tile x %.0f y %.0f attempt %d failed: %s", year, x0, y0, attempt, error)
            time.sleep(10 * attempt)
    raise RuntimeError(f"{year} tile x {x0} y {y0} failed after {MAX_ATTEMPTS} attempts")


def download_year(year: int, bounds: tuple):
    '''download all tiles of one year, merge them, check the result and write one GeoTIFF; return a summary dict'''
    xmin, ymin, xmax, ymax = bounds
    path = os.path.join(OUT_DIR, f"nlcd_{year}_land_cover_maine.tif")
    xs = np.arange(xmin, xmax, TILE_M)
    ys = np.arange(ymin, ymax, TILE_M)
    files = []
    for x0 in xs:
        for y0 in ys:
            files.append(MemoryFile(get_tile(year, x0, min(x0 + TILE_M, xmax), y0, min(y0 + TILE_M, ymax))))
    datasets = [f.open() for f in files]
    for ds in datasets:
        assert ds.crs.to_epsg() == NLCD_EPSG, f"tile CRS {ds.crs}"
        assert ds.res == (RES_M, RES_M), f"tile resolution {ds.res}"
        assert ds.transform.c % RES_M == GRID_OFFSET_M and ds.transform.f % RES_M == GRID_OFFSET_M, "tile off the NLCD grid"
    # tiles share edges on the same 30 m grid, so overlapping pixels hold identical values
    mosaic, transform = rasterio.merge.merge(datasets, bounds=bounds, nodata=NODATA)  # band, row, column
    for ds, f in zip(datasets, files):
        ds.close()
        f.close()
    assert mosaic.shape[0] == 1 and mosaic.dtype == np.uint8, f"mosaic shape {mosaic.shape} dtype {mosaic.dtype}"
    classes = set(np.unique(mosaic).tolist()) - {NODATA}
    assert classes <= VALID_CLASSES, f"unexpected classes {sorted(classes - VALID_CLASSES)}"

    profile = {
        "driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "count": 1, "dtype": "uint8",
        "crs": f"EPSG:{NLCD_EPSG}", "transform": transform, "nodata": NODATA,
        "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512,
    }
    with rasterio.open(path + ".part", "w", **profile) as dst:
        dst.write(mosaic)
    os.replace(path + ".part", path)

    counts = {c: int((mosaic == c).sum()) for c in (41, 42, 43)}
    logger.info("%d written, %d by %d pixels, %d tiles", year, mosaic.shape[2], mosaic.shape[1], len(files))
    return {
        "status": "downloaded", "year": year, "file": os.path.basename(path), "bytes": os.path.getsize(path),
        "width": mosaic.shape[2], "height": mosaic.shape[1], "tiles": len(files),
        "nodata_frac": float((mosaic == NODATA).mean()), "classes": " ".join(str(c) for c in sorted(classes)),
        "pixels_41": counts[41], "pixels_42": counts[42], "pixels_43": counts[43],
    }


def main():
    parser = argparse.ArgumentParser(description="Download NLCD Land Cover L48 over Maine from the MRLC WCS")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS, choices=YEARS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    os.makedirs(OUT_DIR, exist_ok=True)
    xmin, ymin, xmax, ymax = gpd.read_file(REGION_GEOJSON).to_crs(epsg=NLCD_EPSG).total_bounds
    bounds = (snap(xmin - PAD_M, True), snap(ymin - PAD_M, True), snap(xmax + PAD_M, False), snap(ymax + PAD_M, False))
    logger.info("request bounds in EPSG:%d: %s, %.0f by %.0f km", NLCD_EPSG, bounds, (bounds[2] - bounds[0]) / 1e3, (bounds[3] - bounds[1]) / 1e3)

    results = []
    for year in args.years:
        try:
            results.append(download_year(year, bounds))
        except (RuntimeError, AssertionError) as error:
            logger.error("%d failed: %s", year, error)
            results.append({"status": "failed", "year": year})

    job = os.environ.get("SLURM_JOB_ID", "none")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    columns = ["status", "year", "file", "bytes", "width", "height", "tiles", "nodata_frac", "classes", "pixels_41", "pixels_42", "pixels_43"]
    with open(RECORD_TSV, "w") as f:
        f.write("\t".join(columns + ["job", "date_utc"]) + "\n")
        for r in results:
            f.write("\t".join([str(r.get(c, "")) for c in columns] + [job, now]) + "\n")
    logger.info("wrote %s", RECORD_TSV)

    failed = [r for r in results if r["status"] == "failed"]
    logger.info("status %s, %d years, %d failed", "OK" if not failed else "FAIL", len(results), len(failed))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
