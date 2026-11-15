import os
import sys
import json
import time
import netrc
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import shapely
import geopandas as gpd

logger = logging.getLogger(__name__)

API_URL = "https://appeears.earthdatacloud.nasa.gov/api"
EARTHDATA_HOST = "urs.earthdata.nasa.gov"

# AppEEARS request, see data/README.md for the role of each layer
TASK_NAME = "maine-foliage"
PRODUCT = "MOD09A1.061"
LAYERS = [
    "sur_refl_b01",
    "sur_refl_b02",
    "sur_refl_qc_500m",
    "sur_refl_state_500m",
    "sur_refl_day_of_year",
]
START_DATE = "01-01-2001"  # month, day, year as AppEEARS requires
END_DATE = "12-31-2025"
PROJECTION = "native"  # MODIS sinusoidal
FORMAT = "geotiff"

# request region, the Maine boundary widened so pixels on the coast and border are kept
BOUNDARY_EPSG = 26919  # NAD83 UTM zone 19N, units of meters
BUFFER_M = 5000
SIMPLIFY_M = 500

POLL_SECONDS = 600
MAX_ATTEMPTS = 5
CHUNK_BYTES = 1024 * 1024

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BOUNDARY_SHP = os.path.join(
    REPO_ROOT, "data", "raw", "maine_boundary_polygon", "Maine_State_Boundary_Polygon_Feature.shp"
)
REGION_GEOJSON = os.path.join(REPO_ROOT, "configs", "modis_request_region.geojson")
REQUEST_JSON = os.path.join(REPO_ROOT, "configs", "modis_appeears_request.json")
OUT_DIR = os.path.join(REPO_ROOT, "data", "raw", "modis")
TASK_JSON = os.path.join(OUT_DIR, "task.json")
BUNDLE_JSON = os.path.join(OUT_DIR, "bundle.json")
RECORD_TSV = os.path.join(OUT_DIR, "download_record.tsv")


def build_region():
    '''dissolve, buffer and simplify the Maine boundary, then write it as one GeoJSON feature in longitude and latitude'''
    boundary = gpd.read_file(BOUNDARY_SHP)
    assert boundary.crs.to_epsg() == BOUNDARY_EPSG, f"unexpected boundary CRS {boundary.crs}"
    land = boundary[boundary["LAND"] == "y"]
    logger.info("boundary records %d, land records %d", len(boundary), len(land))

    # simplify before the buffer, since buffering the full resolution coastline needs over 16 GB of memory
    # each simplification moves the outline by at most the tolerance, which is much less than the buffer
    maine = land.geometry.simplify(SIMPLIFY_M).union_all()
    region = maine.buffer(BUFFER_M).simplify(SIMPLIFY_M)
    # check against every original polygon at full resolution
    assert shapely.contains(region, land.geometry.values).all(), "region does not contain the Maine boundary"

    num_parts = len(region.geoms) if region.geom_type == "MultiPolygon" else 1
    logger.info(
        "Maine area %.1f square km, %d vertices at full resolution",
        land.geometry.area.sum() / 1e6, shapely.get_num_coordinates(land.geometry.values).sum(),
    )
    logger.info(
        "region area %.1f square km, %d vertices, %d parts",
        region.area / 1e6, shapely.get_num_coordinates(region), num_parts,
    )

    region_lonlat = gpd.GeoSeries([region], crs=boundary.crs).to_crs(epsg=4326)
    lon_min, lat_min, lon_max, lat_max = region_lonlat.total_bounds
    logger.info("region longitude %.4f to %.4f, latitude %.4f to %.4f", lon_min, lon_max, lat_min, lat_max)

    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "maine_buffered", "buffer_m": BUFFER_M, "simplify_m": SIMPLIFY_M},
                "geometry": shapely.geometry.mapping(region_lonlat.iloc[0]),
            }
        ],
    }
    with open(REGION_GEOJSON, "w") as f:
        json.dump(geo, f, indent=1)
    logger.info("wrote %s", REGION_GEOJSON)


def login():
    '''log in to AppEEARS with the Earthdata credentials in ~/.netrc and return request headers holding the token'''
    credentials = netrc.netrc().authenticators(EARTHDATA_HOST)
    if credentials is None:
        raise RuntimeError(f"no entry for {EARTHDATA_HOST} in ~/.netrc")
    response = requests.post(f"{API_URL}/login", auth=(credentials[0], credentials[2]), timeout=60)
    response.raise_for_status()
    return {"Authorization": f"Bearer {response.json()['token']}"}


def logout(headers: dict):
    '''end the AppEEARS session of the given token'''
    response = requests.post(f"{API_URL}/logout", headers=headers, timeout=60)
    logger.info("logout status %d", response.status_code)


def submit_task():
    '''submit the area request to AppEEARS and record the task ID in data/raw/modis/task.json'''
    if os.path.exists(TASK_JSON):
        raise RuntimeError(f"{TASK_JSON} exists, so the task was already submitted; delete it to submit again")
    with open(REGION_GEOJSON) as f:
        geo = json.load(f)

    request = {
        "task_type": "area",
        "task_name": TASK_NAME,
        "params": {
            "dates": [{"startDate": START_DATE, "endDate": END_DATE}],
            "layers": [{"product": PRODUCT, "layer": layer} for layer in LAYERS],
            "geo": geo,
            "output": {"format": {"type": FORMAT}, "projection": PROJECTION},
        },
    }

    headers = login()
    response = requests.post(f"{API_URL}/task", json=request, headers=headers, timeout=120)
    logout(headers)
    if not response.ok:
        logger.error("submission rejected: %s", response.text)
    response.raise_for_status()
    task_id = response.json()["task_id"]
    logger.info("submitted task %s", task_id)

    submitted = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(TASK_JSON, "w") as f:
        json.dump({"task_id": task_id, "submitted_utc": submitted, "request": request}, f, indent=1)
    with open(REQUEST_JSON, "w") as f:
        json.dump(request, f, indent=1)
    logger.info("wrote %s and %s", TASK_JSON, REQUEST_JSON)


def sha256_of(path: str):
    '''sha256 hex digest of a local file'''
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(entry: dict, task_id: str, headers: dict):
    '''download one bundle file and check its size and sha256 against the listing; return skipped, downloaded or failed'''
    name = entry["file_name"]
    expected_sha256 = entry["sha256"].lower()
    path = os.path.join(OUT_DIR, name)

    # file from an earlier run that already matches the listing
    if os.path.exists(path) and os.path.getsize(path) == entry["file_size"] and sha256_of(path) == expected_sha256:
        return "skipped"

    url = f"{API_URL}/bundle/{task_id}/{entry['file_id']}"
    part_path = path + ".part"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        digest = hashlib.sha256()
        try:
            # requests drops the token when AppEEARS redirects to the S3 download link
            with requests.get(url, headers=headers, stream=True, timeout=300) as response:
                response.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        f.write(chunk)
                        digest.update(chunk)
        except requests.RequestException as error:
            logger.warning("%s attempt %d failed: %s", name, attempt, error)
            # longer wait after each failure in case AppEEARS is limiting the request rate
            time.sleep(10 * attempt)
            continue
        if digest.hexdigest() == expected_sha256 and os.path.getsize(part_path) == entry["file_size"]:
            os.replace(part_path, path)
            return "downloaded"
        logger.warning("%s attempt %d does not match the listed size or sha256", name, attempt)
        os.remove(part_path)

    logger.error("%s failed after %d attempts", name, MAX_ATTEMPTS)
    return "failed"


def download_task(task_id: str, wait: bool, max_workers: int):
    '''wait for the task to finish unless wait is False, then download and verify every file in its bundle; return the number of failures'''
    headers = login()

    # wait for AppEEARS to finish processing
    status = "unknown" if wait else "done"
    while status != "done":
        try:
            response = requests.get(f"{API_URL}/task/{task_id}", headers=headers, timeout=60)
            if response.status_code in (401, 403):
                # tokens expire about 48 hours after login
                logger.info("token rejected, logging in again")
                headers = login()
                status = "unknown"
            else:
                response.raise_for_status()
                status = response.json()["status"]
        except requests.RequestException as error:
            logger.warning("status request failed: %s", error)
            status = "unknown"
        logger.info("task %s status %s", task_id, status)
        if status == "error":
            raise RuntimeError(f"task {task_id} failed on the AppEEARS side: {response.text}")
        if status != "done":
            time.sleep(POLL_SECONDS)

    # new token so that it lasts through the downloads
    headers = login()
    response = requests.get(f"{API_URL}/bundle/{task_id}", headers=headers, timeout=120)
    response.raise_for_status()
    bundle = response.json()
    with open(BUNDLE_JSON, "w") as f:
        json.dump(bundle, f, indent=1)
    files = bundle["files"]
    total_bytes = sum(entry["file_size"] for entry in files)
    logger.info("bundle has %d files, %.3f GB, listing written to %s", len(files), total_bytes / 1e9, BUNDLE_JSON)

    for directory in {os.path.dirname(os.path.join(OUT_DIR, entry["file_name"])) for entry in files}:
        os.makedirs(directory, exist_ok=True)

    # downloads in parallel threads, since most of the time per file is waiting on the network
    logger.info("downloading with %d threads", max_workers)
    outcomes = {}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, entry, task_id, headers): entry["file_name"] for entry in files}
        for i, future in enumerate(as_completed(futures)):
            outcomes[futures[future]] = future.result()
            if (i + 1) % 250 == 0:
                logger.info("%d of %d files processed, %.0f seconds", i + 1, len(files), time.perf_counter() - start)
    logout(headers)

    num_downloaded = sum(1 for outcome in outcomes.values() if outcome == "downloaded")
    num_skipped = sum(1 for outcome in outcomes.values() if outcome == "skipped")
    failed = sorted(name for name, outcome in outcomes.items() if outcome == "failed")
    assert len(outcomes) == len(files), "number of outcomes does not match the number of bundle files"

    status = "OK" if not failed else "FAIL"
    logger.info(
        "status %s, %d files, %d downloaded, %d already present, %d failed",
        status, len(files), num_downloaded, num_skipped, len(failed),
    )
    for name in failed:
        logger.error("failed file %s", name)

    job = os.environ.get("SLURM_JOB_ID", "none")
    date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(RECORD_TSV, "w") as f:
        f.write("status\ttask_id\tfiles\tbytes\tdownloaded\talready_present\tfailed\tjob\tdate_utc\n")
        f.write(
            f"{status}\t{task_id}\t{len(files)}\t{total_bytes}\t{num_downloaded}\t{num_skipped}\t{len(failed)}\t{job}\t{date}\n"
        )
    logger.info("wrote %s", RECORD_TSV)
    return len(failed)


def main():
    parser = argparse.ArgumentParser(description="Build the region for, submit, and download the MOD09A1.061 AppEEARS request")
    parser.add_argument("--step", required=True, choices=["region", "submit", "download"])
    parser.add_argument("--task_id", default=None, help="AppEEARS task ID, read from data/raw/modis/task.json by default")
    parser.add_argument(
        "--no_wait", action="store_true", help="download the current bundle listing without waiting for the task status done"
    )
    parser.add_argument("--max_workers", type=int, default=16, help="number of parallel download threads")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    if args.step == "region":
        build_region()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.step == "submit":
        submit_task()
        return

    task_id = args.task_id
    if task_id is None:
        with open(TASK_JSON) as f:
            task_id = json.load(f)["task_id"]
    num_failed = download_task(task_id, not args.no_wait, args.max_workers)
    if num_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
