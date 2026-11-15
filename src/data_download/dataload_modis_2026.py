import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from dataload_modis import API_URL, login, logout, sha256_of

logger = logging.getLogger(__name__)

# AppEEARS request for 2026 MODIS, same product, layers, region and format as dataload_modis.py, see data/README.md
# used only to measure the observed 2026 onset, not for training or the 2024 and 2025 assimilation tests
TASK_NAME = "maine-foliage-2026"
PRODUCT = "MOD09A1.061"
LAYERS = [
    "sur_refl_b01",
    "sur_refl_b02",
    "sur_refl_qc_500m",
    "sur_refl_state_500m",
    "sur_refl_day_of_year",
]
START_DATE = "01-01-2026"  # month, day, year as AppEEARS requires
END_DATE = "11-15-2026"
PROJECTION = "native"  # MODIS sinusoidal, same grid as the 2001 to 2025 download
FORMAT = "geotiff"

POLL_SECONDS = 60
MAX_ATTEMPTS = 5
CHUNK_BYTES = 1024 * 1024

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGION_GEOJSON = os.path.join(REPO_ROOT, "configs", "modis_request_region.geojson")
REQUEST_JSON = os.path.join(REPO_ROOT, "configs", "modis_2026_appeears_request.json")
OUT_DIR = os.path.join(REPO_ROOT, "data", "raw", "modis_2026")
TASK_JSON = os.path.join(OUT_DIR, "task.json")
BUNDLE_JSON = os.path.join(OUT_DIR, "bundle.json")
RECORD_TSV = os.path.join(OUT_DIR, "download_record.tsv")


def submit_task():
    '''submit the area request to AppEEARS and record the task ID in data/raw/modis_2026/task.json'''
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


def download_one(entry: dict, task_id: str, headers: dict):
    '''download one bundle file and check its size and sha256 against the listing; return skipped, downloaded or failed'''
    name = entry["file_name"]
    expected_sha256 = entry["sha256"].lower()
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) == entry["file_size"] and sha256_of(path) == expected_sha256:
        return "skipped"
    url = f"{API_URL}/bundle/{task_id}/{entry['file_id']}"
    part_path = path + ".part"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=300) as response:
                response.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        f.write(chunk)
        except requests.RequestException as error:
            logger.warning("%s attempt %d failed: %s", name, attempt, error)
            time.sleep(10 * attempt)
            continue
        if os.path.getsize(part_path) == entry["file_size"] and sha256_of(part_path) == expected_sha256:
            os.replace(part_path, path)
            return "downloaded"
        logger.warning("%s attempt %d does not match the listed size or sha256", name, attempt)
        os.remove(part_path)
    logger.error("%s failed after %d attempts", name, MAX_ATTEMPTS)
    return "failed"


def download_task(task_id: str, max_workers: int):
    '''wait for the task to finish, then download and verify every file in its bundle; return the number of failures'''
    headers = login()
    status = "unknown"
    while status != "done":
        try:
            response = requests.get(f"{API_URL}/task/{task_id}", headers=headers, timeout=60)
            if response.status_code in (401, 403):
                # a logout from any other session of the same account ends this token too
                logger.info("token rejected, logging in again")
                headers = login()
                continue
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
    logger.info("bundle has %d files, %.3f GB", len(files), total_bytes / 1e9)
    for directory in {os.path.dirname(os.path.join(OUT_DIR, entry["file_name"])) for entry in files}:
        os.makedirs(directory, exist_ok=True)

    outcomes = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, entry, task_id, headers): entry["file_name"] for entry in files}
        for future in as_completed(futures):
            outcomes[futures[future]] = future.result()
    logout(headers)
    assert len(outcomes) == len(files), "number of outcomes does not match the number of bundle files"

    failed = sorted(name for name, outcome in outcomes.items() if outcome == "failed")
    num_downloaded = sum(outcome == "downloaded" for outcome in outcomes.values())
    num_skipped = sum(outcome == "skipped" for outcome in outcomes.values())
    status = "OK" if not failed else "FAIL"
    logger.info("status %s, %d files, %d downloaded, %d already present, %d failed", status, len(files), num_downloaded, num_skipped, len(failed))
    for name in failed:
        logger.error("failed file %s", name)

    job = os.environ.get("SLURM_JOB_ID", "none")
    date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(RECORD_TSV, "w") as f:
        f.write("status\ttask_id\tfiles\tbytes\tdownloaded\talready_present\tfailed\tjob\tdate_utc\n")
        f.write(f"{status}\t{task_id}\t{len(files)}\t{total_bytes}\t{num_downloaded}\t{num_skipped}\t{len(failed)}\t{job}\t{date}\n")
    logger.info("wrote %s", RECORD_TSV)
    return len(failed)


def main():
    parser = argparse.ArgumentParser(description="Submit and download the 2026 MOD09A1.061 AppEEARS request")
    parser.add_argument("--step", required=True, choices=["submit", "download"])
    parser.add_argument("--max_workers", type=int, default=4, help="number of parallel download threads")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.step == "submit":
        submit_task()
        return
    with open(TASK_JSON) as f:
        task_id = json.load(f)["task_id"]
    if download_task(task_id, args.max_workers) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
