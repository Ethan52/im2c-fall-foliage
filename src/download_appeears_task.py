#!/usr/bin/env python3
"""
Download all GeoTIFF outputs from a completed NASA AppEEARS task.

Example:
python src/download_appeears_task.py \
  --task-id YOUR_TASK_ID \
  --out data/raw/modis/appeears-YOUR_TASK_ID \
  --username YOUR_EARTHDATA_USERNAME
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://appeears.earthdatacloud.nasa.gov/api"


def get_auth_header(username: str | None) -> dict[str, str]:
    """Use a temporary token if supplied; otherwise log in interactively."""
    token = os.environ.get("APPEEARS_TOKEN")

    if not token:
        username = username or os.environ.get("EARTHDATA_USERNAME")
        if not username:
            username = input("Earthdata username: ").strip()

        password = getpass.getpass("Earthdata password: ")

        response = requests.post(
            f"{API}/login",
            auth=(username, password),
            timeout=60,
        )
        response.raise_for_status()
        token = response.json()["token"]

    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"

    return {"Authorization": token}


def get_bundle(task_id: str, headers: dict[str, str]) -> dict:
    response = requests.get(
        f"{API}/bundle/{task_id}",
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def download_one(
    task_id: str,
    file_info: dict,
    headers: dict[str, str],
    out_dir: Path,
    retries: int = 4,
) -> str:
    filename = Path(file_info["file_name"]).name
    file_id = file_info["file_id"]
    target = out_dir / filename
    partial = out_dir / f"{filename}.part"

    # A completed file is not downloaded again.
    if target.exists() and target.stat().st_size > 0:
        return f"skipped: {filename}"

    url = f"{API}/bundle/{task_id}/{file_id}"

    for attempt in range(retries):
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 600),
            ) as response:
                response.raise_for_status()

                with partial.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)

            partial.replace(target)
            return f"downloaded: {filename}"

        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Could not download {filename}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--username")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download metadata/support files too, not only .tif files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = get_auth_header(args.username)
    bundle = get_bundle(args.task_id, headers)

    # Save the API file manifest for reproducibility.
    (out_dir / "bundle.json").write_text(
        json.dumps(bundle, indent=2),
        encoding="utf-8",
    )

    files = bundle.get("files", [])
    if not args.all_files:
        files = [
            f for f in files
            if f.get("file_name", "").lower().endswith(".tif")
        ]

    if not files:
        raise RuntimeError(
            "No matching files found. Check that the task is complete "
            "and that the task ID is correct."
        )

    print(f"Downloading {len(files)} files to: {out_dir}")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                args.task_id,
                file_info,
                headers,
                out_dir,
            ): file_info
            for file_info in files
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as error:
                failures.append((future_to_name := futures[future]["file_name"], str(error)))

    if failures:
        print("\nFailed downloads:")
        for filename, error in failures:
            print(f"  {filename}: {error}")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()