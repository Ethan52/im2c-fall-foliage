import os
import re
import sys
import glob
import time
import logging
import argparse

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

# quality masking, EVI2 and aggregation of MOD09A1 to the 1 km forest cells, step 2 of references/preprocessing_plan.md
# bit positions from the MOD09 Collection 6 user guide (Vermote et al. 2020), table 10 for QC and table 13 for state
LAYERS = ["sur_refl_b01", "sur_refl_b02", "sur_refl_qc_500m", "sur_refl_state_500m", "sur_refl_day_of_year"]
DTYPES = {"sur_refl_b01": np.int16, "sur_refl_b02": np.int16, "sur_refl_qc_500m": np.uint32,
          "sur_refl_state_500m": np.uint16, "sur_refl_day_of_year": np.uint16}
REFL_FILL = -28672
QC_FILL = 4294967295
STATE_FILL = 65535
DOY_FILL = 65535
REFL_MIN, REFL_MAX = -100, 16000  # valid range of the stored reflectance integers
REFL_SCALE = 0.0001
COMPOSITE_DAYS = 8
RULES = ["fill", "range", "qc", "cloud", "shadow", "land", "aerosol", "cirrus", "intcloud", "snow", "adjacent", "evi2"]
YEARS = list(range(2001, 2027))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODIS_DIR = os.path.join(REPO_ROOT, "data", "raw", "modis", "MOD09A1.061_2000360_to_2025365")
MODIS_2026_DIR = os.path.join(REPO_ROOT, "data", "raw", "modis_2026")
MASK_NPZ = os.path.join(REPO_ROOT, "data", "processed", "forest_mask", "forest_mask.npz")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "modis")
FILE_RE = re.compile(r"MOD09A1\.061_(sur_refl_\w+?)_doy(\d{4})(\d{3})000000_aid0001\.tif$")


def list_composites(directory: str):
    '''map (start year, start day of year) to a dict of layer to file path, for every complete composite in a directory'''
    composites = {}
    for path in glob.glob(os.path.join(directory, "**", "MOD09A1.061_sur_refl_*.tif"), recursive=True):
        m = FILE_RE.search(os.path.basename(path))
        if m is None or m.group(1) not in LAYERS:
            continue
        composites.setdefault((int(m.group(2)), int(m.group(3))), {})[m.group(1)] = path
    for key, layers in composites.items():
        assert sorted(layers) == sorted(LAYERS), f"composite {key} is missing layers: {sorted(set(LAYERS) - set(layers))}"
    return composites


def read_kept(paths: dict, kept_idx: np.ndarray):
    '''read the five layers of one composite and return 1-D arrays at the kept pixels, dtypes asserted'''
    out = {}
    for layer in LAYERS:
        with rasterio.open(paths[layer]) as src:
            arr = src.read(1)  # MODIS row, MODIS column
            assert arr.shape == (1099, 1353), f"{paths[layer]} shape {arr.shape}"
        assert arr.dtype == DTYPES[layer], f"{paths[layer]} dtype {arr.dtype}"
        out[layer] = arr.ravel()[kept_idx]
    return out


def process_composite(paths: dict, start_year: int, start_doy: int, kept_idx: np.ndarray, kept_slot: np.ndarray, n_cells: int):
    '''quality masking, EVI2 and cell means of one composite; return cell EVI2, cell date offsets, pixel counts and diagnostics'''
    d = read_kept(paths, kept_idx)
    b1 = d["sur_refl_b01"].astype(np.int32)
    b2 = d["sur_refl_b02"].astype(np.int32)
    qc = d["sur_refl_qc_500m"]
    st = d["sur_refl_state_500m"]
    doy = d["sur_refl_day_of_year"].astype(np.int32)

    fill = (b1 == REFL_FILL) | (b2 == REFL_FILL) | (qc == QC_FILL) | (st == STATE_FILL) | (doy == DOY_FILL)
    ok = ~fill
    bad = {"fill": fill}
    bad["range"] = ok & ((b1 < REFL_MIN) | (b1 > REFL_MAX) | (b2 < REFL_MIN) | (b2 > REFL_MAX))
    # band 1 quality in QC bits 2 to 5 and band 2 quality in bits 6 to 9, 0 is highest quality
    bad["qc"] = ok & ((((qc >> 2) & 0xF) != 0) | (((qc >> 6) & 0xF) != 0))
    cloud_state = st & 0b11
    bad["cloud"] = ok & (cloud_state != 0)  # only 00 clear is accepted (choice A)
    bad["shadow"] = ok & (((st >> 2) & 1) == 1)
    # land (001) and ocean coastline or lake shoreline (010) are accepted, every water code is rejected (user decision of 2026-11-15)
    land_water = (st >> 3) & 0b111
    bad["land"] = ok & (land_water != 0b001) & (land_water != 0b010)
    bad["aerosol"] = ok & (((st >> 6) & 0b11) == 0b11)
    bad["cirrus"] = ok & (((st >> 8) & 0b11) >= 0b10)
    bad["intcloud"] = ok & (((st >> 10) & 1) == 1)
    bad["snow"] = ok & ((((st >> 12) & 1) == 1) | (((st >> 15) & 1) == 1))
    bad["adjacent"] = ok & (((st >> 13) & 1) == 1)  # pixels next to cloud are dropped (choice B)

    red = b1 * REFL_SCALE
    nir = b2 * REFL_SCALE
    with np.errstate(divide="ignore", invalid="ignore"):
        evi2 = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)  # float64
    bad["evi2"] = ok & ~((evi2 >= -1.0) & (evi2 <= 1.0))

    any_bad = np.zeros(len(b1), dtype=bool)
    for rule in RULES:
        any_bad |= bad[rule]
    valid = ~any_bad
    # extra observations the alternatives to choices A and B would have added
    fails_other_than_cloud = np.any([bad[r] for r in RULES if r != "cloud"], axis=0)
    fails_other_than_adjacent = np.any([bad[r] for r in RULES if r != "adjacent"], axis=0)
    extra_a = int((~fails_other_than_cloud & (cloud_state == 0b11)).sum())  # cloud state 11, not set, assumed clear
    extra_b = int((bad["adjacent"] & ~fails_other_than_adjacent).sum())

    # day offset of each observation from the composite start, wrapping into the next year for late-December composites
    days_in_start_year = 366 if (start_year % 4 == 0 and (start_year % 100 != 0 or start_year % 400 == 0)) else 365
    offset = np.where(doy >= start_doy, doy - start_doy, doy + days_in_start_year - start_doy)
    assert np.all((offset[valid] >= 0) & (offset[valid] < COMPOSITE_DAYS)), "observation day outside the 8-day composite"

    slots = kept_slot[valid]
    n_pix = np.bincount(slots, minlength=n_cells)
    sum_evi2 = np.bincount(slots, weights=evi2[valid], minlength=n_cells)
    sum_offset = np.bincount(slots, weights=offset[valid], minlength=n_cells)
    assert n_pix.sum() == valid.sum(), "cell counts do not add up to the valid pixels"
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_evi2 = np.where(n_pix > 0, sum_evi2 / n_pix, np.nan)
        cell_offset = np.where(n_pix > 0, np.rint(sum_offset / n_pix), -1).astype(np.int64)

    valid_shoreline = int((valid & (land_water == 0b010)).sum())
    diag = [int(bad[r].sum()) for r in RULES] + [len(b1), int(valid.sum()), extra_a, extra_b, valid_shoreline]
    return cell_evi2.astype(np.float32), cell_offset, n_pix, diag


def main():
    parser = argparse.ArgumentParser(description="Quality mask, EVI2 and 1 km cell means of MOD09A1, one file per year")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(OUT_DIR, exist_ok=True)

    mask = np.load(MASK_NPZ)
    kept = mask["kept"]  # MODIS row, MODIS column
    pixel_cell = mask["pixel_cell"]  # MODIS row, MODIS column
    forest_cell_id = mask["forest_cell_id"]  # sorted analysis cell IDs with at least one kept pixel
    kept_idx = np.flatnonzero(kept)
    kept_slot = np.searchsorted(forest_cell_id, pixel_cell.ravel()[kept_idx])  # position of each kept pixel's cell in forest_cell_id
    assert np.array_equal(forest_cell_id[kept_slot], pixel_cell.ravel()[kept_idx]), "kept pixel cell not in the forest cell list"
    n_cells = len(forest_cell_id)
    logger.info("%d kept MODIS pixels in %d forest cells", len(kept_idx), n_cells)

    main_comp = list_composites(MODIS_DIR)
    comp_2026 = list_composites(MODIS_2026_DIR)
    # the 2025-361 composite is in both downloads and must be identical
    dup = (2025, 361)
    if dup in comp_2026:
        for layer in LAYERS:
            with rasterio.open(main_comp[dup][layer]) as a, rasterio.open(comp_2026[dup][layer]) as b:
                assert np.array_equal(a.read(1), b.read(1)), f"2025-361 {layer} differs between the two downloads"
        logger.info("2025-361 composite identical in both downloads")
    composites = dict(main_comp)
    composites.update({k: v for k, v in comp_2026.items() if k[0] == 2026})

    for year in args.years:
        start = time.perf_counter()
        keys = sorted(k for k in composites if k[0] == year)
        assert keys, f"no composites for {year}"
        n_comp = len(keys)
        evi2 = np.full((n_cells, n_comp), np.nan, dtype=np.float32)  # cell, composite
        obs_date = np.full((n_cells, n_comp), np.datetime64("NaT"), dtype="datetime64[D]")  # cell, composite
        n_pix = np.zeros((n_cells, n_comp), dtype=np.uint8)  # cell, composite
        diag = np.zeros((n_comp, len(RULES) + 5), dtype=np.int64)  # composite, rule counts then kept, valid, extra A, extra B, valid shoreline
        composite_start = np.array([np.datetime64(f"{y}-01-01") + np.timedelta64(d - 1, "D") for y, d in keys], dtype="datetime64[D]")
        for j, (y, d) in enumerate(keys):
            cell_evi2, cell_offset, cell_n, diag[j] = process_composite(composites[(y, d)], y, d, kept_idx, kept_slot, n_cells)
            assert cell_n.max() <= 255, "pixel count does not fit in uint8"
            has = cell_n > 0
            evi2[:, j] = cell_evi2
            obs_date[has, j] = composite_start[j] + cell_offset[has].astype("timedelta64[D]")
            n_pix[:, j] = cell_n
        out = os.path.join(OUT_DIR, f"evi2_{year}.npz")
        np.savez_compressed(
            out, cell_id=forest_cell_id, composite_start=composite_start, evi2=evi2, obs_date=obs_date, n_pix=n_pix,
            diag=diag, diag_names=np.array(RULES + ["kept", "valid", "extra_cloud_state_11", "extra_adjacent", "valid_shoreline"]),
        )
        valid_frac = diag[:, len(RULES) + 1].sum() / diag[:, len(RULES)].sum()
        logger.info("%d: %d composites, valid share of kept pixel-composites %.3f, %.0f seconds, wrote %s",
                    year, n_comp, valid_frac, time.perf_counter() - start, out)


if __name__ == "__main__":
    main()
