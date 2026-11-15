import os
import sys
import logging

import numpy as np

logger = logging.getLogger(__name__)

# observed onset (O2) and spring green-up (D17) of each forest cell and year, with the fixed U and L of D2
# both are the first linearly interpolated crossing of a threshold after which the next observations in the window stay at or above it
# green-up needs both of the next two observations; onset needs every remaining one of the next two, at least one (O2 as amended by the user)
ONSET_THRESHOLD = 0.6  # brownness, decision D1
ONSET_WINDOW = ("08-01", "12-31")  # month-day, same window as the provisional onset of check_preprocessing.py
GREENUP_THRESHOLD = 0.5  # greenness, one minus brownness, the 50 percent amplitude crossing of MCD12Q2 MidGreenup
GREENUP_WINDOW = ("03-01", "07-31")
PERSIST = 2
ONSET_MIN_LATER = 1
GREENUP_MIN_LATER = 2
YEARS = list(range(2001, 2027))
PROVISIONAL_YEARS = list(range(2001, 2024))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BROWN_DIR = os.path.join(REPO_ROOT, "data", "processed", "brownness")
PROVISIONAL_NPZ = os.path.join(BROWN_DIR, "onset_provisional.npz")
OUT_NPZ = os.path.join(REPO_ROOT, "data", "processed", "phenology", "phenology_dates.npz")


def first_crossing(z: np.ndarray, obs_date: np.ndarray, year: int, window: tuple, threshold: float, min_later: int):
    '''first day of year the interpolated series reaches the threshold, with the next PERSIST observations in the window that exist, at least min_later of them, at or above it
    returns the dates (NaN if undefined), and masks of cells already above at the first observation and of cells never crossing'''
    lo = np.datetime64(f"{year}-{window[0]}")
    hi = np.datetime64(f"{year}-{window[1]}")
    valid = np.isfinite(z) & (obs_date >= lo) & (obs_date <= hi)  # cell, composite
    day = (obs_date - np.datetime64(f"{year}-01-01")).astype("timedelta64[D]").astype(np.float64) + 1  # day of year
    # move each cell's valid observations to the left, keeping their time order
    order = np.argsort(~valid, axis=1, kind="stable")
    v = np.take_along_axis(np.where(valid, z, np.nan), order, axis=1)
    t = np.take_along_axis(np.where(valid, day, np.nan), order, axis=1)
    above = v >= threshold  # NaN compares False
    n_obs = valid.sum(axis=1)
    m = v.shape[1]
    persist = np.zeros((v.shape[0], m), dtype=bool)  # cell, observation position
    for j in range(m):
        later = np.arange(j + 1, min(j + 1 + PERSIST, m))
        exists = later[None, :] < n_obs[:, None]  # cell, later position
        persist[:, j] = above[:, j] & (exists.sum(axis=1) >= min_later) & np.all(~exists | above[:, later], axis=1)
    has = persist.any(axis=1)
    k = persist.argmax(axis=1)
    # the observation before the first persistent one is below the threshold, so position 0 means the crossing is before the window's first observation
    ok = has & (k >= 1)
    rows = np.flatnonzero(ok)
    k0, k1 = k[rows] - 1, k[rows]
    v0, v1 = v[rows, k0], v[rows, k1]
    t0, t1 = t[rows, k0], t[rows, k1]
    date = np.full(z.shape[0], np.nan)
    date[rows] = t0 + (threshold - v0) / (v1 - v0) * (t1 - t0)
    return date, has & (k == 0), ~has


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)

    ul = np.load(os.path.join(BROWN_DIR, "u_l.npz"))
    cell_id = ul["cell_id"]
    ul_valid = ul["ul_valid"]
    n_cells = len(cell_id)
    onset = np.full((n_cells, len(YEARS)), np.nan)  # cell, year, day of year
    greenup = np.full((n_cells, len(YEARS)), np.nan)  # cell, year, day of year
    strict_onset = np.full((n_cells, len(PROVISIONAL_YEARS)), np.nan)  # cell, year, onset with both later observations required
    counts = np.zeros((len(YEARS), 4), dtype=np.int64)  # year, then the four counts named in count_names, over cells with valid U and L
    for y, year in enumerate(YEARS):
        d = np.load(os.path.join(BROWN_DIR, f"brownness_{year}.npz"))
        assert np.array_equal(d["cell_id"], cell_id), f"{year} cell list differs"
        brown = d["brownness"]  # cell, composite, float32 as stored
        assert brown.dtype == np.float32
        onset[:, y], early, never = first_crossing(brown, d["obs_date"], year, ONSET_WINDOW, ONSET_THRESHOLD, ONSET_MIN_LATER)
        counts[y, 0], counts[y, 1] = (early & ul_valid).sum(), (never & ul_valid).sum()
        greenup[:, y], early, never = first_crossing(1.0 - brown, d["obs_date"], year, GREENUP_WINDOW, GREENUP_THRESHOLD, GREENUP_MIN_LATER)
        if year in PROVISIONAL_YEARS:
            strict_onset[:, y] = first_crossing(brown, d["obs_date"], year, ONSET_WINDOW, ONSET_THRESHOLD, PERSIST)[0]
        counts[y, 2], counts[y, 3] = (early & ul_valid).sum(), (never & ul_valid).sum()
        logger.info("%d: onset in %d valid cells (median day %.1f), already above at first observation %d, never %d; "
                    "green-up in %d valid cells (median day %.1f), already above %d, never %d",
                    year, np.isfinite(onset[ul_valid, y]).sum(), np.nanmedian(onset[ul_valid, y]), counts[y, 0], counts[y, 1],
                    np.isfinite(greenup[ul_valid, y]).sum(), np.nanmedian(greenup[ul_valid, y]), counts[y, 2], counts[y, 3])
    assert np.all(np.isnan(onset[~ul_valid])) and np.all(np.isnan(greenup[~ul_valid])), "a date where U and L are not valid"

    # with both later observations required the rule is the provisional rule of check_preprocessing.py, which must be reproduced exactly,
    # and the amended onset must equal it wherever the provisional onset is defined
    prov = np.load(PROVISIONAL_NPZ)
    assert np.array_equal(prov["cell_id"], cell_id) and list(prov["years"]) == PROVISIONAL_YEARS
    assert np.array_equal(strict_onset, prov["onset_doy"], equal_nan=True), "strict onset differs from the provisional onset"
    defined = np.isfinite(strict_onset)
    assert np.array_equal(onset[:, : len(PROVISIONAL_YEARS)][defined], strict_onset[defined]), "amended onset differs where the provisional one is defined"
    logger.info("2001 to 2023: strict rule identical to %s; amended onset identical where it is defined, and defined in %d more cell-years",
                PROVISIONAL_NPZ, int((np.isfinite(onset[:, : len(PROVISIONAL_YEARS)]) & ~defined).sum()))

    np.savez_compressed(
        OUT_NPZ, cell_id=cell_id, years=np.array(YEARS), onset_doy=onset, greenup_doy=greenup, ul_valid=ul_valid,
        counts=counts, count_names=np.array(["onset_above_at_first", "onset_never", "greenup_above_at_first", "greenup_never"]),
        onset_threshold=np.array(ONSET_THRESHOLD), greenup_threshold=np.array(GREENUP_THRESHOLD), persist=np.array(PERSIST),
        onset_min_later=np.array(ONSET_MIN_LATER), greenup_min_later=np.array(GREENUP_MIN_LATER),
    )
    logger.info("wrote %s; dates are fractional days of year, NaN where undefined", OUT_NPZ)


if __name__ == "__main__":
    main()
