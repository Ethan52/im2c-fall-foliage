import os
import sys
import glob
import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# U, L and observed brownness, step 3 of references/preprocessing_plan.md
# decision D2: U and L of a cell are the means over the U and L years of the yearly August and late-November medians of EVI2
UL_YEARS = list(range(2001, 2024))
U_WINDOW = ("08-01", "08-31")  # month-day, inclusive
L_WINDOW = ("11-16", "11-30")
MIN_YEARS = 5  # choice C for L, applied to U as well

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVI2_DIR = os.path.join(REPO_ROOT, "data", "processed", "modis")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "brownness")


def window_median(evi2: np.ndarray, obs_date: np.ndarray, year: int, window: tuple):
    '''median over composites of the cell EVI2 whose observation date lies in the window of that year; NaN if none'''
    lo = np.datetime64(f"{year}-{window[0]}")
    hi = np.datetime64(f"{year}-{window[1]}")
    in_window = (obs_date >= lo) & (obs_date <= hi)  # NaT compares False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows give NaN
        return np.nanmedian(np.where(in_window, evi2, np.nan), axis=1)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(OUT_DIR, exist_ok=True)

    cell_id = None
    U_year = []
    L_year = []
    for year in UL_YEARS:
        d = np.load(os.path.join(EVI2_DIR, f"evi2_{year}.npz"))
        if cell_id is None:
            cell_id = d["cell_id"]
        assert np.array_equal(d["cell_id"], cell_id), f"{year} cell list differs"
        U_year.append(window_median(d["evi2"], d["obs_date"], year, U_WINDOW))
        L_year.append(window_median(d["evi2"], d["obs_date"], year, L_WINDOW))
    U_year = np.stack(U_year, axis=1)  # cell, year
    L_year = np.stack(L_year, axis=1)  # cell, year
    n_years_U = np.isfinite(U_year).sum(axis=1)
    n_years_L = np.isfinite(L_year).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        U = np.nanmean(U_year, axis=1)
        L = np.nanmean(L_year, axis=1)
    ul_valid = (n_years_U >= MIN_YEARS) & (n_years_L >= MIN_YEARS) & (U - L > 0)
    logger.info("%d forest cells: %d with at least %d U years, %d with at least %d L years, %d with U above L, %d valid",
                len(cell_id), (n_years_U >= MIN_YEARS).sum(), MIN_YEARS, (n_years_L >= MIN_YEARS).sum(), MIN_YEARS,
                (U - L > 0).sum(), ul_valid.sum())
    logger.info("valid cells: U median %.3f (10th %.3f, 90th %.3f), L median %.3f (10th %.3f, 90th %.3f)",
                np.median(U[ul_valid]), *np.percentile(U[ul_valid], [10, 90]),
                np.median(L[ul_valid]), *np.percentile(L[ul_valid], [10, 90]))
    np.savez_compressed(
        os.path.join(OUT_DIR, "u_l.npz"), cell_id=cell_id, U=U.astype(np.float32), L=L.astype(np.float32),
        n_years_U=n_years_U, n_years_L=n_years_L, ul_valid=ul_valid,
        U_year=U_year.astype(np.float32), L_year=L_year.astype(np.float32), ul_years=np.array(UL_YEARS), min_years=np.array(MIN_YEARS),
    )

    # observed brownness for every year with an EVI2 file, with the same fixed U and L
    denom = np.where(ul_valid, U - L, np.nan)[:, None]  # cell, 1
    for path in sorted(glob.glob(os.path.join(EVI2_DIR, "evi2_*.npz"))):
        d = np.load(path)
        year = int(os.path.basename(path)[5:9])
        assert np.array_equal(d["cell_id"], cell_id), f"{year} cell list differs"
        brown = ((U[:, None] - d["evi2"]) / denom).astype(np.float32)  # cell, composite
        np.savez_compressed(
            os.path.join(OUT_DIR, f"brownness_{year}.npz"), cell_id=cell_id, composite_start=d["composite_start"],
            obs_date=d["obs_date"], brownness=brown,
        )
        if year in UL_YEARS:
            # by construction the window medians of brownness average near 0 in August and near 1 in late November
            bu = (U - U_year[:, UL_YEARS.index(year)]) / (U - L)
            bl = (U - L_year[:, UL_YEARS.index(year)]) / (U - L)
            logger.info("%d: mean August brownness %.3f, mean late-November brownness %.3f over valid cells",
                        year, np.nanmean(bu[ul_valid]), np.nanmean(bl[ul_valid]))
    logger.info("wrote %s", OUT_DIR)


if __name__ == "__main__":
    main()
