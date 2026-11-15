import os
import sys
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# empirical forecast intervals: the error distribution of the August 1 forecasts on the held-out seasons 2024 and 2025,
# all valid cells, gives an 80% interval that is applied to the 2026 forecast and checked against the observed 2026 onset
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(REPO_ROOT, "data", "processed")
FORECAST_DIR = os.path.join(PROC, "foliage", "forecasts")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "prospective")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prospective_config.json")
MAPPING, MODEL, VARIANT = "delta", "gam", "fit"
QUANTILES = (10, 90)


def median_onset(t: str):
    '''ensemble median forecast peak foliage onset per cell, NaN where undefined or not reached'''
    d = np.load(os.path.join(FORECAST_DIR, f"forecast_{t}.npz"))
    i, j, v = list(d["mappings"]).index(MAPPING), list(d["models"]).index(MODEL), list(d["variants"]).index(VARIANT)
    x = d["onset_doy"][i, j, v].astype(np.float64)  # member, cell
    x[np.isinf(x)] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(x, axis=0), np.nanpercentile(x, QUANTILES, axis=0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    ul = np.load(os.path.join(PROC, "brownness", "u_l.npz"))
    ok = ul["ul_valid"]
    ph = np.load(os.path.join(PROC, "phenology", "phenology_dates.npz"))
    years = list(ph["years"])
    issues = [t[:10] for t in json.load(open(CONFIG_PATH))["weather"]["issue_times"]]
    rows = []

    # error quantiles by issue date (nearest issue date of each year), pooled over 2024 and 2025 and all valid cells
    for md in ["08-01", "09-05", "09-21", "10-07"]:
        errs = []
        for y in (2024, 2025):
            t = min([s for s in issues if s.startswith(str(y))], key=lambda s: abs(np.datetime64(s) - np.datetime64(f"{y}-{md}")))
            e = median_onset(t)[0] - ph["onset_doy"][:, years.index(y)]
            errs.append(e[ok & np.isfinite(e)])
        e = np.concatenate(errs)
        q = np.percentile(e, QUANTILES)
        rows.append({"issue_month_day": md, "n_cells": len(e), "mae_days": np.abs(e).mean(), "bias_days": e.mean(),
                     f"error_q{QUANTILES[0]}": q[0], f"error_q{QUANTILES[1]}": q[1]})
        logger.info("issue near %s: %d cell-years, MAE %.2f, bias %.2f, 80%% error range %.1f to %.1f days", md, len(e), np.abs(e).mean(), e.mean(), *q)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "calibrated_error_by_issue_date.tsv"), sep="\t", index=False, float_format="%.2f")

    # the August 1 interval: observed onset expected in [forecast minus q90, forecast minus q10]
    lo_off, hi_off = -table.iloc[0][f"error_q{QUANTILES[1]}"], -table.iloc[0][f"error_q{QUANTILES[0]}"]
    f26, (p_lo, p_hi) = median_onset("2026-08-01")
    o26 = ph["onset_doy"][:, years.index(2026)]
    m = ok & np.isfinite(f26) & np.isfinite(o26)
    cov_cal = float(((o26[m] >= f26[m] + lo_off) & (o26[m] <= f26[m] + hi_off)).mean())
    mm = m & np.isfinite(p_lo)
    cov_ens = float(((o26[mm] >= p_lo[mm]) & (o26[mm] <= p_hi[mm])).mean())
    logger.info("2026: calibrated 80%% interval [forecast %+.1f, forecast %+.1f] covers %.3f of %d cells; raw ensemble 10th to 90th percentile covers %.3f",
                lo_off, hi_off, cov_cal, int(m.sum()), cov_ens)

    s = pd.read_csv(os.path.join(RESULTS_DIR, "prospective_site_onsets_2026-08-01.tsv"), sep="\t")
    s = s[(s["model"] == MODEL) & (s["variant"] == VARIANT) & (s["mapping"] == MAPPING)]
    date = lambda x: (pd.Timestamp("2026-01-01") + pd.Timedelta(days=round(x) - 1)).strftime("%Y-%m-%d")
    sites = []
    for _, r in s.iterrows():
        c = r["forecast_median"]
        sites.append({"site": r["site"], "forecast_median_doy": c, "forecast_median_date": date(c),
                      "window_start": date(c + lo_off), "window_end": date(c + hi_off),
                      "ensemble_p10_doy": r["forecast_p10"], "ensemble_p90_doy": r["forecast_p90"],
                      "observed_onset_doy": r["observed_onset_doy"], "observed_date": date(r["observed_onset_doy"]),
                      "observed_in_window": bool(c + lo_off <= r["observed_onset_doy"] <= c + hi_off)})
    out = pd.DataFrame(sites)
    out["interval_offsets_days"] = f"{lo_off:+.1f} to {hi_off:+.1f}"
    out["coverage_2026_calibrated"] = cov_cal
    out["coverage_2026_ensemble"] = cov_ens
    out.to_csv(os.path.join(RESULTS_DIR, "calibrated_site_windows_2026.tsv"), sep="\t", index=False, float_format="%.2f")
    logger.info("2026 site windows:\n%s", out.to_string(index=False))


if __name__ == "__main__":
    main()
