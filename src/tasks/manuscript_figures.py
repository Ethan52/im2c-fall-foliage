import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# three summary figures for the manuscript results; reads outputs of prospective.py, prospective_calibration.py and
# retrospective_report.py and writes one table of the numbers shown
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(REPO_ROOT, "data", "processed")
FORECAST_DIR = os.path.join(PROC, "foliage", "forecasts")
RES_RETRO = os.path.join(REPO_ROOT, "results", "retrospective")
RES_PROSP = os.path.join(REPO_ROOT, "results", "prospective")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "manuscript")
MAPPING, VARIANT = "delta", "fit"
AUG1 = {2024: "2024-08-01", 2025: "2025-08-01", 2026: "2026-08-01"}
SITE_NAMES = {"fort_kent": "Fort Kent", "rangeley": "Rangeley", "acadia_bar_harbor": "Bar Harbor"}


def onset_percentiles(t: str, model: str):
    '''ensemble 10th, 50th and 90th percentile forecast onset per cell, NaN where undefined or not reached'''
    d = np.load(os.path.join(FORECAST_DIR, f"forecast_{t}.npz"))
    i, j, v = list(d["mappings"]).index(MAPPING), list(d["models"]).index(model), list(d["variants"]).index(VARIANT)
    x = d["onset_doy"][i, j, v].astype(np.float64)  # member, cell
    x[np.isinf(x)] = np.nan
    with np.errstate(all="ignore"):
        return np.nanpercentile(x, [10, 50, 90], axis=0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(FIG_DIR, exist_ok=True)
    ok = np.load(os.path.join(PROC, "brownness", "u_l.npz"))["ul_valid"]
    ph = np.load(os.path.join(PROC, "phenology", "phenology_dates.npz"))
    years = list(ph["years"])

    # August 1 forecast errors per cell, forecast median minus observed onset
    rows, errs = [], {}
    for model in ("gam", "lr"):
        for y, t in AUG1.items():
            p10, med, p90 = onset_percentiles(t, model)
            obs = ph["onset_doy"][:, years.index(y)]
            m = ok & np.isfinite(med) & np.isfinite(obs)
            e = med[m] - obs[m]
            inside = (obs[m] >= p10[m]) & (obs[m] <= p90[m])
            errs[(model, y)] = e
            q = np.percentile(e, [10, 90])
            rows.append({"model": model, "year": y, "n_cells": int(m.sum()), "bias_days": e.mean(), "mae_days": np.abs(e).mean(),
                         "frac_within_7_days": (np.abs(e) <= 7).mean(), "error_q10": q[0], "error_q90": q[1],
                         "ensemble_p10_p90_coverage": inside.mean(),
                         # spatial spread across cells of forecast and observed onset, and their correlation
                         "sd_forecast_days": med[m].std(), "sd_observed_days": obs[m].std(), "corr": np.corrcoef(med[m], obs[m])[0, 1]})
    table = pd.DataFrame(rows)
    out = os.path.join(RES_PROSP, "aug1_error_summary.tsv")
    table.to_csv(out, sep="\t", index=False, float_format="%.3f")
    logger.info("August 1 forecast errors, %s mapping:\n%s", MAPPING, table.to_string(index=False))

    # figure 1, error histograms of the GAM August 1 forecast with the 2024 and 2025 pooled 80 percent range
    cal = pd.read_csv(os.path.join(RES_PROSP, "calibrated_error_by_issue_date.tsv"), sep="\t", dtype={"issue_month_day": str})
    q10, q90 = cal.loc[cal["issue_month_day"] == "08-01", ["error_q10", "error_q90"]].to_numpy()[0]
    bins = np.arange(-30, 31, 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for k, (label, e) in enumerate([("2024 and 2025 (held out)", np.concatenate([errs[("gam", 2024)], errs[("gam", 2025)]])),
                                    ("2026", errs[("gam", 2026)])]):
        ax[k].hist(np.clip(e, bins[0], bins[-1]), bins=bins, density=True, color="0.5")
        for q in (q10, q90):
            ax[k].axvline(q, color="k", ls="--", lw=1)
        ax[k].axvline(0, color="r", lw=1)
        ax[k].set_title(f"{label}: MAE {np.abs(e).mean():.1f}, bias {e.mean():+.1f} days, {len(e)} cell-years", fontsize=9)
        ax[k].set_xlabel("forecast minus observed peak foliage onset, days")
    ax[0].set_ylabel("fraction of cells per day")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "aug1_error_histograms.png"), dpi=200)
    plt.close(fig)

    # figure 2, DACF peak report date minus median MODIS onset of each zone and year
    dacf = pd.read_csv(os.path.join(RES_RETRO, "dacf_peak_weeks_vs_modis_onset.tsv"), sep="\t")
    dacf = dacf[dacf["dacf_after_last_report"] == "no"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rng = np.random.default_rng(0)
    ax.axhspan(0, 7, color="0.9")
    ax.scatter(dacf["year"] + rng.uniform(-0.15, 0.15, len(dacf)), dacf["report_minus_onset_days"], s=14, color="k")
    ax.axhline(0, color="r", lw=1)
    ax.set_xlabel("year")
    ax.set_ylabel("DACF peak report minus MODIS onset, days")
    ax.set_title(f"{len(dacf)} zone-years, median {dacf['report_minus_onset_days'].median():+.1f} days; shaded: report within the week after onset", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dacf_report_minus_onset.png"), dpi=200)
    plt.close(fig)
    d = dacf["report_minus_onset_days"]
    logger.info("DACF: %d zone-years, median %.2f, mean %.2f, abs within 7 days %.3f, report in week after onset %.3f; 2020 to 2025 only: median %.2f, report in week after onset %.3f",
                len(d), d.median(), d.mean(), (d.abs() <= 7).mean(), dacf["onset_in_week_before_report"].mean(),
                d[dacf["year"] <= 2025].median(), dacf.loc[dacf["year"] <= 2025, "onset_in_week_before_report"].mean())

    # figure 3, the 2026 site forecasts, ensemble range, empirical 80 percent window and observed onset
    s = pd.read_csv(os.path.join(RES_PROSP, "calibrated_site_windows_2026.tsv"), sep="\t")
    doy = lambda x: (pd.Timestamp(x) - pd.Timestamp("2026-01-01")).days + 1
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for k, r in s.iterrows():
        ax.plot([doy(r["window_start"]), doy(r["window_end"])], [k, k], color="0.7", lw=8, solid_capstyle="butt",
                label="80% window from 2024 and 2025 errors" if k == 0 else None)
        ax.plot([r["ensemble_p10_doy"], r["ensemble_p90_doy"]], [k, k], color="tab:blue", lw=3, label="ensemble 10th to 90th percentile" if k == 0 else None)
        ax.plot(r["forecast_median_doy"], k, "o", color="tab:blue", label="ensemble median" if k == 0 else None)
        ax.plot(r["observed_onset_doy"], k, "*", color="r", ms=11, label="observed (MODIS)" if k == 0 else None)
    ax.set_yticks(range(len(s)), [SITE_NAMES[x] for x in s["site"]])
    ticks = [doy(x) for x in ["2026-09-20", "2026-10-01", "2026-10-10", "2026-10-20", "2026-10-30"]]
    ax.set_xticks(ticks, ["Sep 20", "Oct 1", "Oct 10", "Oct 20", "Oct 30"])
    ax.set_ylim(-0.6, len(s) - 0.4)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sites_2026.png"), dpi=200)
    plt.close(fig)
    logger.info("wrote figures to %s", FIG_DIR)


if __name__ == "__main__":
    main()
