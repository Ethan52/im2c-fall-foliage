import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.foliage import features
from reporting import maps, dacf

logger = logging.getLogger(__name__)

# figures and tables of the prospective pipeline (AIFS ENS to ACE2 weather forecasts, Kalman updates of D26) and the DACF
# comparison of the test years; reads outputs of src/tasks/prospective.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(REPO_ROOT, "data", "processed")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prospective_config.json")
FORECAST_DIR = os.path.join(features.FOLIAGE_DIR, "forecasts")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "prospective")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "prospective")
MAP_ISSUES = ["08-01", "09-05", "10-07"]  # month-day; the nearest issue date of each year is mapped
MAIN_MODEL, MAIN_MAPPING = "gam", "delta"
COMPOSITE_DAYS = 8


def median_onset(d, mapping: str, model: str, variant: str):
    '''ensemble median forecast onset per cell (day of year), with infinite (no onset by season end) set to NaN, and that mask'''
    i, j, v = list(d["mappings"]).index(mapping), list(d["models"]).index(model), list(d["variants"]).index(variant)
    med = np.nanmedian(d["onset_doy"][i, j, v].astype(np.float64), axis=0)
    never = np.isinf(med)
    med[never] = np.nan
    return med, never


def zone_mask(zone: np.ndarray, z: str):
    '''cells of a report zone; a whole zone (for example 4) covers both of its halves'''
    return (zone == z) | (np.char.rstrip(zone.astype(str), "NS") == z)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    years = cfg["foliage"]["years"]
    issues = np.array([np.datetime64(t[:10]) for t in cfg["weather"]["issue_times"] if int(t[:4]) in years])
    ul = np.load(os.path.join(PROC, "brownness", "u_l.npz"))
    cell_id, ul_valid = ul["cell_id"], ul["ul_valid"]
    ph = np.load(os.path.join(PROC, "phenology", "phenology_dates.npz"))
    assert np.array_equal(ph["cell_id"], cell_id)
    grid, extent = maps.load_grid()
    epsg = int(grid["epsg"])
    boundary = maps.load_boundary(epsg)
    under = maps.underlay(cell_id, ul_valid, grid)
    sites = pd.read_csv(os.path.join(REPO_ROOT, "configs", "sites.csv"))
    scores = pd.read_csv(os.path.join(RESULTS_DIR, "prospective_scores.tsv"), sep="\t", parse_dates=["issue_date"])
    site_onsets = pd.read_csv(os.path.join(RESULTS_DIR, "prospective_site_onsets.tsv"), sep="\t", parse_dates=["issue_date"])
    cat, _, towns = dacf.load_tables()
    pos = np.searchsorted(grid["cell_id"], cell_id)
    zone = dacf.cell_zones(grid["x"][pos], grid["y"][pos], towns, epsg)
    zone_rows = []

    for year in years:
        year_issues = issues[issues.astype("datetime64[Y]").astype(int) + 1970 == year]
        picks = [year_issues[np.argmin(np.abs(year_issues - np.datetime64(f"{year}-{md}")))] for md in MAP_ISSUES]
        observed = np.where(ul_valid, ph["onset_doy"][:, list(ph["years"]).index(year)], np.nan)
        a = np.load(os.path.join(features.FOLIAGE_DIR, f"assimilation_{year}.npz"))
        fc = {t: np.load(os.path.join(FORECAST_DIR, f"forecast_{t}.npz")) for t in year_issues}
        lo, hi = np.nanpercentile(observed, [2, 98])

        # figure 5, forecast onset maps at three issue dates next to the observed onset, both mappings
        for mapping in cfg["foliage"]["mappings"]:
            fig, axes = plt.subplots(2, len(picks) + 1, figsize=(3.2 * (len(picks) + 1), 9))
            med_r, err_r, med_t, err_t = [], [], [], []
            for t in picks:
                med, never = median_onset(fc[t], mapping, MAIN_MODEL, "fit")
                med = np.where(ul_valid, med, np.nan)
                both = np.isfinite(med) & np.isfinite(observed)
                err = med - observed
                med_r.append(maps.to_raster(med, cell_id, grid))
                err_r.append(maps.to_raster(err, cell_id, grid))
                med_t.append(f"issued {t}\nno peak foliage onset in {int((never & ul_valid).sum())} cells")
                err_t.append(f"issued {t}\nbias {np.mean(err[both]):.1f}, MAE {np.mean(np.abs(err[both])):.1f} days")
            med_r.append(maps.to_raster(observed, cell_id, grid))
            med_t.append("observed peak foliage onset (MODIS)")
            maps.map_panels(axes[0], med_r, med_t, extent, boundary, under, "viridis", lo, hi, sites)
            fig.colorbar(axes[0][-1].images[-1], ax=axes[0], shrink=0.7, label="peak foliage onset, day of year")
            im = maps.map_panels(axes[1][:-1], err_r, err_t, extent, boundary, under, "RdBu_r", -20, 20, sites)
            axes[1][-1].axis("off")
            fig.colorbar(im, ax=axes[1], shrink=0.7, label="forecast minus observed, days")
            fig.suptitle(f"{year}: ensemble median forecast peak foliage onset, {MAIN_MODEL.upper()}, {mapping} mapping, retrospective fit coefficients")
            fig.savefig(os.path.join(FIG_DIR, f"forecast_onset_maps_{year}_{mapping}.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)

        # figure 6, change of the forecast onset by the Kalman updates at the same issue dates
        fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 5.5))
        rr, tt = [], []
        for t in picks:
            fit, _ = median_onset(fc[t], MAIN_MAPPING, MAIN_MODEL, "fit")
            asm, _ = median_onset(fc[t], MAIN_MAPPING, MAIN_MODEL, "assimilated")
            k = int(np.flatnonzero(a["issue_dates"] == t)[0])
            n_cum = int(a[f"n_new_{MAIN_MODEL}"][: k + 1].sum())
            ch = np.where(ul_valid, asm - fit, np.nan)
            rr.append(maps.to_raster(ch, cell_id, grid))
            tt.append(f"issued {t}\n{n_cum} observations assimilated, beta change {a[f'relative_change_{MAIN_MODEL}'][k]:.3f}\nmedian absolute change {np.nanmedian(np.abs(ch)):.2f} days")
        vmax = max(1.0, float(np.nanpercentile(np.abs(np.concatenate([r[np.isfinite(r)] for r in rr])), 98)))
        im = maps.map_panels(axes, rr, tt, extent, boundary, under, "PuOr_r", -vmax, vmax, sites)
        fig.colorbar(im, ax=axes, shrink=0.7, label="assimilated minus fit peak foliage onset, days")
        fig.suptitle(f"{year}: effect of data assimilation on the ensemble median peak foliage onset, {MAIN_MODEL.upper()}, {MAIN_MAPPING} mapping")
        fig.savefig(os.path.join(FIG_DIR, f"assimilation_onset_change_maps_{year}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

        # figure 10, DACF categories on each report date: DACF zones, forecast brownness, observed brownness
        b = np.load(os.path.join(PROC, "brownness", f"brownness_{year}.npz"))
        starts = b["composite_start"]
        sub = dacf.expand_subzones(cat[cat["report_date"].dt.year == year])
        reports = sorted(sub["report_date"].unique())
        fig, axes = plt.subplots(3, len(reports), figsize=(2.6 * len(reports), 11), squeeze=False)
        conf = np.zeros((5, 5), dtype=int)  # DACF category (Past Peak merged into Peak), model category of the zone median
        for c, rd in enumerate(reports):
            rd = np.datetime64(pd.Timestamp(rd).date())
            k = int(np.flatnonzero((starts <= rd) & (rd < starts + COMPOSITE_DAYS))[0])  # composite containing the report date
            t = year_issues[year_issues <= starts[k]].max()  # latest issue date before that composite
            i, j, v = list(fc[t]["mappings"]).index(MAIN_MAPPING), list(fc[t]["models"]).index(MAIN_MODEL), list(fc[t]["variants"]).index("assimilated")
            pred_b = np.where(ul_valid, fc[t]["pred_brownness_median"][i, j, v][:, k], np.nan)
            obs_b = np.where(ul_valid, b["brownness"][:, k], np.nan)
            d_cat = np.full(len(cell_id), np.nan)
            for _, r in sub[sub["report_date"] == pd.Timestamp(rd)].iterrows():
                d_cat[(zone == r["zone"]) & ul_valid] = dacf.CATEGORIES.index(r["color_change"])
                m = dacf.brownness_category(np.array([np.nanmedian(pred_b[(zone == r["zone"]) & ul_valid])]))[0]
                if np.isfinite(m):
                    conf[min(dacf.CATEGORIES.index(r["color_change"]), 4), int(m)] += 1
            panels = [d_cat, dacf.brownness_category(pred_b), dacf.brownness_category(obs_b)]
            labels = [f"DACF report {rd}", f"forecast issued {t}\ncomposite {starts[k]}", f"observed MODIS\ncomposite {starts[k]}"]
            im = None
            for rrow in range(3):
                im = maps.map_panels([axes[rrow, c]], [maps.to_raster(panels[rrow], cell_id, grid)], [labels[rrow]], extent, boundary, under,
                                     "YlOrRd", -0.5, 5.5)
        cb = fig.colorbar(im, ax=axes, shrink=0.5, ticks=range(6))
        cb.ax.set_yticklabels(dacf.CATEGORIES)
        fig.suptitle(f"{year}: DACF color-change categories against brownness categories (brownness read as the DACF percent ranges)")
        fig.savefig(os.path.join(FIG_DIR, f"dacf_category_maps_{year}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        conf_df = pd.DataFrame(conf, index=[f"DACF {c}" for c in dacf.CATEGORIES[:4] + ["Peak or Past Peak"]], columns=[f"model {c}" for c in dacf.CATEGORIES[:5]])
        conf_df.to_csv(os.path.join(RESULTS_DIR, f"dacf_category_confusion_{year}.tsv"), sep="\t")
        logger.info("%d DACF category confusion, zone-report pairs (rows DACF, columns zone median of the forecast):\n%s", year, conf_df.to_string())

        # table 10, forecast onset per DACF zone against the DACF peak report date
        peaks = dacf.report_peak_dates(cat)
        for _, r in peaks[peaks["year"] == year].iterrows():
            in_zone = (zone == r["zone"]) & ul_valid
            row = {"year": year, "zone": r["zone"], "dacf_peak_report_doy": float(pd.Timestamp(r["peak_report_date"]).day_of_year),
                   "observed_onset_median_doy": float(np.nanmedian(observed[in_zone]))}
            for t in picks:
                for variant in ("fit", "assimilated"):
                    med, _ = median_onset(fc[t], MAIN_MAPPING, MAIN_MODEL, variant)
                    row[f"forecast_{variant}_{t}_median_doy"] = float(np.nanmedian(med[in_zone]))
            zone_rows.append(row)

    zt = pd.DataFrame(zone_rows)
    zt.to_csv(os.path.join(RESULTS_DIR, "dacf_peak_weeks_vs_forecast_onset.tsv"), sep="\t", index=False, float_format="%.1f")

    # figure 7, scores against issue date, all models, mappings and variants
    fig, axes = plt.subplots(2, len(years), figsize=(7 * len(years), 8), squeeze=False)
    for y, year in enumerate(years):
        for (mapping, model, variant), g in scores[scores["year"] == year].groupby(["mapping", "model", "variant"]):
            style = dict(marker="o", ms=3, ls="-" if variant == "fit" else "--", label=f"{model} {mapping} {variant}")
            axes[0, y].plot(g["issue_date"], g["onset_mae_days"], **style)
            axes[1, y].plot(g["issue_date"], g["brownness_rmse_after_issue"], **style)
        axes[0, y].set_title(f"{year}: peak foliage onset MAE of the ensemble median")
        axes[0, y].set_ylabel("days")
        axes[1, y].set_title(f"{year}: brownness RMSE on composites after the issue date (a shrinking set)")
        axes[1, y].set_ylabel("brownness")
        axes[0, y].legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scores_by_issue_date.png"), dpi=130)
    plt.close(fig)
    # the assimilation gain, assimilated minus fit, on identical observation sets
    piv = scores.pivot_table(index=["year", "issue_date", "mapping", "model"], columns="variant", values=["onset_mae_days", "brownness_rmse_after_issue"])
    gain = pd.DataFrame({"onset_mae_change_days": piv[("onset_mae_days", "assimilated")] - piv[("onset_mae_days", "fit")],
                         "brownness_rmse_change": piv[("brownness_rmse_after_issue", "assimilated")] - piv[("brownness_rmse_after_issue", "fit")]}).reset_index()
    gain.to_csv(os.path.join(RESULTS_DIR, "assimilation_gain.tsv"), sep="\t", index=False, float_format="%.5f")
    logger.info("assimilation gain, assimilated minus fit, mean over issue dates:\n%s",
                gain.groupby(["year", "mapping", "model"])[["onset_mae_change_days", "brownness_rmse_change"]].mean().to_string())

    # figure 8, relative change of beta and the observations assimilated, against issue date
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for year in years:
        a = np.load(os.path.join(features.FOLIAGE_DIR, f"assimilation_{year}.npz"))
        d = pd.to_datetime(a["issue_dates"]).strftime("%m-%d")
        for model in ("lr", "gam"):
            axes[0].plot(d, a[f"relative_change_{model}"], marker="o", ms=3, label=f"{year} {model}")
            axes[1].plot(d, np.cumsum(a[f"n_new_{model}"]), marker="o", ms=3, label=f"{year} {model}")
    axes[0].set_ylabel("norm of beta change over norm of beta")
    axes[1].set_ylabel("cumulative observations assimilated")
    for ax in axes:
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "assimilation_beta_change.png"), dpi=130)
    plt.close(fig)

    # figure 9, site fans: 10th, 50th and 90th percentile of the forecast onset against issue date, observed onset as a line
    s = site_onsets[(site_onsets["model"] == MAIN_MODEL) & (site_onsets["mapping"] == MAIN_MAPPING)]
    site_names = list(s["site"].unique())
    fig, axes = plt.subplots(len(years), len(site_names), figsize=(4.5 * len(site_names), 3.5 * len(years)), squeeze=False)
    for y, year in enumerate(years):
        for c, site in enumerate(site_names):
            ax = axes[y, c]
            for variant, color in (("fit", "tab:blue"), ("assimilated", "tab:orange")):
                g = s[(s["year"] == year) & (s["site"] == site) & (s["variant"] == variant)]
                ax.fill_between(g["issue_date"], g["forecast_p10"], g["forecast_p90"], color=color, alpha=0.2)
                ax.plot(g["issue_date"], g["forecast_median"], color=color, marker="o", ms=3, label=variant)
                obs = g["observed_onset_doy"].iloc[0]
            if np.isfinite(obs):
                ax.axhline(obs, color="k", ls=":", label="observed")
            ax.set_title(f"{site} {year}", fontsize=9)
            ax.set_ylabel("peak foliage onset, day of year")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "site_onset_fans.png"), dpi=130)
    plt.close(fig)

    # figure 11, the 2026 forecast if the prospective task has produced it
    path_2026 = os.path.join(FORECAST_DIR, "forecast_2026-08-01.npz")
    if os.path.exists(path_2026):
        d = np.load(path_2026)
        obs26 = np.where(ul_valid, ph["onset_doy"][:, list(ph["years"]).index(2026)], np.nan)
        med, never = median_onset(d, MAIN_MAPPING, MAIN_MODEL, "fit")
        med = np.where(ul_valid, med, np.nan)
        lo, hi = np.nanpercentile(obs26, [2, 98])
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        maps.map_panels(axes[:2], [maps.to_raster(med, cell_id, grid), maps.to_raster(obs26, cell_id, grid)],
                        [f"forecast issued 2026-08-01\nno onset reached in {int((never & ul_valid).sum())} cells", "observed 2026\n(MODIS to Nov 8)"],
                        extent, boundary, under, "viridis", lo, hi, sites)
        fig.colorbar(axes[1].images[-1], ax=axes[:2], shrink=0.7, label="peak foliage onset, day of year")
        both = np.isfinite(med) & np.isfinite(obs26)
        im = maps.map_panels(axes[2:], [maps.to_raster(med - obs26, cell_id, grid)],
                             [f"forecast minus observed\nbias {np.mean((med - obs26)[both]):.1f}, MAE {np.mean(np.abs(med - obs26)[both]):.1f} days"],
                             extent, boundary, under, "RdBu_r", -20, 20, sites)
        fig.colorbar(im, ax=axes[2], shrink=0.7, label="days")
        fig.savefig(os.path.join(FIG_DIR, "forecast_onset_maps_2026.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    else:
        logger.info("no 2026 forecast file, figure 11 skipped")
    logger.info("wrote %s and %s", RESULTS_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
