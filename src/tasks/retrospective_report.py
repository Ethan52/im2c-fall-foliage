import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.foliage import features, outcome, evaluation
from reporting import maps, dacf

logger = logging.getLogger(__name__)

# figures and tables of the retrospective pipeline (R3 on the test years with Daymet weather), the DACF comparison of
# the observed MODIS onset, and the preprocessing counts; reads outputs of src/tasks/retrospective.py and the data session
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(REPO_ROOT, "data", "processed")
FIT_NPZ = os.path.join(features.FOLIAGE_DIR, "fits.npz")
SPLIT_FIT_NPZ = os.path.join(features.FOLIAGE_DIR, "fits_split.npz")  # fit on the training cells of the within-year split
SPLIT_NPZ = os.path.join(features.FOLIAGE_DIR, "split_cells.npz")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "retrospective")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "retrospective")
TEST_YEARS = [2024, 2025]
COMPOSITE_DAYS = 8
MAP_DATES = ["09-13", "09-29", "10-15", "10-31"]  # month-day; the composite whose start is nearest is mapped
DACF_YEARS = list(range(2020, 2027))
PANEL_DATES = ["09-13", "09-21", "09-29", "10-07", "10-15", "10-23"]  # month-day, six composites of the year panel figure


def doy_of(dates: np.ndarray, year: int):
    '''day of year, 1 for January 1, NaN for missing dates'''
    out = (dates - np.datetime64(f"{year}-01-01")).astype("timedelta64[D]").astype(float) + 1
    out[np.isnat(dates)] = np.nan
    return out


def gam_daily(year: int, cell_id: np.ndarray, ul_valid: np.ndarray, gam: dict):
    '''GAM daily brownness from Daymet covariates, exactly as in the evaluate stage of retrospective.py; day of year, cell by day, onset'''
    weather, dates = features.load_daymet(year, cell_id, features.SEASON[1])
    static, defined, _ = features.static_covariates(cell_id, year)
    cube, season_dates = features.covariate_cube(weather, dates, static)  # cell, day, covariate
    day = doy_of(season_dates, year)
    pred = outcome.predict(cube.reshape(-1, cube.shape[2]), gam["beta"], gam["smooths"]).reshape(cube.shape[:2])  # cell, day
    pred[~(defined & ul_valid)] = np.nan
    onset, _ = evaluation.first_crossing(np.broadcast_to(day, pred.shape), pred)
    return day, pred, onset


def year_panel_figure(panel_years: list, cell_id: np.ndarray, show: np.ndarray, gam: dict, ph, grid, extent, boundary, under, tag: str):
    '''observed and GAM brownness at six composites and peak foliage onset, rows observed then GAM for each year; only cells with show true'''
    ul_valid = show
    under = maps.underlay(cell_id, show, grid)  # gray only for shown cells without a value
    years = list(ph["years"])
    fig, axes = plt.subplots(2 * len(panel_years), len(PANEL_DATES) + 1, figsize=(2.3 * (len(PANEL_DATES) + 1), 3.3 * 2 * len(panel_years)))
    for y, year in enumerate(panel_years):
        day, pred, onset = gam_daily(year, cell_id, ul_valid, gam)
        b = np.load(os.path.join(PROC, "brownness", f"brownness_{year}.npz"))
        assert np.array_equal(b["cell_id"], cell_id)
        starts = b["composite_start"]
        obs_doy = doy_of(b["obs_date"], year)  # cell, composite
        for c, md in enumerate(PANEL_DATES):
            k = int(np.argmin(np.abs(starts - np.datetime64(f"{year}-{md}"))))
            o = np.where(ul_valid, b["brownness"][:, k], np.nan)
            # prediction on each cell's observation date, or on the composite middle day where the cell was not observed
            mid = doy_of(np.array([starts[k] + np.timedelta64(COMPOSITE_DAYS // 2, "D")]), year)[0]
            target = np.where(np.isfinite(obs_doy[:, k]), obs_doy[:, k], mid)
            idx = (target - day[0]).astype(int)
            assert idx.min() >= 0 and idx.max() < len(day), "composite outside the predicted season"
            p = pred[np.arange(len(cell_id)), idx]
            for r, v in ((2 * y, o), (2 * y + 1, p)):
                im_b = maps.map_panels([axes[r, c]], [maps.to_raster(v, cell_id, grid)], [str(starts[k])], extent, boundary, under, "YlOrBr", 0.0, 1.0)
            logger.info("%d composite %s: %d observed cells, GAM minus observed brownness mean %.3f", year, starts[k], int(np.isfinite(o).sum()),
                        float(np.nanmean(p - o)))
        observed = np.where(ul_valid, ph["onset_doy"][:, years.index(year)], np.nan)
        pon = np.where(ul_valid, onset, np.nan)
        for r, v in ((2 * y, observed), (2 * y + 1, pon)):
            im_o = maps.map_panels([axes[r, -1]], [maps.to_raster(v, cell_id, grid)], ["peak foliage onset"], extent, boundary, under, "viridis", 265, 295)
        axes[2 * y, 0].annotate(f"{year}\nobserved", xy=(-0.12, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=10)
        axes[2 * y + 1, 0].annotate(f"{year}\npredicted", xy=(-0.12, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=10)
        test = np.load(SPLIT_NPZ)["test"]
        for part, cells in (("held-out test", test), ("training", ~test)):
            both = np.isfinite(observed) & np.isfinite(pon) & cells
            logger.info("%d peak foliage onset on %s cells: GAM minus observed bias %.2f, MAE %.2f days over %d cells (%s)",
                        year, part, np.mean((pon - observed)[both]), np.mean(np.abs(pon - observed)[both]), int(both.sum()),
                        "training year" if year <= 2023 else "held-out year")
    fig.colorbar(im_b, ax=axes[:, :-1], shrink=0.4, location="bottom", pad=0.02, label="brownness")
    fig.colorbar(im_o, ax=axes[:, -1], shrink=0.8, location="bottom", pad=0.02, label="peak foliage onset, day of year")
    path = os.path.join(FIG_DIR, f"brownness_and_onset_panels_{tag}_{'_'.join(str(y) for y in panel_years)}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


def main():
    parser = argparse.ArgumentParser(description="Figures and tables of the retrospective pipeline")
    parser.add_argument("--figure", default="all", choices=["all", "panel"], help="all figures and tables, or only the year panel figure")
    parser.add_argument("--years", type=int, nargs=3, default=None,
                        help="training years of the test-cell panel figure; default the early, typical and late years by median observed onset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    ul = np.load(os.path.join(PROC, "brownness", "u_l.npz"))
    cell_id, ul_valid = ul["cell_id"], ul["ul_valid"]
    ph = np.load(os.path.join(PROC, "phenology", "phenology_dates.npz"))
    assert np.array_equal(ph["cell_id"], cell_id)
    years = list(ph["years"])
    grid, extent = maps.load_grid()
    epsg = int(grid["epsg"])
    boundary = maps.load_boundary(epsg)
    under = maps.underlay(cell_id, ul_valid, grid)
    sites = pd.read_csv(os.path.join(REPO_ROOT, "configs", "sites.csv"))
    r3 = pd.read_csv(os.path.join(RESULTS_DIR, "r3_test_scores.tsv"), sep="\t")
    gam = outcome.load_fit(FIT_NPZ)["gam"]

    # year panel figure; the default training year has the statewide median observed onset closest to the 2001 to 2023 median
    # year panel figures of the split fit: training years with early, typical and late statewide median observed onset on the
    # held-out test cells only, and the held-out years 2024 and 2025 on all cells
    train = [y for y in years if y <= 2023]
    med = np.array([np.nanmedian(ph["onset_doy"][ul_valid, years.index(y)]) for y in train])
    order = np.argsort(med)
    picked = [train[order[0]], train[int(np.argmin(np.abs(med - np.median(med))))], train[order[-1]]]
    logger.info("statewide median observed onset per training year %s; early, typical, late %s", dict(zip(train, np.round(med, 1))), picked)
    test = np.load(SPLIT_NPZ)["test"]
    split_gam = outcome.load_fit(SPLIT_FIT_NPZ)["gam"]
    year_panel_figure(args.years if args.years else picked, cell_id, ul_valid & test, split_gam, ph, grid, extent, boundary, under, "training_years_test_cells")
    year_panel_figure(TEST_YEARS, cell_id, ul_valid, split_gam, ph, grid, extent, boundary, under, "heldout_years_all_cells")

    # per-year scores of the split fit, held-out test cells against training cells
    paths = [os.path.join(RESULTS_DIR, "split_by_year", f"split_scores_{y}.tsv") for y in years if y <= 2025]
    split = pd.concat([pd.read_csv(q, sep="\t") for q in paths])
    split.to_csv(os.path.join(RESULTS_DIR, "split_scores.tsv"), sep="\t", index=False, float_format="%.4f")
    summary = split.groupby(["year_role", "cells", "model"])[["onset_bias_days", "onset_mae_days", "brownness_rmse"]].mean()
    summary.to_csv(os.path.join(RESULTS_DIR, "split_scores_summary.tsv"), sep="\t", float_format="%.4f")
    logger.info("split scores, mean over years:\n%s", summary.to_string())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (model, part), g in split.groupby(["model", "cells"]):
        style = dict(marker="o", ms=3, ls="-" if part == "test" else ":", label=f"{model.upper()}, {part} cells")
        axes[0].plot(g["year"], g["onset_mae_days"], **style)
        axes[1].plot(g["year"], g["brownness_rmse"], **style)
    for ax, lab in zip(axes, ["peak foliage onset MAE, days", "brownness RMSE"]):
        ax.axvspan(2023.5, 2025.5, color="0.9")
        ax.set_xlabel("year")
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "split_scores_by_year.png"), dpi=150)
    plt.close(fig)
    if args.figure == "panel":
        return

    for year in TEST_YEARS:
        day, pred, onset = gam_daily(year, cell_id, ul_valid, gam)
        observed = ph["onset_doy"][:, years.index(year)]

        # check against the evaluate stage before any figure is drawn
        bias, mae, n = evaluation.onset_scores(onset[ul_valid], observed[ul_valid])
        ref = r3[(r3["year"] == year) & (r3["model"] == "gam")].iloc[0]
        assert abs(bias - ref["onset_bias_days"]) < 1e-3 and abs(mae - ref["onset_mae_days"]) < 1e-3 and n == ref["n_cells_scored"], \
            f"{year}: recomputed GAM onset scores {bias:.4f}, {mae:.4f}, {n} differ from r3_test_scores.tsv"
        logger.info("%d GAM onset bias %.4f, MAE %.4f days over %d cells, equal to r3_test_scores.tsv", year, bias, mae, n)

        # figure 1, brownness at four composites: observed, GAM at each cell's observation date, and their difference
        b = np.load(os.path.join(PROC, "brownness", f"brownness_{year}.npz"))
        assert np.array_equal(b["cell_id"], cell_id)
        starts = b["composite_start"]
        obs_doy = doy_of(b["obs_date"], year)  # cell, composite
        fig, axes = plt.subplots(3, len(MAP_DATES), figsize=(3.2 * len(MAP_DATES), 12))
        rows = {"observed": [], "GAM": [], "GAM minus observed": []}
        titles = []
        for md in MAP_DATES:
            k = int(np.argmin(np.abs(starts - np.datetime64(f"{year}-{md}"))))
            o = np.where(ul_valid, b["brownness"][:, k], np.nan)
            idx = obs_doy[:, k] - day[0]
            ok = np.isfinite(idx) & (idx >= 0) & (idx < len(day))
            p = np.full(len(cell_id), np.nan)
            p[ok] = pred[np.flatnonzero(ok), idx[ok].astype(int)]
            p[~np.isfinite(o)] = np.nan
            rows["observed"].append(maps.to_raster(o, cell_id, grid))
            rows["GAM"].append(maps.to_raster(p, cell_id, grid))
            rows["GAM minus observed"].append(maps.to_raster(p - o, cell_id, grid))
            titles.append(f"composite {starts[k]}, {int(np.isfinite(o).sum())} cells")
        for i, (name, rr) in enumerate(rows.items()):
            diff = name.endswith("observed") and name != "observed"
            im = maps.map_panels(axes[i], rr, [f"{name}\n{t}" for t in titles], extent, boundary, under,
                                 "RdBu_r" if diff else "YlOrBr", -0.5 if diff else 0.0, 0.5 if diff else 1.0, sites)
            fig.colorbar(im, ax=axes[i], shrink=0.7, label="brownness difference" if diff else "brownness")
        fig.suptitle(f"{year}: brownness, observed MODIS and GAM with Daymet weather (cells without a clear observation in gray)")
        fig.savefig(os.path.join(FIG_DIR, f"r3_brownness_maps_{year}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

        # figure 2, onset maps
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        r_obs = maps.to_raster(np.where(ul_valid, observed, np.nan), cell_id, grid)
        r_gam = maps.to_raster(np.where(ul_valid, onset, np.nan), cell_id, grid)
        lo, hi = np.nanpercentile(np.concatenate([observed[ul_valid], onset[ul_valid]]), [2, 98])
        maps.map_panels(axes[:2], [r_obs, r_gam], ["observed peak foliage onset", "GAM peak foliage onset"], extent, boundary, under, "viridis", lo, hi, sites)
        fig.colorbar(axes[1].images[-1], ax=axes[:2], shrink=0.7, label="peak foliage onset, day of year")
        im = maps.map_panels(axes[2:], [r_gam - r_obs], [f"GAM minus observed, bias {bias:.1f}, MAE {mae:.1f} days"], extent, boundary, under, "RdBu_r", -20, 20, sites)
        fig.colorbar(im, ax=axes[2], shrink=0.7, label="days")
        fig.suptitle(f"{year}: peak foliage onset (brownness 0.6), {n} cells scored")
        fig.savefig(os.path.join(FIG_DIR, f"r3_onset_maps_{year}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

        # figure 3, onset scatter
        both = ul_valid & np.isfinite(onset) & np.isfinite(observed)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.hexbin(observed[both], onset[both], gridsize=60, bins="log", cmap="Greys", extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], "r-", lw=0.8)
        ax.set_xlabel("observed peak foliage onset, day of year")
        ax.set_ylabel("GAM peak foliage onset, day of year")
        ax.set_title(f"{year}: bias {bias:.2f}, MAE {mae:.2f} days, {n} cells", fontsize=9)
        fig.savefig(os.path.join(FIG_DIR, f"r3_onset_scatter_{year}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("%d: wrote brownness maps, onset maps and onset scatter", year)

    # table 4, observed MODIS onset against the DACF peak report week, per zone and year
    cat, peaks, towns = dacf.load_tables()
    pos = np.searchsorted(grid["cell_id"], cell_id)
    zone = dacf.cell_zones(grid["x"][pos], grid["y"][pos], towns, epsg)
    reported = pd.concat([peaks[peaks["year"] < 2024].assign(source="archive"),
                          peaks[peaks["year"] == 2024].assign(source="archive"),
                          dacf.report_peak_dates(cat).query("year > 2024").assign(source="weekly reports", after_last_report="no")])
    out = []
    for _, r in reported.iterrows():
        year = int(r["year"])
        if year not in DACF_YEARS:
            continue
        in_zone = ul_valid & np.array([z == r["zone"] or z.rstrip("NS") == r["zone"] for z in zone])
        o = ph["onset_doy"][in_zone, years.index(year)]
        med = float(np.nanmedian(o))
        report_doy = float(r["peak_report_date"].day_of_year)
        out.append({"year": year, "zone": r["zone"], "source": r["source"], "dacf_peak_report_date": r["peak_report_date"].date(),
                    "dacf_after_last_report": r["after_last_report"], "n_cells": int(in_zone.sum()), "n_cells_with_onset": int(np.isfinite(o).sum()),
                    "modis_onset_median_doy": med, "modis_onset_median_date": (pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=med - 1)).date(),
                    "report_minus_onset_days": report_doy - med, "onset_in_week_before_report": bool(0 <= report_doy - med < 7)})
    table = pd.DataFrame(out)
    table.to_csv(os.path.join(RESULTS_DIR, "dacf_peak_weeks_vs_modis_onset.tsv"), sep="\t", index=False, float_format="%.1f")
    for (period, g) in table.groupby(table["year"] <= 2023):
        logger.info("%s: report minus MODIS onset median %.1f days, mean absolute %.1f days, onset in the week before the report in %d of %d zone-years",
                    "2020 to 2023" if period else "2024 to 2026", g["report_minus_onset_days"].median(), g["report_minus_onset_days"].abs().mean(),
                    int(g["onset_in_week_before_report"].sum()), len(g))
    zone_counts = pd.Series(zone[ul_valid]).value_counts().reindex(dacf.SUBZONES)
    logger.info("cells with valid U and L per nearest-town zone: %s", zone_counts.to_dict())

    # table 12, preprocessing counts, each read from a data-session output
    summary = pd.read_csv(os.path.join(PROC, "preprocessing_summary.tsv"), sep="\t")
    keep = ["grid_cells", "forest_cells_threshold_0.3", "forest_cells_threshold_0.5", "forest_cells_threshold_0.7", "forest_cells",
            "ul_valid_cells", "ul_valid_cells_if_min_L_years_3", "ul_valid_cells_if_min_L_years_10"]
    counts = summary[summary["quantity"].isin(keep)].assign(source="data/processed/preprocessing_summary.tsv")
    names = list(ph["count_names"])
    for i, year in enumerate(years):
        o = ph["onset_doy"][:, i]
        counts = pd.concat([counts, pd.DataFrame([
            {"quantity": f"onset_undefined_{year}", "value": int((ul_valid & ~np.isfinite(o)).sum()), "source": "phenology_dates.npz onset_doy"},
            {"quantity": f"onset_never_{year}", "value": int(ph["counts"][i, names.index("onset_never")]), "source": "phenology_dates.npz counts"},
            {"quantity": f"onset_above_at_first_{year}", "value": int(ph["counts"][i, names.index("onset_above_at_first")]), "source": "phenology_dates.npz counts"},
        ])])
    counts.to_csv(os.path.join(RESULTS_DIR, "preprocessing_counts.tsv"), sep="\t", index=False)
    logger.info("wrote %s and %s", RESULTS_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
