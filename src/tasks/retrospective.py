import os
import sys
import json
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.foliage import features, outcome, evaluation

logger = logging.getLogger(__name__)

# retrospective pipeline R1 to R3 of manuscript section 2.4.2, outcome models of section 3.4, decision D25
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrospective_config.json")
STAGES = ["rows", "fit", "evaluate", "split_fit", "split_evaluate"]
MODELS = ["lr", "gam"]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(REPO_ROOT, "data", "processed")
ROWS_DIR = os.path.join(features.FOLIAGE_DIR, "rows")
FIT_NPZ = os.path.join(features.FOLIAGE_DIR, "fits.npz")
SPLIT_FIT_NPZ = os.path.join(features.FOLIAGE_DIR, "fits_split.npz")  # fit on the training cells of the within-year split
SPLIT_NPZ = os.path.join(features.FOLIAGE_DIR, "split_cells.npz")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "retrospective")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "retrospective")
SITES_TSV = os.path.join(PROC, "site_candidates.tsv")


def load_rows(years: list, keep: np.ndarray = None):
    '''training rows of several years stacked: covariates of shape row, covariate, and the response; only cells with keep true if given'''
    X, y = [], []
    for year in years:
        d = np.load(os.path.join(ROWS_DIR, f"rows_{year}.npz"))
        assert list(d["covariates"]) == features.COVARIATES
        sel = keep[d["cell"]] if keep is not None else np.ones(len(d["y"]), dtype=bool)
        X.append(d["X"][sel])
        y.append(d["y"][sel])
    return np.concatenate(X), np.concatenate(y)


def split_cells(ul_valid: np.ndarray, fraction: float, seed: int):
    '''test mask over the forest cells: a random fraction of the cells with valid U and L, the same cells in every year; cached'''
    if os.path.exists(SPLIT_NPZ):
        d = np.load(SPLIT_NPZ)
        assert float(d["fraction"]) == fraction and int(d["seed"]) == seed and np.array_equal(d["ul_valid"], ul_valid), "split settings changed"
        return d["test"]
    valid = np.flatnonzero(ul_valid)
    rng = np.random.default_rng(seed)
    test = np.zeros(len(ul_valid), dtype=bool)
    test[rng.choice(valid, size=int(round(fraction * len(valid))), replace=False)] = True
    np.savez(SPLIT_NPZ, test=test, fraction=fraction, seed=seed, ul_valid=ul_valid)
    return test


def main():
    parser = argparse.ArgumentParser(description="Retrospective pipeline: training rows, fits of F, evaluation")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--index", type=int, default=None, help="1-based index into the row years, for SLURM arrays")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    with open(CONFIG_PATH) as f:
        config = json.load(f)["foliage"]
    train_years = list(range(config["train_years"][0], config["train_years"][1] + 1))
    test_years = config["test_years"]
    ul = np.load(os.path.join(PROC, "brownness", "u_l.npz"))
    cell_id, ul_valid = ul["cell_id"], ul["ul_valid"]

    if args.stage == "rows":
        # R1: rows of one year with Daymet covariates
        year = (train_years + test_years)[args.index - 1]
        os.makedirs(ROWS_DIR, exist_ok=True)
        r = features.training_rows(year, cell_id, ul_valid)
        np.savez_compressed(os.path.join(ROWS_DIR, f"rows_{year}.npz"), covariates=np.array(features.COVARIATES), **r)
        logger.info("wrote rows_%d.npz: %d rows; covariate ranges %s", year, len(r["y"]),
                    {c: (round(float(r["X"][:, k].min()), 3), round(float(r["X"][:, k].max()), 3)) for k, c in enumerate(features.COVARIATES)})
        return

    if args.stage in ("fit", "split_fit"):
        # R2: linear regression (eq. 10) and GAM (eq. 18) on the training years; split_fit uses only the training cells of the split
        split = args.stage == "split_fit"
        fit_npz = SPLIT_FIT_NPZ if split else FIT_NPZ
        prefix = "split_" if split else ""
        keep = ~split_cells(ul_valid, config["split"]["test_fraction"], config["split"]["seed"]) if split else None
        X, y = load_rows(train_years, keep)
        n = len(y)
        sst = float(((y - y.mean()) ** 2).sum())
        logger.info("%d training rows from %d to %d, response mean %.4f, standard deviation %.4f", n, train_years[0], train_years[-1], y.mean(), y.std())
        G, b, yy, _ = outcome.gram(X, y, outcome.lr_design)
        lr_beta = np.linalg.solve(G, b)
        lr_rss = yy - 2 * lr_beta @ b + lr_beta @ G @ lr_beta
        lr_gamma = lr_rss / (n - G.shape[0])
        lr_Q = G / lr_gamma  # prior precision of D26, the training posterior
        logger.info("linear regression: R squared %.4f, residual standard deviation %.4f", 1 - lr_rss / sst, np.sqrt(lr_gamma))

        smooths = outcome.setup_smooths(X)
        G, b, yy, _ = outcome.gram(X, y, outcome.gam_design, smooths)
        P = outcome.penalty(smooths)
        lam, grid, scores = outcome.choose_lambda(G, b, yy, n, P)
        gam_beta, gam_rss, edf, gcv = outcome.penalized_fit(G, b, yy, n, P, lam)
        gam_gamma = gam_rss / (n - edf)
        gam_Q = (G + lam * P) / gam_gamma
        logger.info("GAM: lambda %.4g by GCV (score %.6g), effective degrees of freedom %.2f of %d, R squared %.4f, residual standard deviation %.4f",
                    lam, gcv, edf, G.shape[0], 1 - gam_rss / sst, np.sqrt(gam_gamma))
        sm = outcome.save_smooths(smooths)
        np.savez_compressed(
            fit_npz, covariates=np.array(features.COVARIATES), train_years=np.array(train_years), n=n,
            lr_beta=lr_beta, lr_gamma=lr_gamma, lr_Q=lr_Q, lr_rss=lr_rss, gam_beta=gam_beta, gam_gamma=gam_gamma, gam_Q=gam_Q,
            gam_rss=gam_rss, gam_lambda=lam, gam_edf=edf, gam_gcv=gcv, gcv_grid=grid, gcv_scores=scores, sst=sst,
            **{f"gam_{k}": v for k, v in sm.items()},
        )
        os.makedirs(RESULTS_DIR, exist_ok=True)
        pd.DataFrame({"term": ["intercept"] + features.COVARIATES, "coefficient": lr_beta}).to_csv(
            os.path.join(RESULTS_DIR, f"{prefix}lr_coefficients.tsv"), sep="\t", index=False, float_format="%.6g")
        pd.DataFrame([{"model": "lr", "n_rows": n, "n_coefficients": len(lr_beta), "edf": len(lr_beta), "lambda": np.nan,
                       "r_squared": 1 - lr_rss / sst, "residual_sd": np.sqrt(lr_gamma)},
                      {"model": "gam", "n_rows": n, "n_coefficients": len(gam_beta), "edf": edf, "lambda": lam,
                       "r_squared": 1 - gam_rss / sst, "residual_sd": np.sqrt(gam_gamma)}]).to_csv(
            os.path.join(RESULTS_DIR, f"{prefix}fit_summary.tsv"), sep="\t", index=False, float_format="%.6g")

        # partial effects of the GAM between the 1st and 99th training percentiles, with the training density as a rug of quantiles
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(4, 4, figsize=(14, 12))
        for c, name in enumerate(features.COVARIATES):
            ax = axes.flat[c]
            lo, hi = np.percentile(X[:, c], [1, 99])
            xs = np.linspace(lo, hi, 200)
            ax.plot(xs, outcome.partial_effect(xs, c, gam_beta, smooths), color="k")
            ax.plot(np.percentile(X[:, c], np.arange(5, 100, 10)), np.full(10, ax.get_ylim()[0]), "|", color="0.5")
            ax.set_title(name)
            ax.axhline(0, color="0.8", lw=0.5)
        for ax in axes.flat[len(features.COVARIATES):]:
            ax.axis("off")
        fig.suptitle(f"GAM partial effects on brownness, 2001 to 2023 fit{', training cells of the split' if split else ''}, lambda {lam:.3g}, edf {edf:.1f}")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{prefix}gam_partial_effects.png"), dpi=120)
        logger.info("wrote %s and the tables in %s", fit_npz, RESULTS_DIR)
        return

    if args.stage == "split_evaluate":
        # scores of the split fit for one year: held-out test cells and training cells, onset and brownness
        year = (train_years + test_years)[args.index - 1]
        test = split_cells(ul_valid, config["split"]["test_fraction"], config["split"]["seed"])
        fits = outcome.load_fit(SPLIT_FIT_NPZ)
        ph = np.load(os.path.join(PROC, "phenology", "phenology_dates.npz"))
        assert np.array_equal(ph["cell_id"], cell_id)
        weather, dates = features.load_daymet(year, cell_id, features.SEASON[1])
        static, defined, _ = features.static_covariates(cell_id, year)
        cube, season_dates = features.covariate_cube(weather, dates, static)  # cell, day, covariate
        doy = (season_dates - np.datetime64(f"{year}-01-01")).astype(int) + 1.0
        observed = ph["onset_doy"][:, list(ph["years"]).index(year)]
        r = np.load(os.path.join(ROWS_DIR, f"rows_{year}.npz"))
        rows_out = []
        for model in MODELS:
            f = fits[model]
            pred = outcome.predict(cube.reshape(-1, cube.shape[2]), f["beta"], f["smooths"]).reshape(cube.shape[:2])  # cell, day
            pred[~(defined & ul_valid)] = np.nan
            onset, _ = evaluation.first_crossing(np.broadcast_to(doy, pred.shape), pred)
            row_pred = outcome.predict(r["X"], f["beta"], f["smooths"])
            for part, cells in (("test", test & ul_valid), ("train", ~test & ul_valid)):
                bias, mae, n_both = evaluation.onset_scores(onset[cells], observed[cells])
                rr = cells[r["cell"]]
                err, n_obs = evaluation.rmse(row_pred[rr], r["y"][rr])
                rows_out.append({"year": year, "year_role": "train" if year in train_years else "test", "cells": part, "model": model,
                                 "onset_bias_days": bias, "onset_mae_days": mae, "n_cells_scored": n_both, "brownness_rmse": err, "n_brownness_obs": n_obs})
                logger.info("%d %s %s cells: onset bias %.2f, MAE %.2f days over %d cells; brownness RMSE %.4f over %d observations",
                            year, model, part, bias, mae, n_both, err, n_obs)
        out_dir = os.path.join(RESULTS_DIR, "split_by_year")
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(os.path.join(out_dir, f"split_scores_{year}.tsv"), sep="\t", index=False, float_format="%.4f")
        return

    if args.stage == "evaluate":
        # R3: predictions for the test years from Daymet covariates
        fits = outcome.load_fit(FIT_NPZ)
        ph = np.load(os.path.join(PROC, "phenology", "phenology_dates.npz"))
        assert np.array_equal(ph["cell_id"], cell_id)
        sites = evaluation.site_cells(cell_id, SITES_TSV)
        rows_out, site_rows = [], []
        fig, axes = plt.subplots(len(test_years), len(sites), figsize=(5 * len(sites), 3.5 * len(test_years)), squeeze=False)
        for i, year in enumerate(test_years):
            weather, dates = features.load_daymet(year, cell_id, features.SEASON[1])
            static, defined, _ = features.static_covariates(cell_id, year)
            cube, season_dates = features.covariate_cube(weather, dates, static)  # cell, day, covariate
            doy = (season_dates - np.datetime64(f"{year}-01-01")).astype(int) + 1.0
            observed = ph["onset_doy"][:, list(ph["years"]).index(year)]
            r = np.load(os.path.join(ROWS_DIR, f"rows_{year}.npz"))
            b = np.load(os.path.join(PROC, "brownness", f"brownness_{year}.npz"))
            for model in MODELS:
                f = fits[model]
                pred = outcome.predict(cube.reshape(-1, cube.shape[2]), f["beta"], f["smooths"]).reshape(cube.shape[:2])  # cell, day
                pred[~(defined & ul_valid)] = np.nan
                onset, first = evaluation.first_crossing(np.broadcast_to(doy, pred.shape), pred)
                bias, mae, n_both = evaluation.onset_scores(onset[ul_valid], observed[ul_valid])
                rmse, n_obs = evaluation.rmse(outcome.predict(r["X"], f["beta"], f["smooths"]), r["y"])
                rows_out.append({"year": year, "model": model, "onset_bias_days": bias, "onset_mae_days": mae, "n_cells_scored": n_both,
                                 "n_valid_cells": int(ul_valid.sum()), "n_predicted_onset": int(np.isfinite(onset[ul_valid]).sum()),
                                 "n_reached_at_aug1": int(first[ul_valid].sum()), "n_observed_onset": int(np.isfinite(observed[ul_valid]).sum()),
                                 "brownness_rmse": rmse, "n_brownness_obs": n_obs,
                                 "predicted_onset_median": float(np.nanmedian(onset[ul_valid])), "observed_onset_median": float(np.nanmedian(observed[ul_valid]))})
                logger.info("%d %s: onset bias %.2f days, MAE %.2f days over %d cells; brownness RMSE %.4f over %d observations",
                            year, model, bias, mae, n_both, rmse, n_obs)
                for j, (site, pos) in enumerate(sites.items()):
                    site_rows.append({"year": year, "model": model, "site": site, "cell_id": int(cell_id[pos]),
                                      "predicted_onset_doy": onset[pos], "observed_onset_doy": observed[pos]})
                    ax = axes[i, j]
                    ax.plot(doy, pred[pos], label=model)
                    if model == MODELS[-1]:
                        od = (b["obs_date"][pos] - np.datetime64(f"{year}-01-01")).astype("timedelta64[D]").astype(float) + 1
                        ax.plot(od, b["brownness"][pos], "o", color="k", ms=3, label="observed")
                        ax.axhline(evaluation.ONSET_THRESHOLD, color="0.7", lw=0.5)
                        ax.axvline(observed[pos], color="k", ls=":", lw=0.8, label="observed peak foliage onset")
                        ax.set_xlim(doy[0], doy[-1])
                        ax.set_title(f"{site} {year}, brownness and peak foliage onset")
                        ax.legend(fontsize=7)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(os.path.join(RESULTS_DIR, "r3_test_scores.tsv"), sep="\t", index=False, float_format="%.4f")
        pd.DataFrame(site_rows).to_csv(os.path.join(RESULTS_DIR, "r3_site_onsets.tsv"), sep="\t", index=False, float_format="%.2f")
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, "r3_site_curves.png"), dpi=120)
        logger.info("wrote %s and %s", RESULTS_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
