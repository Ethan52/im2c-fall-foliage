import os
import sys
import json
import logging
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.foliage import features, outcome, assimilation, evaluation

logger = logging.getLogger(__name__)

# prospective pipeline P1 to P3 of manuscript section 2.4.3 with the data assimilation of section 3.5, decisions D6, D21 and D26
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prospective_config.json")
STAGES = ["assimilate", "forecast", "evaluate"]
MODELS = ["lr", "gam"]
VARIANTS = ["fit", "assimilated"]  # coefficients of the retrospective fit (P3 alone), and after the Kalman updates (stage 2.5)
COMPOSITE_DAYS = 8

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(REPO_ROOT, "data", "processed")
FIT_NPZ = os.path.join(features.FOLIAGE_DIR, "fits.npz")
ROWS_DIR = os.path.join(features.FOLIAGE_DIR, "rows")
FORECAST_OUT = os.path.join(features.FOLIAGE_DIR, "forecasts")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "prospective")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "prospective")
SITES_TSV = os.path.join(PROC, "site_candidates.tsv")


def doy(dates: np.ndarray, year: int):
    '''day of year as float, NaN for NaT'''
    out = (dates - np.datetime64(f"{year}-01-01")).astype("timedelta64[D]").astype(np.float64) + 1
    out[np.isnat(dates)] = np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="Prospective pipeline: data assimilation, forecasts, scores")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--index", type=int, default=None, help="1-based index into the issue dates of the foliage years, for SLURM arrays")
    parser.add_argument("--prior-years", type=int, default=0,
                        help="power prior weight of the training data in seasons, a0 = this over the number of training years; 0 keeps the full training posterior")
    parser.add_argument("--issue-date", default=None, help="a single issue date outside the foliage years, for example 2026-08-01 (D7), in place of --index")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    config = cfg["foliage"]
    years = config["years"]
    issues = np.array([np.datetime64(t[:10]) for t in cfg["weather"]["issue_times"] if int(t[:4]) in years])
    ul = np.load(os.path.join(PROC, "brownness", "u_l.npz"))
    cell_id, ul_valid = ul["cell_id"], ul["ul_valid"]
    fits = outcome.load_fit(FIT_NPZ)
    tag = f"_prior{args.prior_years}yr" if args.prior_years else ""  # output suffix of a power prior run
    if args.issue_date:
        assert int(args.issue_date[:4]) not in years and not args.prior_years, "--issue-date is for a year outside the test years"
        issues = np.array([np.datetime64(args.issue_date)])
        tag = f"_{args.issue_date}"

    if args.stage == "assimilate":
        # Kalman filter with A the identity and Sigma zero, continuing the training fit (D26)
        features.dayl_climatology(cell_id)  # cached here once, before the forecast array reads it
        for year in years:
            r = np.load(os.path.join(ROWS_DIR, f"rows_{year}.npz"))
            comp_end = r["composite_start"] + np.timedelta64(COMPOSITE_DAYS - 1, "D")
            year_issues = issues[issues.astype("datetime64[Y]").astype(int) + 1970 == year]
            out = {"issue_dates": year_issues}
            for model in MODELS:
                f = fits[model]
                design = outcome.lr_design if f["smooths"] is None else outcome.gam_design
                extra = () if f["smooths"] is None else (f["smooths"],)
                m, Q = f["beta"].copy(), f["Q"].copy()
                if args.prior_years:
                    # power prior (Ibrahim and Chen 2000): the training term of the prior precision is multiplied by a0, the GAM roughness term of eq. 32 is not
                    fd = np.load(FIT_NPZ)
                    a0 = args.prior_years / len(fd["train_years"])
                    rough = 0.0 if f["smooths"] is None else float(fd["gam_lambda"]) * outcome.penalty(f["smooths"]) / f["gamma"]
                    Q = a0 * (Q - rough) + rough
                    logger.info("%s %s: power prior a0 %.5f", year, model, a0)
                states, n_new, rel = [], [], []
                prev = None
                for t in year_issues:
                    # composites that ended before t and after the previous issue date; with Sigma zero the prediction step leaves m and Q unchanged
                    new = comp_end < t
                    if prev is not None:
                        new &= comp_end >= prev
                    if new.any():
                        G, b, _, _ = outcome.gram(r["X"][new], r["y"][new], design, *extra)
                        m, Q = assimilation.update_from_sums(m, Q, G, b, f["gamma"])
                    states.append(m.copy())
                    n_new.append(int(new.sum()))
                    rel.append(float(np.linalg.norm(m - f["beta"]) / np.linalg.norm(f["beta"])))
                    logger.info("%s %s %s: %d new observations, relative change of beta from the fit %.3e",
                                year, model, t, n_new[-1], rel[-1])
                    prev = t
                out[f"m_{model}"] = np.stack(states)
                out[f"n_new_{model}"] = np.array(n_new)
                out[f"relative_change_{model}"] = np.array(rel)
            os.makedirs(features.FOLIAGE_DIR, exist_ok=True)
            np.savez_compressed(os.path.join(features.FOLIAGE_DIR, f"assimilation_{year}{tag}.npz"), **out)
        return

    if args.stage == "forecast":
        t = issues[0] if args.issue_date else issues[args.index - 1]
        year = int(str(t)[:4])
        if year in years:
            a = np.load(os.path.join(features.FOLIAGE_DIR, f"assimilation_{year}{tag}.npz"))
            k = int(np.flatnonzero(a["issue_dates"] == t)[0])
            betas = {model: np.stack([fits[model]["beta"], a[f"m_{model}"][k]], axis=1) for model in MODELS}  # coefficient, variant
            observed, obs_dates = features.load_daymet(year, cell_id, features.SEASON[1])
        else:
            # 2026 (D7): no season observations before August 1, so the assimilated variant equals the fit; weather before t is corrected gridMET (D8)
            betas = {model: np.stack([fits[model]["beta"], fits[model]["beta"]], axis=1) for model in MODELS}
            g = np.load(os.path.join(PROC, "gridmet_daymet", f"gridmet_daymet_{year}.npz"))
            assert np.array_equal(g["cell_id"], cell_id)
            obs_dates = g["date"]
            assert obs_dates[0] == np.datetime64(f"{year}-{features.WEATHER_START}"), f"gridMET starts {obs_dates[0]}"
            observed = {v: g[v] for v in ("tmax", "tmin", "prcp", "srad", "vp", "dayl")}
        before = obs_dates < t
        static, defined, _ = features.static_covariates(cell_id, year)
        usable = defined & ul_valid
        clim = features.dayl_climatology(cell_id)
        # observations available at t for the spliced curve, and those after t for the brownness score
        b = np.load(os.path.join(PROC, "brownness", f"brownness_{year}.npz"))
        comp_end = b["composite_start"] + np.timedelta64(COMPOSITE_DAYS - 1, "D")
        season_start = np.datetime64(f"{year}-{features.SEASON[0]}")
        avail = (comp_end < t)[None, :] & (b["obs_date"] >= season_start) & np.isfinite(b["brownness"]) & usable[:, None]
        obs_day = np.where(avail, doy(b["obs_date"], year), np.nan)  # cell, composite
        obs_val = np.where(avail, b["brownness"], np.nan)
        n_map, n_mem = len(config["mappings"]), None
        for i, mapping in enumerate(config["mappings"]):
            fc = np.load(os.path.join(REPO_ROOT, config["forecast_dir"], mapping, f"{year}_{t}.npz"))
            assert np.array_equal(fc["cell_id"], cell_id), f"{mapping} {t}: cell list differs"
            fdates = fc["date"]
            assert fdates[0] == t, f"{mapping} {t}: forecast starts {fdates[0]}"
            dates = np.concatenate([obs_dates[before], fdates])
            assert np.all(np.diff(dates) == np.timedelta64(1, "D")), "spliced dates are not consecutive"
            dayl = np.concatenate([observed["dayl"][:, before], clim[:, features.leap_slot(fdates)]], axis=1)
            after = (b["obs_date"] >= t) & (b["obs_date"] <= fdates[-1]) & np.isfinite(b["brownness"]) & usable[:, None]
            if n_mem is None:
                n_mem = fc["tmax"].shape[0]
                onset = np.full((n_map, len(MODELS), len(VARIANTS), n_mem, len(cell_id)), np.nan, dtype=np.float32)
                pred_obs = np.full((n_map, len(MODELS), len(VARIANTS), n_mem) + b["brownness"].shape, np.nan, dtype=np.float32)
            fcv = {v: fc[v] for v in ("tmax", "tmin", "prcp", "srad", "vp")}  # member, cell, day; read once, each access of the npz decompresses
            assert all(x.shape == (n_mem, len(cell_id), len(fdates)) for x in fcv.values())
            for mem in range(n_mem):
                weather = {v: np.concatenate([observed[v][:, before], fcv[v][mem]], axis=1) for v in fcv}
                weather["dayl"] = dayl
                cube, season_dates = features.covariate_cube(weather, dates, static)  # cell, day, covariate
                fut = season_dates >= t
                pred_day = doy(season_dates[fut], year)
                rows = cube[:, fut].reshape(-1, cube.shape[2])
                for j, model in enumerate(MODELS):
                    pred = outcome.predict(rows, betas[model], fits[model]["smooths"]).reshape(len(cell_id), int(fut.sum()), len(VARIANTS))
                    pred[~usable] = np.nan
                    for v in range(len(VARIANTS)):
                        o, first = evaluation.spliced_onset(obs_day, obs_val, pred_day, pred[:, :, v])
                        # a curve that never reaches the threshold within the forecast is marked infinite, one reached at its first point stays NaN
                        never = usable & np.isnan(o) & ~first
                        o[never] = np.inf
                        onset[i, j, v, mem] = o
                        cr, cc = np.nonzero(after)
                        pidx = (doy(b["obs_date"][cr, cc], year) - pred_day[0]).astype(int)
                        pred_obs[i, j, v, mem, cr, cc] = pred[cr, pidx, v]
                logger.info("%s %s member %d done", t, mapping, mem)
        os.makedirs(FORECAST_OUT, exist_ok=True)
        np.savez_compressed(
            os.path.join(FORECAST_OUT, f"forecast_{t}{'' if args.issue_date else tag}.npz"), issue_date=t, mappings=np.array(config["mappings"]), models=np.array(MODELS),
            variants=np.array(VARIANTS), cell_id=cell_id, onset_doy=onset, pred_brownness_median=np.nanmedian(pred_obs, axis=3),
        )
        logger.info("wrote forecast_%s.npz: onset of shape mapping, model, variant, member, cell", t)
        return

    if args.stage == "evaluate":
        ph = np.load(os.path.join(PROC, "phenology", "phenology_dates.npz"))
        sites = evaluation.site_cells(cell_id, SITES_TSV)
        rows_out, site_rows = [], []
        for t in issues:
            year = int(str(t)[:4])
            d = np.load(os.path.join(FORECAST_OUT, f"forecast_{t}{'' if args.issue_date else tag}.npz"))
            observed = ph["onset_doy"][:, list(ph["years"]).index(year)]
            b = np.load(os.path.join(PROC, "brownness", f"brownness_{year}.npz"))
            for i, mapping in enumerate(d["mappings"]):
                for j, model in enumerate(d["models"]):
                    for v, variant in enumerate(d["variants"]):
                        o = d["onset_doy"][i, j, v].astype(np.float64)  # member, cell
                        med = np.nanmedian(o, axis=0)
                        no_onset = np.isinf(med)
                        med[no_onset] = np.nan
                        bias, mae, n_both = evaluation.onset_scores(med[ul_valid], observed[ul_valid])
                        err, n_obs = evaluation.rmse(d["pred_brownness_median"][i, j, v], b["brownness"])
                        rows_out.append({"year": year, "issue_date": str(t), "mapping": mapping, "model": model, "variant": variant,
                                         "onset_bias_days": bias, "onset_mae_days": mae, "n_cells_scored": n_both,
                                         "n_no_forecast_onset_by_season_end": int((no_onset & ul_valid).sum()),
                                         "brownness_rmse_after_issue": err, "n_brownness_obs": n_obs})
                        for site, pos in sites.items():
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)  # a site cell without any member onset gives NaN
                                q = np.nanpercentile(o[:, pos], [10, 50, 90])
                            site_rows.append({"year": year, "issue_date": str(t), "mapping": mapping, "model": model, "variant": variant,
                                              "site": site, "forecast_p10": q[0], "forecast_median": q[1], "forecast_p90": q[2],
                                              "observed_onset_doy": observed[pos]})
        os.makedirs(RESULTS_DIR, exist_ok=True)
        scores = pd.DataFrame(rows_out)
        scores.to_csv(os.path.join(RESULTS_DIR, f"prospective_scores{tag}.tsv"), sep="\t", index=False, float_format="%.4f")
        site_df = pd.DataFrame(site_rows)
        site_df.to_csv(os.path.join(RESULTS_DIR, f"prospective_site_onsets{tag}.tsv"), sep="\t", index=False, float_format="%.2f")
        logger.info("scores:\n%s", scores.to_string(index=False))

        if args.issue_date:
            return
        # onset mean absolute error against issue date
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, len(years), figsize=(7 * len(years), 4.5), squeeze=False)
        for y, year in enumerate(years):
            ax = axes[0, y]
            for (mapping, model, variant), g in scores[scores["year"] == year].groupby(["mapping", "model", "variant"]):
                ax.plot(pd.to_datetime(g["issue_date"]), g["onset_mae_days"], marker="o", ms=3,
                        ls="-" if variant == "fit" else "--", label=f"{model} {mapping} {variant}")
            ax.set_title(f"{year}: peak foliage onset MAE of the ensemble median, valid cells")
            ax.set_ylabel("days")
            ax.legend(fontsize=7)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"onset_mae_by_issue_date{tag}.png"), dpi=120)
        logger.info("wrote %s and %s", RESULTS_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
