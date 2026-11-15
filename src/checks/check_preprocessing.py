import os
import sys
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# counts, figures, site candidates and the literature check of the MODIS preprocessing, step 4 of references/preprocessing_plan.md
ONSET_THRESHOLD = 0.6  # decision D1
PERSIST = 2  # provisional onset rule for O2: brownness stays at or above the threshold for the next two observations
SEASON = ("08-01", "12-31")  # observations used for the provisional onset, month-day
ONSET_YEARS = list(range(2001, 2024))
NORTH_LAT = 46.0
LIT_RANGE = (250, 280)  # day of year; Liu et al. 2017 figure 3a puts northern Maine in the 260 to 270 class
MIN_ONSET_YEARS = 10
N_CANDIDATES = 5
SERIES_YEARS = [2021, 2022, 2023, 2024, 2025]
L_MIN_OPTIONS = [3, 5, 10]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(REPO_ROOT, "data", "processed")
SITES_CSV = os.path.join(REPO_ROOT, "configs", "sites.csv")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "preprocessing")
SUMMARY_TSV = os.path.join(PROC, "preprocessing_summary.tsv")
CANDIDATES_TSV = os.path.join(PROC, "site_candidates.tsv")
ONSET_NPZ = os.path.join(PROC, "brownness", "onset_provisional.npz")


def cell_map(values: np.ndarray, cells: np.ndarray, grid):
    '''place values of the given analysis cells on the 1 km grid raster, NaN elsewhere'''
    out = np.full(tuple(grid["shape"]), np.nan)  # grid row, grid column
    out[grid["row"][cells], grid["col"][cells]] = values
    return out


def provisional_onset(brown: np.ndarray, obs_date: np.ndarray, year: int):
    '''first date brownness reaches the threshold and stays there for the next PERSIST observations, linearly interpolated; NaN if undefined'''
    lo = np.datetime64(f"{year}-{SEASON[0]}")
    hi = np.datetime64(f"{year}-{SEASON[1]}")
    valid = np.isfinite(brown) & (obs_date >= lo) & (obs_date <= hi)  # cell, composite
    day = (obs_date - np.datetime64(f"{year}-01-01")).astype("timedelta64[D]").astype(np.float64) + 1  # day of year
    # move each cell's valid observations to the left, keeping their time order
    order = np.argsort(~valid, axis=1, kind="stable")
    b = np.take_along_axis(np.where(valid, brown, np.nan), order, axis=1)
    t = np.take_along_axis(np.where(valid, day, np.nan), order, axis=1)
    above = b >= ONSET_THRESHOLD  # NaN compares False
    persist = above[:, : above.shape[1] - PERSIST].copy()
    for k in range(1, PERSIST + 1):
        persist &= above[:, k: above.shape[1] - PERSIST + k]
    has = persist.any(axis=1)
    k = persist.argmax(axis=1)
    # the observation before k is below the threshold by construction, so k equal to 0 means the crossing is before the first observation
    ok = has & (k >= 1)
    rows = np.flatnonzero(ok)
    k0, k1 = k[rows] - 1, k[rows]
    b0, b1 = b[rows, k0], b[rows, k1]
    t0, t1 = t[rows, k0], t[rows, k1]
    onset = np.full(brown.shape[0], np.nan)
    onset[rows] = t0 + (ONSET_THRESHOLD - b0) / (b1 - b0) * (t1 - t0)
    return onset, int((has & (k == 0)).sum()), int((~has).sum())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(FIG_DIR, exist_ok=True)
    grid = np.load(os.path.join(PROC, "grid", "analysis_grid.npz"))
    mask = np.load(os.path.join(PROC, "forest_mask", "forest_mask.npz"))
    ul = np.load(os.path.join(PROC, "brownness", "u_l.npz"))
    cells = ul["cell_id"]
    valid = ul["ul_valid"]
    summary = []

    # counts
    summary.append(("grid_cells", len(grid["cell_id"])))
    for t, kp, kc in zip(mask["thresholds"], mask["kept_pixels"], mask["kept_cells"]):
        summary += [(f"kept_pixels_threshold_{t:.1f}", kp), (f"forest_cells_threshold_{t:.1f}", kc)]
    summary.append(("forest_mass_balance_relative_difference", float(mask["mass_balance_rel"])))
    summary.append(("forest_cells", len(cells)))
    summary.append(("ul_valid_cells", int(valid.sum())))
    base = (ul["n_years_U"] >= ul["min_years"]) & (ul["U"] - ul["L"] > 0)
    for m in L_MIN_OPTIONS:
        summary.append((f"ul_valid_cells_if_min_L_years_{m}", int((base & (ul["n_years_L"] >= m)).sum())))
    hist = np.bincount(ul["n_years_L"], minlength=len(ul["ul_years"]) + 1)
    for n, c in enumerate(hist):
        summary.append((f"cells_with_{n}_L_years", int(c)))

    # share of kept pixel-composites failing each rule, August to November and full year, over all EVI2 files
    diag_rows = []
    for path in sorted(glob.glob(os.path.join(PROC, "modis", "evi2_*.npz"))):
        d = np.load(path)
        months = d["composite_start"].astype("datetime64[M]").astype(int) % 12 + 1
        diag_rows.append(pd.DataFrame(d["diag"], columns=d["diag_names"]).assign(month=months))
    diag = pd.concat(diag_rows, ignore_index=True)
    rule_names = [c for c in diag.columns if c not in ("kept", "valid", "extra_cloud_state_11", "extra_adjacent", "valid_shoreline", "month")]
    for label, sub in [("aug_nov", diag[diag["month"].between(8, 11)]), ("full_year", diag)]:
        kept_total = sub["kept"].sum()
        for r in rule_names:
            summary.append((f"fail_share_{r}_{label}", sub[r].sum() / kept_total))
        summary.append((f"valid_share_{label}", sub["valid"].sum() / kept_total))
        summary.append((f"extra_share_of_valid_if_cloud_state_11_{label}", sub["extra_cloud_state_11"].sum() / sub["valid"].sum()))
        summary.append((f"extra_share_of_valid_if_adjacent_kept_{label}", sub["extra_adjacent"].sum() / sub["valid"].sum()))
        summary.append((f"shoreline_share_of_valid_{label}", sub["valid_shoreline"].sum() / sub["valid"].sum()))

    # maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 14))
    im = axes[0, 0].imshow(mask["frac_stable"], vmin=0, vmax=1, cmap="Greens")
    axes[0, 0].set_title("stable deciduous or mixed fraction, MODIS grid")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.6)
    for ax, (name, vals, vmin, vmax, cmap) in zip(
        axes.ravel()[1:],
        [("U", ul["U"], 0.3, 0.8, "viridis"), ("L", ul["L"], 0.1, 0.6, "viridis"),
         ("U minus L", ul["U"] - ul["L"], 0, 0.5, "magma"), ("number of L years", ul["n_years_L"], 0, 23, "cividis"),
         ("valid U and L", valid.astype(float), 0, 1, "Greys")],
    ):
        im = ax.imshow(cell_map(vals, cells, grid), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f"{name}, 1 km forest cells")
        fig.colorbar(im, ax=ax, shrink=0.6)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "maps.png"), dpi=110)
    plt.close(fig)

    # site candidates: cell means of the forest fractions over the MODIS pixels whose centers fall in the cell
    pc = mask["pixel_cell"].ravel()
    on = pc >= 0
    n_cell_pix = np.bincount(pc[on], minlength=len(grid["cell_id"]))
    cell_mean = {name: np.bincount(pc[on], weights=np.nan_to_num(mask[name].ravel()[on]), minlength=len(grid["cell_id"])) / np.maximum(n_cell_pix, 1)
                 for name in ("frac_stable", "frac41", "frac42", "frac43")}
    n_kept = np.bincount(pc[mask["kept"].ravel()], minlength=len(grid["cell_id"]))
    sites = pd.read_csv(SITES_CSV)
    vc = cells[valid]
    rows = []
    for _, s in sites.iterrows():
        dist = np.hypot(grid["x"][vc] - s["easting_m"], grid["y"][vc] - s["northing_m"])
        for rank, i in enumerate(np.argsort(dist)[:N_CANDIDATES], start=1):
            c = vc[i]
            j = np.searchsorted(cells, c)
            rows.append({
                "site_id": s["site_id"], "rank": rank, "cell_id": c, "distance_km": dist[i] / 1000,
                "lon": grid["lon"][c], "lat": grid["lat"][c], "kept_500m_pixels": n_kept[c],
                "frac_stable": cell_mean["frac_stable"][c], "frac41_2021": cell_mean["frac41"][c],
                "frac42_2021": cell_mean["frac42"][c], "frac43_2021": cell_mean["frac43"][c],
                "U": ul["U"][j], "L": ul["L"][j], "n_years_L": ul["n_years_L"][j],
            })
    candidates = pd.DataFrame(rows)
    candidates.to_csv(CANDIDATES_TSV, sep="\t", index=False, float_format="%.4f")
    logger.info("site candidates:\n%s", candidates.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # brownness time series at the nearest candidate of each site
    fig, axes = plt.subplots(1, len(sites), figsize=(6 * len(sites), 4.5), sharey=True)
    for ax, (_, s) in zip(np.atleast_1d(axes), sites.iterrows()):
        c = candidates[(candidates["site_id"] == s["site_id"]) & (candidates["rank"] == 1)]["cell_id"].iloc[0]
        j = np.searchsorted(cells, c)
        for year in SERIES_YEARS:
            d = np.load(os.path.join(PROC, "brownness", f"brownness_{year}.npz"))
            ok = np.isfinite(d["brownness"][j])
            doy = (d["obs_date"][j][ok] - np.datetime64(f"{year}-01-01")).astype(int) + 1
            ax.plot(doy, d["brownness"][j][ok], marker="o", ms=3, label=str(year))
        ax.axhline(ONSET_THRESHOLD, color="k", lw=0.8, ls="--")
        ax.set_xlim(180, 340)
        ax.set_ylim(-1, 2)
        ax.set_title(f"{s['site_name']}, cell {c}")
        ax.set_xlabel("day of year")
    np.atleast_1d(axes)[0].set_ylabel("observed brownness")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "site_brownness.png"), dpi=110)
    plt.close(fig)

    # provisional onset for the literature check, valid cells only
    onset = np.full((len(cells), len(ONSET_YEARS)), np.nan)  # cell, year
    for i, year in enumerate(ONSET_YEARS):
        d = np.load(os.path.join(PROC, "brownness", f"brownness_{year}.npz"))
        onset[:, i], n_before_first, n_none = provisional_onset(d["brownness"], d["obs_date"], year)
        onset[~valid, i] = np.nan
        summary += [(f"onset_defined_cells_{year}", int(np.isfinite(onset[:, i]).sum())),
                    (f"onset_before_first_observation_{year}", n_before_first), (f"onset_never_{year}", n_none)]
    np.savez_compressed(ONSET_NPZ, cell_id=cells, years=np.array(ONSET_YEARS), onset_doy=onset,
                        note=np.array("provisional, O2 open: interpolated first crossing of 0.6 that persists for 2 observations"))
    lat = grid["lat"][cells]
    n_def = np.isfinite(onset).sum(axis=1)
    cell_mean_onset = np.where(n_def >= MIN_ONSET_YEARS, np.nanmean(np.where(np.isfinite(onset), onset, np.nan), axis=1), np.nan)
    for label, sel in [("north", lat >= NORTH_LAT), ("all", np.ones(len(cells), dtype=bool))]:
        pooled = onset[sel][np.isfinite(onset[sel])]
        means = cell_mean_onset[sel][np.isfinite(cell_mean_onset[sel])]
        p10, p50, p90 = np.percentile(pooled, [10, 50, 90])
        summary += [(f"onset_{label}_pooled_p10", p10), (f"onset_{label}_pooled_median", p50), (f"onset_{label}_pooled_p90", p90),
                    (f"onset_{label}_cell_mean_median", np.median(means)), (f"onset_{label}_cells_with_{MIN_ONSET_YEARS}_years", len(means))]
        logger.info("provisional onset, %s: pooled cell-years median day %.1f (10th %.1f, 90th %.1f); median of cell means %.1f over %d cells",
                    label, p50, p10, p90, np.median(means), len(means))
    north_median = np.median(cell_mean_onset[(lat >= NORTH_LAT) & np.isfinite(cell_mean_onset)])
    within = LIT_RANGE[0] <= north_median <= LIT_RANGE[1]
    summary.append(("literature_check_north_within_range", within))
    logger.info("literature check: northern median of cell mean onsets %.1f, expected range %d to %d: %s",
                north_median, LIT_RANGE[0], LIT_RANGE[1], "WITHIN" if within else "OUTSIDE, stop and report before using the data")

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(cell_map(cell_mean_onset, cells, grid), vmin=240, vmax=300, cmap="plasma")
    ax.set_title("provisional mean onset day of year, 2001 to 2023")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "onset_mean_provisional.png"), dpi=110)
    plt.close(fig)

    pd.DataFrame(summary, columns=["quantity", "value"]).to_csv(SUMMARY_TSV, sep="\t", index=False)
    logger.info("wrote %s, %s and figures in %s", SUMMARY_TSV, CANDIDATES_TSV, FIG_DIR)


if __name__ == "__main__":
    main()
