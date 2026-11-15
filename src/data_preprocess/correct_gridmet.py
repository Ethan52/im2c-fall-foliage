import os
import sys
import logging

import numpy as np
import pandas as pd
import pyproj
from scipy.io import netcdf_file
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# 2026 gridMET on the forest cells, corrected to Daymet, decision D8
# each forest cell takes the nearest gridMET cell with data (O7); a per-cell linear map from gridMET to Daymet is fitted on
# June 1 to November 30 of the fit years; precipitation uses a slope with no intercept, so dry days stay dry
GRIDMET_VARS = {"tmax": "tmmx", "tmin": "tmmn", "prcp": "pr", "srad": "srad", "vpd": "vpd"}  # our name, gridMET file name
VARIABLES = ["tmax", "tmin", "prcp", "srad", "vp"]  # Daymet variables delivered
NO_INTERCEPT = ["prcp"]
FIT_YEARS = list(range(2001, 2026))  # D8, every year with both datasets
TRAIN_YEARS = list(range(2001, 2024))  # D5, refit for the held-out check on 2024 and 2025
TEST_YEARS = [2024, 2025]
SEASON = ("06-01", "11-30")  # month-day; the weather features reach back to June 1 (O12)
TARGET_YEAR = 2026
GRIDMET_FILL = 32767

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRIDMET_DIR = os.path.join(REPO_ROOT, "data", "raw", "gridmet")
DAYMET_DIR = os.path.join(REPO_ROOT, "data", "processed", "daymet")
GRID_NPZ = os.path.join(REPO_ROOT, "data", "processed", "grid", "analysis_grid.npz")
MASK_NPZ = os.path.join(REPO_ROOT, "data", "processed", "forest_mask", "forest_mask.npz")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "gridmet_daymet")


def read_gridmet(var: str, year: int, iy: np.ndarray, ix: np.ndarray):
    '''one gridMET variable and year at the given grid indices, unpacked to physical units; returns dates and values of shape cell, day'''
    with netcdf_file(os.path.join(GRIDMET_DIR, f"gridmet_maine_{var}_{year}.nc"), "r", mmap=False) as nc:
        names = [k for k in nc.variables if k not in ("day", "lat", "lon", "crs")]
        assert len(names) == 1, f"{var} {year} data variables {names}"
        v = nc.variables[names[0]]
        raw = v.data  # day, lat, lon; the file's dimensions attribute is wrong (V4)
        assert raw.dtype == np.dtype(">i2") and v._attributes.get("_Unsigned") == b"true", f"{var} {year} storage"
        scale, offset = float(v._attributes["scale_factor"]), float(v._attributes["add_offset"])
        days = nc.variables["day"].data.copy()  # days since 1900-01-01
        assert raw.shape[0] == len(days)
        stored = raw[:, iy, ix].astype(np.int32).T  # cell, day
    stored = np.where(stored < 0, stored + 65536, stored)  # the stored integers are unsigned
    n_fill = int((stored == GRIDMET_FILL).sum())
    assert n_fill == 0, f"{var} {year}: {n_fill} cell-days are fill at the chosen gridMET cells"
    dates = np.datetime64("1900-01-01") + days.astype(np.int64).astype("timedelta64[D]")
    return dates, stored * scale + offset


def saturation_vp(t: np.ndarray):
    '''saturation vapor pressure in Pa over water at temperature in degrees C, FAO-56 (Allen et al. 1998) equation 11'''
    return 610.8 * np.exp(17.27 * t / (t + 237.3))


def gridmet_as_daymet(year: int, iy: np.ndarray, ix: np.ndarray, dayl: np.ndarray):
    '''gridMET of one year in Daymet units before correction; dayl is the daylength in seconds of shape cell, day'''
    out = {}
    dates = None
    for ours, var in GRIDMET_VARS.items():
        d, values = read_gridmet(var, year, iy, ix)
        if dates is None:
            dates = d
        assert np.array_equal(d, dates), f"{var} {year} dates differ"
        out[ours] = values
    tmax = out["tmax"] - 273.15  # K to degrees C
    tmin = out["tmin"] - 273.15
    # Daymet srad is the mean over the daylight period, gridMET srad is the mean over 24 hours
    srad = out["srad"] * 86400.0 / dayl
    # vapor pressure from the deficit, assuming gridMET's deficit uses the mean of saturation at tmax and tmin (FAO-56 equation 12)
    vp = 0.5 * (saturation_vp(tmax) + saturation_vp(tmin)) - out["vpd"] * 1000.0
    return dates, {"tmax": tmax, "tmin": tmin, "prcp": out["prcp"], "srad": srad, "vp": vp}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.makedirs(OUT_DIR, exist_ok=True)

    grid = np.load(GRID_NPZ)
    cell_id = np.load(MASK_NPZ)["forest_cell_id"]
    n_cells = len(cell_id)

    # gridMET cells with data in every day of the first fit year, nearest to each forest cell center in meters
    with netcdf_file(os.path.join(GRIDMET_DIR, f"gridmet_maine_tmmx_{FIT_YEARS[0]}.nc"), "r", mmap=False) as nc:
        lat = nc.variables["lat"].data.copy()
        lon = nc.variables["lon"].data.copy()
        raw = nc.variables["air_temperature"].data  # day, lat, lon
        has_data = ~np.any(raw == GRIDMET_FILL, axis=0)  # lat, lon
    for var in GRIDMET_VARS.values():
        for year in FIT_YEARS + [TARGET_YEAR]:
            with netcdf_file(os.path.join(GRIDMET_DIR, f"gridmet_maine_{var}_{year}.nc"), "r", mmap=False) as nc:
                assert np.array_equal(nc.variables["lat"].data, lat) and np.array_equal(nc.variables["lon"].data, lon), f"{var} {year} grid differs"
    glat, glon = np.meshgrid(lat, lon, indexing="ij")  # lat, lon
    gy, gx = np.nonzero(has_data)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{int(grid['epsg'])}", always_xy=True)
    ux, uy = to_utm.transform(glon[gy, gx], glat[gy, gx])
    dist, k = cKDTree(np.column_stack([ux, uy])).query(np.column_stack([grid["x"][cell_id], grid["y"][cell_id]]))
    iy, ix = gy[k], gx[k]
    east_edge = lon.max() + 0.5 * abs(lon[1] - lon[0])  # east edge of the last gridMET column
    east = grid["lon"][cell_id] > east_edge
    logger.info("%d forest cells: nearest gridMET center median %.0f m, max %.0f m; %d cells east of the gridMET edge, max distance among them %.0f m; %d distinct gridMET cells",
                n_cells, np.median(dist), dist.max(), east.sum(), dist[east].max() if east.any() else 0.0, len(np.unique(iy * len(lon) + ix)))

    # Daymet daylength climatology by month and day over 2001 to 2025, used to convert gridMET srad in every year
    dayl_sum = np.zeros((n_cells, 366))  # cell, day of a leap year
    dayl_n = np.zeros(366)
    dayl_dev = 0.0
    daymet_dates = {}
    for year in FIT_YEARS:
        dm = np.load(os.path.join(DAYMET_DIR, f"daymet_{year}.npz"))
        assert np.array_equal(dm["cell_id"], cell_id), f"Daymet {year} cell list differs"
        md = pd.to_datetime(dm["date"]).strftime("2000-%m-%d").to_numpy(dtype="datetime64[D]")
        slot = (md - np.datetime64("2000-01-01")).astype(int)  # position in a leap year
        dayl_sum[:, slot] += dm["dayl"]
        dayl_n[slot] += 1
        daymet_dates[year] = dm["date"]
    assert np.all(dayl_n > 0), "a calendar day has no Daymet daylength"
    dayl_clim = dayl_sum / dayl_n  # cell, day of a leap year, seconds
    for year in FIT_YEARS:
        dm = np.load(os.path.join(DAYMET_DIR, f"daymet_{year}.npz"))
        md = pd.to_datetime(dm["date"]).strftime("2000-%m-%d").to_numpy(dtype="datetime64[D]")
        dayl_dev = max(dayl_dev, float(np.abs(dm["dayl"] - dayl_clim[:, (md - np.datetime64("2000-01-01")).astype(int)]).max()))
    logger.info("Daymet daylength: largest difference of any year from the climatology %.1f seconds", dayl_dev)

    def clim_for(dates):
        md = pd.to_datetime(dates).strftime("2000-%m-%d").to_numpy(dtype="datetime64[D]")
        return dayl_clim[:, (md - np.datetime64("2000-01-01")).astype(int)]

    # sums for the per-cell least squares fits, one set over all fit years and one over the training years
    sums = {fit: {v: np.zeros((6, n_cells)) for v in VARIABLES} for fit in ("all", "train")}  # count, x, y, xx, xy, yy
    test_pairs = {v: [] for v in VARIABLES}
    for year in FIT_YEARS:
        lo, hi = np.datetime64(f"{year}-{SEASON[0]}"), np.datetime64(f"{year}-{SEASON[1]}")
        g_dates, _ = read_gridmet("tmmx", year, iy[:1], ix[:1])
        season_dates = g_dates[(g_dates >= lo) & (g_dates <= hi)]
        dm = np.load(os.path.join(DAYMET_DIR, f"daymet_{year}.npz"))
        d_idx = np.searchsorted(dm["date"], season_dates)
        assert np.array_equal(dm["date"][d_idx], season_dates), f"{year}: Daymet lacks a season date"
        g_dates, g = gridmet_as_daymet(year, iy, ix, clim_for(g_dates))
        g_idx = np.searchsorted(g_dates, season_dates)
        for v in VARIABLES:
            x = g[v][:, g_idx].astype(np.float64)  # cell, day
            y = dm[v][:, d_idx].astype(np.float64)  # cell, day
            s = np.stack([np.full(n_cells, x.shape[1], dtype=np.float64), x.sum(1), y.sum(1), (x * x).sum(1), (x * y).sum(1), (y * y).sum(1)])
            sums["all"][v] += s
            if year in TRAIN_YEARS:
                sums["train"][v] += s
            if year in TEST_YEARS:
                test_pairs[v].append((x, y))
        logger.info("%d: %d season days paired", year, len(season_dates))

    coef = {}
    for fit in ("all", "train"):
        for v in VARIABLES:
            n, sx, sy, sxx, sxy, syy = sums[fit][v]
            if v in NO_INTERCEPT:
                b = sxy / sxx
                a = np.zeros(n_cells)
            else:
                b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
                a = (sy - b * sx) / n
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"{fit} {v}: fit not finite"
            coef[(fit, v)] = (a, b)
            if fit == "all":
                # in-sample R squared of the fit, per cell
                sse = syy - 2 * a * sy - 2 * b * sxy + n * a * a + 2 * a * b * sx + b * b * sxx
                sst = syy - sy * sy / n
                r2 = 1 - sse / sst
                logger.info("fit %s on %d to %d: slope median %.3f (1st %.3f, 99th %.3f), intercept median %.3f, per-cell R squared median %.3f (1st %.3f)",
                            v, FIT_YEARS[0], FIT_YEARS[-1], np.median(b), *np.percentile(b, [1, 99]), np.median(a), np.median(r2), np.percentile(r2, 1))

    # held-out check: fit on the training years, error on the 2024 and 2025 seasons against Daymet
    rows = []
    for v in VARIABLES:
        a, b = coef[("train", v)]
        x = np.concatenate([p[0] for p in test_pairs[v]], axis=1)
        y = np.concatenate([p[1] for p in test_pairs[v]], axis=1)
        pred = a[:, None] + b[:, None] * x
        for name, est in (("uncorrected", x), ("corrected", pred)):
            err = est - y
            rows.append({"variable": v, "version": name, "bias": err.mean(), "rmse": np.sqrt((err ** 2).mean()),
                         "median_cell_rmse": np.median(np.sqrt((err ** 2).mean(axis=1))), "n_cell_days": err.size})
    check = pd.DataFrame(rows)
    check_path = os.path.join(OUT_DIR, "heldout_check_2024_2025.tsv")
    check.to_csv(check_path, sep="\t", index=False, float_format="%.4f")
    logger.info("held-out check, fit on %d to %d, error against Daymet on %s, June 1 to November 30:\n%s",
                TRAIN_YEARS[0], TRAIN_YEARS[-1], TEST_YEARS, check.to_string(index=False))

    # 2026 from June 1 to the last gridMET day, corrected with the fit over all fit years
    g_dates, g = gridmet_as_daymet(TARGET_YEAR, iy, ix, clim_for(read_gridmet("tmmx", TARGET_YEAR, iy[:1], ix[:1])[0]))
    keep = g_dates >= np.datetime64(f"{TARGET_YEAR}-{SEASON[0]}")
    dates_out = g_dates[keep]
    out = {}
    for v in VARIABLES:
        a, b = coef[("all", v)]
        corrected = a[:, None] + b[:, None] * g[v][:, keep]  # cell, day
        n_neg = int((corrected < 0).sum()) if v in ("prcp", "srad", "vp") else 0
        if n_neg:
            logger.info("%s %d: %d negative cell-days after correction set to 0 (%.5f of cell-days)", v, TARGET_YEAR, n_neg, n_neg / corrected.size)
            corrected = np.maximum(corrected, 0.0)
        out[v] = corrected.astype(np.float32)
    dayl_out = clim_for(dates_out).astype(np.float32)
    path = os.path.join(OUT_DIR, f"gridmet_daymet_{TARGET_YEAR}.npz")
    np.savez_compressed(
        path, cell_id=cell_id, date=dates_out, dayl=dayl_out, gridmet_iy=iy, gridmet_ix=ix, gridmet_distance_m=dist,
        **out, **{f"intercept_{v}": coef[("all", v)][0] for v in VARIABLES}, **{f"slope_{v}": coef[("all", v)][1] for v in VARIABLES},
    )
    logger.info("wrote %s: %s to %s, %d days; arrays of shape cell, day in Daymet units; dayl is the 2001 to 2025 Daymet climatology",
                path, dates_out[0], dates_out[-1], len(dates_out))


if __name__ == "__main__":
    main()
