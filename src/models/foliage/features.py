import os
import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# covariates of the outcome model F, decision D25, computed per cell and day from daily weather in Daymet units
COVARIATES = ["T60", "P60", "R30", "CDD", "GSI", "greenup", "S", "h",
              "Z_hardwood", "Z_red_maple", "Z_sugar_maple", "Z_birch", "Z_aspen"]
SPECIES_GROUPS = ["hardwood", "red_maple", "sugar_maple", "birch", "aspen"]
WEATHER_START = "06-01"  # month-day of the first weather day loaded, so the 60 day windows are full on August 1
SEASON = ("08-01", "11-30")  # month-day, the season of the response
T_WINDOW = 60  # days, mean temperature
P_WINDOW = 60  # days, precipitation sum
R_WINDOW = 30  # days, mean daily shortwave energy
CDD_BASE = 20.0  # degrees C, user choice
CDD_START = "08-01"
GSI_WINDOW = 21  # days, trailing mean
GSI_TMIN = (-2.0, 5.0)  # degrees C, Jolly et al. 2005
GSI_VPD = (900.0, 4100.0)  # Pa, Jolly et al. 2005
GSI_PHOTO = (10.0, 11.0)  # hours, Jolly et al. 2005
TRAIN_YEARS = list(range(2001, 2024))  # D5
ALL_YEARS = list(range(2001, 2027))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROC = os.path.join(REPO_ROOT, "data", "processed")
DAYMET_DIR = os.path.join(PROC, "daymet")
BROWN_DIR = os.path.join(PROC, "brownness")
FOLIAGE_DIR = os.path.join(PROC, "foliage")
DAYL_CLIM_NPZ = os.path.join(FOLIAGE_DIR, "dayl_climatology.npz")


def saturation_vp(t: np.ndarray):
    '''saturation vapor pressure in Pa over water at temperature in degrees C, FAO-56 (Allen et al. 1998) equation 11'''
    return 610.8 * np.exp(17.27 * t / (t + 237.3))


def trailing_mean(x: np.ndarray, window: int):
    '''mean over the window ending on each day, last axis is day; NaN where the window is not full'''
    c = np.cumsum(x, axis=-1, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    out[..., window - 1] = c[..., window - 1]
    out[..., window:] = c[..., window:] - c[..., :-window]
    out[..., window - 1:] /= window
    return out


def weather_covariates(tmax: np.ndarray, tmin: np.ndarray, prcp: np.ndarray, srad: np.ndarray, vp: np.ndarray,
                       dayl: np.ndarray, dates: np.ndarray):
    '''T60, P60, R30, CDD and GSI from daily weather of shape cell, day in Daymet units, consecutive dates
    returns a dict of arrays of shape cell, day; NaN on days whose window is not full or before August 1 for CDD'''
    assert tmax.shape == tmin.shape == prcp.shape == srad.shape == vp.shape == dayl.shape
    assert tmax.shape[-1] == len(dates) and np.all(np.diff(dates) == np.timedelta64(1, "D")), "dates are not consecutive"
    tmean = 0.5 * (tmax.astype(np.float64) + tmin.astype(np.float64))
    out = {
        "T60": trailing_mean(tmean, T_WINDOW),
        "P60": trailing_mean(prcp, P_WINDOW) * P_WINDOW,  # sum, mm
        "R30": trailing_mean(srad.astype(np.float64) * dayl.astype(np.float64) / 1e6, R_WINDOW),  # MJ per square meter per day
    }
    # cold degree days, summed from August 1 of the year of the first date
    year = dates[0].astype("datetime64[Y]").astype(int) + 1970
    start = np.datetime64(f"{year}-{CDD_START}")
    since = dates >= start
    cold = np.where(since, np.maximum(0.0, CDD_BASE - tmean), 0.0)
    out["CDD"] = np.where(since, np.cumsum(cold, axis=-1), np.nan)
    # Jolly et al. 2005 growing season index, trailing 21 day mean of the product of the three ramps
    i_tmin = np.clip((tmin - GSI_TMIN[0]) / (GSI_TMIN[1] - GSI_TMIN[0]), 0.0, 1.0)
    vpd = saturation_vp(tmean) - vp  # Pa
    i_vpd = 1.0 - np.clip((vpd - GSI_VPD[0]) / (GSI_VPD[1] - GSI_VPD[0]), 0.0, 1.0)
    i_photo = np.clip((dayl / 3600.0 - GSI_PHOTO[0]) / (GSI_PHOTO[1] - GSI_PHOTO[0]), 0.0, 1.0)
    out["GSI"] = trailing_mean(i_tmin * i_vpd * i_photo, GSI_WINDOW)
    return out


def load_daymet(year: int, cell_id: np.ndarray, last: str):
    '''daily Daymet from June 1 to the given last month-day of a year, shape cell, day, and the dates'''
    d = np.load(os.path.join(DAYMET_DIR, f"daymet_{year}.npz"))
    assert np.array_equal(d["cell_id"], cell_id), f"Daymet {year} cell list differs"
    dates = d["date"]
    keep = (dates >= np.datetime64(f"{year}-{WEATHER_START}")) & (dates <= np.datetime64(f"{year}-{last}"))
    out = {v: d[v][:, keep] for v in ("tmax", "tmin", "prcp", "srad", "vp", "dayl")}
    return out, dates[keep]


def dayl_climatology(cell_id: np.ndarray):
    '''Daymet daylength climatology over 2001 to 2025 by calendar day, shape cell, day of a leap year, seconds; cached'''
    if os.path.exists(DAYL_CLIM_NPZ):
        c = np.load(DAYL_CLIM_NPZ)
        assert np.array_equal(c["cell_id"], cell_id)
        return c["dayl"]
    total = np.zeros((len(cell_id), 366))
    count = np.zeros(366)
    for year in range(2001, 2026):
        d = np.load(os.path.join(DAYMET_DIR, f"daymet_{year}.npz"))
        assert np.array_equal(d["cell_id"], cell_id)
        slot = leap_slot(d["date"])
        total[:, slot] += d["dayl"]
        count[slot] += 1
    assert np.all(count > 0)
    clim = total / count
    os.makedirs(FOLIAGE_DIR, exist_ok=True)
    np.savez_compressed(DAYL_CLIM_NPZ, cell_id=cell_id, dayl=clim)
    return clim


def leap_slot(dates: np.ndarray):
    '''position of each date's month and day in a leap year, 0 for January 1'''
    month = dates.astype("datetime64[M]").astype(int) % 12
    day = (dates - dates.astype("datetime64[M]")).astype(int)
    first = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])  # day before the first of each month in a leap year
    return first[month] + day


def static_covariates(cell_id: np.ndarray, year: int):
    '''green-up, elevation and species shares of one year per cell, and the mask of cells with all of them defined'''
    ph = np.load(os.path.join(PROC, "phenology", "phenology_dates.npz"))
    assert np.array_equal(ph["cell_id"], cell_id)
    years = list(ph["years"])
    train = [years.index(y) for y in TRAIN_YEARS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # cells without any training green-up give NaN
        greenup_mean = np.nanmean(ph["greenup_doy"][:, train], axis=1)  # cell, mean over training years (D25)
    greenup = ph["greenup_doy"][:, years.index(year)].copy()
    filled = np.isnan(greenup)
    greenup[filled] = greenup_mean[filled]
    elev = np.load(os.path.join(PROC, "static", "elevation.npz"))
    assert np.array_equal(elev["cell_id"], cell_id)
    sp = np.load(os.path.join(PROC, "static", "species.npz"))
    assert np.array_equal(sp["cell_id"], cell_id) and list(sp["groups"]) == SPECIES_GROUPS
    share = sp["share_year"][:, :, list(sp["years"]).index(year)].astype(np.float64)  # group, cell
    out = {"greenup": greenup, "h": elev["elevation_m"].astype(np.float64)}
    for g, name in enumerate(SPECIES_GROUPS):
        out[f"Z_{name}"] = share[g]
    defined = np.all([np.isfinite(v) for v in out.values()], axis=0)
    return out, defined, int(filled.sum())


def covariate_cube(weather: dict, dates: np.ndarray, static: dict):
    '''all 13 covariates on season days, shape cell, day, covariate, and the season dates'''
    w = weather_covariates(weather["tmax"], weather["tmin"], weather["prcp"], weather["srad"], weather["vp"], weather["dayl"], dates)
    year = dates[0].astype("datetime64[Y]").astype(int) + 1970
    in_season = dates >= np.datetime64(f"{year}-{SEASON[0]}")
    n_cells = weather["tmax"].shape[0]
    n_days = int(in_season.sum())
    cube = np.empty((n_cells, n_days, len(COVARIATES)))
    for k, name in enumerate(COVARIATES):
        if name in w:
            cube[:, :, k] = w[name][:, in_season]
        elif name == "S":
            cube[:, :, k] = weather["dayl"][:, in_season] / 3600.0  # hours
        else:
            cube[:, :, k] = static[name][:, None]
    return cube, dates[in_season]


def training_rows(year: int, cell_id: np.ndarray, ul_valid: np.ndarray):
    '''rows of one year: observed brownness and covariates at the observation dates in the season, from Daymet'''
    weather, dates = load_daymet(year, cell_id, SEASON[1])
    static, defined, n_filled = static_covariates(cell_id, year)
    cube, season_dates = covariate_cube(weather, dates, static)  # cell, day, covariate
    b = np.load(os.path.join(BROWN_DIR, f"brownness_{year}.npz"))
    assert np.array_equal(b["cell_id"], cell_id)
    obs_date = b["obs_date"]  # cell, composite
    brown = b["brownness"]  # cell, composite
    lo, hi = season_dates[0], season_dates[-1]
    keep = ul_valid[:, None] & defined[:, None] & np.isfinite(brown) & (obs_date >= lo) & (obs_date <= hi)
    rows_cell, rows_comp = np.nonzero(keep)
    day = (obs_date[rows_cell, rows_comp] - lo).astype(int)
    X = cube[rows_cell, day, :]  # row, covariate
    assert np.all(np.isfinite(X)), f"{year}: covariates not finite at a training row"
    logger.info("%d: %d rows from %d cells; %d cells with green-up filled by the training mean; %d valid cells without all static covariates",
                year, len(rows_cell), len(np.unique(rows_cell)), n_filled, int((ul_valid & ~defined).sum()))
    return {"cell": rows_cell.astype(np.int32), "composite": rows_comp.astype(np.int16), "composite_start": b["composite_start"][rows_comp],
            "obs_date": obs_date[rows_cell, rows_comp], "y": brown[rows_cell, rows_comp].astype(np.float64), "X": X}
