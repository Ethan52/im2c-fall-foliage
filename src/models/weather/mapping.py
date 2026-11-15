import os
import glob
import logging

import numpy as np
import xarray as xr

from models.weather import era5

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FORECAST_DIR = os.path.join(REPO_ROOT, "data", "interim", "weather_forecast")
GRID_PATH = os.path.join(REPO_ROOT, "data", "processed", "grid", "analysis_grid.npz")
FOREST_PATH = os.path.join(REPO_ROOT, "data", "processed", "forest_mask", "forest_mask.npz")
DAYMET_DIR = os.path.join(REPO_ROOT, "data", "processed", "daymet")
CACHE_DIR = os.path.join(REPO_ROOT, "data", "processed", "weather_mapping")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "weather_forecast")

EPS = 0.621981  # ratio of the gas constants of dry air and water vapor, as in Ai2's pipeline
# season of the linear-map fit, and the climatology window half width in days
FIT_MONTHS = (8, 9, 10, 11)
HALF_WINDOW = 15
# days of year covered by the cached climatologies, with a margin around August 1 to November 30
CACHE_DOYS = np.arange(200, 346)

# Daymet variable, ERA5 or model variable, and how the anomaly is applied in the delta method
VARIABLES = [
    ("tmax", "tmax", "difference"),
    ("tmin", "tmin", "difference"),
    ("prcp", "prcp", "ratio"),
    ("srad", "ssrd", "ratio"),  # Daymet daylight-mean shortwave against the 24 hour mean; the ratio needs no conversion
    ("vp", "vp", "ratio"),
]


def daily(times: np.ndarray, snapshots: dict, windows: dict, last_day: np.datetime64):
    '''daily values over 06 UTC to 06 UTC from 6-hourly arrays of shape (members, times, points)

    snapshots are instantaneous at the time stamps (12, 18, 00 and 06 UTC are used); windows are 6 hour quantities
    ending at the time stamps (the windows ending 12, 18, 00 and 06 UTC are used); only complete days are returned
    '''
    index = {t: k for k, t in enumerate(times)}
    first = (times[0] - np.timedelta64(7, "h")).astype("datetime64[D]")
    days = []
    for day in np.arange(first, last_day + 1):
        stamps = [day.astype("datetime64[h]") + np.timedelta64(h, "h") for h in (12, 18, 24, 30)]
        if all(s in index for s in stamps):
            days.append(day)
    days = np.array(days)
    out = {}
    for name, values in {**snapshots, **windows}.items():
        out[name] = np.stack([values[:, [index[day.astype("datetime64[h]") + np.timedelta64(h, "h")] for h in (12, 18, 24, 30)], :] for day in days], axis=1)
    return days, out  # arrays of shape (members, days, 4, points)


def segment_fields(days: np.ndarray, d: dict, point_lat: np.ndarray, point_lon: np.ndarray):
    '''daily tmax, tmin, tmean (C), prcp (mm/day), ssrd (24 hour mean W/m2) and vp (Pa) of one model segment'''
    return dict(
        date=days, point_lat=point_lat, point_lon=point_lon,
        tmax=d["t2m"].max(axis=2) - 273.15, tmin=d["t2m"].min(axis=2) - 273.15, tmean=d["t2m"].mean(axis=2) - 273.15,
        prcp=d["prcp"].sum(axis=2), ssrd=d["ssrd"].mean(axis=2), vp=d["vp"].mean(axis=2),
    )


def forecast_daily(start: str):
    '''daily values of the AIFS ENS segment (days 0 to 14) and the ACE2 segment (from the hand-off), written as one npz'''
    last_day = np.datetime64(f"{start[:4]}-11-30")
    segments = {}

    # AIFS ENS: 2 m temperature and dewpoint snapshots, precipitation (m) and shortwave (J/m2) per 6 hours
    aifs = xr.open_dataset(os.path.join(FORECAST_DIR, "aifs_ens", start[:13], "maine.nc"))
    times = aifs.time.values.astype("datetime64[h]")
    snaps = {"t2m": aifs["2t"].values, "vp": era5.vapor_pressure_from_dewpoint(aifs["2d"].values.astype(np.float64))}
    wins = {"prcp": aifs["tp"].values * 1000.0, "ssrd": aifs["ssrd"].values / (6 * 3600.0)}  # mm per 6 h, W/m2 mean
    days, d = daily(times, snaps, wins, last_day)
    segments["aifs"] = segment_fields(days, d, aifs.latitude.values, aifs.longitude.values)
    logger.info("AIFS ENS: %d members, %d days from %s to %s, %d points", d["t2m"].shape[0], len(days), days[0], days[-1], d["t2m"].shape[-1])

    # ACE2: TMP2m, Q2m and PRESsfc snapshots; PRATEsfc (kg/m2/s) and DSWRFsfc (W/m2) 6 hour means
    paths = sorted(glob.glob(os.path.join(FORECAST_DIR, "ace2", start[:13], "member*", "maine.nc")))
    if paths:
        members = [xr.open_dataset(p, decode_timedelta=True).isel(sample=0) for p in paths]
        times = members[0].valid_time.values.astype("datetime64[h]")
        for m in members:
            assert np.array_equal(m.valid_time.values.astype("datetime64[h]"), times), "ACE2 members have different times"
        stack = {v: np.stack([m[v].values.reshape(len(times), -1) for m in members]) for v in ["TMP2m", "Q2m", "PRESsfc", "PRATEsfc", "DSWRFsfc"]}
        q, p = stack["Q2m"].astype(np.float64), stack["PRESsfc"].astype(np.float64)
        snaps = {"t2m": stack["TMP2m"], "vp": q * p / (EPS + (1 - EPS) * q)}
        wins = {"prcp": stack["PRATEsfc"] * 6 * 3600.0, "ssrd": stack["DSWRFsfc"]}  # mm per 6 h, W/m2 mean
        days, d = daily(times, snaps, wins, last_day)
        assert days[0] == segments["aifs"]["date"][-1] + 1, f"ACE2 starts on {days[0]}, AIFS ends on {segments['aifs']['date'][-1]}"
        lat2, lon2 = np.meshgrid(members[0].lat.values, members[0].lon.values, indexing="ij")
        segments["ace2"] = segment_fields(days, d, lat2.ravel(), lon2.ravel())
        assert segments["ace2"]["tmax"].shape[0] == segments["aifs"]["tmax"].shape[0], "different member counts in AIFS and ACE2"
        logger.info("ACE2: %d members, %d days from %s to %s, %d points", len(paths), len(days), days[0], days[-1], lat2.size)

    out_dir = os.path.join(FORECAST_DIR, "daily")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{start[:13]}.npz")
    arrays = {f"{seg}_{k}": v for seg, fields in segments.items() for k, v in fields.items()}
    np.savez(path, issue_time=np.datetime64(start, "h"), segments=np.array(list(segments)), **arrays)
    logger.info("wrote %s", path)
    return path


def nearest_point(lat: np.ndarray, lon: np.ndarray, point_lat: np.ndarray, point_lon: np.ndarray):
    '''index of the nearest model grid point for each location, distance on the sphere in the small-angle form'''
    coslat = np.cos(np.radians(lat))[:, None]
    d2 = (lat[:, None] - point_lat[None, :]) ** 2 + (((lon[:, None] - point_lon[None, :] + 180) % 360 - 180) * coslat) ** 2
    return np.argmin(d2, axis=1)


def window_climatology(values: np.ndarray, doy: np.ndarray, target_doy: np.ndarray):
    '''mean over all years of the days within the half window of each target day of year, values of shape (days, ...)'''
    out = np.empty((len(target_doy),) + values.shape[1:])
    for j, d in enumerate(target_doy):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)
        sel = dist <= HALF_WINDOW
        assert sel.sum() >= 20, f"too few days in the window of day {d}"
        out[j] = np.nanmean(values[sel], axis=0)
    return out


def forest_cells():
    '''forest cell IDs (D11) and their latitude and longitude in degrees east, in the order of the pipeline session's files'''
    grid = np.load(GRID_PATH)
    cell_id = np.load(FOREST_PATH)["forest_cell_id"]
    row = np.searchsorted(grid["cell_id"], cell_id)
    assert np.array_equal(grid["cell_id"][row], cell_id), "forest cells missing from the analysis grid"
    return cell_id, grid["lat"][row], grid["lon"][row] % 360


def load_daymet(cell_id: np.ndarray, years: range):
    '''daily Daymet on the forest cells for the given years, from the pipeline session's files (D19)'''
    dates, data = [], {name: [] for name, _, _ in VARIABLES}
    for year in years:
        f = np.load(os.path.join(DAYMET_DIR, f"daymet_{year}.npz"))
        assert np.array_equal(f["cell_id"], cell_id), f"Daymet {year} cell_id differs from the forest cells"
        dates.append(f["date"].astype("datetime64[D]"))
        for name, _, _ in VARIABLES:
            values = f[name]  # shape (cells, days)
            assert values.shape == (len(cell_id), len(f["date"])), f"Daymet {year} {name} has shape {values.shape}"
            data[name].append(values.T)  # shape (days, cells)
    return np.concatenate(dates), {name: np.concatenate(v) for name, v in data.items()}


def era5_on_points(point_lat: np.ndarray, point_lon: np.ndarray):
    '''ERA5 daily values averaged over the 0.25 degree cells whose nearest model point is each point, shape (days, points)'''
    ref = xr.open_dataset(era5.DAILY_PATH)
    lat2, lon2 = np.meshgrid(ref.latitude.values, ref.longitude.values, indexing="ij")
    owner = nearest_point(lat2.ravel(), lon2.ravel(), point_lat, point_lon)  # model point of each ERA5 cell
    weights = np.cos(np.radians(lat2.ravel()))
    out = {}
    for _, name, _ in VARIABLES:
        values = ref[name].values.reshape(ref.sizes["time"], -1)  # shape (days, ERA5 cells)
        num = np.zeros((values.shape[0], len(point_lat)))
        den = np.zeros(len(point_lat))
        np.add.at(den, owner, weights)
        for k in range(len(point_lat)):
            sel = owner == k
            if sel.any():
                num[:, k] = (values[:, sel] * weights[sel]).sum(axis=1)
        out[name] = num / np.where(den > 0, den, np.nan)
    return ref.time.values.astype("datetime64[D]"), out


def prepare_segment(name: str, point_lat: np.ndarray, point_lon: np.ndarray, cell_lat: np.ndarray, cell_lon: np.ndarray,
                    daymet_dates: np.ndarray, daymet: dict, fit_years: range):
    '''climatologies and per-cell linear fits for one model grid, from the fit years only; saved for reuse by every issue time'''
    owner = nearest_point(cell_lat, cell_lon, point_lat, point_lon)  # model point of each forest cell
    era5_dates, ref = era5_on_points(point_lat, point_lon)
    assert era5_dates.astype("datetime64[Y]").astype(int).max() + 1970 <= max(fit_years), "ERA5 beyond the fit years"
    assert daymet_dates.astype("datetime64[Y]").astype(int).max() + 1970 <= max(fit_years), "Daymet beyond the fit years"
    daymet_doy = (daymet_dates - daymet_dates.astype("datetime64[Y]")).astype(int) + 1
    era5_doy = (era5_dates - era5_dates.astype("datetime64[Y]")).astype(int) + 1
    fit_daymet = np.isin(daymet_dates.astype("datetime64[M]").astype(int) % 12 + 1, FIT_MONTHS)
    era5_index = {d: k for k, d in enumerate(era5_dates)}
    both = np.array([d for d in daymet_dates[fit_daymet] if d in era5_index])
    cache = {"owner": owner, "point_lat": point_lat, "point_lon": point_lon, "doys": CACHE_DOYS}
    for daymet_name, model_name, _ in VARIABLES:
        cache[f"daymet_clim_{daymet_name}"] = window_climatology(daymet[daymet_name], daymet_doy, CACHE_DOYS).astype(np.float32)  # (doys, cells)
        cache[f"era5_clim_{daymet_name}"] = window_climatology(ref[model_name], era5_doy, CACHE_DOYS)[:, owner].astype(np.float32)  # (doys, cells)
        # per-cell least squares of Daymet on ERA5 at the cell's model point, August to November of the fit years
        x = ref[model_name][[era5_index[d] for d in both]][:, owner]  # shape (fit days, cells)
        y = daymet[daymet_name][np.searchsorted(daymet_dates, both)]  # shape (fit days, cells)
        ok = np.isfinite(x) & np.isfinite(y)
        n = ok.sum(axis=0)
        xm = np.where(ok, x, 0).sum(axis=0) / n
        ym = np.where(ok, y, 0).sum(axis=0) / n
        slope = np.where(ok, (x - xm) * (y - ym), 0).sum(axis=0) / np.where(ok, (x - xm) ** 2, 0).sum(axis=0)
        cache[f"slope_{daymet_name}"] = slope
        cache[f"intercept_{daymet_name}"] = ym - slope * xm
        logger.info("%s %s: linear map slope median %.3f, intercept median %.3f, %d fit days, %d to %d",
                    name, daymet_name, np.median(slope), np.median(ym - slope * xm), len(both), both[0].astype("datetime64[Y]").astype(int) + 1970,
                    both[-1].astype("datetime64[Y]").astype(int) + 1970)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.npz")
    np.savez(path, **cache)
    logger.info("wrote %s", path)


def load_segments(start: str):
    '''daily forecast values per model segment, as written by forecast_daily'''
    model = np.load(os.path.join(FORECAST_DIR, "daily", f"{start[:13]}.npz"))
    segments = {}
    for name in model["segments"]:
        segments[str(name)] = {k[len(name) + 1:]: model[k] for k in model.files if k.startswith(f"{name}_")}
    return model["issue_time"], segments


def prepare(start: str, fit_years: range):
    '''climatologies and fits for every model grid present in the daily forecast of the given issue time (D15)'''
    _, segments = load_segments(start)
    cell_id, cell_lat, cell_lon = forest_cells()
    daymet_dates, daymet = load_daymet(cell_id, fit_years)
    for name, seg in segments.items():
        prepare_segment(name, seg["point_lat"], seg["point_lon"] % 360, cell_lat, cell_lon, daymet_dates, daymet, fit_years)


def map_segment(name: str, seg: dict):
    '''both mappings (D15) for one model segment, using the cached climatologies and fits; returns arrays (members, days, cells)'''
    cache = np.load(os.path.join(CACHE_DIR, f"{name}.npz"))
    assert np.allclose(cache["point_lat"], seg["point_lat"]) and np.allclose(cache["point_lon"], seg["point_lon"] % 360), f"{name} grid differs from the cache"
    owner = cache["owner"]
    dates = seg["date"].astype("datetime64[D]")
    doy = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
    k = np.searchsorted(cache["doys"], doy)
    assert np.array_equal(cache["doys"][k], doy), "forecast days outside the cached days of year"
    num_members = seg["tmax"].shape[0]
    outputs = {"delta": {}, "linear": {}}
    for daymet_name, model_name, kind in VARIABLES:
        m = seg[model_name][:, :, owner]  # shape (members, days, cells)
        assert m.shape == (num_members, len(dates), len(owner)), f"{model_name} has shape {m.shape}"
        d_clim = cache[f"daymet_clim_{daymet_name}"][k]  # shape (days, cells)
        e_clim = cache[f"era5_clim_{daymet_name}"][k]
        if kind == "difference":
            outputs["delta"][daymet_name] = d_clim[None] + (m - e_clim[None])
        else:
            outputs["delta"][daymet_name] = d_clim[None] * m / e_clim[None]
        mapped = cache[f"intercept_{daymet_name}"][None, None] + cache[f"slope_{daymet_name}"][None, None] * m
        outputs["linear"][daymet_name] = mapped if kind == "difference" else np.maximum(mapped, 0.0)
    logger.info("%s: %d members, %s to %s, %d model points", name, num_members, dates[0], dates[-1], len(np.unique(owner)))
    return dates, outputs


def map_to_daymet(start: str):
    '''both mappings of one issue time, written as the hand-off files O11: arrays (members, cells, days) in Daymet units'''
    issue_time, segments = load_segments(start)
    cell_id = np.load(FOREST_PATH)["forest_cell_id"]
    pieces = [map_segment(name, seg) for name, seg in segments.items()]
    dates = np.concatenate([p[0] for p in pieces])
    assert np.all(np.diff(dates) == np.timedelta64(1, "D")), "forecast days are not consecutive"
    num_members = pieces[0][1]["delta"]["tmax"].shape[0]
    for method in ("delta", "linear"):
        out_dir = os.path.join(OUT_DIR, "aifs_ens_ace2", method)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{str(dates[0])[:4]}_{start[:10]}.npz")
        arrays = {}
        for daymet_name, _, _ in VARIABLES:
            values = np.concatenate([p[1][method][daymet_name] for p in pieces], axis=1)  # shape (members, days, cells)
            arrays[daymet_name] = np.transpose(values, (0, 2, 1)).astype(np.float32)
            assert arrays[daymet_name].shape == (num_members, len(cell_id), len(dates)), f"{daymet_name} has shape {arrays[daymet_name].shape}"
            assert np.isfinite(arrays[daymet_name]).all(), f"{method} {daymet_name} has missing values"
        np.savez(path, cell_id=cell_id, date=dates, issue_time=issue_time, **arrays)
        logger.info("wrote %s", path)
