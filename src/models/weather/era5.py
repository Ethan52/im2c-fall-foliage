import os
import time
import logging
from datetime import datetime, timezone

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# public ARCO ERA5 stores on Google Cloud, read anonymously
FULL_37_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
MODEL_LEVEL_URL = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HOURLY_DIR = os.path.join(REPO_ROOT, "data", "raw", "era5")
DAILY_PATH = os.path.join(REPO_ROOT, "data", "processed", "era5", "era5_maine_daily_2001_2023.nc")
AIFS_INPUT_DIR = os.path.join(REPO_ROOT, "data", "interim", "aifs_inputs")

# Maine box with a margin of at least one 2.8 degree model cell, longitudes in degrees east
LAT_MIN, LAT_MAX = 40.0, 50.0
LON_MIN, LON_MAX = 285.0, 296.0

# hourly near-surface variables of the Maine reference, used for the delta-method climatology and the linear map (D15)
HOURLY_VARIABLES = {
    "2m_temperature": "t2m",  # K
    "2m_dewpoint_temperature": "d2m",  # K
    "total_precipitation": "tp",  # m, accumulated over the hour ending at the time stamp
    "mean_surface_downward_short_wave_radiation_flux": "ssrd",  # W/m2, mean over the hour ending at the time stamp
}
# season window of the hourly reference, with a margin of 16 days on each side for the 31 day climatology window
SEASON_START = "07-15"
SEASON_END = "12-16"

# daily window from 06 UTC of day d to 06 UTC of day d+1, i.e. 01 EST to 01 EST, aligned with 6-hourly model output
WINDOW_START_HOUR = 6
# snapshot hours inside the window, the times at which 6-hourly models give instantaneous fields
SNAPSHOT_HOURS = [12, 18, 24, 30]

# AIFS ENS input fields: AIFS short names and ARCO ERA5 names, from the AIFS ENS example notebook parameter lists
AIFS_SURFACE = {
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "2d": "2m_dewpoint_temperature",
    "2t": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "skt": "skin_temperature",
    "sp": "surface_pressure",
    "tcw": "total_column_water",
    "lsm": "land_sea_mask",
    "z": "geopotential_at_surface",
    "slor": "slope_of_sub_gridscale_orography",
    "sdor": "standard_deviation_of_orography",
    "stl1": "soil_temperature_level_1",
    "stl2": "soil_temperature_level_2",
}
AIFS_PRESSURE = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}
AIFS_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# surface fields of an ACE2 initial condition, and the water species summed into ACE2's total water
SURFACE_NAMES = ["surface_pressure", "2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind",
                 "10m_v_component_of_wind", "skin_temperature"]
WATER_SPECIES = [
    "specific_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "specific_rain_water_content",
    "specific_snow_water_content",
]


def open_full_37(chunks=None):
    '''the ARCO ERA5 store of single-level and 37 pressure-level fields, hourly, 0.25 degrees'''
    return xr.open_zarr(FULL_37_URL, chunks=chunks, storage_options=dict(token="anon"))


def vapor_pressure_from_dewpoint(dewpoint: np.ndarray):
    '''vapor pressure in Pa from dewpoint in K, the saturation formula over water used in Ai2's ERA5 pipeline'''
    return 611.21 * np.exp(17.502 * (dewpoint - 273.16) / (dewpoint - 32.19))


def specific_humidity_from_dewpoint(dewpoint: np.ndarray, pressure: np.ndarray):
    '''2 m specific humidity in kg/kg from dewpoint in K and surface pressure in Pa, the formula in Ai2's pipeline'''
    ewsat = vapor_pressure_from_dewpoint(dewpoint)
    eps = 0.621981
    return eps * ewsat / (pressure - (1 - eps) * ewsat)


def download_maine_hourly(year: int):
    '''download one season of hourly near-surface ERA5 over the Maine box to data/raw/era5/, with checks'''
    start, end = f"{year}-{SEASON_START}T00", f"{year}-{SEASON_END}T23"
    path = os.path.join(HOURLY_DIR, f"era5_maine_hourly_{year}.nc")
    os.makedirs(HOURLY_DIR, exist_ok=True)
    box = open_full_37()[list(HOURLY_VARIABLES)].sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))

    # one month at a time to bound memory and log progress
    t0 = time.perf_counter()
    months = []
    for month_start in np.arange(np.datetime64(start[:7]), np.datetime64(end[:7]) + 1):
        month_end = (month_start + 1).astype("datetime64[h]") - np.timedelta64(1, "h")
        lo = max(np.datetime64(start), month_start.astype("datetime64[h]"))
        hi = min(np.datetime64(end), month_end)
        months.append(box.sel(time=slice(lo, hi)).load())
        logger.info("%d: read %s to %s, %.0f s", year, lo, hi, time.perf_counter() - t0)
    ds = xr.concat(months, dim="time").rename(HOURLY_VARIABLES)

    # checks on the time axis, the grid and the values
    hours = (ds.time.values - ds.time.values[0]) / np.timedelta64(1, "h")
    assert np.array_equal(hours, np.arange(ds.sizes["time"])), "time steps are not consecutive hours"
    assert ds.sizes["latitude"] == 41 and ds.sizes["longitude"] == 45, f"unexpected box shape {dict(ds.sizes)}"
    for name in HOURLY_VARIABLES.values():
        num_nan = int(ds[name].isnull().sum())
        assert num_nan == 0, f"{name} has {num_nan} missing values"
    assert float(ds["t2m"].min()) > 200 and float(ds["t2m"].max()) < 330, "2 m temperature outside 200 to 330 K"
    # ERA5 accumulations and mean fluxes can hold tiny negative values from packing; they are logged and kept unclipped
    for name, tolerance in (("tp", -1e-6), ("ssrd", -1.0)):
        num_negative = int((ds[name] < 0).sum())
        logger.info("%s minimum %.3g, %d negative values of %d", name, float(ds[name].min()), num_negative, ds[name].size)
        assert float(ds[name].min()) >= tolerance, f"{name} has values below {tolerance}"

    ds.attrs["source"] = FULL_37_URL
    ds.attrs["downloaded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ds.attrs["slurm_job"] = os.environ.get("SLURM_JOB_ID", "none")
    encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 4} for name in HOURLY_VARIABLES.values()}
    ds.to_netcdf(path, encoding=encoding)
    logger.info("wrote %s, dims %s, %.1f MB", path, dict(ds.sizes), os.path.getsize(path) / 1e6)


def daily_from_hourly(ds: xr.Dataset):
    '''daily tmax, tmin, tmean (C), prcp (mm/day), ssrd (24 hour mean W/m2) and vp (Pa) from one year of hourly ERA5'''
    t0 = ds.time.values[0]
    first_day = (t0 + np.timedelta64(23, "h")).astype("datetime64[D]")  # first day whose window is fully inside the file
    last_day = (ds.time.values[-1] - np.timedelta64(30, "h")).astype("datetime64[D]")
    days = np.arange(first_day, last_day + 1)
    hours = (ds.time.values - days[0].astype("datetime64[h]")) / np.timedelta64(1, "h")
    hours = hours.astype(int)
    index = {h: k for k, h in enumerate(hours)}

    t2m, d2m, tp, ssrd = (ds[name].values.astype(np.float64) for name in ("t2m", "d2m", "tp", "ssrd"))
    out = {name: np.zeros((len(days),) + t2m.shape[1:]) for name in ("tmax", "tmin", "tmean", "prcp", "ssrd", "vp")}
    for j in range(len(days)):
        base = 24 * j
        snaps = [index[base + h] for h in SNAPSHOT_HOURS]
        # hourly accumulations and means are labeled by the end of their hour, so the window is hours 7 to 30 after midnight
        window = [index[base + h] for h in range(WINDOW_START_HOUR + 1, WINDOW_START_HOUR + 25)]
        out["tmax"][j] = t2m[snaps].max(axis=0) - 273.15
        out["tmin"][j] = t2m[snaps].min(axis=0) - 273.15
        out["tmean"][j] = t2m[snaps].mean(axis=0) - 273.15
        out["prcp"][j] = tp[window].sum(axis=0) * 1000.0  # m to mm
        out["ssrd"][j] = ssrd[window].mean(axis=0)
        out["vp"][j] = vapor_pressure_from_dewpoint(d2m[snaps]).mean(axis=0)
    return days, out


def build_daily_reference(years: range):
    '''daily ERA5 reference over the Maine box for the given years, from the hourly files, written to data/processed/era5/'''
    pieces = []
    for year in years:
        ds = xr.open_dataset(os.path.join(HOURLY_DIR, f"era5_maine_hourly_{year}.nc"))
        days, out = daily_from_hourly(ds)
        pieces.append(xr.Dataset(
            {name: (("time", "latitude", "longitude"), values.astype(np.float32)) for name, values in out.items()},
            coords={"time": days, "latitude": ds.latitude.values, "longitude": ds.longitude.values},
        ))
        logger.info("%d: %d days from %s to %s", year, len(days), days[0], days[-1])
    daily = xr.concat(pieces, dim="time")
    for name in daily.data_vars:
        assert np.isfinite(daily[name].values).all(), f"{name} has missing values"
    assert (daily["tmax"] >= daily["tmin"]).all(), "tmax below tmin"
    daily.attrs["day_definition"] = "06 UTC of the date to 06 UTC of the next date (01 EST to 01 EST)"
    daily.attrs["snapshots"] = "tmax, tmin, tmean and vp from 2 m snapshots at 12, 18, 00 and 06 UTC"
    os.makedirs(os.path.dirname(DAILY_PATH), exist_ok=True)
    daily.to_netcdf(DAILY_PATH)
    logger.info("wrote %s, dims %s", DAILY_PATH, dict(daily.sizes))


def save_aifs_inputs(start: str):
    '''the 0.25 degree ERA5 fields AIFS ENS needs at an issue time and 6 hours before, saved as one npz'''
    t = np.datetime64(start, "h")
    times = [t - np.timedelta64(6, "h"), t]
    era5 = open_full_37()
    assert float(era5.latitude[0]) == 90.0 and float(era5.longitude[0]) == 0.0, "unexpected ERA5 grid orientation"

    fields = {}
    for short, name in AIFS_SURFACE.items():
        da = era5[name]
        # time-invariant fields are repeated for both input times
        values = da.sel(time=times).values if "time" in da.dims else np.stack([da.values, da.values])
        fields[short] = values.astype(np.float32)  # shape (2, 721, 1440)
    for short, name in AIFS_PRESSURE.items():
        block = era5[name].sel(time=times, level=AIFS_LEVELS).values  # shape (2, levels, 721, 1440)
        for k, level in enumerate(AIFS_LEVELS):
            fields[f"{short}_{level}"] = block[:, k].astype(np.float32)
    for key, values in fields.items():
        assert values.shape == (2, 721, 1440), f"{key} has shape {values.shape}"
        assert np.isfinite(values).all() or key in ("stl1", "stl2"), f"{key} has missing values"
    logger.info("read %d fields at %s and %s", len(fields), times[0], times[1])

    os.makedirs(AIFS_INPUT_DIR, exist_ok=True)
    path = os.path.join(AIFS_INPUT_DIR, f"era5_aifs_inputs_{start[:13]}.npz")
    np.savez(path, times=np.array(times), **fields)
    logger.info("wrote %s", path)
    return path


def surface_fields(t: np.datetime64):
    '''ERA5 surface fields of an ACE2 initial condition at one time, latitude south to north, each of shape (721, 1440)'''
    full = open_full_37()
    return {name: full[name].sel(time=t).values[::-1].astype(np.float64) for name in SURFACE_NAMES}


def pressure_level_fields(t: np.datetime64, levels_hpa: list):
    '''ERA5 temperature, specific humidity and winds on the given pressure levels, latitude south to north, shape (levels, 721, 1440)'''
    full = open_full_37()
    fields = {"levels_pa": np.array(levels_hpa, dtype=np.float64) * 100.0}
    for name in ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]:
        fields[name] = full[name].sel(time=t, level=list(levels_hpa)).values[:, ::-1].astype(np.float64)
    return fields


def latest_model_level_time(t: np.datetime64):
    '''latest 6-hourly time at or before t with ERA5 model-level data; the ARCO store has values only through 2026-07-31T18'''
    model = xr.open_zarr(MODEL_LEVEL_URL, chunks=None, storage_options=dict(token="anon"))
    t_found = t
    while not np.isfinite(model["temperature"].sel(time=t_found, hybrid=1).values).all():
        t_found = t_found - np.timedelta64(6, "h")
        assert t - t_found <= np.timedelta64(24, "h"), "no ERA5 model-level data within 24 hours before the issue time"
    return t_found


def model_level_columns(t: np.datetime64):
    '''ERA5 temperature, total water (five species) and winds on the 137 model levels, and the hybrid coefficients

    arrays have shape (137, 721, 1440), top to bottom, latitude south to north; ak and bk have the 138 interface values
    '''
    # dask chunks so the model-level chunks of each field are fetched in parallel threads
    model = xr.open_zarr(MODEL_LEVEL_URL, chunks={}, storage_options=dict(token="anon"))
    assert np.array_equal(model.hybrid.values, np.arange(1, 138)), "model levels are not 1 to 137, top to bottom"
    # ERA5 hybrid coefficients; the top interface is kept at 0 Pa as in the published ACE2-ERA5 coordinate
    pv = np.array(model["temperature"].attrs["GRIB_pv"])
    total_water = 0.0
    for name in WATER_SPECIES:
        total_water = total_water + model[name].sel(time=t).values[:, ::-1].astype(np.float64)
    columns = {
        "air_temperature": model["temperature"].sel(time=t).values[:, ::-1].astype(np.float64),
        "specific_total_water": total_water,
        "eastward_wind": model["u_component_of_wind"].sel(time=t).values[:, ::-1].astype(np.float64),
        "northward_wind": model["v_component_of_wind"].sel(time=t).values[:, ::-1].astype(np.float64),
    }
    for name, values in columns.items():
        assert values.shape == (137, 721, 1440), f"{name} has shape {values.shape}"
    return columns, pv[:138], pv[138:]


def skin_and_ice(times: np.ndarray):
    '''ERA5 skin temperature (K) and sea ice cover at the given times, latitude south to north, shape (times, 721, 1440)'''
    full = open_full_37()
    skin = full["skin_temperature"].sel(time=times).values[:, ::-1].astype(np.float64)
    # sea ice is undefined over land, where it is set to zero as in Ai2's pipeline
    ice = np.nan_to_num(full["sea_ice_cover"].sel(time=times).values[:, ::-1].astype(np.float64), nan=0.0)
    return skin, ice


def toa_shortwave_6h(full: xr.Dataset, t: np.datetime64):
    '''6 hour mean of ERA5 top-of-atmosphere downward shortwave, the window ending at time t, latitude south to north

    full is the opened store from open_full_37, passed in so it is opened once for many times
    '''
    hours = t - np.arange(5, -1, -1) * np.timedelta64(1, "h")
    hourly = full["mean_top_downward_short_wave_radiation_flux"].sel(time=hours).values[:, ::-1].astype(np.float64)
    return hourly.mean(axis=0)
