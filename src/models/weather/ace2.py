import os
import sys
import glob
import logging
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

from models.weather import era5

logger = logging.getLogger(__name__)

# ACE2-ERA5 checkpoint and Ai2's published forcing files (huggingface.co/allenai/ACE2-ERA5)
ACE_DIR = "/hpc/group/youlab/xw262/models/ace2_era5"
CHECKPOINT = os.path.join(ACE_DIR, "ace2_era5_ckpt.tar")
TEMPLATE_PATH = os.path.join(ACE_DIR, "forcing_2022.nc")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "interim", "ace2_inputs")
AIFS_DIR = os.path.join(REPO_ROOT, "data", "interim", "weather_forecast", "aifs_ens")
OUT_DIR = os.path.join(REPO_ROOT, "data", "interim", "weather_forecast", "ace2")

# ACE2-ERA5 conventions, from Ai2's ERA5 pipeline (github.com/ai2cm/ace, scripts/era5/pipeline/xr-beam-pipeline.py)
# interfaces of the 8 coarse layers among the 138 ERA5 model-level interfaces, top to bottom
LAYER_INDICES = [0, 48, 67, 79, 90, 100, 109, 119, 137]
NUM_LAYERS = 8

CONFIG = """experiment_dir: {exp_dir}
n_forward_steps: {steps}
forward_steps_in_memory: 50
checkpoint_path: {checkpoint}
logging:
  log_to_screen: true
  log_to_wandb: false
  log_to_file: true
  project: ace
initial_condition:
  path: {ic_path}
  start_indices:
    times: [{ic_times}]
forcing_loader:
  dataset:
    data_path: {forcing_dir}
  num_data_workers: 2
data_writer:
  save_prediction_files: false
  save_monthly_files: false
  files:
    - label: maine
      names: [TMP2m, Q2m, PRESsfc, PRATEsfc, DSWRFsfc, surface_temperature]
      lat_extent: [40.0, 50.0]
      lon_extent: [285.0, 296.0]
      save_reference: false
"""


def overlap_weights(dst_lo: np.ndarray, dst_hi: np.ndarray, src_lo: np.ndarray, src_hi: np.ndarray):
    '''matrix of overlap lengths between destination and source intervals, shape (destination, source)'''
    return np.clip(np.minimum(dst_hi[:, None], src_hi[None, :]) - np.maximum(dst_lo[:, None], src_lo[None, :]), 0, None)


def regrid_weights(dst_lat: np.ndarray, dst_lon: np.ndarray):
    '''separable conservative weights from the 0.25 degree ERA5 grid (latitude south to north) to a rectilinear grid'''
    src_lat = np.linspace(-90, 90, 721)
    src_lat_b = np.concatenate([[-90], 0.5 * (src_lat[:-1] + src_lat[1:]), [90]])
    dst_lat_b = np.concatenate([[-90], 0.5 * (dst_lat[:-1] + dst_lat[1:]), [90]])
    # latitude overlaps are areas, so they are measured in sine of latitude
    w_lat = overlap_weights(
        np.sin(np.radians(dst_lat_b[:-1])), np.sin(np.radians(dst_lat_b[1:])),
        np.sin(np.radians(src_lat_b[:-1])), np.sin(np.radians(src_lat_b[1:])),
    )
    src_lon = np.arange(1440) * 0.25
    src_lo, src_hi = src_lon - 0.125, src_lon + 0.125
    dlon = dst_lon[1] - dst_lon[0]
    dst_lo, dst_hi = dst_lon - dlon / 2, dst_lon + dlon / 2
    # source cells are also shifted by one full turn so the cell centered on 0 degrees is counted on both sides
    w_lon = sum(overlap_weights(dst_lo, dst_hi, src_lo + shift, src_hi + shift) for shift in (-360.0, 0.0, 360.0))
    return w_lat, w_lon


def regrid(field: np.ndarray, w_lat: np.ndarray, w_lon: np.ndarray):
    '''conservative regridding of one or more fields of shape (..., 721, 1440), latitude south to north; NaN is skipped'''
    valid = np.isfinite(field)
    # two matrix products, latitude then longitude; broadcasting handles leading dimensions
    num = w_lat @ np.where(valid, field, 0.0) @ w_lon.T
    den = w_lat @ valid.astype(np.float64) @ w_lon.T
    # an output cell is defined if any part of it is covered by valid input, as xESMF with na_thres 1.0
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan).astype(np.float32)


def load_grid():
    '''the ACE2-ERA5 grid (1 degree Gaussian, latitude south to north) from Ai2's 2022 forcing file, and the regridding weights'''
    template = xr.open_dataset(TEMPLATE_PATH)
    w_lat, w_lon = regrid_weights(template.latitude.values, template.longitude.values)
    assert w_lat.shape == (180, 721) and w_lon.shape == (360, 1440), "unexpected weight shapes"
    return template, w_lat, w_lon


def surface_to_ace2(fields: dict, surface: dict, w_lat: np.ndarray, w_lon: np.ndarray):
    '''add the regridded surface fields of an ACE2 initial condition to fields; 2 m humidity from regridded dewpoint and pressure'''
    fields["PRESsfc"] = regrid(surface["surface_pressure"], w_lat, w_lon)
    fields["TMP2m"] = regrid(surface["2m_temperature"], w_lat, w_lon)
    fields["UGRD10m"] = regrid(surface["10m_u_component_of_wind"], w_lat, w_lon)
    fields["VGRD10m"] = regrid(surface["10m_v_component_of_wind"], w_lat, w_lon)
    fields["surface_temperature"] = regrid(surface["skin_temperature"], w_lat, w_lon)
    # 2 m humidity from the regridded dewpoint and surface pressure, in that order as in Ai2's pipeline
    dewpoint = regrid(surface["2m_dewpoint_temperature"], w_lat, w_lon)
    fields["Q2m"] = era5.specific_humidity_from_dewpoint(dewpoint.astype(np.float64), fields["PRESsfc"].astype(np.float64)).astype(np.float32)


def to_dataset(fields: dict, t: np.datetime64, template: xr.Dataset):
    '''ACE2 initial condition dataset at one time, with a check for missing values'''
    ds = xr.Dataset(
        {name: (("time", "latitude", "longitude"), values[None]) for name, values in fields.items()},
        coords={"time": [t], "latitude": template.latitude.values, "longitude": template.longitude.values},
    )
    for name in ds.data_vars:
        assert np.isfinite(ds[name].values).all(), f"{name} has missing values"
    return ds


def initial_condition_from_model_levels(t: np.datetime64, template: xr.Dataset, w_lat: np.ndarray, w_lon: np.ndarray):
    '''ACE2-ERA5 initial condition at one time from ERA5 model levels and surface fields, as in Ai2's pipeline (V12)'''
    columns, ak, bk = era5.model_level_columns(t)
    surface = era5.surface_fields(t)
    ps = surface["surface_pressure"]
    dp = (ak[1:] - ak[:-1])[:, None, None] + (bk[1:] - bk[:-1])[:, None, None] * ps[None]  # shape (137, 721, 1440)
    assert (dp > 0).all(), "non-positive layer thickness"
    fields = {}
    for name, values in columns.items():
        # pressure-weighted mean over the fine levels of each coarse layer, at 0.25 degrees, then regridding
        for i in range(NUM_LAYERS):
            k0, k1 = LAYER_INDICES[i], LAYER_INDICES[i + 1]
            layer = (values[k0:k1] * dp[k0:k1]).sum(axis=0) / dp[k0:k1].sum(axis=0)
            fields[f"{name}_{i}"] = regrid(layer, w_lat, w_lon)
        logger.info("coarsened and regridded %s", name)
    del columns
    surface_to_ace2(fields, surface, w_lat, w_lon)
    return to_dataset(fields, t, template)


def layer_means_from_pressure_levels(values: np.ndarray, levels: np.ndarray, surface: np.ndarray, ps: np.ndarray, ak: np.ndarray, bk: np.ndarray):
    '''pressure-weighted mean over each ACE2 layer of a profile given on pressure levels plus its surface value

    values has shape (levels, lat, lon) on levels in Pa ascending; levels below ground are replaced by the surface node,
    the profile is linear in pressure between nodes and constant above the top level; returns shape (layers, lat, lon)
    '''
    num_samples = 40  # sample pressures per layer, midpoints of equal sub-intervals
    out = np.empty((len(ak) - 1,) + ps.shape)
    for i in range(len(ak) - 1):
        p_top, p_bot = ak[i] + bk[i] * ps, ak[i + 1] + bk[i + 1] * ps
        frac = (np.arange(num_samples) + 0.5) / num_samples
        p = p_top[None] + frac[:, None, None] * (p_bot - p_top)[None]  # shape (samples, lat, lon)
        k = np.searchsorted(levels, p)  # number of levels with pressure below p
        k_lo = np.clip(k - 1, 0, len(levels) - 1)
        p_lo = levels[k_lo]
        v_lo = np.take_along_axis(values, k_lo, axis=0)
        # the upper node is the next level if it is above ground, otherwise the surface
        k_hi = np.clip(k, 0, len(levels) - 1)
        use_surface = (k >= len(levels)) | (levels[k_hi] >= ps[None])
        p_hi = np.where(use_surface, ps[None], levels[k_hi])
        v_hi = np.where(use_surface, surface[None], np.take_along_axis(values, k_hi, axis=0))
        weight = np.clip((p - p_lo) / np.where(p_hi > p_lo, p_hi - p_lo, 1.0), 0.0, 1.0)
        sample = v_lo + weight * (v_hi - v_lo)
        # above the top level the top value is held
        sample = np.where(k == 0, values[0][None], sample)
        out[i] = sample.mean(axis=0)
    return out


def initial_condition_from_pressure_levels(fields_pl: dict, surface: dict, t: np.datetime64, template: xr.Dataset, w_lat: np.ndarray, w_lon: np.ndarray):
    '''ACE2-ERA5 initial condition from 0.25 degree fields on pressure levels, used for the AIFS ENS hand-off (D20, V15)

    fields_pl maps temperature, specific_humidity, u_component_of_wind, v_component_of_wind to arrays of shape
    (levels, 721, 1440) on the pressure levels of fields_pl["levels_pa"]; surface maps era5.SURFACE_NAMES to
    (721, 1440); all latitude south to north. Total water is taken as specific humidity, since cloud water is not available.
    '''
    ak = np.array([float(template[f"ak_{i}"]) for i in range(NUM_LAYERS + 1)])
    bk = np.array([float(template[f"bk_{i}"]) for i in range(NUM_LAYERS + 1)])
    levels = np.asarray(fields_pl["levels_pa"], dtype=np.float64)
    assert np.all(np.diff(levels) > 0), "pressure levels must be ascending"
    ps = surface["surface_pressure"]
    q2m = era5.specific_humidity_from_dewpoint(surface["2m_dewpoint_temperature"], ps)
    profiles = {
        "air_temperature": (fields_pl["temperature"], surface["2m_temperature"]),
        "specific_total_water": (fields_pl["specific_humidity"], q2m),
        "eastward_wind": (fields_pl["u_component_of_wind"], surface["10m_u_component_of_wind"]),
        "northward_wind": (fields_pl["v_component_of_wind"], surface["10m_v_component_of_wind"]),
    }
    fields = {}
    for name, (values, surface_value) in profiles.items():
        assert values.shape == (len(levels),) + ps.shape, f"{name} has shape {values.shape}"
        layers = layer_means_from_pressure_levels(values.astype(np.float64), levels, surface_value, ps, ak, bk)
        for i in range(NUM_LAYERS):
            fields[f"{name}_{i}"] = regrid(layers[i], w_lat, w_lon)
    surface_to_ace2(fields, surface, w_lat, w_lon)
    return to_dataset(fields, t, template)


def issue_time_correction(start: np.datetime64, levels_pa: np.ndarray, template: xr.Dataset, w_lat: np.ndarray, w_lon: np.ndarray):
    '''conversion error at the issue time, exact model-level build minus the pressure-level conversion, both from ERA5 (V15)'''
    levels_hpa = (np.asarray(levels_pa) / 100).astype(int).tolist()
    exact = initial_condition_from_model_levels(start, template, w_lat, w_lon).isel(time=0)
    converted = initial_condition_from_pressure_levels(
        era5.pressure_level_fields(start, levels_hpa), era5.surface_fields(start), start, template, w_lat, w_lon
    ).isel(time=0)
    correction = (exact - converted).drop_vars("time")
    logger.info("correction at the issue time: RMS of air_temperature_0 %.3g K, air_temperature_7 %.3g K",
                float(np.sqrt((correction["air_temperature_0"] ** 2).mean())), float(np.sqrt((correction["air_temperature_7"] ** 2).mean())))
    return correction


def forcing_climatology(name: str, times: np.ndarray, years: range):
    '''mean over years and a 31 day window of an ACE2 forcing variable, for each requested time of year'''
    # the window is applied on Ai2's 6-hourly forcing files, at the same hour of day
    out = np.zeros((len(times), 180, 360))
    count = np.zeros(len(times))
    for year in years:
        f = xr.open_dataset(os.path.join(ACE_DIR, "forcing_data", f"forcing_{year}.nc"))
        ftimes = pd.DatetimeIndex(f.time.values)
        values = f[name].values  # shape (times of that year, 180, 360)
        for j, t in enumerate(pd.DatetimeIndex(times)):
            same_hour = ftimes.hour == t.hour
            # day-of-year distance with wraparound, so late December and early January are neighbors
            dist = np.abs(ftimes.dayofyear - t.dayofyear)
            dist = np.minimum(dist, 365 - dist)
            sel = same_hour & (dist <= 15)
            out[j] += values[sel].sum(axis=0)
            count[j] += sel.sum()
        f.close()
    assert (count > 0).all(), "empty climatology window"
    return (out / count[:, None, None]).astype(np.float32)


def build_forcing(start: str, first_time: np.datetime64, end: np.datetime64, mode: str, clim_years: range,
                  template: xr.Dataset, w_lat: np.ndarray, w_lon: np.ndarray):
    '''6-hourly ACE2 forcing from first_time to end; returns the directory written

    mode persisted: ERA5 up to the issue time, then the sea surface temperature and sea ice anomaly of the issue time held
    fixed and added to the climatology (D13); mode observed: ERA5 throughout, for validation only
    '''
    t_issue = np.datetime64(start, "h")
    times = np.arange(first_time, end + np.timedelta64(6, "h"), np.timedelta64(6, "h"))
    past = times <= t_issue
    land = template["land_fraction"].values
    if mode == "observed":
        skin_raw, ice_raw = era5.skin_and_ice(times)
        skin, ice = regrid(skin_raw, w_lat, w_lon), regrid(ice_raw, w_lat, w_lon)
    else:
        skin_raw, ice_raw = era5.skin_and_ice(times[past])
        skin_obs, ice_obs = regrid(skin_raw, w_lat, w_lon), regrid(ice_raw, w_lat, w_lon)
        skin_clim = forcing_climatology("surface_temperature", times, clim_years)
        ice_clim = forcing_climatology("sea_ice_fraction", times, clim_years)
        k = int(np.flatnonzero(past)[-1])  # index of the issue time
        skin = skin_clim + (skin_obs[k] - skin_clim[k])[None]
        ice = np.clip(ice_clim + (ice_obs[k] - ice_clim[k])[None], 0.0, 1.0)
        skin[past], ice[past] = skin_obs, ice_obs
        logger.info("persisted anomaly after %s, climatology years %d to %d", t_issue, clim_years[0], clim_years[-1])

    # ocean fraction as in Ai2's pipeline, with sea ice reduced where land plus ice exceeds one
    ocean = 1.0 - land[None] - ice
    ice = ice + np.minimum(ocean, 0.0)
    ocean = np.maximum(ocean, 0.0)

    # global mean CO2 from the 2022 file, extended by a constant growth rate for later years (D22)
    co2_2022 = template["global_mean_co2"].to_series()
    growth = 2.4e-6  # mol/mol per year, approximate recent global growth rate
    co2 = []
    for t in pd.DatetimeIndex(times):
        ref = co2_2022.index[np.argmin(np.abs(co2_2022.index.dayofyear - t.dayofyear))]
        co2.append(co2_2022[ref] + growth * (t.year - 2022))
    logger.info("global mean CO2 from %.3g to %.3g mol/mol, growth %.2g per year after 2022", co2[0], co2[-1], growth)

    full = era5.open_full_37()
    toa = np.stack([regrid(era5.toa_shortwave_6h(full, t), w_lat, w_lon) for t in times])
    # top-of-atmosphere shortwave is astronomical; where ERA5 has no value yet (2026 after the ERA5T end), the 2022 value
    # of Ai2's forcing file at the same day of year and hour is used
    missing = ~np.isfinite(toa).all(axis=(1, 2))
    if missing.any():
        ref_times = pd.DatetimeIndex(template.time.values)
        for j in np.flatnonzero(missing):
            t = pd.Timestamp(times[j])
            k = np.flatnonzero((ref_times.dayofyear == t.dayofyear) & (ref_times.hour == t.hour))[0]
            toa[j] = template["DSWRFtoa"].values[k]
        logger.info("top-of-atmosphere shortwave from the 2022 forcing file at %d times without ERA5, %s to %s", missing.sum(), times[missing][0], times[missing][-1])
    forcing = xr.Dataset(
        {
            "DSWRFtoa": (("time", "latitude", "longitude"), toa),
            "surface_temperature": (("time", "latitude", "longitude"), skin.astype(np.float32)),
            "sea_ice_fraction": (("time", "latitude", "longitude"), ice.astype(np.float32)),
            "ocean_fraction": (("time", "latitude", "longitude"), ocean.astype(np.float32)),
            "global_mean_co2": (("time",), np.array(co2, dtype=np.float32)),
            "HGTsfc": template["HGTsfc"],
            "land_fraction": template["land_fraction"],
        },
        coords={"time": times, "latitude": template.latitude.values, "longitude": template.longitude.values},
    )
    for i in range(NUM_LAYERS + 1):
        forcing[f"ak_{i}"] = template[f"ak_{i}"]
        forcing[f"bk_{i}"] = template[f"bk_{i}"]
    for name in ["DSWRFtoa", "surface_temperature", "sea_ice_fraction", "ocean_fraction"]:
        assert np.isfinite(forcing[name].values).all(), f"{name} has missing values"
    forcing_dir = os.path.join(INPUT_DIR, f"forcing_{start[:13]}_{mode}")
    os.makedirs(forcing_dir, exist_ok=True)
    forcing.to_netcdf(os.path.join(forcing_dir, "forcing.nc"))
    logger.info("wrote %s, %d times from %s to %s", forcing_dir, len(times), times[0], times[-1])
    return forcing_dir


def run_inference(exp_dir: str, ic_path: str, ic_times: list, steps: int, forcing_dir: str):
    '''write the fme configuration and run ACE2-ERA5 inference from the initial times in ic_path; returns the Maine output path'''
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(CONFIG.format(exp_dir=exp_dir, steps=steps, checkpoint=CHECKPOINT, ic_path=ic_path,
                              ic_times=", ".join(f'"{np.datetime64(t, "s")}"' for t in ic_times), forcing_dir=forcing_dir))
    result = subprocess.run([sys.executable, "-m", "fme.ace.inference", config_path], capture_output=True, text=True)
    out_path = os.path.join(exp_dir, "maine.nc")
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"fme inference failed in {exp_dir}\n{result.stderr[-3000:]}")
    out = xr.open_dataset(out_path, decode_timedelta=True)
    num_nan = sum(int(out[v].isnull().sum()) for v in out.data_vars)
    assert num_nan == 0, f"{out_path} has missing values"
    return out_path


def run_from_handoff(start: str, corrected_names: list, clim_years: range):
    '''persisted forcing from the issue time, then ACE2 from each AIFS ENS member's hand-off state to 06 UTC on December 1 (D20)'''
    t_issue = np.datetime64(start, "h")
    # the last daily window, November 30, ends at 06 UTC on December 1
    end = np.datetime64(f"{start[:4]}-12-01T06", "h")
    template, w_lat, w_lon = load_grid()
    # the forcing runs past December 1 by two days, as for the lagged members of the validation runs
    forcing_dir = build_forcing(start, t_issue, np.datetime64(f"{start[:4]}-12-03T00", "h"), "persisted", clim_years, template, w_lat, w_lon)

    paths = sorted(glob.glob(os.path.join(AIFS_DIR, start[:13], "handoff_member*.npz")))
    assert len(paths) > 0, f"no hand-off files for {start}"
    correction = None
    if corrected_names:
        t_corr = era5.latest_model_level_time(t_issue)
        logger.info("correction measured at %s", t_corr)
        correction = issue_time_correction(t_corr, np.load(paths[0])["levels_pa"], template, w_lat, w_lon)
    for path in paths:
        member = os.path.basename(path).replace("handoff_", "").replace(".npz", "")
        handoff = np.load(path)
        t = np.datetime64(handoff["time"].item(), "h")
        steps = int((end - t) / np.timedelta64(6, "h"))
        assert steps > 0, f"hand-off time {t} is not before {end}"

        # ACE2 initial condition from the member's own state on 13 pressure levels plus surface fields
        fields_pl = {name: handoff[name].astype(np.float64) for name in ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]}
        fields_pl["levels_pa"] = handoff["levels_pa"]
        surface = {name: handoff[name].astype(np.float64) for name in era5.SURFACE_NAMES}
        ic = initial_condition_from_pressure_levels(fields_pl, surface, t, template, w_lat, w_lon)
        if correction is not None:
            # only the layers at and above the highest AIFS level, where the conversion error is a persistent profile shape (V15)
            for name in corrected_names:
                ic[name] = ic[name] + correction[name].expand_dims(time=ic.time)
            ic["specific_total_water_0"] = ic["specific_total_water_0"].clip(min=0.0)
        exp_dir = os.path.join(OUT_DIR, start[:13], member)
        os.makedirs(exp_dir, exist_ok=True)
        ic_path = os.path.join(exp_dir, "ic.nc")
        ic.to_netcdf(ic_path)
        logger.info("%s: hand-off %s, %d steps to %s", member, t, steps, end)
        out_path = run_inference(exp_dir, ic_path, [t], steps, forcing_dir)
        logger.info("%s: wrote %s, 0 NaN values", member, out_path)
