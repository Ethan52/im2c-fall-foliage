import os
import time
import logging

import numpy as np
import torch
import xarray as xr
import earthkit.regrid as ekr
from anemoi.inference.runners.simple import SimpleRunner

logger = logging.getLogger(__name__)

# AIFS ENS 1.0 weights (ECMWF, CC BY 4.0), cached by Hugging Face under HF_HOME
CHECKPOINT = {"huggingface": "ecmwf/aifs-ens-1.0"}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "data", "interim", "aifs_inputs")
OUT_DIR = os.path.join(REPO_ROOT, "data", "interim", "weather_forecast", "aifs_ens")

# Maine box with a margin, longitudes in degrees east
LAT_MIN, LAT_MAX = 40.0, 50.0
LON_MIN, LON_MAX = 285.0, 296.0
# 6-hourly fields kept over Maine: 2 m temperature and dewpoint (K), surface pressure (Pa),
# precipitation (m) and surface downward shortwave (J/m2), both accumulated over the 6 hours ending at the time stamp
MAINE_FIELDS = ["2t", "2d", "sp", "tp", "ssrd"]
# global fields at the hand-off time, converted to an ACE2 initial condition (D20)
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
HANDOFF_SURFACE = {"sp": "surface_pressure", "2t": "2m_temperature", "2d": "2m_dewpoint_temperature",
                   "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind", "skt": "skin_temperature"}
HANDOFF_PRESSURE = {"t": "temperature", "q": "specific_humidity", "u": "u_component_of_wind", "v": "v_component_of_wind"}


def to_n320(input_path: str):
    '''0.25 degree ERA5 input fields (from era5.save_aifs_inputs) interpolated to N320 as in ECMWF's example notebook'''
    raw = np.load(input_path)
    fields = {k: np.stack([ekr.interpolate(v, {"grid": (0.25, 0.25)}, {"grid": "N320"}) for v in raw[k]]) for k in raw.files if k != "times"}
    logger.info("regridded %d input fields to N320, times %s", len(fields), raw["times"])
    return fields


def save_handoff(state: dict, handoff_time: np.datetime64, path: str):
    '''global state at the hand-off, interpolated from N320 to 0.25 degrees and flipped to latitude south to north'''
    handoff = {"levels_pa": np.array(LEVELS) * 100.0}
    for short, name in HANDOFF_SURFACE.items():
        handoff[name] = ekr.interpolate(np.asarray(state["fields"][short]), {"grid": "N320"}, {"grid": (0.25, 0.25)})[::-1]
    for short, name in HANDOFF_PRESSURE.items():
        handoff[name] = np.stack([
            ekr.interpolate(np.asarray(state["fields"][f"{short}_{level}"]), {"grid": "N320"}, {"grid": (0.25, 0.25)})[::-1]
            for level in LEVELS
        ])
    for key, values in handoff.items():
        assert np.isfinite(values).all(), f"hand-off field {key} has missing values"
        if key != "levels_pa":
            assert values.shape[-2:] == (721, 1440), f"hand-off field {key} has shape {values.shape}"
    np.savez(path, time=handoff_time, **handoff)


def run_members(start: str, members: int, lead_hours: int, out_dir: str):
    '''AIFS ENS members from the ERA5 state at the issue time; writes the Maine box (6-hourly) and each member's state at the last step

    each member is a different realization of the model noise, seeded by its index; returns the path of maine.nc
    '''
    logger.info("torch %s, device %s, ANEMOI_INFERENCE_NUM_CHUNKS %s", torch.__version__, torch.cuda.get_device_name(0),
                os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS"))
    fields = to_n320(os.path.join(INPUT_DIR, f"era5_aifs_inputs_{start[:13]}.npz"))
    runner = SimpleRunner(CHECKPOINT, device="cuda")
    t_issue = np.datetime64(start, "h")
    handoff_time = t_issue + np.timedelta64(lead_hours, "h")
    os.makedirs(out_dir, exist_ok=True)

    maine = []
    for member in range(members):
        torch.manual_seed(member)
        t0 = time.perf_counter()
        records, step_times, keep = [], [], None
        for state in runner.run(input_state=dict(date=t_issue.astype("datetime64[s]").item(), fields=dict(fields)), lead_time=lead_hours):
            if keep is None:
                lat, lon = state["latitudes"], state["longitudes"]
                keep = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
            records.append(np.stack([np.asarray(state["fields"][n])[keep] for n in MAINE_FIELDS]))
            step_times.append(np.datetime64(state["date"], "h"))
            if step_times[-1] == handoff_time:
                save_handoff(state, handoff_time, os.path.join(out_dir, f"handoff_member{member:02d}.npz"))
        out = np.stack(records)  # shape (steps, fields, points)
        assert step_times[-1] == handoff_time, f"last step {step_times[-1]} is not the hand-off time {handoff_time}"
        assert np.isfinite(out).all(), f"member {member} has missing values"
        maine.append(out)
        logger.info("member %d: %.0f s, %d steps to %s", member, time.perf_counter() - t0, len(step_times), step_times[-1])

    stacked = np.stack(maine)  # shape (members, steps, fields, points)
    ds = xr.Dataset(
        {n: (("member", "time", "point"), stacked[:, :, j, :]) for j, n in enumerate(MAINE_FIELDS)},
        coords={"time": np.array(step_times), "latitude": ("point", lat[keep]), "longitude": ("point", lon[keep])},
        attrs={"issue_time": str(t_issue), "units": "2t, 2d K; sp Pa; tp m and ssrd J/m2 accumulated over the 6 hours ending at each time"},
    )
    path = os.path.join(out_dir, "maine.nc")
    ds.to_netcdf(path)
    logger.info("wrote %s and %d hand-off files, peak GPU memory %.2f GB", path, members, torch.cuda.max_memory_allocated() / 1e9)
    return path
