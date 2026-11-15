import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from models.weather import era5, ace2, mapping

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SITES_CSV = os.path.join(REPO_ROOT, "configs", "sites.csv")
SEASON_DIR = os.path.join(REPO_ROOT, "data", "interim", "weather_smoke")
# Ai2's published initial condition for 2020-01-01T00, downloaded from huggingface.co/allenai/ACE2-ERA5
PUBLISHED_IC = os.path.join(ace2.ACE_DIR, "ic_2020.nc")
# lead-time windows in days for the drift and spread summaries
WINDOWS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 122)]
# a hand-off 15 days after an issue time, both in a training year (V15)
CORRECTION_ISSUE = np.datetime64("2019-12-17T00", "h")
CORRECTION_HANDOFF = np.datetime64("2020-01-01T00", "h")


def compare(name: str, ours: np.ndarray, theirs: np.ndarray, maine=None, height=None):
    '''log the difference between two fields, globally, over the Maine box, and against terrain height'''
    diff = ours.astype(np.float64) - theirs.astype(np.float64)
    logger.info(
        "%-24s max abs diff %.4g, RMS diff %.4g, RMS diff over std of reference field %.2e",
        name, np.abs(diff).max(), np.sqrt((diff ** 2).mean()), np.sqrt((diff ** 2).mean()) / np.std(theirs),
    )
    if maine is not None:
        d = diff[..., maine[0], maine[1]] if diff.ndim == 2 else diff[:, maine[0], maine[1]]
        r = np.corrcoef(np.abs(diff.reshape(-1, height.size)).mean(axis=0), height.ravel())[0, 1]
        logger.info("%-24s   Maine box max abs diff %.4g, RMS diff %.4g; correlation of abs diff with terrain height %.2f", "", np.abs(d).max(), np.sqrt((d ** 2).mean()), r)


def check_builder():
    '''our ERA5-to-ACE2 initial condition and forcing against Ai2's published files, and the hand-off conversion (V12, V15)'''
    template, w_lat, w_lon = ace2.load_grid()
    lat, lon = template.latitude.values, template.longitude.values
    maine = np.ix_((lat >= 42) & (lat <= 48), (lon >= 288) & (lon <= 294))
    height = template["HGTsfc"].values

    t = np.datetime64("2020-01-01T00", "h")
    published = xr.open_dataset(PUBLISHED_IC).sel(time=t)
    ours = ace2.initial_condition_from_model_levels(t, template, w_lat, w_lon).isel(time=0)
    assert set(ours.data_vars) == set(published.data_vars), f"variable sets differ: {set(ours.data_vars) ^ set(published.data_vars)}"
    logger.info("model-level build against the published initial condition")
    for name in sorted(published.data_vars):
        compare(name, ours[name].values, published[name].values, maine, height)

    # hand-off conversion (D20): the same time built from ERA5 on the 13 AIFS ENS pressure levels only
    from_pl = ace2.initial_condition_from_pressure_levels(
        era5.pressure_level_fields(t, sorted(era5.AIFS_LEVELS)), era5.surface_fields(t), t, template, w_lat, w_lon
    ).isel(time=0)
    logger.info("hand-off conversion from 13 pressure levels, against the published initial condition")
    for name in sorted(published.data_vars):
        compare(name, from_pl[name].values, published[name].values, maine, height)
    logger.info("hand-off conversion from 13 pressure levels, against our model-level build")
    for name in sorted(published.data_vars):
        compare(name, from_pl[name].values, ours[name].values, maine, height)

    # forcing for four times on 2022-08-01 against the published 2022 forcing
    times = np.arange(np.datetime64("2022-08-01T00", "h"), np.datetime64("2022-08-02T00", "h"), np.timedelta64(6, "h"))
    skin_raw, ice_raw = era5.skin_and_ice(times)
    ref = template.sel(time=times)
    compare("surface_temperature", ace2.regrid(skin_raw, w_lat, w_lon), ref["surface_temperature"].values)
    compare("sea_ice_fraction", ace2.regrid(ice_raw, w_lat, w_lon), ref["sea_ice_fraction"].values)
    full = era5.open_full_37()
    toa = np.stack([ace2.regrid(era5.toa_shortwave_6h(full, s), w_lat, w_lon) for s in times])
    compare("DSWRFtoa window ending", toa, ref["DSWRFtoa"].values)
    # the same comparison with the 6 hour window starting at each time, to check the labeling convention
    toa_next = np.stack([ace2.regrid(era5.toa_shortwave_6h(full, s + np.timedelta64(6, "h")), w_lat, w_lon) for s in times])
    compare("DSWRFtoa window starting", toa_next, ref["DSWRFtoa"].values)


def check_handoff_correction():
    '''the issue-time correction of the hand-off conversion on a training-year case (V15)'''
    template, w_lat, w_lon = ace2.load_grid()
    lat, lon = template.latitude.values, template.longitude.values
    maine = np.ix_((lat >= 42) & (lat <= 48), (lon >= 288) & (lon <= 294))
    levels = sorted(era5.AIFS_LEVELS)
    truth = ace2.initial_condition_from_model_levels(CORRECTION_HANDOFF, template, w_lat, w_lon).isel(time=0)
    raw = ace2.initial_condition_from_pressure_levels(
        era5.pressure_level_fields(CORRECTION_HANDOFF, levels), era5.surface_fields(CORRECTION_HANDOFF), CORRECTION_HANDOFF, template, w_lat, w_lon
    ).isel(time=0)
    correction = ace2.issue_time_correction(CORRECTION_ISSUE, np.array(levels) * 100.0, template, w_lat, w_lon)
    corrected = raw + correction
    logger.info("RMS difference from the exact model-level build at %s: raw conversion, then conversion corrected with the error at %s", CORRECTION_HANDOFF, CORRECTION_ISSUE)
    for name in sorted(truth.data_vars):
        e_raw = (raw[name] - truth[name]).values
        e_cor = (corrected[name] - truth[name]).values
        logger.info(
            "%-24s global raw %.4g, corrected %.4g; Maine raw %.4g, corrected %.4g",
            name, np.sqrt((e_raw ** 2).mean()), np.sqrt((e_cor ** 2).mean()),
            np.sqrt((e_raw[maine] ** 2).mean()), np.sqrt((e_cor[maine] ** 2).mean()),
        )


def season_ace2(start: str, lags: int, clim_years: range):
    '''ACE2-ERA5 alone for one season from lagged ERA5 initial conditions, persisted forcing (V11); returns the Maine output path'''
    template, w_lat, w_lon = ace2.load_grid()
    t_issue = np.datetime64(start, "h")
    # member k starts from ERA5 at the issue time minus 6 k hours, all known at the issue time
    ic_times = t_issue - np.arange(lags)[::-1] * np.timedelta64(6, "h")
    ic = xr.concat([ace2.initial_condition_from_model_levels(t, template, w_lat, w_lon) for t in ic_times], dim="time")
    os.makedirs(ace2.INPUT_DIR, exist_ok=True)
    ic_path = os.path.join(ace2.INPUT_DIR, f"ic_{start[:13]}_lags{lags}.nc")
    ic.to_netcdf(ic_path)
    forcing_dir = ace2.build_forcing(start, ic_times[0], np.datetime64(f"{start[:4]}-12-03T00", "h"), "persisted", clim_years, template, w_lat, w_lon)
    steps = int((np.datetime64(f"{start[:4]}-12-01T06", "h") - ic_times[0]) / np.timedelta64(6, "h"))
    return ace2.run_inference(os.path.join(SEASON_DIR, f"ace2_{start[:13]}_lags{lags}"), ic_path, list(ic_times), steps, forcing_dir)


def season_aifs(start: str, members: int):
    '''AIFS ENS alone for one season from ERA5 (V11), needs era5.save_aifs_inputs first; returns the Maine output path'''
    from models.weather import aifs  # torch, anemoi and earthkit exist only in the AIFS environment
    lead_hours = int((np.datetime64(f"{start[:4]}-12-01T06", "h") - np.datetime64(start, "h")) / np.timedelta64(1, "h"))
    return aifs.run_members(start, members, lead_hours, os.path.join(SEASON_DIR, f"aifs_ens_{start[:13]}"))


def site_series(path: str, kind: str):
    '''2 m temperature at the model point nearest each site, members by times, from a season run; also the times'''
    sim = xr.open_dataset(path, decode_timedelta=True)
    if kind == "ace2":
        # fme writes lead time per lagged member; keep the valid times from the latest start onward, common to all members
        start = sim.valid_time.values[:, 0].max()
        series = []
        for m in range(sim.sizes["sample"]):
            member = sim.isel(sample=m).swap_dims(time="valid_time").drop_vars("time").rename(valid_time="time")
            series.append(member.sel(time=member.time >= start))
        common = np.intersect1d(series[0].time.values, series[-1].time.values)
        sim = xr.concat([m.sel(time=common) for m in series], dim="member").rename(lat="latitude", lon="longitude")
    out = {}
    for _, site in pd.read_csv(SITES_CSV).iterrows():
        lon_east = site.longitude % 360
        if kind == "aifs":
            dist = (sim.latitude.values - site.latitude) ** 2 + ((sim.longitude.values - lon_east) * np.cos(np.radians(site.latitude))) ** 2
            j = int(np.argmin(dist))
            out[site.site_name] = (sim["2t"].isel(point=j).values, float(sim.latitude[j]), float(sim.longitude[j]))
        else:
            point = sim["TMP2m"].sel(latitude=site.latitude, longitude=lon_east, method="nearest")
            out[site.site_name] = (point.transpose("member", "time").values, float(point.latitude), float(point.longitude))
    return sim.time.values, out


def drift_check(path: str, kind: str):
    '''2 m temperature of a season run against ERA5 at the same point, by lead-time window (V11)'''
    times, sites = site_series(path, kind)
    t2m = xr.open_dataset(os.path.join(era5.HOURLY_DIR, f"era5_maine_hourly_{str(times[0])[:4]}.nc"))["t2m"]
    lead_days = (times - times[0]) / np.timedelta64(1, "D")
    logger.info("%s: %s to %s", path, times[0], times[-1])
    for site, (model_t, grid_lat, grid_lon) in sites.items():
        ref = t2m.sel(time=times).sel(latitude=grid_lat, longitude=grid_lon, method="nearest").values
        assert np.isfinite(model_t).all() and np.isfinite(ref).all(), "missing values in the comparison"
        logger.info("%s, model point %.2f N %.2f E, 2 m temperature against ERA5 at the same point", site, grid_lat, grid_lon)
        for lo, hi in WINDOWS:
            sel = (lead_days >= lo) & (lead_days < hi)
            spread = np.abs(model_t[0, sel] - model_t[1, sel]).mean() if model_t.shape[0] > 1 else np.nan
            logger.info("  lead %3d to %3d days: mean model minus ERA5 %+.2f K, RMSE %.2f K, mean absolute member difference %.2f K",
                        lo, hi, (model_t[:, sel] - ref[None, sel]).mean(), np.sqrt(((model_t[:, sel] - ref[None, sel]) ** 2).mean()), spread)
        aug, nov = pd.DatetimeIndex(times).month == 8, pd.DatetimeIndex(times).month == 11
        logger.info("  August mean %.1f C (ERA5 %.1f C), November mean %.1f C (ERA5 %.1f C)",
                    model_t[:, aug].mean() - 273.15, ref[aug].mean() - 273.15, model_t[:, nov].mean() - 273.15, ref[nov].mean() - 273.15)


def chain_check(start: str):
    '''daily values of the full forecast (AIFS ENS then ACE2) against the daily ERA5 reference at the nearest model point of each site'''
    _, segments = mapping.load_segments(start)
    ref = xr.open_dataset(era5.DAILY_PATH)
    t_issue = np.datetime64(start[:10], "D")
    for _, site in pd.read_csv(SITES_CSV).iterrows():
        lon_east = site.longitude % 360
        logger.info("%s", site.site_name)
        for name, seg in segments.items():
            j = int(mapping.nearest_point(np.array([site.latitude]), np.array([lon_east]), seg["point_lat"], seg["point_lon"] % 360)[0])
            e = ref.sel(time=seg["date"]).sel(latitude=seg["point_lat"][j], longitude=seg["point_lon"][j] % 360, method="nearest")
            lead = (seg["date"].astype("datetime64[D]") - t_issue).astype(int)
            for variable in ["tmean", "tmax", "tmin", "prcp", "ssrd", "vp"]:
                diff = seg[variable][:, :, j] - e[variable].values[None]  # shape (members, days)
                spread = seg[variable][:, :, j].std(axis=0).mean()
                logger.info("  %-4s days %3d to %3d, %-6s mean model minus ERA5 %+9.3f, RMSE %8.3f, ensemble std %8.3f",
                            name, lead[0], lead[-1], variable, diff.mean(), np.sqrt((diff ** 2).mean()), spread)


def main():
    parser = argparse.ArgumentParser(description="Validation of the weather pipeline (V11, V12, V15)")
    parser.add_argument("--check", required=True, choices=["builder", "handoff", "season", "chain"])
    parser.add_argument("--model", choices=["aifs", "ace2"], help="model of a season run")
    parser.add_argument("--start", default="2023-08-01T00", help="issue time of a season run or chain check; use training years only")
    parser.add_argument("--members", type=int, default=2, help="AIFS ENS members or ACE2 lagged initial conditions of a season run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    assert int(args.start[:4]) <= 2023, "validation uses training years only (D5)"

    if args.check == "builder":
        check_builder()
    elif args.check == "handoff":
        check_handoff_correction()
    elif args.check == "season":
        path = season_aifs(args.start, args.members) if args.model == "aifs" else season_ace2(args.start, args.members, range(2001, 2023))
        drift_check(path, args.model)
    else:
        chain_check(args.start)


if __name__ == "__main__":
    main()
