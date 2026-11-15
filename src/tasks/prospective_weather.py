import os
import sys
import json
import logging
import argparse

import numpy as np

from models.weather import era5

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prospective_config.json")
STAGES = ["era5_hourly", "era5_daily", "aifs_inputs", "aifs", "ace2", "prepare_mapping", "postprocess"]


def main():
    parser = argparse.ArgumentParser(description="Prospective pipeline, weather forecast stages (D20 to D22)")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--index", type=int, default=None, help="1-based index into the issue times of the config, for SLURM arrays")
    parser.add_argument("--start", default=None, help="issue time, UTC, overrides --index, for example 2024-08-01T00")
    parser.add_argument("--year", type=int, default=None, help="year of the era5_hourly stage")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    with open(CONFIG_PATH) as f:
        config = json.load(f)["weather"]
    fit_years = range(config["fit_years"][0], config["fit_years"][1] + 1)
    start = args.start if args.start else (config["issue_times"][args.index - 1] if args.index else None)
    logger.info("stage %s, issue time %s", args.stage, start)

    # ERA5 reference over the Maine box, fit years only (D15)
    if args.stage == "era5_hourly":
        assert args.year in fit_years, f"{args.year} is outside the fit years"
        era5.download_maine_hourly(args.year)
        return
    if args.stage == "era5_daily":
        era5.build_daily_reference(fit_years)
        return

    assert start is not None, "this stage needs --start or --index"
    # hand-off at 06 UTC on day 15 (issue times are 00 UTC), so every daily window, 06 UTC to 06 UTC, lies in one model
    lead_hours = 24 * config["handoff_days"] + 6
    handoff = np.datetime64(start, "h") + np.timedelta64(lead_hours, "h")

    if args.stage == "aifs_inputs":
        # read in the ACE2 environment, since the pinned AIFS stack has zarr 2 and the ARCO store is zarr 3
        era5.save_aifs_inputs(start)
    elif args.stage == "aifs":
        from models.weather import aifs  # torch, anemoi and earthkit exist only in the AIFS environment
        aifs.run_members(start, config["members"], lead_hours, os.path.join(aifs.OUT_DIR, start[:13]))
    elif args.stage == "ace2":
        from models.weather import ace2
        if handoff >= np.datetime64(f"{start[:4]}-12-01T00", "h"):
            logger.info("hand-off %s is after November 30, AIFS ENS covers the season, nothing to do", handoff)
            return
        clim = range(config["sst_climatology_years"][0], config["sst_climatology_years"][1] + 1)
        ace2.run_from_handoff(start, config["corrected_names"], clim)
    elif args.stage == "prepare_mapping":
        from models.weather import mapping
        mapping.prepare(start, fit_years)
    elif args.stage == "postprocess":
        from models.weather import mapping
        mapping.forecast_daily(start)
        mapping.map_to_daymet(start)


if __name__ == "__main__":
    main()
