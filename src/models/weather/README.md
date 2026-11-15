# Weather model (prospective pipeline)

This folder holds the functions of the weather covariate model W (decisions D12 to D15 and D20 to D22 in `references/project_decisions.md`). Nothing here is run directly, except `validation.py`. The runnable entry point is `src/tasks/prospective_weather.py`, with settings under the `weather` key of `src/tasks/prospective_config.json`.

## What the forecast is

For each issue time t (every 8 days in 2024 and 2025, and 2026-08-01):

1. AIFS ENS 1.0 (ECMWF) runs 10 stochastic members from the ERA5 state at t and t minus 6 hours, to 06 UTC on day 15.
2. Each member's own state on day 15 is converted to an ACE2-ERA5 initial condition (8 layers, 1 degree).
3. ACE2-ERA5 (Ai2) continues each member to December 1, forced by the sea surface temperature and sea ice anomaly of t held fixed (D13).
4. Daily values (06 to 06 UTC) are mapped to the 1 km forest cells in Daymet units, by the delta method and by a per-cell linear map, both fitted on 2001 to 2023 only (D15).

## Modules

| Module | Contents |
|---|---|
| `era5.py` | All ERA5 access from the public ARCO bucket: hourly Maine reference and its daily aggregation, AIFS input state, ERA5 fields for ACE2 (model levels, pressure levels, surface, skin temperature, sea ice, top-of-atmosphere shortwave), humidity formulas |
| `ace2.py` | ACE2-ERA5 grid and conventions: conservative regridding, initial conditions from ERA5 model levels or from pressure levels (hand-off), issue-time correction, persisted-anomaly forcing, fme inference |
| `aifs.py` | AIFS ENS member loop, Maine-box output and hand-off states (AIFS environment only) |
| `mapping.py` | Daily aggregation of the two forecast segments; climatologies, linear fits and both mappings to the forest cells |
| `validation.py` | Runnable checks, training years only: `builder` (V12, V15), `handoff` (V15), `season` (V11), `chain` |

## Order of runs

All from the repository root; each SLURM script lists its options in its header.

| Step | Command |
|---|---|
| ERA5 hourly reference, 2001 to 2023 | `STAGE=era5_hourly sbatch --array=1-23 src/shell_scripts/prospective/prospective_weather_cpu.sh` |
| ERA5 daily reference | `STAGE=era5_daily sbatch --array=1 src/shell_scripts/prospective/prospective_weather_cpu.sh` |
| AIFS ENS stage | `sbatch src/shell_scripts/prospective/prospective_weather_aifs.sh` |
| ACE2 stage | `sbatch --dependency=aftercorr:<AIFS job> src/shell_scripts/prospective/prospective_weather_ace2.sh` |
| Mapping climatologies and fits, once | `STAGE=prepare_mapping START=2024-08-01T00 sbatch --array=1 --mem=96G src/shell_scripts/prospective/prospective_weather_cpu.sh` |
| Daily values and mappings | `STAGE=postprocess sbatch src/shell_scripts/prospective/prospective_weather_cpu.sh` |
| Validation | `CHECK=builder sbatch src/shell_scripts/prospective/validate_weather.sh` (see the script header) |

## Outputs

- `data/interim/aifs_inputs/`, `data/interim/ace2_inputs/`: model inputs.
- `data/interim/weather_forecast/aifs_ens/<issue time>/`, `.../ace2/<issue time>/member<m>/`: raw model output over Maine.
- `data/interim/weather_forecast/daily/<issue time>.npz`: daily values per model segment.
- `data/processed/weather_mapping/`: cached climatologies and fits.
- `data/processed/weather_forecast/aifs_ens_ace2/<delta|linear>/<year>_<issue date>.npz`: the hand-off files (O11), arrays (member, cell, day) of tmax, tmin (C), prcp (mm/day), srad (W/m2, daylight mean), vp (Pa), on the forest cells.

## Environments

- `/hpc/group/youlab/xw262/im2c-weather-ace-env`: fme (ACE2) and all ERA5 reading (zarr 3).
- `/hpc/group/youlab/xw262/im2c-weather-aifs-env`: anemoi-inference 0.6.0, anemoi-models 0.6.0, torch 2.5.0, flash-attn 2.7.4; zarr 2, so it cannot read the ARCO store.
