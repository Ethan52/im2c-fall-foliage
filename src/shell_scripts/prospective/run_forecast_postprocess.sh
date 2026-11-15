#!/bin/bash
#SBATCH -A youlab
#SBATCH -J wx-post
#SBATCH -o slurm_output/weather/run_forecast_postprocess-%A_%a.out
#SBATCH -e slurm_output/weather/run_forecast_postprocess-%A_%a.out
#SBATCH -t 2:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --array=1-33

# Post-processing of the weather forecast for one issue time per array task: daily values from the AIFS ENS and
# ACE2 segments (src/weather/forecast_daily.py), then both mappings to Daymet-compatible values on the forest cells
# (src/weather/map_to_daymet.py, D15), using climatologies and fits cached by a prepare run. Requests no GPU.
#
# Output: data/processed/weather_forecast/aifs_ens_ace2/<delta|linear>/<year>_<issue date>.npz (hand-off O11)
#
# Submit from the repository root, after the AIFS ENS and ACE2 stages:
#   PREPARE=1 START=2024-08-01T00 sbatch --array=1 --mem=96G src/shell_scripts/prospective/run_forecast_postprocess.sh   # once
#   sbatch src/shell_scripts/prospective/run_forecast_postprocess.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-ace-env}

cd "$REPO_ROOT" || exit 1
# START can be set in the environment for a development run outside the issue list
START=${START:-$(grep -v "^#" configs/weather_issue_times.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")}
echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), array task $SLURM_ARRAY_TASK_ID, issue time $START"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export TMPDIR=/tmp

python src/weather/forecast_daily.py --start "$START" || exit 1
if [[ -n "$PREPARE" ]]; then
    python src/weather/map_to_daymet.py --start "$START" --prepare || exit 1
fi
python src/weather/map_to_daymet.py --start "$START"
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
