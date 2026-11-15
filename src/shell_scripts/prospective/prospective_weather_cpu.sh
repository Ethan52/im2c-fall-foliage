#!/bin/bash
#SBATCH -A youlab
#SBATCH -J wx-cpu
#SBATCH -o slurm_output/weather/prospective_weather_cpu-%A_%a.out
#SBATCH -e slurm_output/weather/prospective_weather_cpu-%A_%a.out
#SBATCH -t 4:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --array=1-33

# CPU stages of the weather forecast, selected by the STAGE variable; requests no GPU.
#   era5_hourly      hourly ERA5 over the Maine box for one year; array task i is year 2000 + i (2001 to 2023)
#   era5_daily       daily ERA5 reference for 2001 to 2023, single task
#   prepare_mapping  climatologies and per-cell linear fits (D15) on the grids of one issue time, single task, 96 GB
#   postprocess      daily values and both mappings to the forest cells for one issue time per array task (hand-off O11)
#
# Submit from the repository root:
#   STAGE=era5_hourly sbatch --array=1-23 src/shell_scripts/prospective/prospective_weather_cpu.sh
#   STAGE=era5_daily sbatch --array=1 src/shell_scripts/prospective/prospective_weather_cpu.sh
#   STAGE=prepare_mapping START=2024-08-01T00 sbatch --array=1 --mem=96G src/shell_scripts/prospective/prospective_weather_cpu.sh
#   STAGE=postprocess sbatch src/shell_scripts/prospective/prospective_weather_cpu.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ACE2_ENV=${ACE2_ENV:-/hpc/group/youlab/xw262/im2c-weather-ace-env}

cd "$REPO_ROOT" || exit 1
ISSUE=${START:+--start $START}
ISSUE=${ISSUE:---index $SLURM_ARRAY_TASK_ID}
echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), stage $STAGE, array task $SLURM_ARRAY_TASK_ID"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ACE2_ENV" || exit 1
export PYTHONPATH=$REPO_ROOT/src
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export TMPDIR=/tmp

if [[ "$STAGE" == "era5_hourly" ]]; then
    python src/tasks/prospective_weather.py --stage era5_hourly --year $((2000 + SLURM_ARRAY_TASK_ID))
elif [[ "$STAGE" == "era5_daily" ]]; then
    python src/tasks/prospective_weather.py --stage era5_daily
else
    python src/tasks/prospective_weather.py --stage "$STAGE" $ISSUE
fi
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
