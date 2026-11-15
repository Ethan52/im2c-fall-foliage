#!/bin/bash
#SBATCH -A youlab
#SBATCH -J wx-ace2
#SBATCH -o slurm_output/weather/prospective_weather_ace2-%A_%a.out
#SBATCH -e slurm_output/weather/prospective_weather_ace2-%A_%a.out
#SBATCH -t 3:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
# all GPUs on this node failed with "No devices were found" on 2026-11-15
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --array=1-33

# ACE2-ERA5 stage of the weather forecast (D20), one issue time per array task, after the AIFS ENS stage:
# persisted-anomaly forcing from ERA5 at the issue time (D13), each AIFS member's hand-off state converted to an ACE2
# initial condition with the issue-time correction (V15), and ACE2 to 06 UTC on December 1. Issue times whose hand-off is after
# November 30 exit at once. Output: data/interim/weather_forecast/ace2/<issue time>/member<m>/maine.nc
#
# Submit from the repository root:
#   sbatch --dependency=aftercorr:<AIFS job id> src/shell_scripts/prospective/prospective_weather_ace2.sh
#   START=2023-08-01T00 sbatch --array=1 src/shell_scripts/prospective/prospective_weather_ace2.sh   # one issue time

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ACE2_ENV=${ACE2_ENV:-/hpc/group/youlab/xw262/im2c-weather-ace-env}

cd "$REPO_ROOT" || exit 1
ISSUE=${START:+--start $START}
ISSUE=${ISSUE:---index $SLURM_ARRAY_TASK_ID}
echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), array task $SLURM_ARRAY_TASK_ID, $ISSUE"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ACE2_ENV" || exit 1
export PYTHONPATH=$REPO_ROOT/src
# caches go to group storage since the home quota is full; temporary files on local disk avoid NFS cleanup errors
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export TMPDIR=/tmp

python src/tasks/prospective_weather.py --stage ace2 $ISSUE
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
