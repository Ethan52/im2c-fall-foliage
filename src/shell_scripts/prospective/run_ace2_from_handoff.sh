#!/bin/bash
#SBATCH -A youlab
#SBATCH -J ace2-cont
#SBATCH -o slurm_output/weather/run_ace2_from_handoff-%A_%a.out
#SBATCH -e slurm_output/weather/run_ace2_from_handoff-%A_%a.out
#SBATCH -t 3:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
# one GPU on this node failed with "No devices were found" on 2026-11-15
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --array=1-33

# ACE2-ERA5 stage of the weather forecast (D20): for one issue time per array task, builds the persisted-anomaly
# forcing from ERA5 at the issue time (D13), converts each AIFS ENS member's day-15 state to an ACE2 initial
# condition, and runs ACE2 to December 1. Issue times whose hand-off is after November 30 are skipped.
# Run after src/shell_scripts/prospective/run_aifs_ens.sh has finished for the same task.
#
# Output: data/interim/weather_forecast/ace2/<issue time>/member<m>/maine.nc
#
# Submit from the repository root:
#   sbatch --dependency=aftercorr:<run_aifs_ens job id> src/shell_scripts/prospective/run_ace2_from_handoff.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-ace-env}

cd "$REPO_ROOT" || exit 1
# START can be set in the environment for a development run outside the issue list
START=${START:-$(grep -v "^#" configs/weather_issue_times.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")}
echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), array task $SLURM_ARRAY_TASK_ID, issue time $START"

HANDOFF_DAY=$(python3 -c "import datetime as d; print((d.datetime.fromisoformat('${START}:00') + d.timedelta(days=15, hours=6)).date())")
if [[ "$HANDOFF_DAY" > "${START:0:4}-11-30" ]]; then
    echo "hand-off $HANDOFF_DAY is after November 30, AIFS ENS covers the season, nothing to do"
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1
# the home directory quota is full, so caches go to group storage; temporary files on local disk avoid NFS cleanup errors
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export TMPDIR=/tmp

python src/weather/era5_inputs_ace2.py --start "$START" --lags 0 --forcing persisted || exit 1
python src/weather/run_ace2_from_handoff.py --start "$START" --correction
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
