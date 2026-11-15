#!/bin/bash
#SBATCH -A youlab
#SBATCH -J wx-aifs
#SBATCH -o slurm_output/weather/prospective_weather_aifs-%A_%a.out
#SBATCH -e slurm_output/weather/prospective_weather_aifs-%A_%a.out
#SBATCH -t 3:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
# all GPUs on this node failed with "No devices were found" on 2026-11-15
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --array=1-33

# AIFS ENS stage of the weather forecast (D20), one issue time per array task (index into the issue times of
# src/tasks/prospective_config.json): ERA5 input state, then 10 stochastic members to 06 UTC on day 15, writing
# the Maine box and each member's hand-off state to data/interim/weather_forecast/aifs_ens/<issue time>/.
#
# Submit from the repository root:
#   mkdir -p slurm_output/weather
#   sbatch src/shell_scripts/prospective/prospective_weather_aifs.sh
#   START=2023-08-01T00 sbatch --array=1 src/shell_scripts/prospective/prospective_weather_aifs.sh   # one issue time

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
AIFS_ENV=${AIFS_ENV:-/hpc/group/youlab/xw262/im2c-weather-aifs-env}
ACE2_ENV=${ACE2_ENV:-/hpc/group/youlab/xw262/im2c-weather-ace-env}

cd "$REPO_ROOT" || exit 1
ISSUE=${START:+--start $START}
ISSUE=${ISSUE:---index $SLURM_ARRAY_TASK_ID}
echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), array task $SLURM_ARRAY_TASK_ID, $ISSUE"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONPATH=$REPO_ROOT/src
# memory settings from the AIFS ENS model card, 16 chunks for a 24 GB GPU
export ANEMOI_INFERENCE_NUM_CHUNKS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the home directory quota is full, so caches and temporary files go to group storage
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export HF_HOME=/hpc/group/youlab/xw262/.cache/huggingface
export TMPDIR=/hpc/group/youlab/xw262/tmp

# step 1: ERA5 inputs, read in the ACE2 environment since the pinned AIFS stack has zarr 2 and the ARCO store is zarr 3
conda activate "$ACE2_ENV" || exit 1
python src/tasks/prospective_weather.py --stage aifs_inputs $ISSUE || exit 1
# step 2: the model run, with a private home on local disk: earthkit keeps its cache index in SQLite, which fails over NFS
conda activate "$AIFS_ENV" || exit 1
LOCAL_HOME=/tmp/ekhome_${SLURM_JOB_ID}
mkdir -p "$LOCAL_HOME"
HOME=$LOCAL_HOME XDG_CACHE_HOME=$LOCAL_HOME/.cache XDG_CONFIG_HOME=$LOCAL_HOME/.config python src/tasks/prospective_weather.py --stage aifs $ISSUE
STATUS=$?
rm -rf "$LOCAL_HOME"

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
