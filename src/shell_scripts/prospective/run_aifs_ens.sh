#!/bin/bash
#SBATCH -A youlab
#SBATCH -J aifs-ens
#SBATCH -o slurm_output/weather/run_aifs_ens-%A_%a.out
#SBATCH -e slurm_output/weather/run_aifs_ens-%A_%a.out
#SBATCH -t 3:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
# one GPU on this node failed with "No devices were found" on 2026-11-15
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --array=1-33

# AIFS ENS stage of the weather forecast (D20): for one issue time per array task, reads the ERA5 input state,
# runs 10 stochastic members for 15 days, and writes the Maine box (6-hourly) and each member's global state at
# day 15 for the hand-off to ACE2. Issue times are the non-comment lines of configs/weather_issue_times.txt.
#
# Output: data/interim/weather_forecast/aifs_ens/<issue time>/maine.nc and handoff_member<m>.npz
#
# Submit from the repository root:
#   mkdir -p slurm_output/weather
#   sbatch src/shell_scripts/prospective/run_aifs_ens.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-aifs-env}
INPUT_ENV_PATH=${INPUT_ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-ngcm-env}

cd "$REPO_ROOT" || exit 1
# START can be set in the environment for a development run outside the issue list
START=${START:-$(grep -v "^#" configs/weather_issue_times.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")}
echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), array task $SLURM_ARRAY_TASK_ID, issue time $START"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$(conda info --base)/etc/profile.d/conda.sh"
# memory settings from the AIFS ENS model card, 16 chunks for a 24 GB GPU
export ANEMOI_INFERENCE_NUM_CHUNKS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the home directory quota is full, so caches, configs and temporary files go to group storage
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export XDG_CONFIG_HOME=/hpc/group/youlab/xw262/.config
export HF_HOME=/hpc/group/youlab/xw262/.cache/huggingface
export TMPDIR=/hpc/group/youlab/xw262/tmp

# step 1: ERA5 inputs, read in the NeuralGCM environment since the pinned AIFS stack has zarr 2 and the ARCO store is zarr 3
conda activate "$INPUT_ENV_PATH" || exit 1
python src/weather/era5_inputs_aifs.py --start "$START" || exit 1
# step 2: the model run
conda activate "$ENV_PATH" || exit 1
# a private home on local disk per task: earthkit keeps its cache index in SQLite, which fails when tasks share it over NFS
LOCAL_HOME=/tmp/ekhome_${SLURM_JOB_ID}
mkdir -p "$LOCAL_HOME"
HOME=$LOCAL_HOME XDG_CACHE_HOME=$LOCAL_HOME/.cache XDG_CONFIG_HOME=$LOCAL_HOME/.config python src/weather/run_aifs_ens.py --start "$START" --members 10 --days 15
STATUS=$?
rm -rf "$LOCAL_HOME"

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
