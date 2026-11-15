#!/bin/bash
#SBATCH -A youlab
#SBATCH -J smoke-aifs
#SBATCH -o slurm_output/weather/smoke_aifs_ens-%j.out
#SBATCH -e slurm_output/weather/smoke_aifs_ens-%j.out
#SBATCH -t 6:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=96G

# Smoke test of AIFS ENS 1.0: builds the two input states (issue time and 6 hours before) from the public ARCO
# ERA5 bucket, regrids them to N320 as in ECMWF's example notebook, runs a few members for one season, and
# saves the Maine box to data/interim/weather_smoke/. The A5000 has 24 GB, so the model mapper is chunked.
#
# Submit from the repository root:
#   mkdir -p slurm_output/weather
#   sbatch src/shell_scripts/prospective/smoke_aifs_ens.sh
#
# Arguments after the script name (only --start is shared) are passed to both Python steps.

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-aifs-env}
INPUT_ENV_PATH=${INPUT_ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-ngcm-env}

echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "$REPO_ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"

# memory settings from the AIFS ENS model card, 16 chunks for a 24 GB GPU
export ANEMOI_INFERENCE_NUM_CHUNKS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the home directory quota is full, so caches, configs and temporary files go to group storage
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export XDG_CONFIG_HOME=/hpc/group/youlab/xw262/.config
export HF_HOME=/hpc/group/youlab/xw262/.cache/huggingface
export TMPDIR=/hpc/group/youlab/xw262/tmp
mkdir -p "$XDG_CONFIG_HOME"

echo "Extra arguments: $*"
# step 1: ERA5 inputs are read in the NeuralGCM environment, since the pinned AIFS stack has zarr 2 and the ARCO store is zarr 3
conda activate "$INPUT_ENV_PATH" || exit 1
python src/weather/era5_inputs_aifs.py "$@" || exit 1
# step 2: regridding and the model run in the AIFS environment
conda activate "$ENV_PATH" || exit 1
# earthkit writes its config and regridding matrices under the home directory, which is full
HOME=/hpc/group/youlab/xw262/home_redirect python src/weather/smoke_aifs_ens.py "$@"
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
