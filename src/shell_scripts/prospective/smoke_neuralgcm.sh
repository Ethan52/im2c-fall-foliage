#!/bin/bash
#SBATCH -A youlab
#SBATCH -J smoke-ngcm
#SBATCH -o slurm_output/weather/smoke_neuralgcm-%j.out
#SBATCH -e slurm_output/weather/smoke_neuralgcm-%j.out
#SBATCH -t 4:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=64G

# Smoke test of a NeuralGCM checkpoint: reads the ERA5 state at the issue time from the public ARCO bucket,
# runs a few ensemble members for one season, and saves the Maine box to data/interim/weather_smoke/.
# Logs input, forcing and output variables, seconds per simulated day, NaN counts and peak GPU memory.
#
# Submit from the repository root:
#   mkdir -p slurm_output/weather
#   sbatch src/shell_scripts/prospective/smoke_neuralgcm.sh
#   sbatch src/shell_scripts/prospective/smoke_neuralgcm.sh --model v1/stochastic_1_4_deg.pkl
#
# Arguments after the script name are passed to smoke_neuralgcm.py.

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-ngcm-env}

echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "$REPO_ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1

# let JAX allocate GPU memory on demand so the logged peak is the real requirement
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# the home directory quota is full, so caches and temporary files go to group storage
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export TMPDIR=/hpc/group/youlab/xw262/tmp

echo "Extra arguments: $*"
python src/weather/smoke_neuralgcm.py "$@"
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
