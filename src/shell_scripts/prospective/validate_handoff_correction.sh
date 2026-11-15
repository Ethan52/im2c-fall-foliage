#!/bin/bash
#SBATCH -A youlab
#SBATCH -J val-hand
#SBATCH -o slurm_output/weather/validate_handoff_correction-%j.out
#SBATCH -e slurm_output/weather/validate_handoff_correction-%j.out
#SBATCH -t 2:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=48G

# Tests the hand-off correction (D20): the pressure-level conversion error measured at an issue time, added at the hand-off 15 days later.
# Training-year case: issue 2019-12-17, hand-off 2020-01-01. Requests no GPU.
#
# Submit from the repository root:
#   sbatch src/shell_scripts/prospective/validate_handoff_correction.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-ace-env}

echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"

cd "$REPO_ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1
# the home directory quota is full, so caches and temporary files go to group storage
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export TMPDIR=/hpc/group/youlab/xw262/tmp

python src/weather/validate_handoff_correction.py
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
