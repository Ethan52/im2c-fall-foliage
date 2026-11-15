#!/bin/bash
#SBATCH -A youlab
#SBATCH -J retrospective
#SBATCH -o slurm_output/retrospective/retrospective-%A_%a.out
#SBATCH -e slurm_output/retrospective/retrospective-%A_%a.out
#SBATCH -t 2:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --array=1

# Retrospective pipeline of the foliage model (manuscript 2.4.2 and 3.4, D25), stage chosen by the STAGE variable; requests no GPU.
#   rows      R1 training rows of one year from Daymet; array task i is the i-th year of train_years then test_years (2001 to 2025)
#   fit       R2 linear regression and GAM on the training years, single task
#   evaluate  R3 predictions and scores on the test years, single task
# Settings: the "foliage" key of src/tasks/retrospective_config.json.
#
# Submit from the repository root:
#   mkdir -p slurm_output/retrospective
#   STAGE=rows sbatch --array=1-25 src/shell_scripts/retrospective/retrospective.sh
#   STAGE=fit sbatch src/shell_scripts/retrospective/retrospective.sh
#   STAGE=evaluate sbatch src/shell_scripts/retrospective/retrospective.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-env}

cd "$REPO_ROOT" || exit 1
echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), stage $STAGE, array task $SLURM_ARRAY_TASK_ID"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1
export PYTHONPATH=$REPO_ROOT/src
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export MPLCONFIGDIR=/hpc/group/youlab/xw262/.cache/matplotlib

python src/tasks/retrospective.py --stage "$STAGE" --index "$SLURM_ARRAY_TASK_ID"
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
