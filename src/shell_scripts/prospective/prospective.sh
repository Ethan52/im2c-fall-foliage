#!/bin/bash
#SBATCH -A youlab
#SBATCH -J prospective
#SBATCH -o slurm_output/prospective/prospective-%A_%a.out
#SBATCH -e slurm_output/prospective/prospective-%A_%a.out
#SBATCH -t 3:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --array=1

# Prospective pipeline of the foliage model (manuscript 2.4.3 and 3.5, D6, D21, D26), stage chosen by the STAGE variable; requests no GPU.
#   assimilate  Kalman filter states at every issue date of 2024 and 2025, single task
#   forecast    forecasts of one issue date for both mappings and all members; array task i is the i-th issue date (1 to 32)
#   evaluate    scores and figures, single task
# Needs the retrospective fit (src/shell_scripts/retrospective/retrospective.sh). Settings: the "foliage" key of src/tasks/prospective_config.json.
#
# Submit from the repository root:
#   mkdir -p slurm_output/prospective
#   STAGE=assimilate sbatch src/shell_scripts/prospective/prospective.sh
#   STAGE=forecast sbatch --array=1-32 src/shell_scripts/prospective/prospective.sh
#   STAGE=evaluate sbatch src/shell_scripts/prospective/prospective.sh
# 2026 forecast (D7), observed weather before August 1 from corrected gridMET: ISSUE_DATE=2026-08-01 with STAGE=forecast, then STAGE=evaluate.
# Power prior run (D27), outputs suffixed _prior1yr: add PRIOR_YEARS=1 to each of the three submissions above.

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

python src/tasks/prospective.py --stage "$STAGE" --index "$SLURM_ARRAY_TASK_ID" ${PRIOR_YEARS:+--prior-years $PRIOR_YEARS} ${ISSUE_DATE:+--issue-date $ISSUE_DATE}
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
