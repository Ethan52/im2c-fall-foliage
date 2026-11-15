#!/bin/bash
#SBATCH -A youlab
#SBATCH -J prosp-report
#SBATCH -o slurm_output/prospective/prospective_report-%j.out
#SBATCH -e slurm_output/prospective/prospective_report-%j.out
#SBATCH -t 1:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=48G

# Figures and tables of the prospective pipeline (src/tasks/prospective_report.py): forecast onset maps, data assimilation
# maps and curves, scores against issue date, site fans and the DACF category comparison of the test years.
# Run after the evaluate stage of src/shell_scripts/prospective/prospective.sh; requests no GPU.
#
# Submit from the repository root:
#   mkdir -p slurm_output/prospective
#   sbatch src/shell_scripts/prospective/prospective_report.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-env}

cd "$REPO_ROOT" || exit 1
echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1
export PYTHONPATH=$REPO_ROOT/src
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export MPLCONFIGDIR=/hpc/group/youlab/xw262/.cache/matplotlib
# the environment's own PROJ data directory is missing, so pyproj's bundled one is used
export PROJ_DATA=$(python -c "import pyproj; print(pyproj.datadir.get_data_dir())")

python src/tasks/prospective_report.py
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
