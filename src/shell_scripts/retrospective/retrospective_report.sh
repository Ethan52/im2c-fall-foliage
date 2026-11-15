#!/bin/bash
#SBATCH -A youlab
#SBATCH -J retro-report
#SBATCH -o slurm_output/retrospective/retrospective_report-%j.out
#SBATCH -e slurm_output/retrospective/retrospective_report-%j.out
#SBATCH -t 1:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=48G

# Figures and tables of the retrospective pipeline (src/tasks/retrospective_report.py): R3 brownness and onset maps
# of the test years, the DACF peak-week comparison of the observed MODIS onset, and the preprocessing counts.
# Run after the fit and evaluate stages of src/shell_scripts/retrospective/retrospective.sh; requests no GPU.
#
# Submit from the repository root:
#   mkdir -p slurm_output/retrospective
#   sbatch src/shell_scripts/retrospective/retrospective_report.sh

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

# ARGS can pass options, for example ARGS="--figure panel --years 2010 2024 2025"
python src/tasks/retrospective_report.py $ARGS
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
