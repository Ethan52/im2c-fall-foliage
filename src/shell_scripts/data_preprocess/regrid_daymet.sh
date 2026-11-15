#!/bin/bash
#SBATCH -A youlab
#SBATCH -J Daymet-Regrid
#SBATCH -o slurm_output/data_preprocess/regrid_daymet-%j.out
#SBATCH -e slurm_output/data_preprocess/regrid_daymet-%j.out
#SBATCH -t 2:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=16G

# Puts daily Daymet (tmax, tmin, prcp, srad, vp, dayl; 2001 to 2025) on the 1 km forest cells by nearest Daymet cell (D19),
# one file per year in data/processed/daymet/daymet_<year>.npz, arrays of shape (cell, day) in Daymet units and calendar.
# Needs data/processed/grid/ and data/processed/forest_mask/ from preprocess_modis.sh. Requests no GPU.
#
# Submit from the repository root:
#   mkdir -p slurm_output/data_preprocess
#   sbatch src/shell_scripts/data_preprocess/regrid_daymet.sh
#   sbatch src/shell_scripts/data_preprocess/regrid_daymet.sh --years 2025    # one year
#
# Arguments after the script name are passed to regrid_daymet.py.

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-env}

echo "Regridding started: $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Environment: $ENV_PATH"
echo "Node: $(hostname)"

cd "$REPO_ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1

echo "Extra arguments: $*"

python src/data_preprocess/regrid_daymet.py "$@"
STATUS=$?

echo "Regridding finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
