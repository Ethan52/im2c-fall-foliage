#!/bin/bash
#SBATCH -A youlab
#SBATCH -J Covariates
#SBATCH -o slurm_output/data_preprocess/preprocess_covariates-%j.out
#SBATCH -e slurm_output/data_preprocess/preprocess_covariates-%j.out
#SBATCH -t 2:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=48G

# Runs one covariate preprocessing step, named by the first argument:
#   elevation   src/data_preprocess/build_elevation.py        NASADEM mean elevation of each forest cell (D7)
#   species     src/data_preprocess/build_species.py          TreeMap basal-area shares of each forest cell (D23)
#   gridmet     src/data_preprocess/correct_gridmet.py        2026 gridMET corrected to Daymet on the forest cells (D8, O7)
#   phenology   src/data_preprocess/build_phenology_dates.py  observed onset (O2) and spring green-up (D17) of each cell and year
# Outputs go to data/processed/static/, data/processed/gridmet_daymet/ and data/processed/phenology/. Requests no GPU.
# Needs the outputs of preprocess_modis.sh, and of regrid_daymet.sh for the gridmet step.
#
# Submit from the repository root:
#   mkdir -p slurm_output/data_preprocess
#   sbatch src/shell_scripts/data_preprocess/preprocess_covariates.sh elevation
#   sbatch src/shell_scripts/data_preprocess/preprocess_covariates.sh species --editions 2023    # test on one edition
#
# Arguments after the step name are passed to its script.

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-env}

STEP=$1
shift
case "$STEP" in
    elevation) SCRIPT=src/data_preprocess/build_elevation.py ;;
    species) SCRIPT=src/data_preprocess/build_species.py ;;
    gridmet) SCRIPT=src/data_preprocess/correct_gridmet.py ;;
    phenology) SCRIPT=src/data_preprocess/build_phenology_dates.py ;;
    *) echo "unknown step: $STEP"; exit 1 ;;
esac

echo "Step $STEP started: $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Environment: $ENV_PATH"
echo "Node: $(hostname)"

cd "$REPO_ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1

echo "Extra arguments: $*"

python "$SCRIPT" "$@"
STATUS=$?

echo "Step $STEP finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
