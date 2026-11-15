#!/bin/bash
#SBATCH -A youlab
#SBATCH -J MODIS-Preprocess
#SBATCH -o slurm_output/data_preprocess/preprocess_modis-%j.out
#SBATCH -e slurm_output/data_preprocess/preprocess_modis-%j.out
#SBATCH -t 4:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G

# Runs the MODIS preprocessing of references/preprocessing_plan.md in order, stopping at the first failure:
#   step 0  src/data_preprocess/build_grid.py          1 km analysis grid over Maine
#   step 1  src/data_preprocess/build_forest_mask.py   forest fraction on the MODIS grid and the pixel to cell map (D11)
#   step 2  src/data_preprocess/preprocess_modis.py    quality masks, EVI2, 1 km cell means, one file per year
#   step 3  src/data_preprocess/build_brownness.py     U, L and observed brownness (D2)
#   step 4  src/checks/check_preprocessing.py          counts, figures, site candidates, literature check
# Outputs go to data/processed/ and figures/preprocessing/. Requests no GPU.
#
# Submit from the repository root:
#   mkdir -p slurm_output/data_preprocess
#   sbatch src/shell_scripts/data_preprocess/preprocess_modis.sh                 # all years, steps 0 to 4
#   sbatch src/shell_scripts/data_preprocess/preprocess_modis.sh --years 2023    # test: steps 0 to 2 for one year only
#
# Arguments after the script name are passed to preprocess_modis.py; when any are given, steps 3 and 4 are skipped,
# since U and L need every year from 2001 to 2023.

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-env}

echo "Preprocessing started: $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Environment: $ENV_PATH"
echo "Node: $(hostname)"

cd "$REPO_ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1

echo "Extra arguments: $*"

run_step() {
    echo "Step: $* ($(date --iso-8601=seconds))"
    python "$@"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "Preprocessing failed at $1: $(date --iso-8601=seconds), exit status $STATUS"
        exit $STATUS
    fi
}

run_step src/data_preprocess/build_grid.py
run_step src/data_preprocess/build_forest_mask.py
run_step src/data_preprocess/preprocess_modis.py "$@"
if [ $# -eq 0 ]; then
    run_step src/data_preprocess/build_brownness.py
    run_step src/checks/check_preprocessing.py
fi

echo "Preprocessing finished: $(date --iso-8601=seconds), exit status 0"
exit 0
