#!/bin/bash
#SBATCH -A youlab
#SBATCH -J DEM-Download
#SBATCH -o slurm_output/data_download/dataload_dem-%j.out
#SBATCH -e slurm_output/data_download/dataload_dem-%j.out
#SBATCH -t 6:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=8G

# Waits for the NASADEM AppEEARS task to finish processing, then downloads every file of its bundle to
# data/raw/dem/ and checks each file against the size and sha256 in the bundle listing. Requests no GPU.
#
# Run this first, once, interactively from the repository root with the environment active:
#   python src/data_download/dataload_dem.py --step submit    # writes data/raw/dem/task.json
#
# Submit from the repository root:
#   mkdir -p slurm_output/data_download
#   sbatch src/shell_scripts/data_download/dataload_dem.sh
#
# Arguments after the script name are passed to dataload_dem.py.

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-env}

echo "Download started: $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Environment: $ENV_PATH"
echo "Node: $(hostname)"

cd "$REPO_ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1

echo "Extra arguments: $*"

python src/data_download/dataload_dem.py --step download "$@"
STATUS=$?

echo "Download finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
