#!/bin/bash
#SBATCH -A youlab
#SBATCH -J ERA5-Download
#SBATCH -o slurm_output/data_download/dataload_era5-%A_%a.out
#SBATCH -e slurm_output/data_download/dataload_era5-%A_%a.out
#SBATCH -t 6:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --array=1-23

# Downloads hourly ERA5 2 m temperature, 2 m dewpoint, precipitation and surface downward shortwave over the
# Maine box (40 to 50 N, 285 to 296 E) from the public ARCO ERA5 bucket, July 15 to December 16 of each year.
# One array task per year: task 1 is 2001, task 23 is 2023. Requests no GPU.
# Used for the ERA5 daily climatology (delta method) and the per-cell ERA5 to Daymet linear map (D15).
#
# Output: data/raw/era5/era5_maine_hourly_<year>.nc
#
# Submit from the repository root:
#   mkdir -p slurm_output/data_download
#   sbatch src/shell_scripts/data_download/dataload_era5.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-ngcm-env}

echo "Download started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), array task $SLURM_ARRAY_TASK_ID"

cd "$REPO_ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1
# the home directory quota is full, so caches and temporary files go to group storage
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export TMPDIR=/hpc/group/youlab/xw262/tmp

python src/data_download/dataload_era5.py "$@"
STATUS=$?

echo "Download finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
