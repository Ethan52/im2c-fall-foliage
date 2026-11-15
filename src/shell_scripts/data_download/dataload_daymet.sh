#!/bin/bash
#SBATCH -A youlab
#SBATCH -J Daymet-Download
#SBATCH -o slurm_output/data_download/dataload_daymet-%j.out
#SBATCH -e slurm_output/data_download/dataload_daymet-%j.out
#SBATCH -t 12:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=8G

# Downloads Daymet V4R1 daily (tmax, tmin, prcp, srad, vp, dayl; 2001 to 2025) over the Maine request region
# from NASA Earthdata OPeNDAP, as one classic NetCDF-3 file per variable and year in data/raw/daymet/.
# Each file is checked for shape, calendar, fill pattern and value range; results go to data/raw/daymet/download_record.tsv.
# Requests no GPU. Earthdata credentials are read from ~/.netrc (machine urs.earthdata.nasa.gov).
# A rerun skips files that already pass the checks.
#
# Submit from the repository root:
#   mkdir -p slurm_output/data_download
#   sbatch src/shell_scripts/data_download/dataload_daymet.sh
#   sbatch src/shell_scripts/data_download/dataload_daymet.sh --variables tmax --years 2024 2025    # subset
#
# Arguments after the script name are passed to dataload_daymet.py.

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

python src/data_download/dataload_daymet.py "$@"
STATUS=$?

echo "Download finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
