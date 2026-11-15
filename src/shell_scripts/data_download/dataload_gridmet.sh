#!/bin/bash
#SBATCH -A youlab
#SBATCH -J gridMET-Download
#SBATCH -o slurm_output/data_download/dataload_gridmet-%j.out
#SBATCH -e slurm_output/data_download/dataload_gridmet-%j.out
#SBATCH -t 2:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=8G

# Downloads gridMET daily (tmmx, tmmn, pr, srad, vpd; 2001 to 2026) over the Maine request region from the
# Climatology Lab THREDDS server, as one classic NetCDF-3 file per variable and year in data/raw/gridmet/.
# Each file is checked for shape, calendar, packing and value range; results go to data/raw/gridmet/download_record.tsv.
# 2026 runs to the latest day on the server, and the record keeps which 2026 days are permanent or provisional.
# Requests no GPU and no login. A rerun skips files that already pass the checks.
#
# Submit from the repository root:
#   mkdir -p slurm_output/data_download
#   sbatch src/shell_scripts/data_download/dataload_gridmet.sh
#   sbatch src/shell_scripts/data_download/dataload_gridmet.sh --variables tmmx --years 2026    # subset
#
# Arguments after the script name are passed to dataload_gridmet.py.

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

python src/data_download/dataload_gridmet.py "$@"
STATUS=$?

echo "Download finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
