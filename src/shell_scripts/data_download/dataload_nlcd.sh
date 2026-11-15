#!/bin/bash
#SBATCH -A youlab
#SBATCH -J NLCD-Download
#SBATCH -o slurm_output/data_download/dataload_nlcd-%j.out
#SBATCH -e slurm_output/data_download/dataload_nlcd-%j.out
#SBATCH -t 2:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=16G

# Downloads NLCD Land Cover L48 (2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021) over the Maine request
# region from the MRLC GeoServer WCS, in 100 km tiles merged into one GeoTIFF per year in data/raw/nlcd/
# (EPSG:5070, 30 m, uint8, nodata 0). Each year is checked for CRS, grid alignment and valid classes;
# results go to data/raw/nlcd/download_record.tsv. Requests no GPU and no login. A rerun downloads again.
#
# Submit from the repository root:
#   mkdir -p slurm_output/data_download
#   sbatch src/shell_scripts/data_download/dataload_nlcd.sh
#   sbatch src/shell_scripts/data_download/dataload_nlcd.sh --years 2021    # subset
#
# Arguments after the script name are passed to dataload_nlcd.py.

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

python src/data_download/dataload_nlcd.py "$@"
STATUS=$?

echo "Download finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
