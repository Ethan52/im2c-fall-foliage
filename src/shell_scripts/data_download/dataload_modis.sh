#!/bin/bash
#SBATCH -A youlab
#SBATCH -J MODIS-Download
#SBATCH -o slurm_output/data_download/dataload_modis-%j.out
#SBATCH -e slurm_output/data_download/dataload_modis-%j.out
#SBATCH -t 3-00:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=8G

# Waits for the AppEEARS task of the MOD09A1.061 request to finish processing, then downloads every file of
# its bundle to data/raw/modis/ and checks each file against the size and sha256 in the bundle listing.
# Requests no GPU; the download only needs network access. Files download in 16 parallel threads by default
# (--max_workers), and the 4 CPUs cover the sha256 computations.
#
# Run these first, once, interactively from the repository root with the environment active:
#   python src/data_download/dataload_modis.py --step region    # writes configs/modis_request_region.geojson
#   python src/data_download/dataload_modis.py --step submit    # writes data/raw/modis/task.json
#
# Earthdata credentials are read from ~/.netrc (machine urs.earthdata.nasa.gov).
#
# Outputs in data/raw/modis/:
#   bundle.json             AppEEARS listing of the bundle files with sizes and sha256
#   download_record.tsv     status, number of files, bytes, failures, job ID, date
#
# A rerun skips files already downloaded and verified, so it resumes an interrupted download.
#
# Submit from the repository root (the output directory must exist before sbatch opens the log file):
#   mkdir -p slurm_output/data_download
#   sbatch src/shell_scripts/data_download/dataload_modis.sh
#   sbatch src/shell_scripts/data_download/dataload_modis.sh --no_wait    # skip waiting for the task status done
#
# Arguments after the script name are passed to dataload_modis.py.

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

python src/data_download/dataload_modis.py --step download "$@"
STATUS=$?

echo "Download finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
