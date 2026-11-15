#!/bin/bash
#SBATCH -A youlab
#SBATCH -J Foliage-Reports
#SBATCH -o slurm_output/data_download/dataload_foliage_reports-%j.out
#SBATCH -e slurm_output/data_download/dataload_foliage_reports-%j.out
#SBATCH -t 0:30:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=2G

# Saves the Maine DACF fall foliage report archive page and lists the 2024 to 2026 weekly report URLs in
# data/raw/maine_foliage_reports/report_urls.tsv. The weekly reports themselves refuse scripted requests and
# are saved by hand (see MANUAL_STEPS.md). Requests no GPU and no login. Takes seconds; can also be run directly.
#
# Submit from the repository root:
#   mkdir -p slurm_output/data_download
#   sbatch src/shell_scripts/data_download/dataload_foliage_reports.sh
#
# Arguments after the script name are passed to dataload_foliage_reports.py.

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

python src/data_download/dataload_foliage_reports.py "$@"
STATUS=$?

echo "Download finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
