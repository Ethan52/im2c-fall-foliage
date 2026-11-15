#!/bin/bash
#SBATCH -A youlab
#SBATCH -J wx-valid
#SBATCH -o slurm_output/weather/validate_weather-%j.out
#SBATCH -e slurm_output/weather/validate_weather-%j.out
#SBATCH -t 4:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
# all GPUs on this node failed with "No devices were found" on 2026-11-15
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=96G

# Validation of the weather pipeline (src/models/weather/validation.py), training years only. CHECK is one of
#   builder   ERA5-to-ACE2 build and hand-off conversion against Ai2's published files (V12, V15)
#   handoff   issue-time correction of the hand-off conversion (V15)
#   season    one model alone for a season from ERA5 plus the drift check (V11); MODEL is aifs or ace2
#   chain     daily forecast of both models against the daily ERA5 reference (needs the postprocess daily file)
#
# Submit from the repository root:
#   CHECK=builder sbatch src/shell_scripts/prospective/validate_weather.sh
#   CHECK=season MODEL=ace2 sbatch src/shell_scripts/prospective/validate_weather.sh
#   CHECK=chain START=2023-08-01T00 sbatch src/shell_scripts/prospective/validate_weather.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
AIFS_ENV=${AIFS_ENV:-/hpc/group/youlab/xw262/im2c-weather-aifs-env}
ACE2_ENV=${ACE2_ENV:-/hpc/group/youlab/xw262/im2c-weather-ace-env}
START=${START:-2023-08-01T00}

cd "$REPO_ROOT" || exit 1
echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), check $CHECK, model $MODEL, issue time $START"

source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONPATH=$REPO_ROOT/src
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export HF_HOME=/hpc/group/youlab/xw262/.cache/huggingface
export TMPDIR=/tmp
export ANEMOI_INFERENCE_NUM_CHUNKS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$CHECK" == "season" && "$MODEL" == "aifs" ]]; then
    conda activate "$ACE2_ENV" || exit 1
    python src/tasks/prospective_weather.py --stage aifs_inputs --start "$START" || exit 1
    conda activate "$AIFS_ENV" || exit 1
    LOCAL_HOME=/tmp/ekhome_${SLURM_JOB_ID}
    mkdir -p "$LOCAL_HOME"
    HOME=$LOCAL_HOME XDG_CACHE_HOME=$LOCAL_HOME/.cache XDG_CONFIG_HOME=$LOCAL_HOME/.config \
        python src/models/weather/validation.py --check season --model aifs --start "$START"
    STATUS=$?
    rm -rf "$LOCAL_HOME"
else
    conda activate "$ACE2_ENV" || exit 1
    python src/models/weather/validation.py --check "$CHECK" ${MODEL:+--model $MODEL} --start "$START"
    STATUS=$?
fi

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
