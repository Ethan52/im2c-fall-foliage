#!/bin/bash
#SBATCH -A youlab
#SBATCH -J smoke-ace2
#SBATCH -o slurm_output/weather/smoke_ace2-%j.out
#SBATCH -e slurm_output/weather/smoke_ace2-%j.out
#SBATCH -t 6:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=128G

# Smoke test of ACE2-ERA5: builds lagged initial conditions and persisted-anomaly forcing from ERA5 for one issue
# time (src/weather/era5_inputs_ace2.py), then runs Ai2's fme inference to December 1 and writes the Maine box.
#
# Submit from the repository root:
#   mkdir -p slurm_output/weather
#   sbatch src/shell_scripts/prospective/smoke_ace2.sh                  # issue time 2023-08-01T00, 2 lagged members
#   START=2023-08-01T00 LAGS=4 sbatch src/shell_scripts/prospective/smoke_ace2.sh

REPO_ROOT=${REPO_ROOT:-/hpc/group/youlab/xw262/im2c-fall-foliage}
ENV_PATH=${ENV_PATH:-/hpc/group/youlab/xw262/im2c-weather-ace-env}
START=${START:-2023-08-01T00}
LAGS=${LAGS:-2}
CKPT=/hpc/group/youlab/xw262/models/ace2_era5/ace2_era5_ckpt.tar

echo "Started: $(date --iso-8601=seconds)"
echo "Node: $(hostname), issue time $START, lags $LAGS"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "$REPO_ROOT" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || exit 1
# the home directory quota is full, so caches and temporary files go to group storage
export XDG_CACHE_HOME=/hpc/group/youlab/xw262/.cache
export TMPDIR=/hpc/group/youlab/xw262/tmp

# step 1: initial conditions and forcing from ERA5
python src/weather/era5_inputs_ace2.py --start "$START" --lags "$LAGS" --forcing persisted || exit 1

# step 2: fme inference; members start 6 hours apart and all run to December 1 of the start year or later
IN_DIR=data/interim/ace2_inputs
EXP_DIR=data/interim/weather_smoke/ace2_${START}_lags${LAGS}
mkdir -p "$EXP_DIR"
TIMES=$(python -c "
import numpy as np
t = np.datetime64('$START', 'h')
print(', '.join('\"' + str(t - np.timedelta64(6 * k, 'h')) + ':00:00\"' for k in range($LAGS - 1, -1, -1)))
")
STEPS=$(python -c "
import numpy as np
t0 = np.datetime64('$START', 'h') - np.timedelta64(6 * ($LAGS - 1), 'h')
print(int((np.datetime64('${START:0:4}-12-01T00', 'h') - t0) / np.timedelta64(6, 'h')))
")
echo "initial times $TIMES, $STEPS forward steps"
cat > "$EXP_DIR/config.yaml" << EOF
experiment_dir: $EXP_DIR
n_forward_steps: $STEPS
forward_steps_in_memory: 50
checkpoint_path: $CKPT
logging:
  log_to_screen: true
  log_to_wandb: false
  log_to_file: true
  project: ace
initial_condition:
  path: $IN_DIR/ic_${START}_lags${LAGS}.nc
  start_indices:
    times: [$TIMES]
forcing_loader:
  dataset:
    data_path: $IN_DIR/forcing_${START}_persisted
  num_data_workers: 2
data_writer:
  save_prediction_files: false
  save_monthly_files: false
  files:
    - label: maine
      names: [TMP2m, Q2m, PRESsfc, PRATEsfc, DSWRFsfc, surface_temperature]
      lat_extent: [40.0, 50.0]
      lon_extent: [285.0, 296.0]
      save_reference: false
EOF
python -m fme.ace.inference "$EXP_DIR/config.yaml"
STATUS=$?

echo "Finished: $(date --iso-8601=seconds), exit status $STATUS"
exit $STATUS
