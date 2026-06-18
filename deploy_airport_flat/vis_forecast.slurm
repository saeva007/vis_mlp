#!/bin/bash
#SBATCH --job-name=vis_forecast
#SBATCH --output=/data2/share/chenxi/PuTS/tianji/forecast/logs/vis_forecast_%j.out
#SBATCH --error=/data2/share/chenxi/PuTS/tianji/forecast/logs/vis_forecast_%j.err
#SBATCH --partition=dcu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6-23:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@60

set -euo pipefail

WORK_DIR="/data2/share/chenxi/PuTS/tianji/forecast"
LOG_DIR="${WORK_DIR}/logs"
MODEL_DIR="${WORK_DIR}/model"
CONFIG_FILE="${WORK_DIR}/forecast_config.json"
PYTHON_SCRIPT="${WORK_DIR}/mlp_test3.py"
STOP_FILE="${WORK_DIR}/.stop_forecast"

mkdir -p "${LOG_DIR}" "${MODEL_DIR}"
cd "${WORK_DIR}"

TIMEOUT_FLAG=0

on_term() {
    echo "[$(date)] Received SIGTERM before time limit."
    TIMEOUT_FLAG=1
}
trap on_term TERM

echo "========================================"
echo "Job started      : $(date)"
echo "Job ID           : ${SLURM_JOB_ID}"
echo "Restart count    : ${SLURM_RESTART_COUNT:-0}"
echo "Work dir         : ${WORK_DIR}"
echo "Config           : ${CONFIG_FILE}"
echo "========================================"

if [[ -f "${STOP_FILE}" ]]; then
    echo "Stop file found. Exit without requeue."
    rm -f "${STOP_FILE}"
    exit 0
fi

source /lustre/chenxi/yliang/miniforge3/bin/activate
module purge
module load compiler/dtk/23.10
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
source /lustre/chenxi/yliang/work/cast3/env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

required_files=(
    "${CONFIG_FILE}"
    "${PYTHON_SCRIPT}"
    "${WORK_DIR}/xiahang_forecast_system.py"
    "${WORK_DIR}/airport_visibility_common.py"
)

for f in "${required_files[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: missing required file: ${f}" >&2
        exit 1
    fi
done

python - "${CONFIG_FILE}" <<'PY'
import json
import os
import sys

config_file = sys.argv[1]
with open(config_file, "r", encoding="utf-8") as f:
    cfg = json.load(f)

required_config_files = [
    "model_path",
    "scaler_path",
    "dataset_metadata_path",
]

missing = []
for key in required_config_files:
    path = cfg.get(key)
    if not path:
        missing.append(f"{key}: not set")
    elif not os.path.isfile(path):
        missing.append(f"{key}: {path}")

if missing:
    print("ERROR: missing file(s) from forecast_config.json:", file=sys.stderr)
    for item in missing:
        print(f"  {item}", file=sys.stderr)
    sys.exit(1)

print("Config file path check ok")
PY

python -c "import torch, joblib, xarray; import xiahang_forecast_system; print('Python import check ok')"

echo "$(date): Python started" >> "${LOG_DIR}/job_history.log"

set +e
python -u "${PYTHON_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --process-existing \
    2>&1 | tee -a "${LOG_DIR}/forecast_runtime_${SLURM_JOB_ID}.log"
PY_EXIT=${PIPESTATUS[0]}
set -e

if [[ ${TIMEOUT_FLAG} -eq 1 ]]; then
    echo "[$(date)] Time-limit signal received. Requeue this job."
    scontrol requeue "${SLURM_JOB_ID}" || true
    exit 0
fi

if [[ ${PY_EXIT} -ne 0 ]]; then
    echo "[$(date)] Python exited with code ${PY_EXIT}. Do not requeue."
    exit "${PY_EXIT}"
fi

echo "[$(date)] Python exited normally."
exit 0
