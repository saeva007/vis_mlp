#!/bin/bash
#SBATCH -J pangu_onnx_2025
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:1
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH -D /data2/share/chenxi/PuTS/mlp
#SBATCH -o /data2/share/chenxi/PuTS/mlp/logs/%j_%x.out
#SBATCH -e /data2/share/chenxi/PuTS/mlp/logs/%j_%x.err

set -euo pipefail

echo "Pangu ONNX 2025 inference started at: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"

module purge
module load compiler/devtoolset/7.3.1
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

WORK_DIR="${WORK_DIR:-/data2/share/chenxi/PuTS/mlp}"
cd "${WORK_DIR}"
mkdir -p logs

SCRIPT_PATH="${SCRIPT_PATH:-${WORK_DIR}/run_pangu_onnx_2025.py}"
MODEL_PATH="${MODEL_PATH:-${WORK_DIR}/pangu_models/pangu_weather_24.onnx}"

# INPUT_MODE=era5-dirs reads /sharedata/dataset/.../{single,pressure}-hourly variable folders.
# INPUT_MODE=pair keeps the old path for pre-merged pressure/surface GRIB pairs.
INPUT_MODE="${INPUT_MODE:-era5-dirs}"
ERA5_ROOT="${ERA5_ROOT:-/sharedata/dataset/realtime/SD005-ERA5/0p25}"
INIT_HOURS="${INIT_HOURS:-0,12}"
SKIP_MISSING_INIT="${SKIP_MISSING_INIT:-0}"
PAIR_FILE="${PAIR_FILE:-}"
UPPER_DIR="${UPPER_DIR:-${WORK_DIR}/pangu_init/era5_2025/pressure}"
SURFACE_DIR="${SURFACE_DIR:-${WORK_DIR}/pangu_init/era5_2025/surface}"
UPPER_GLOB="${UPPER_GLOB:-*.grib*}"
SURFACE_GLOB="${SURFACE_GLOB:-*.grib*}"

START_DATE="${START_DATE:-2025-01-01}"
END_DATE="${END_DATE:-2025-12-31}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/pangu_2025_china_monthly}"
SAVE_STEPS="${SAVE_STEPS:-1}"
SMOKE_COUNT="${SMOKE_COUNT:-0}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
    python "${SCRIPT_PATH}"
    --model-path "${MODEL_PATH}"
    --input-mode "${INPUT_MODE}"
    --start-date "${START_DATE}"
    --end-date "${END_DATE}"
    --output-dir "${OUTPUT_DIR}"
    --save-steps "${SAVE_STEPS}"
    --smoke-count "${SMOKE_COUNT}"
    --device "${DEVICE}"
)

if [[ "${INPUT_MODE}" == "era5-dirs" ]]; then
    CMD+=(
        --era5-root "${ERA5_ROOT}"
        --init-hours "${INIT_HOURS}"
    )
elif [[ -n "${PAIR_FILE}" ]]; then
    CMD+=(--pair-file "${PAIR_FILE}")
else
    CMD+=(
        --upper-dir "${UPPER_DIR}"
        --surface-dir "${SURFACE_DIR}"
        --upper-glob "${UPPER_GLOB}"
        --surface-glob "${SURFACE_GLOB}"
    )
fi
if [[ "${SKIP_MISSING_INIT}" == "1" || "${SKIP_MISSING_INIT}" == "true" ]]; then
    CMD+=(--skip-missing-init)
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Pangu ONNX 2025 inference finished at: $(date)"
