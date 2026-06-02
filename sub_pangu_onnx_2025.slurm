#!/bin/bash
#SBATCH -J pangu_onnx_2025
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:1
#SBATCH --mem=180G
#SBATCH -t 72:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err

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

cd /public/home/putianshu/vis_mlp/train
mkdir -p logs

SCRIPT_PATH="${SCRIPT_PATH:-/public/home/putianshu/vis_mlp/train/run_pangu_onnx_2025.py}"
MODEL_PATH="${MODEL_PATH:-/public/home/putianshu/vis_mlp/pangu_models/pangu_weather_24.onnx}"

# Use either PAIR_FILE or sorted UPPER_DIR/SURFACE_DIR globs.
PAIR_FILE="${PAIR_FILE:-}"
UPPER_DIR="${UPPER_DIR:-/public/home/putianshu/vis_mlp/pangu_init/era5_2025/pressure}"
SURFACE_DIR="${SURFACE_DIR:-/public/home/putianshu/vis_mlp/pangu_init/era5_2025/surface}"
UPPER_GLOB="${UPPER_GLOB:-*.grib*}"
SURFACE_GLOB="${SURFACE_GLOB:-*.grib*}"

START_DATE="${START_DATE:-2025-01-01}"
END_DATE="${END_DATE:-2025-12-31}"
OUTPUT_DIR="${OUTPUT_DIR:-/public/home/putianshu/vis_mlp/pangu_2025_china_monthly}"
SAVE_STEPS="${SAVE_STEPS:-1}"
SMOKE_COUNT="${SMOKE_COUNT:-0}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
    python "${SCRIPT_PATH}"
    --model-path "${MODEL_PATH}"
    --start-date "${START_DATE}"
    --end-date "${END_DATE}"
    --output-dir "${OUTPUT_DIR}"
    --save-steps "${SAVE_STEPS}"
    --smoke-count "${SMOKE_COUNT}"
    --device "${DEVICE}"
)

if [[ -n "${PAIR_FILE}" ]]; then
    CMD+=(--pair-file "${PAIR_FILE}")
else
    CMD+=(
        --upper-dir "${UPPER_DIR}"
        --surface-dir "${SURFACE_DIR}"
        --upper-glob "${UPPER_GLOB}"
        --surface-glob "${SURFACE_GLOB}"
    )
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGV=(${EXTRA_ARGS})
    CMD+=("${EXTRA_ARGV[@]}")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Pangu ONNX 2025 inference finished at: $(date)"
