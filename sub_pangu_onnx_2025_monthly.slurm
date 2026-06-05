#!/bin/bash
#SBATCH -J pangu_2025_monthly
#SBATCH --partition=dcu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=dcu:2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH --array=0-5
#SBATCH -D /data2/share/chenxi/PuTS/mlp
#SBATCH -o /data2/share/chenxi/PuTS/mlp/logs/%A_%a_%x.out
#SBATCH -e /data2/share/chenxi/PuTS/mlp/logs/%A_%a_%x.err

set -euo pipefail

echo "Pangu monthly array started at: $(date)"
echo "Node: $(hostname)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"

source /lustre/chenxi/yliang/miniforge3/bin/activate
module purge
module load compiler/dtk/23.10
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
source /lustre/chenxi/yliang/work/cast3/env.sh

WORK_DIR="${WORK_DIR:-/data2/share/chenxi/PuTS/mlp}"
cd "${WORK_DIR}"
mkdir -p logs
PYTHON_PACKAGES="${PYTHON_PACKAGES:-${WORK_DIR}/python_packages}"
if [[ -d "${PYTHON_PACKAGES}" ]]; then
    export PYTHONPATH="${PYTHON_PACKAGES}:${PYTHONPATH:-}"
fi

SCRIPT_PATH="${SCRIPT_PATH:-${WORK_DIR}/run_pangu_onnx_2025.py}"
MODEL_PATH="${MODEL_PATH:-${WORK_DIR}/pangu_models/pangu_weather_24.onnx}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/pangu_2025_china_monthly}"
ERA5_ROOT="${ERA5_ROOT:-/sharedata/dataset/realtime/SD005-ERA5/0p25}"
INIT_HOURS="${INIT_HOURS:-0:12}"
SAVE_STEPS="${SAVE_STEPS:-1}"
DEVICE="${DEVICE:-auto}"
SMOKE_COUNT="${SMOKE_COUNT:-0}"
SKIP_MISSING_INIT="${SKIP_MISSING_INIT:-1}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-2}"

worker_threads="$(( ${SLURM_CPUS_PER_TASK:-64} / WORKERS_PER_NODE ))"
if [[ "${worker_threads}" -lt 1 ]]; then
    worker_threads=1
fi

MONTH_STARTS=(
    2025-01-01 2025-02-01 2025-03-01 2025-04-01
    2025-05-01 2025-06-01 2025-07-01 2025-08-01
    2025-09-01 2025-10-01 2025-11-01 2025-12-01
)
MONTH_ENDS=(
    2025-01-31 2025-02-28 2025-03-31 2025-04-30
    2025-05-31 2025-06-30 2025-07-31 2025-08-31
    2025-09-30 2025-10-31 2025-11-30 2025-12-31
)

group_id="${SLURM_ARRAY_TASK_ID:-0}"
pids=()

for ((local_rank = 0; local_rank < WORKERS_PER_NODE; local_rank++)); do
    month_idx=$(( group_id * WORKERS_PER_NODE + local_rank ))
    if [[ "${month_idx}" -ge 12 ]]; then
        continue
    fi

    start_date="${MONTH_STARTS[${month_idx}]}"
    end_date="${MONTH_ENDS[${month_idx}]}"
    month_num="$(printf "%02d" "$(( month_idx + 1 ))")"
    worker_log="${WORK_DIR}/logs/pangu_2025_${month_num}_${SLURM_JOB_ID}_${group_id}_${local_rank}.log"

    (
        export PYTHONUNBUFFERED=1
        export HIP_VISIBLE_DEVICES="${local_rank}"
        export CUDA_VISIBLE_DEVICES="${local_rank}"
        export ROCR_VISIBLE_DEVICES="${local_rank}"
        export OMP_NUM_THREADS="${worker_threads}"
        export OPENBLAS_NUM_THREADS="${worker_threads}"
        export MKL_NUM_THREADS="${worker_threads}"
        export NUMEXPR_NUM_THREADS="${worker_threads}"

        CMD=(
            python "${SCRIPT_PATH}"
            --model-path "${MODEL_PATH}"
            --input-mode era5-dirs
            --era5-root "${ERA5_ROOT}"
            --init-hours "${INIT_HOURS}"
            --start-date "${start_date}"
            --end-date "${end_date}"
            --output-dir "${OUTPUT_DIR}"
            --save-steps "${SAVE_STEPS}"
            --smoke-count "${SMOKE_COUNT}"
            --device "${DEVICE}"
            --manifest-name "pangu_onnx_manifest_${month_num}.json"
        )

        if [[ "${SKIP_MISSING_INIT}" == "1" || "${SKIP_MISSING_INIT}" == "true" ]]; then
            CMD+=(--skip-missing-init)
        fi
        if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
            CMD+=(--overwrite)
        fi
        if [[ -n "${EXTRA_ARGS}" ]]; then
            # shellcheck disable=SC2206
            EXTRA_ARGV=(${EXTRA_ARGS})
            CMD+=("${EXTRA_ARGV[@]}")
        fi

        echo "Month ${month_num}: ${CMD[*]}"
        "${CMD[@]}"
    ) > "${worker_log}" 2>&1 &
    pids+=("$!")
    echo "Launched month ${month_num} on local DCU ${local_rank}; log=${worker_log}"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [[ "${status}" -ne 0 ]]; then
    echo "At least one monthly worker failed; inspect logs/pangu_2025_*.log"
    exit "${status}"
fi

echo "Pangu monthly array finished at: $(date)"
