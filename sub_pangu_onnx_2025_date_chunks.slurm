#!/bin/bash
#SBATCH -J pangu_2025_chunks
#SBATCH --partition=dcu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=dcu:2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH --array=0-23
#SBATCH -D /data2/share/chenxi/PuTS/mlp
#SBATCH -o /data2/share/chenxi/PuTS/mlp/logs/%A_%a_%x.out
#SBATCH -e /data2/share/chenxi/PuTS/mlp/logs/%A_%a_%x.err

set -euo pipefail

echo "Pangu date-chunk array started at: $(date)"
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
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/pangu_2025_china_chunks}"
ERA5_ROOT="${ERA5_ROOT:-/sharedata/dataset/realtime/SD005-ERA5/0p25}"
YEAR="${YEAR:-2025}"
TOTAL_CHUNKS="${TOTAL_CHUNKS:-48}"
INIT_HOURS="${INIT_HOURS:-0:12}"
SAVE_STEPS="${SAVE_STEPS:-1}"
DEVICE="${DEVICE:-auto}"
SKIP_MISSING_INIT="${SKIP_MISSING_INIT:-1}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-2}"

worker_threads="$(( ${SLURM_CPUS_PER_TASK:-64} / WORKERS_PER_NODE ))"
if [[ "${worker_threads}" -lt 1 ]]; then
    worker_threads=1
fi

echo "[preflight] python=$(command -v python)"
echo "[preflight] script=${SCRIPT_PATH}"
echo "[preflight] model=${MODEL_PATH}"
echo "[preflight] era5_root=${ERA5_ROOT}"
[[ -f "${SCRIPT_PATH}" ]] || { echo "[FATAL] missing script: ${SCRIPT_PATH}"; exit 2; }
[[ -f "${MODEL_PATH}" ]] || { echo "[FATAL] missing model: ${MODEL_PATH}"; exit 2; }
[[ -d "${ERA5_ROOT}" ]] || { echo "[FATAL] missing ERA5 root: ${ERA5_ROOT}"; exit 2; }
python - <<'PY'
import sys
import importlib.util
import torch
import xarray
import onnx2torch
import cfgrib

print(f"[preflight] python={sys.executable}")
print(f"[preflight] torch={torch.__version__}")
print(f"[preflight] torch.cuda.is_available={torch.cuda.is_available()}")
print(f"[preflight] torch.cuda.device_count={torch.cuda.device_count()}")
print(f"[preflight] xarray={xarray.__version__}")
print(f"[preflight] onnx2torch={getattr(onnx2torch, '__version__', 'unknown')}")
print(f"[preflight] onnx2torch_file={onnx2torch.__file__}")
print(f"[preflight] cfgrib={getattr(cfgrib, '__version__', 'unknown')}")
netcdf_backends = [
    name for name in ("h5netcdf", "netCDF4", "scipy")
    if importlib.util.find_spec(name) is not None
]
print(f"[preflight] netcdf_backends={netcdf_backends}")
if not netcdf_backends:
    raise RuntimeError("No NetCDF writer found; need h5netcdf, netCDF4, or scipy.")
if not torch.cuda.is_available():
    raise RuntimeError("DCU backend is not visible through torch.cuda.")
if torch.cuda.device_count() < 2:
    raise RuntimeError(f"Expected 2 visible DCUs, got {torch.cuda.device_count()}.")
PY

group_id="${SLURM_ARRAY_TASK_ID:-0}"
pids=()
worker_logs=()
worker_tags=()

for ((local_rank = 0; local_rank < WORKERS_PER_NODE; local_rank++)); do
    chunk_id=$(( group_id * WORKERS_PER_NODE + local_rank ))
    if [[ "${chunk_id}" -ge "${TOTAL_CHUNKS}" ]]; then
        continue
    fi

    read -r start_date end_date < <(
        python -c 'from datetime import date,timedelta; import sys; year=int(sys.argv[1]); total=int(sys.argv[2]); idx=int(sys.argv[3]); s0=date(year,1,1); n=(date(year+1,1,1)-s0).days; a=idx*n//total; b=((idx+1)*n//total)-1; print((s0+timedelta(days=a)).isoformat(), (s0+timedelta(days=b)).isoformat())' \
            "${YEAR}" "${TOTAL_CHUNKS}" "${chunk_id}"
    )
    chunk_tag="$(printf "chunk%03d" "${chunk_id}")"
    worker_log="${WORK_DIR}/logs/pangu_2025_${chunk_tag}_${SLURM_JOB_ID}_${group_id}_${local_rank}.log"

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
            --device "${DEVICE}"
            --output-label-suffix "_${chunk_tag}"
            --manifest-name "pangu_onnx_manifest_${chunk_tag}.json"
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

        echo "${chunk_tag}: ${start_date} to ${end_date}"
        echo "${chunk_tag}: ${CMD[*]}"
        "${CMD[@]}"
    ) > "${worker_log}" 2>&1 &
    pids+=("$!")
    worker_logs+=("${worker_log}")
    worker_tags+=("${chunk_tag}")
    echo "Launched ${chunk_tag} (${start_date}..${end_date}) on local DCU ${local_rank}; log=${worker_log}"
done

status=0
for idx in "${!pids[@]}"; do
    pid="${pids[${idx}]}"
    worker_log="${worker_logs[${idx}]}"
    worker_tag="${worker_tags[${idx}]}"
    if wait "${pid}"; then
        echo "[OK] ${worker_tag} completed."
    else
        rc=$?
        status=1
        echo "[FAILED] ${worker_tag} exit_code=${rc}; log=${worker_log}"
        echo "----- tail -n 120 ${worker_log} -----"
        tail -n 120 "${worker_log}" || true
        echo "----- end ${worker_tag} log tail -----"
    fi
done

if [[ "${status}" -ne 0 ]]; then
    echo "At least one date-chunk worker failed; inspect logs/pangu_2025_chunk*.log"
    exit "${status}"
fi

echo "Pangu date-chunk array finished at: $(date)"
