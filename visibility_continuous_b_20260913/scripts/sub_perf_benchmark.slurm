#!/bin/bash
#SBATCH -J r4b_perf
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=100G
#SBATCH -t 03:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/r4b_perf_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/r4b_perf_%j.err
set -euo pipefail
ROOT=/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913
source "${ROOT}/scripts/runtime.sh"
OUT="${ROOT}/benchmark/perf_${SLURM_JOB_ID}"
mkdir -p "${OUT}"
export MIOPEN_USER_DB_PATH="/tmp/miopen_r4b_perf_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
LOCAL_X="/tmp/r4b_perf_subset_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_X}"
trap 'test "${LOCAL_X}" = "/tmp/r4b_perf_subset_${SLURM_JOB_ID}" && rm -rf -- "${LOCAL_X}"' EXIT
python -c "import numpy as np; x=np.load('/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2/X_train.npy', mmap_mode='r'); np.save('${LOCAL_X}/X_train.npy', np.asarray(x[:500000]))"
export VISCAST_X_DIR="${LOCAL_X}"
export VISCAST_BENCHMARK_POOL=500000

run_one () {
  local devices=$1
  local nproc=$2
  local batch=$3
  local amp=$4
  local tag="${amp}_dcu${nproc}_b${batch}"
  local amp_arg="${amp}"
  if [ "${amp}" = "fp32" ]; then amp_arg=none; fi
  HIP_VISIBLE_DEVICES="${devices}" python -m torch.distributed.run --standalone --nproc_per_node="${nproc}" \
    "${ROOT}/scripts/viscast_continuous_b_perf.py" benchmark --route R4b --seed 1 \
    --batch-size "${batch}" --num-workers 16 --amp "${amp_arg}" \
    --benchmark-tag "${tag}" --output-dir "${OUT}"
}

# Pure Stage-B training only; validation is never called by benchmark mode.
run_one 0 1 2048 fp32
run_one 0,1 2 1024 fp32
run_one 0,1,2,3 4 512 fp32
NPROC=$(python "${ROOT}/scripts/select_perf_benchmark.py" "${OUT}" dcu)
case "${NPROC}" in
  1) DEVICES=0 ;;
  2) DEVICES=0,1 ;;
  4) DEVICES=0,1,2,3 ;;
  *) exit 2 ;;
esac
CURRENT_BATCH=$((2048 / NPROC))
DOUBLE_BATCH=$((4096 / NPROC))
run_one "${DEVICES}" "${NPROC}" "${DOUBLE_BATCH}" fp32
BATCH=$(python "${ROOT}/scripts/select_perf_benchmark.py" "${OUT}" batch --dcu "${NPROC}")
run_one "${DEVICES}" "${NPROC}" "${BATCH}" fp16
python "${ROOT}/scripts/select_perf_benchmark.py" "${OUT}" final --dcu "${NPROC}" --batch "${BATCH}"
