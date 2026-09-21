#!/bin/bash
#SBATCH -J vcb_bench
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=100G
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/benchmark_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/benchmark_%j.err
set -u
ROOT=/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913
source ${ROOT}/scripts/runtime.sh
SOURCE_DATA=/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2
LOCAL_X=/tmp/viscast_benchmark_${SLURM_JOB_ID}
test "${LOCAL_X}" = "/tmp/viscast_benchmark_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_X}"
trap 'test "${LOCAL_X}" = "/tmp/viscast_benchmark_${SLURM_JOB_ID}" && rm -rf -- "${LOCAL_X}"' EXIT
cp "${SOURCE_DATA}/X_train.npy" "${LOCAL_X}/X_train.npy"
export VISCAST_X_DIR="${LOCAL_X}"
mkdir -p "${ROOT}/benchmark/${SLURM_JOB_ID}"

run_one () {
  local devices=$1
  local workers=$2
  local batch=$3
  local nproc=$4
  local tag="dcu${nproc}_w${workers}_b${batch}"
  echo "BENCHMARK_START ${tag}"
  HIP_VISIBLE_DEVICES=${devices} python -m torch.distributed.run --standalone --nproc_per_node=${nproc} \
    "${ROOT}/scripts/viscast_continuous_b.py" benchmark --route R4b --seed 1 \
    --batch-size ${batch} --num-workers ${workers} --benchmark-tag ${tag} \
    --output-dir "${ROOT}/benchmark/${SLURM_JOB_ID}" || echo "BENCHMARK_FAILED ${tag}"
}

run_one 0 12 512 1
run_one 0 16 512 1
run_one 0 16 1024 1
run_one 0 16 2048 1
run_one 0,1 16 512 2
run_one 0,1,2,3 16 512 4
python "${ROOT}/scripts/select_resources.py" "${ROOT}/benchmark/${SLURM_JOB_ID}"
