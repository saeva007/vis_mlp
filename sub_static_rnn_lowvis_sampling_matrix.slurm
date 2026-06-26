#!/bin/bash
#SBATCH -J static_rnn_sampling
#SBATCH -p kshdexclu01
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 240:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
DCUS_PER_NODE="${DCUS_PER_NODE:-4}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

base_model_args() {
    echo "--encoder gru --pooling mean --rnn-layers 1 --hidden-dim 256 --static-hidden-dim 96 --fusion-hidden-dim 256 --dropout 0.2 --selection-metric recall_csi"
}

get_experiment() {
    case "$1" in
        0)
            EXP_NAME="natural_shuffle"
            MODEL_ARGS="$(base_model_args) --sampler-mode natural_shuffle"
            ;;
        1)
            EXP_NAME="current_stratified"
            MODEL_ARGS="$(base_model_args) --sampler-mode stratified_balanced --fog-ratio-s1 0.20 --mist-ratio-s1 0.20 --fog-ratio-s2 0.18 --mist-ratio-s2 0.22"
            ;;
        2)
            EXP_NAME="light_lowvis_oversample"
            MODEL_ARGS="$(base_model_args) --sampler-mode stratified_balanced --fog-ratio-s1 0.10 --mist-ratio-s1 0.15 --fog-ratio-s2 0.10 --mist-ratio-s2 0.15"
            ;;
        3)
            EXP_NAME="heavy_lowvis_oversample"
            MODEL_ARGS="$(base_model_args) --sampler-mode stratified_balanced --fog-ratio-s1 0.25 --mist-ratio-s1 0.25 --fog-ratio-s2 0.25 --mist-ratio-s2 0.25"
            ;;
        4)
            EXP_NAME="mild_lowvis_oversample"
            MODEL_ARGS="$(base_model_args) --sampler-mode stratified_balanced --fog-ratio-s1 0.15 --mist-ratio-s1 0.15 --fog-ratio-s2 0.14 --mist-ratio-s2 0.16"
            ;;
        *)
            echo "Unknown sampling experiment id: $1" >&2
            exit 2
            ;;
    esac
}

echo "Static-RNN sampling matrix started at: $(date)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "Nodes=${SLURM_NNODES} CPUs/task=${CPUS_PER_TASK} DCUs/node=${DCUS_PER_NODE}"

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:$LD_LIBRARY_PATH
source /public/home/jarvis226/miniconda3/bin/activate
conda activate torch
export LD_LIBRARY_PATH=/public/home/xichen/.conda/envs/py310_ppy/openssl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release/:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c '
  for c in /sbin/ip /usr/sbin/ip; do
    if [ -x "$c" ]; then
      v=$("$c" -4 -o addr show ib0 2>/dev/null | awk "NR==1{print \$4}" | cut -d/ -f1)
      if [ -n "$v" ]; then
        echo "$v"
        exit 0
      fi
    fi
  done
  hostname -I 2>/dev/null | awk "{print \$1}"
' 2>/dev/null | head -1 | tr -d "\r") || true

if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=${head_node}
fi

BASE_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

export MASTER_ADDR
export LOWVIS_RNN_MODE="${LOWVIS_RNN_MODE:-both}"
export LOWVIS_RNN_S1_STEPS="${LOWVIS_RNN_S1_STEPS:-15000}"
export LOWVIS_RNN_S2_A_STEPS="${LOWVIS_RNN_S2_A_STEPS:-8000}"
export LOWVIS_RNN_S2_B_STEPS="${LOWVIS_RNN_S2_B_STEPS:-22000}"
export LOWVIS_RNN_VAL_INTERVAL="${LOWVIS_RNN_VAL_INTERVAL:-500}"
export LOWVIS_RNN_BATCH_SIZE="${LOWVIS_RNN_BATCH_SIZE:-512}"
export LOWVIS_RNN_GRAD_ACCUM="${LOWVIS_RNN_GRAD_ACCUM:-2}"
export LOWVIS_RNN_NUM_WORKERS="${LOWVIS_RNN_NUM_WORKERS:-0}"
export LOWVIS_RNN_PATIENCE="${LOWVIS_RNN_PATIENCE:-10}"
export LOWVIS_RNN_PRETRAINED_CKPT="${LOWVIS_RNN_PRETRAINED_CKPT:-}"
export LOWVIS_RNN_THRESHOLD_MODE="${LOWVIS_RNN_THRESHOLD_MODE:-val_search}"
export LOWVIS_RNN_EXTRA_ARGS="${LOWVIS_RNN_EXTRA_ARGS:-}"
export LOWVIS_RNN_SAMPLING_EXPERIMENTS="${LOWVIS_RNN_SAMPLING_EXPERIMENTS:-0 1}"
EXPERIMENT_LIST="${LOWVIS_RNN_SAMPLING_EXPERIMENTS//,/ }"
EXPERIMENT_LIST="${EXPERIMENT_LIST//:/ }"
export LOWVIS_RNN_RUN_PREFIX="${LOWVIS_RNN_RUN_PREFIX:-${LOWVIS_RNN_RUN_ID:-exp_${SLURM_JOB_ID}_static_rnn_sampling_matrix}}"
export LOWVIS_RNN_LOCAL_CACHE_ID="${LOWVIS_RNN_LOCAL_CACHE_ID:-${LOWVIS_RNN_RUN_PREFIX}_shared_data}"
export LOWVIS_RNN_LOCAL_CACHE_DIR="${LOWVIS_RNN_LOCAL_CACHE_DIR:-/tmp}"
export LOWVIS_RNN_CLEAN_LOCAL_CACHE="${LOWVIS_RNN_CLEAN_LOCAL_CACHE:-1}"
export LOWVIS_RNN_SLEEP_BETWEEN="${LOWVIS_RNN_SLEEP_BETWEEN:-15}"
export LOWVIS_RNN_S1_DATA_DIR="${LOWVIS_RNN_S1_DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_pmst_v5_aligned_12h_pm10_pm25}"
export LOWVIS_RNN_S2_DATA_DIR="${LOWVIS_RNN_S2_DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
export LOWVIS_RNN_CKPT_DIR="${LOWVIS_RNN_CKPT_DIR:-/public/home/putianshu/vis_mlp/checkpoints}"

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "BASE_PORT       : ${BASE_PORT}"
echo "MODE            : ${LOWVIS_RNN_MODE}"
echo "PRETRAINED_CKPT : ${LOWVIS_RNN_PRETRAINED_CKPT}"
echo "RUN_PREFIX      : ${LOWVIS_RNN_RUN_PREFIX}"
echo "LOCAL_CACHE_ID  : ${LOWVIS_RNN_LOCAL_CACHE_ID}"
echo "LOCAL_CACHE_DIR : ${LOWVIS_RNN_LOCAL_CACHE_DIR}"
echo "CLEAN_LOCAL_CACHE: ${LOWVIS_RNN_CLEAN_LOCAL_CACHE}"
echo "S1_DATA_DIR     : ${LOWVIS_RNN_S1_DATA_DIR}"
echo "S2_DATA_DIR     : ${LOWVIS_RNN_S2_DATA_DIR}"
echo "CKPT_DIR        : ${LOWVIS_RNN_CKPT_DIR}"
echo "THRESHOLD_MODE  : ${LOWVIS_RNN_THRESHOLD_MODE}"
echo "EXPERIMENTS     : ${EXPERIMENT_LIST}"
echo "EXTRA_ARGS      : ${LOWVIS_RNN_EXTRA_ARGS}"

if [ "${LOWVIS_RNN_MODE}" = "s2" ] && [ -z "${LOWVIS_RNN_PRETRAINED_CKPT}" ]; then
    echo "[WARN] LOWVIS_RNN_MODE=s2 but LOWVIS_RNN_PRETRAINED_CKPT is empty; S2 experiments will train without S1 initialization."
fi

if [ "${LOWVIS_RNN_CLEAN_LOCAL_CACHE}" = "1" ]; then
    echo "Cleaning user-owned stale Static-RNN local cache files on allocated nodes..."
    srun --nodes=${SLURM_NNODES} \
         --ntasks=${SLURM_NNODES} \
         --ntasks-per-node=1 \
         /bin/bash -c '
             cache_dir="${LOWVIS_RNN_LOCAL_CACHE_DIR:-/tmp}"
             mkdir -p "${cache_dir}" 2>/dev/null || cache_dir="/tmp"
             echo "[$(hostname)] local cache dir: ${cache_dir}"
             df -h "${cache_dir}" | tail -1 || true
             find "${cache_dir}" -maxdepth 1 -user "$USER" -type f \
                 \( -name "X_train_*.npy" -o -name "X_val_*.npy" \
                    -o -name "y_train_*.npy" -o -name "y_val_*.npy" \
                    -o -name "X_train_*.npy.tmp" -o -name "X_val_*.npy.tmp" \
                    -o -name "y_train_*.npy.tmp" -o -name "y_val_*.npy.tmp" \
                    -o -name "*.nfs_fallback" \) \
                 -print -delete 2>/dev/null || true
             df -h "${cache_dir}" | tail -1 || true
         '
fi

run_one_experiment() {
    local exp_id="$1"
    get_experiment "${exp_id}"

    local rdzv_port
    local dist_store_port
    local run_id
    rdzv_port=$(expr ${BASE_PORT} + ${exp_id} \* 2)
    dist_store_port=$(expr ${rdzv_port} + 1)
    run_id="${LOWVIS_RNN_RUN_PREFIX}_${exp_id}_${EXP_NAME}"

    export DIST_STORE_PORT="${dist_store_port}"

    echo ""
    echo "================================================================"
    echo "Sampling experiment ${exp_id}: ${EXP_NAME}"
    echo "Started at       : $(date)"
    echo "RUN_ID           : ${run_id}"
    echo "RDZV_PORT        : ${rdzv_port}"
    echo "DIST_STORE_PORT  : ${DIST_STORE_PORT}"
    echo "MODEL_ARGS       : ${MODEL_ARGS}"
    echo "================================================================"

    echo "Cleaning /dev/shm on all allocated nodes before ${EXP_NAME}..."
    srun --nodes=${SLURM_NNODES} \
         --ntasks=${SLURM_NNODES} \
         --ntasks-per-node=1 \
         bash -c "rm -rf /dev/shm/torch_* /dev/shm/*static_rnn_lowvis* /dev/shm/*lowvis_rnn* 2>/dev/null; \
                  echo \"\$(hostname) SHM Cleaned\""

    srun --nodes=${SLURM_NNODES} \
         --ntasks=${SLURM_NNODES} \
         --ntasks-per-node=1 \
         --cpus-per-task=${CPUS_PER_TASK} \
         bash -c "
             set -euo pipefail
             cd \"${SCRIPT_DIR}\"
             export MIOPEN_USER_DB_PATH=/tmp/miopen-\${USER}-\${SLURM_JOB_ID}-${exp_id}-\${SLURM_NODEID}
             mkdir -p \${MIOPEN_USER_DB_PATH}

             torchrun \
                 --nnodes=${SLURM_NNODES} \
                 --nproc_per_node=${DCUS_PER_NODE} \
                 --node_rank=\${SLURM_NODEID} \
                 --rdzv_backend=static \
                 --rdzv_endpoint=${MASTER_ADDR}:${rdzv_port} \
                 --rdzv_id=${SLURM_JOBID}_sampling_${exp_id} \
                 --max_restarts=0 \
                 \"${SCRIPT_DIR}/train_static_rnn_lowvis.py\" \
                 --mode \"${LOWVIS_RNN_MODE}\" \
                 --run-id \"${run_id}\" \
                 --local-cache-id \"${LOWVIS_RNN_LOCAL_CACHE_ID}\" \
                 --s1-data-dir \"${LOWVIS_RNN_S1_DATA_DIR}\" \
                 --s2-data-dir \"${LOWVIS_RNN_S2_DATA_DIR}\" \
                 --ckpt-dir \"${LOWVIS_RNN_CKPT_DIR}\" \
                 --pretrained-ckpt \"${LOWVIS_RNN_PRETRAINED_CKPT}\" \
                 --threshold-mode \"${LOWVIS_RNN_THRESHOLD_MODE}\" \
                 ${MODEL_ARGS} \
                 ${LOWVIS_RNN_EXTRA_ARGS}
         "

    echo "Sampling experiment ${exp_id}: ${EXP_NAME} finished at $(date)"
    sleep "${LOWVIS_RNN_SLEEP_BETWEEN}"
}

for exp_id in ${EXPERIMENT_LIST}; do
    run_one_experiment "${exp_id}"
done

echo "Static-RNN sampling matrix finished at: $(date)"
