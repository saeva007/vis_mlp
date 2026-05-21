#!/bin/bash
#SBATCH -J static_rnn_lowvis_boundary
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
            EXP_NAME="focal_baseline"
            MODEL_ARGS="$(base_model_args)"
            ;;
        1)
            EXP_NAME="boundary_weight_focal"
            MODEL_ARGS="$(base_model_args) --boundary-weight 1.0 --boundary-fog-sigma 100 --boundary-mist-sigma 200 --sample-weight-cap 3.5"
            ;;
        2)
            EXP_NAME="physical_hard_focal"
            MODEL_ARGS="$(base_model_args) --physical-hard-weight 1.0 --aerosol-hard-weight 0.5 --sample-weight-cap 3.5"
            ;;
        3)
            EXP_NAME="boundary_physical_focal"
            MODEL_ARGS="$(base_model_args) --boundary-weight 0.8 --physical-hard-weight 0.8 --aerosol-hard-weight 0.4 --sample-weight-cap 3.5"
            ;;
        4)
            EXP_NAME="ordinal_focal"
            MODEL_ARGS="$(base_model_args) --ordinal-cost-weight 0.35"
            ;;
        5)
            EXP_NAME="simple_full_boundary_focal"
            MODEL_ARGS="$(base_model_args) --boundary-weight 0.7 --physical-hard-weight 0.7 --aerosol-hard-weight 0.35 --ordinal-cost-weight 0.25 --alpha-clear-fp 2.5 --sample-weight-cap 3.0"
            ;;
        *)
            echo "Unknown experiment id: $1" >&2
            exit 2
            ;;
    esac
}

echo "Boundary data-centric matrix started at: $(date)"
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
export LOWVIS_RNN_MODE="${LOWVIS_RNN_MODE:-s2}"
export LOWVIS_RNN_S1_STEPS="${LOWVIS_RNN_S1_STEPS:-15000}"
export LOWVIS_RNN_S2_A_STEPS="${LOWVIS_RNN_S2_A_STEPS:-8000}"
export LOWVIS_RNN_S2_B_STEPS="${LOWVIS_RNN_S2_B_STEPS:-22000}"
export LOWVIS_RNN_VAL_INTERVAL="${LOWVIS_RNN_VAL_INTERVAL:-500}"
export LOWVIS_RNN_BATCH_SIZE="${LOWVIS_RNN_BATCH_SIZE:-512}"
export LOWVIS_RNN_GRAD_ACCUM="${LOWVIS_RNN_GRAD_ACCUM:-2}"
export LOWVIS_RNN_NUM_WORKERS="${LOWVIS_RNN_NUM_WORKERS:-0}"
export LOWVIS_RNN_PATIENCE="${LOWVIS_RNN_PATIENCE:-10}"
export LOWVIS_RNN_PRETRAINED_CKPT="${LOWVIS_RNN_PRETRAINED_CKPT:-}"
export LOWVIS_RNN_EXTRA_ARGS="${LOWVIS_RNN_EXTRA_ARGS:-}"
export LOWVIS_RNN_EXPERIMENTS="${LOWVIS_RNN_EXPERIMENTS:-0 1 2 3 4 5}"
EXPERIMENT_LIST="${LOWVIS_RNN_EXPERIMENTS//,/ }"
EXPERIMENT_LIST="${EXPERIMENT_LIST//:/ }"
export LOWVIS_RNN_RUN_PREFIX="${LOWVIS_RNN_RUN_PREFIX:-${LOWVIS_RNN_RUN_ID:-exp_${SLURM_JOB_ID}_static_rnn_boundary_matrix}}"
export LOWVIS_RNN_SLEEP_BETWEEN="${LOWVIS_RNN_SLEEP_BETWEEN:-15}"

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "BASE_PORT       : ${BASE_PORT}"
echo "MODE            : ${LOWVIS_RNN_MODE}"
echo "PRETRAINED_CKPT : ${LOWVIS_RNN_PRETRAINED_CKPT}"
echo "RUN_PREFIX      : ${LOWVIS_RNN_RUN_PREFIX}"
echo "EXPERIMENTS     : ${EXPERIMENT_LIST}"
echo "EXTRA_ARGS      : ${LOWVIS_RNN_EXTRA_ARGS}"

if [ "${LOWVIS_RNN_MODE}" = "s2" ] && [ -z "${LOWVIS_RNN_PRETRAINED_CKPT}" ]; then
    echo "[WARN] LOWVIS_RNN_PRETRAINED_CKPT is empty; S2 experiments will train without S1 initialization."
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
    echo "Experiment ${exp_id}: ${EXP_NAME}"
    echo "Started at       : $(date)"
    echo "RUN_ID           : ${run_id}"
    echo "RDZV_PORT        : ${rdzv_port}       (torchrun rendezvous)"
    echo "DIST_STORE_PORT  : ${DIST_STORE_PORT} (Python dist.TCPStore)"
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
                 --rdzv_id=${SLURM_JOBID}_${exp_id} \
                 --max_restarts=0 \
                 \"${SCRIPT_DIR}/train_static_rnn_lowvis.py\" \
                 --mode \"${LOWVIS_RNN_MODE}\" \
                 --run-id \"${run_id}\" \
                 ${MODEL_ARGS} \
                 --pretrained-ckpt \"${LOWVIS_RNN_PRETRAINED_CKPT}\" \
                 ${LOWVIS_RNN_EXTRA_ARGS}
         "

    echo "Experiment ${exp_id}: ${EXP_NAME} finished at $(date)"
    sleep "${LOWVIS_RNN_SLEEP_BETWEEN}"
}

for exp_id in ${EXPERIMENT_LIST}; do
    run_one_experiment "${exp_id}"
done

echo "Boundary data-centric matrix finished at: $(date)"
