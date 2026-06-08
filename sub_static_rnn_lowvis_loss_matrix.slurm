#!/bin/bash
#SBATCH -J static_rnn_lowvis_loss
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
            EXP_NAME="simple_ce_classification"
            MODEL_ARGS="$(base_model_args) --loss-mode ce --no-label-smoothing --alpha-clear-fp 0.0 --alpha-recall-boost 0.0 --ordinal-cost-weight 0.0 --aux-reg-weight 0.0"
            ;;
        1)
            EXP_NAME="simple_logvis_regression"
            MODEL_ARGS="$(base_model_args) --loss-mode regression --no-label-smoothing --alpha-clear-fp 0.0 --alpha-recall-boost 0.0 --ordinal-cost-weight 0.0 --aux-reg-weight 0.0"
            ;;
        2)
            EXP_NAME="proposed_rare_event_focal"
            MODEL_ARGS="$(base_model_args) --loss-mode designed_focal"
            ;;
        3)
            PLAIN_FOCAL_ALPHA_FOG="${LOWVIS_RNN_PLAIN_FOCAL_ALPHA_FOG:-1.0}"
            PLAIN_FOCAL_ALPHA_MIST="${LOWVIS_RNN_PLAIN_FOCAL_ALPHA_MIST:-1.0}"
            PLAIN_FOCAL_ALPHA_CLEAR="${LOWVIS_RNN_PLAIN_FOCAL_ALPHA_CLEAR:-1.0}"
            PLAIN_FOCAL_GAMMA_FOG="${LOWVIS_RNN_PLAIN_FOCAL_GAMMA_FOG:-2.0}"
            PLAIN_FOCAL_GAMMA_MIST="${LOWVIS_RNN_PLAIN_FOCAL_GAMMA_MIST:-2.0}"
            PLAIN_FOCAL_GAMMA_CLEAR="${LOWVIS_RNN_PLAIN_FOCAL_GAMMA_CLEAR:-2.0}"
            EXP_NAME="plain_focal_loss"
            MODEL_ARGS="$(base_model_args) --loss-mode designed_focal --no-label-smoothing --class-weight-fog ${PLAIN_FOCAL_ALPHA_FOG} --class-weight-mist ${PLAIN_FOCAL_ALPHA_MIST} --class-weight-clear ${PLAIN_FOCAL_ALPHA_CLEAR} --focal-gamma-fog ${PLAIN_FOCAL_GAMMA_FOG} --focal-gamma-mist ${PLAIN_FOCAL_GAMMA_MIST} --focal-gamma-clear ${PLAIN_FOCAL_GAMMA_CLEAR} --alpha-clear-fp 0.0 --alpha-recall-boost 0.0 --ordinal-cost-weight 0.0 --aux-reg-weight 0.0 --boundary-weight 0.0 --physical-hard-weight 0.0 --aerosol-hard-weight 0.0 --sample-weight-cap 1.0"
            ;;
        *)
            echo "Unknown loss experiment id: $1" >&2
            exit 2
            ;;
    esac
}

echo "Static-RNN loss-function matrix started at: $(date)"
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
export LOWVIS_RNN_EXTRA_ARGS="${LOWVIS_RNN_EXTRA_ARGS:-}"
export LOWVIS_RNN_EXPERIMENTS="${LOWVIS_RNN_EXPERIMENTS:-0 1 2}"
EXPERIMENT_LIST="${LOWVIS_RNN_EXPERIMENTS//,/ }"
EXPERIMENT_LIST="${EXPERIMENT_LIST//:/ }"
export LOWVIS_RNN_RUN_PREFIX="${LOWVIS_RNN_RUN_PREFIX:-${LOWVIS_RNN_RUN_ID:-exp_${SLURM_JOB_ID}_static_rnn_loss_matrix}}"
export LOWVIS_RNN_LOCAL_CACHE_ID="${LOWVIS_RNN_LOCAL_CACHE_ID:-${LOWVIS_RNN_RUN_PREFIX}_shared_data}"
export LOWVIS_RNN_SLEEP_BETWEEN="${LOWVIS_RNN_SLEEP_BETWEEN:-15}"

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "BASE_PORT       : ${BASE_PORT}"
echo "MODE            : ${LOWVIS_RNN_MODE}"
echo "PRETRAINED_CKPT : ${LOWVIS_RNN_PRETRAINED_CKPT}"
echo "RUN_PREFIX      : ${LOWVIS_RNN_RUN_PREFIX}"
echo "LOCAL_CACHE_ID  : ${LOWVIS_RNN_LOCAL_CACHE_ID}"
echo "EXPERIMENTS     : ${EXPERIMENT_LIST}"
echo "EXTRA_ARGS      : ${LOWVIS_RNN_EXTRA_ARGS}"

if [ "${LOWVIS_RNN_MODE}" = "s2" ] && [ -z "${LOWVIS_RNN_PRETRAINED_CKPT}" ]; then
    echo "[WARN] LOWVIS_RNN_MODE=s2 but LOWVIS_RNN_PRETRAINED_CKPT is empty; S2 experiments will train without S1 initialization."
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
    echo "Loss experiment ${exp_id}: ${EXP_NAME}"
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
                 --rdzv_id=${SLURM_JOBID}_loss_${exp_id} \
                 --max_restarts=0 \
                 \"${SCRIPT_DIR}/train_static_rnn_lowvis.py\" \
                 --mode \"${LOWVIS_RNN_MODE}\" \
                 --run-id \"${run_id}\" \
                 --local-cache-id \"${LOWVIS_RNN_LOCAL_CACHE_ID}\" \
                 ${MODEL_ARGS} \
                 --pretrained-ckpt \"${LOWVIS_RNN_PRETRAINED_CKPT}\" \
                 ${LOWVIS_RNN_EXTRA_ARGS}
         "

    echo "Loss experiment ${exp_id}: ${EXP_NAME} finished at $(date)"
    sleep "${LOWVIS_RNN_SLEEP_BETWEEN}"
}

for exp_id in ${EXPERIMENT_LIST}; do
    run_one_experiment "${exp_id}"
done

echo "Static-RNN loss-function matrix finished at: $(date)"
