#!/bin/bash
#SBATCH -J lowvis_hard_pm10pm25
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 240:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/log/lowvis_hard_samples/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/log/lowvis_hard_samples/%j_%x.err
#SBATCH --exclusive
#
# Inference-only hard-sample pool export. No training and no checkpoint writes.
#
# Submit:
#   mkdir -p /public/home/putianshu/vis_mlp/log/lowvis_hard_samples
#   sbatch /public/home/putianshu/vis_mlp/train/sub_export_lowvis_hard_samples_pm10_pm25.slurm
#
# Useful overrides:
#   sbatch --export=ALL,SPLIT=train,MAX_PER_POOL=200000 sub_export_lowvis_hard_samples_pm10_pm25.slurm
#   sbatch --export=ALL,SPLIT=val,LIMIT_SAMPLES=200000 sub_export_lowvis_hard_samples_pm10_pm25.slurm

set -euo pipefail

DCUS_PER_NODE=4
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Partition=${SLURM_JOB_PARTITION:-kshdexclu01} Nodes=${SLURM_NNODES:-1} CPUs=${CPUS_PER_TASK} DCUs/node=${DCUS_PER_NODE}"
if [ "${SLURM_NNODES:-1}" -ne 1 ]; then
    echo "Refusing to launch: this export script is intentionally single-node, but SLURM_NNODES=${SLURM_NNODES:-unset}."
    exit 1
fi

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"

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

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c '
  for c in /sbin/ip /usr/sbin/ip; do
    if [ -x "$c" ]; then
      v=$("$c" -4 -o addr show ib0 2>/dev/null | awk "NR==1{print \$4}" | cut -d/ -f1)
      if [ -n "$v" ]; then echo "$v"; exit 0; fi
    fi
  done
  if command -v ip >/dev/null 2>&1; then
    v=$(ip -4 -o addr show ib0 2>/dev/null | awk "NR==1{print \$4}" | cut -d/ -f1)
    if [ -n "$v" ]; then echo "$v"; exit 0; fi
  fi
  h=$(hostname -I 2>/dev/null | awk "{print \$1}")
  [ -n "$h" ] && echo "$h" && exit 0
  exit 1
' 2>/dev/null | head -1 | tr -d "\r") || true

if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=${head_node}
fi

BASE_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
RDZV_PORT=${BASE_PORT}

export MASTER_ADDR

BASE_DIR="${BASE_DIR:-/public/home/putianshu/vis_mlp}"
RUN_ID="${RUN_ID:-exp_1778563813_pm10_more_temp_search_utc}"
SPLIT="${SPLIT:-test}"
DATA_DIR="${DATA_DIR:-ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
CKPT_PATH="${CKPT_PATH:-checkpoints/${RUN_ID}_S2_PhaseB_best_score.pt}"
SCALER_PATH="${SCALER_PATH:-checkpoints/robust_scaler_${RUN_ID}_w12_dyn27_s2_48h_pm10.pkl}"
OUT_DIR="${OUT_DIR:-/public/home/putianshu/vis_mlp/hard_samples/${RUN_ID}_${SPLIT}_hard_samples}"
LOWVIS_GATE_TH="${LOWVIS_GATE_TH:-0.81}"
FINE_FOG_TH="${FINE_FOG_TH:-0.34}"
FINE_MIST_TH="${FINE_MIST_TH:-0.56}"
MAX_PER_POOL="${MAX_PER_POOL:-0}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
SEED="${SEED:-42}"
DIST_BACKEND="${DIST_BACKEND:-gloo}"

mkdir -p /public/home/putianshu/vis_mlp/log/lowvis_hard_samples

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "RDZV_PORT       : ${RDZV_PORT}       (torchrun rendezvous)"
echo "SLURM_NNODES    : ${SLURM_NNODES}"
echo "Total DCUs      : $((SLURM_NNODES * DCUS_PER_NODE))"
echo "DIST_BACKEND    : ${DIST_BACKEND}"
echo "RUN_ID          : ${RUN_ID}"
echo "SPLIT           : ${SPLIT}"
echo "OUT_DIR         : ${OUT_DIR}"
echo "MAX_PER_POOL    : ${MAX_PER_POOL}"

echo "Cleaning /dev/shm on all allocated nodes..."
srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     bash -c "rm -rf /dev/shm/torch_* /dev/shm/*lowvis_hard_samples* 2>/dev/null; \
              echo \"\$(hostname) SHM Cleaned\""

echo "Starting hard-sample export on ${SLURM_NNODES} node (${DCUS_PER_NODE} DCUs)..."

srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     --cpus-per-task=${CPUS_PER_TASK} \
     --gres=dcu:${DCUS_PER_NODE} \
     --kill-on-bad-exit=1 \
     bash -c "
         cd \"${SCRIPT_DIR}\" || exit 1
         export MIOPEN_USER_DB_PATH=/tmp/miopen_db_${SLURM_JOBID}_\${SLURM_NODEID}
         export MIOPEN_CUSTOM_CACHE_DIR=\${MIOPEN_USER_DB_PATH}
         mkdir -p \${MIOPEN_USER_DB_PATH}

         torchrun \
             --nnodes=${SLURM_NNODES} \
             --nproc_per_node=${DCUS_PER_NODE} \
             --node_rank=\${SLURM_NODEID} \
             --rdzv_backend=static \
             --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT} \
             --rdzv_id=${SLURM_JOBID} \
             --max_restarts=0 \
             \"${SCRIPT_DIR}/export_lowvis_hard_samples_pm10_pm25.py\" \
             --distributed \
             --base \"${BASE_DIR}\" \
             --run-id \"${RUN_ID}\" \
             --data-dir \"${DATA_DIR}\" \
             --split \"${SPLIT}\" \
             --ckpt-path \"${CKPT_PATH}\" \
             --scaler-path \"${SCALER_PATH}\" \
             --out-dir \"${OUT_DIR}\" \
             --lowvis-gate-th \"${LOWVIS_GATE_TH}\" \
             --fine-fog-th \"${FINE_FOG_TH}\" \
             --fine-mist-th \"${FINE_MIST_TH}\" \
             --max-per-pool \"${MAX_PER_POOL}\" \
             --batch-size \"${BATCH_SIZE}\" \
             --limit-samples \"${LIMIT_SAMPLES}\" \
             --seed \"${SEED}\" \
             --device auto \
             --dist-backend \"${DIST_BACKEND}\"
     " 2>&1 | tee "/public/home/putianshu/vis_mlp/log/lowvis_hard_samples/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"

echo "Finished at $(date)"
