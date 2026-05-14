#!/bin/bash
#SBATCH -J lowvis_diag_pm10pm25
#SBATCH -p kshdexclu01
# Match the S2 launcher: each allocated node must use all 4 DCUs.
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 240:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/log/lowvis_diag/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/log/lowvis_diag/%j_%x.err
#SBATCH --exclusive
#
# Post-hoc diagnostic only: no training, no checkpoint writes.
#
# Submit:
#   mkdir -p /public/home/putianshu/vis_mlp/log/lowvis_diag
#   sbatch /public/home/putianshu/vis_mlp/train/sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm
#
# Useful overrides:
#   sbatch --export=ALL,SPLIT=val,SKIP_FINE_GRID=1 sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm
#   sbatch --export=ALL,LIMIT_SAMPLES=200000 sub_diagnose_lowvis_checkpoint_pm10_pm25.slurm

set -euo pipefail

DCUS_PER_NODE=4
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Partition=${SLURM_JOB_PARTITION:-kshdexclu01} Nodes=${SLURM_NNODES:-5} CPUs=${CPUS_PER_TASK} DCUs/node=${DCUS_PER_NODE}"

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"

# Match the DCU PyTorch runtime used by the S2 training launcher.
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

# RCCL/NCCL settings copied from the S2 training launcher.
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
SEASON_TH_PATH="${SEASON_TH_PATH:-checkpoints/${RUN_ID}_season_thresholds.pt}"
OUT_DIR="${OUT_DIR:-diagnostics/${RUN_ID}_${SPLIT}_lowvis_diag}"
FOG_TH="${FOG_TH:-0.34}"
MIST_TH="${MIST_TH:-0.56}"
THRESHOLD_RULE="${THRESHOLD_RULE:-mutual}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-0}"
SKIP_FINE_GRID="${SKIP_FINE_GRID:-0}"
USE_CALIBRATION="${USE_CALIBRATION:-0}"

EXTRA_ARGS=""
if [[ "${SKIP_FINE_GRID}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --skip-fine-grid"
fi
if [[ "${USE_CALIBRATION}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --use-calibration"
fi

mkdir -p /public/home/putianshu/vis_mlp/log/lowvis_diag

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "RDZV_PORT       : ${RDZV_PORT}       (torchrun rendezvous)"
echo "SLURM_NNODES    : ${SLURM_NNODES}"
echo "Total DCUs      : $((SLURM_NNODES * DCUS_PER_NODE))"
echo "RUN_ID          : ${RUN_ID}"
echo "SPLIT           : ${SPLIT}"
echo "OUT_DIR         : ${OUT_DIR}"

echo "Cleaning /dev/shm on all allocated nodes..."
srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     bash -c "rm -rf /dev/shm/torch_* /dev/shm/*lowvis_diag* 2>/dev/null; \
              echo \"\$(hostname) SHM Cleaned\""

echo "Starting diagnostics on ${SLURM_NNODES} nodes (${SLURM_NNODES}x${DCUS_PER_NODE} DCUs)..."

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
             \"${SCRIPT_DIR}/diagnose_lowvis_checkpoint_pm10_pm25.py\" \
             --distributed \
             --base \"${BASE_DIR}\" \
             --data-dir \"${DATA_DIR}\" \
             --split \"${SPLIT}\" \
             --ckpt-path \"${CKPT_PATH}\" \
             --scaler-path \"${SCALER_PATH}\" \
             --season-th-path \"${SEASON_TH_PATH}\" \
             --out-dir \"${OUT_DIR}\" \
             --fog-th \"${FOG_TH}\" \
             --mist-th \"${MIST_TH}\" \
             --threshold-rule \"${THRESHOLD_RULE}\" \
             --batch-size \"${BATCH_SIZE}\" \
             --limit-samples \"${LIMIT_SAMPLES}\" \
             --device auto \
             ${EXTRA_ARGS}
     " 2>&1 | tee "/public/home/putianshu/vis_mlp/log/lowvis_diag/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"

echo "Finished at $(date)"
