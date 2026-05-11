#!/bin/bash
#SBATCH -J PMST_airport_metar
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

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Queue=${SLURM_JOB_PARTITION:-kshdexclu01} CPUs=${SLURM_CPUS_PER_TASK:-32}"

MAX_STABLE_NNODES=10
if [ "${SLURM_NNODES}" -gt "${MAX_STABLE_NNODES}" ]; then
    echo "Refusing to launch on ${SLURM_NNODES} nodes: validated up to ${MAX_STABLE_NNODES} nodes on the current RCCL/DCU cluster."
    exit 1
fi

SCRIPT_DIR="/public/home/putianshu/vis_mlp/train"

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

# Keep the same conservative DCU/RCCL environment as the S2 PMST template,
# even though this job uses only one process.
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

cd "${SCRIPT_DIR}"
mkdir -p logs checkpoints

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
DIST_STORE_PORT=$(expr ${BASE_PORT} + 1)

export MASTER_ADDR
export DIST_STORE_PORT

echo "Head Node       : ${head_node}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "RDZV_PORT       : ${RDZV_PORT}"
echo "DIST_STORE_PORT : ${DIST_STORE_PORT}"
echo "SLURM_NNODES    : ${SLURM_NNODES}"
echo "Total DCUs      : $((SLURM_NNODES * 4))"

echo "Cleaning /dev/shm on all allocated nodes..."
srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     bash -c "rm -rf /dev/shm/torch_* /dev/shm/*X_train* /dev/shm/*y_train* 2>/dev/null; \
              echo \"\$(hostname) SHM Cleaned\""

echo "Starting airport METAR PMST full training on ${SLURM_NNODES} nodes (${SLURM_NNODES}x4 DCUs)..."
srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     --cpus-per-task=32 \
     --gres=dcu:4 \
     --kill-on-bad-exit=1 \
     bash -c "
         cd \"${SCRIPT_DIR}\" || exit 1
         export MIOPEN_USER_DB_PATH=/tmp/miopen_db_${SLURM_JOBID}_\${SLURM_NODEID}
         export MIOPEN_CUSTOM_CACHE_DIR=\${MIOPEN_USER_DB_PATH}
         mkdir -p \${MIOPEN_USER_DB_PATH}

         torchrun \
             --nnodes=${SLURM_NNODES} \
             --nproc_per_node=4 \
             --node_rank=\${SLURM_NODEID} \
             --rdzv_backend=static \
             --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT} \
             --rdzv_id=${SLURM_JOBID} \
             --max_restarts=0 \
             \"${SCRIPT_DIR}/PMST_net_airport_metar.py\" \
             --data-dir /public/home/putianshu/vis_mlp/ml_dataset_airport_metar_2025_12h \
             --save-ckpt-dir /public/home/putianshu/vis_mlp/checkpoints \
             --exp-id airport_metar_2025 \
             --run-suffix full_from_scratch \
             --total-steps 60000 \
             --batch-size 512 \
             --grad-accum 2 \
             --val-interval 500 \
             --num-workers 0 \
             --lr-backbone 1e-4 \
             --lr-head 2e-4 \
             --val-ratio 0.10
     "

echo "Job finished at: $(date)"
