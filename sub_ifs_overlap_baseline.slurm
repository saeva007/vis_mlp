#!/bin/bash
#SBATCH -J pmst_ifs_overlap
#SBATCH -p kshdexclu01
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 240:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --exclusive
#
# IFS / Tianji overlap 基线训练（ifs_baseline 下 train_PMST_* 脚本），环境与通信配置对齐 sub_s1.slurm。
#
# 用法（建议在 train 目录下提交，以便 logs/ 落在 train/logs）：
#   cd /public/home/putianshu/vis_mlp/train && mkdir -p logs
#
#   S1（默认数据 ifs_baseline/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_overlap）：
#     sbatch --export=EXPERIMENT=s1_overlap sub_ifs_overlap_baseline.slurm
#
#   S2 overlap — Tianji 数据集：
#     sbatch --export=EXPERIMENT=s2_tianji sub_ifs_overlap_baseline.slurm
#
#   S2 overlap — IFS 数据集：
#     sbatch --export=EXPERIMENT=s2_ifs sub_ifs_overlap_baseline.slurm
#
# 可选环境变量（传给训练脚本或覆盖默认）：
#   OVERLAP_S1_BEST_CKPT   — S2 加载的 S1 权重（默认 ifs_baseline/checkpoints/*_S1_best_score.pt）
#   OVERLAP_S2_DATA_DIR    — 覆盖 S2 数据目录（非空时等价于 --s2_data_dir）
#   TORCHRUN_EXTRA_ARGS    — 追加到 torchrun 的参数（例如最大重启等）
#   TRAIN_EXTRA_ARGS       — 追加到 Python 入口的参数（例如 --s2_data_dir /path）
#
# 示例：自定义 S2 数据路径与 S1 权重（默认 S2 目录为 *_pm10_pm25_baseline）
#   sbatch --export=EXPERIMENT=s2_tianji,OVERLAP_S1_BEST_CKPT=/public/home/putianshu/vis_mlp/ifs_baseline/checkpoints/exp_s1_overlap_pm10_baseline_S1_best_score.pt sub_ifs_overlap_baseline.slurm

set -euo pipefail

echo "Job started at: $(date)"
echo "EXPERIMENT=${EXPERIMENT:-s1_overlap}"

MAX_STABLE_NNODES=10
if [ "${SLURM_NNODES}" -gt "${MAX_STABLE_NNODES}" ]; then
    echo "Refusing to launch on ${SLURM_NNODES} nodes"
    exit 1
fi

SCRIPT_DIR="/public/home/putianshu/vis_mlp/train"
BASELINE_DIR="/public/home/putianshu/vis_mlp/ifs_baseline"

EXPERIMENT="${EXPERIMENT:-s1_overlap}"
TORCHRUN_EXTRA_ARGS="${TORCHRUN_EXTRA_ARGS:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

case "${EXPERIMENT}" in
    s1_overlap)
        TRAIN_SCRIPT="${BASELINE_DIR}/train_PMST_s1_overlap_baseline.py"
        PY_ARGS=()
        ;;
    s2_tianji)
        TRAIN_SCRIPT="${BASELINE_DIR}/train_PMST_overlap_baseline_s2.py"
        PY_ARGS=(--data_source tianji)
        ;;
    s2_ifs)
        TRAIN_SCRIPT="${BASELINE_DIR}/train_PMST_overlap_baseline_s2.py"
        PY_ARGS=(--data_source ifs)
        ;;
    *)
        echo "Unknown EXPERIMENT=${EXPERIMENT} (use s1_overlap | s2_tianji | s2_ifs)"
        exit 1
        ;;
esac

if [ -n "${OVERLAP_S2_DATA_DIR:-}" ]; then
    PY_ARGS+=(--s2_data_dir "${OVERLAP_S2_DATA_DIR}")
fi

# shellcheck disable=SC2206
EXTRA_TOKENS=( ${TRAIN_EXTRA_ARGS} )

# ==========================================
# 1. Environment（与 sub_s1.slurm 一致）
# ==========================================
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

# ==========================================
# 2. RCCL / NCCL Communication
# ==========================================
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=0
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# ==========================================
# 3. Node Discovery
# ==========================================
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

# 计算节点上 PATH 常不含 ip；用 /sbin/ip 或回退 hostname -I（避免 execve(): ip: No such file or directory）
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
echo "TRAIN_SCRIPT    : ${TRAIN_SCRIPT}"
echo "PY_ARGS         : ${PY_ARGS[*]} ${EXTRA_TOKENS[*]}"

# ==========================================
# 4. Clean /dev/shm
# ==========================================
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 \
     bash -c "rm -rf /dev/shm/torch_* /dev/shm/*X_train* /dev/shm/*y_train* 2>/dev/null; echo \"\$(hostname) SHM Cleaned\""

# ==========================================
# 5. Launch Training
# ==========================================
echo "Starting ${EXPERIMENT} on ${SLURM_NNODES} nodes (${SLURM_NNODES}x4 DCUs)..."

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
             ${TORCHRUN_EXTRA_ARGS} \
             \"${TRAIN_SCRIPT}\" \
             ${PY_ARGS[@]} \
             ${EXTRA_TOKENS[@]}
     "

echo "Job finished at: $(date)"
