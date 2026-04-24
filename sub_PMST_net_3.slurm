#!/bin/bash
#SBATCH -J PMST_net_test_8
#SBATCH -p kshdexclu01
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 240:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --exclusive

echo "Job started at: $(date)"

# ==========================================
# 环境加载
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
# RCCL / NCCL 通信配置
#
# 根本原因：系统缺少 iommu=pt，导致 IB Verbs 内存注册（ibv_reg_mr）
# 失败，RCCL 在 Ring 拓扑的 QP 建立阶段挂死。
#
# 核心修复：NCCL_IB_DISABLE=1
#   完全禁用 IB Verbs/RDMA 层（无需 iommu=pt），
#   改用 TCP socket 走 ib0 高速网络接口。
#   带宽仍为 IB 物理线速，延迟稍高但完全可靠。
#
# 若未来管理员在内核启动参数中加入 iommu=pt，
#   可将 NCCL_IB_DISABLE 改回 0，并取消下方
#   "IB Verbs 高性能模式" 注释块的注释。
# ==========================================

# ---- 核心修复 ----
export NCCL_IB_DISABLE=1          # 禁用 IB Verbs，改用 socket 模式
export NCCL_SOCKET_IFNAME=ib0      # socket 走 IB 高速网络接口

# ---- AMD DCU 专项设置 ----
export HSA_ENABLE_SDMA=0           # 禁用 SDMA（AMD DCU NUMA 失衡时不稳定）
export HSA_FORCE_FINE_GRAIN_PCIE=1

# ---- 其他稳定性设置 ----
export NCCL_P2P_LEVEL=0            # 禁用节点内 GPU P2P（DCU 不支持）
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN             # INFO 会产生大量日志，训练稳定后用 WARN

# ---- 异步错误上报：将挂死转为可见错误 ----
export NCCL_ASYNC_ERROR_HANDLING=1

# ==============================================================
# IB Verbs 高性能模式（需要内核 iommu=pt，当前环境不支持，保留备用）
# ==============================================================
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=0       # 禁用 GPU Direct over IB
# export NCCL_IB_HCA=mlx5_0:1       # 只用一张 HCA 的一个端口，避免 QP 建立复杂度
# export NCCL_IB_GID_INDEX=3        # 使用 RoCEv2 GID，对 IOMMU 要求更低
# export NCCL_IB_TIMEOUT=22         # IB 操作超时（单位：约 4us * 2^timeout）
# export NCCL_IB_RETRY_CNT=7        # QP 重传次数
# ==============================================================

# ==========================================
# 节点发现
# ==========================================
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

# 直接使用主机名作为 master_addr，避免多网卡环境下 IP 选择不一致的问题
# torchrun 会通过 DNS/hosts 解析，适用于集群内网
MASTER_ADDR=${head_node}

# 用作业ID末4位生成端口，避免同机器多作业端口冲突
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Head Node: ${head_node}"
echo "Master Addr: ${MASTER_ADDR}, Master Port: ${MASTER_PORT}"

# ==========================================
# 验证 IB 网卡状态（只跑一次，用于确认网络可用）
# ==========================================
echo "--- IB Status on head node ---"
srun --ntasks=1 -w "$head_node" bash -c "ibstat | grep -E 'CA|Port|State|GUID'"

# ==========================================
# 清理各节点 /dev/shm
# ==========================================
echo "Cleaning /dev/shm on all allocated nodes..."
srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     bash -c "rm -rf /dev/shm/torch_* /dev/shm/*X_train* /dev/shm/*y_train* 2>/dev/null; echo \"\$(hostname) SHM Cleaned\""

# ==========================================
# 启动训练
# ==========================================
echo "Starting Training with torchrun on ${SLURM_NNODES} nodes (${SLURM_NNODES}x4 DCUs)..."

srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     --cpus-per-task=32 \
     --gres=dcu:4 \
     bash -c "
         # 为每个节点分配独立的 MIOpen 缓存，避免 NFS 上的缓存竞争
         export MIOPEN_USER_DB_PATH=/tmp/miopen_db_${SLURM_JOBID}_\${SLURM_NODEID}
         export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH
         mkdir -p \$MIOPEN_USER_DB_PATH

         torchrun \
             --nnodes=${SLURM_NNODES} \
             --nproc_per_node=4 \
             --node_rank=\${SLURM_NODEID} \
             --master_addr=${MASTER_ADDR} \
             --master_port=${MASTER_PORT} \
             --rdzv_id=${SLURM_JOBID} \
             --rdzv_backend=c10d \
             --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
             --max_restarts=0 \
             PMST_net_test_8_s2.py
     "

echo "Job finished at: $(date)"