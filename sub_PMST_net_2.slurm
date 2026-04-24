#!/bin/bash
#SBATCH -J PMST_net_test_6
#SBATCH -p kshdexclu01
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem 0
#SBATCH -t 240:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --exclusive 
echo "Job started at: `date`"

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

# ★★★ 核心修复：RCCL/ROCm DCU 多节点通信稳定性配置 ★★★

# 1. 禁用 GPU Direct RDMA（GDR），这是 iommu=pt 缺失时 hang 的根源
#    等系统管理员修好 iommu=pt 后，可以恢复为 NCCL_NET_GDR_LEVEL=3
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_GDR_LEVEL=0        # 禁用GPU Direct RDMA，避免iommu=pt缺失导致的hang
export NCCL_P2P_DISABLE=1          # 禁用P2P，改用更稳定的Host内存中转

# 2. 强制使用 Ring 算法，避免 Tree 算法在 NUMA 不均衡时退化
export NCCL_ALGO=Ring
export NCCL_TREE_THRESHOLD=0

# 3. 禁用 ROCm SDMA（AMD DCU 特有，NUMA 失衡时 SDMA 会加剧问题）
export HSA_ENABLE_SDMA=0

# 4. 指定正确的 IB 网卡（关键：在超算上通常是 ib0 或 mlx5_0）
#    先用 `ibstat` 或 `ip addr` 确认节点间 IB 网卡名，再修改
export NCCL_SOCKET_IFNAME=ib0      # 如果 ibstat 显示的是 mlx5_0 则改成 mlx5_0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3         # RoCEv2 配置，根据实际情况调整

# 5. 超时和调试
export NCCL_DEBUG=INFO             # 临时改为 INFO 来排查通信问题，稳定后改回 WARN
export NCCL_TIMEOUT=1800 
# 注意：下面这个变量对PyTorch的ProcessGroup超时无效，超时在代码中用 timedelta 控制
# 保留它只影响底层NCCL socket超时
export NCCL_SOCKET_TIMEOUT=120

# 6. 防止 NUMA 干扰
export NCCL_IGNORE_CPU_AFFINITY=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
echo "Head Node: $head_node, IP: $head_node_ip"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# 在 srun 之前，先验证 IB 网卡名（只需要一次，确认后可以删掉这行）
srun --ntasks=1 -w "$head_node" bash -c "ibstat | grep -E 'CA|Port|State'"

echo "Cleaning /dev/shm on all allocated nodes..."
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -c "rm -rf /dev/shm/torch_* 2>/dev/null; rm -rf /dev/shm/*X_train* 2>/dev/null; echo \"\$(hostname) SHM Cleaned\""

echo "Starting Training with torchrun on ${SLURM_NNODES} nodes..."
srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     --cpus-per-task=32 \
     --gres=dcu:4 \
     bash -c "
         export MIOPEN_USER_DB_PATH=/tmp/miopen_db_${SLURM_JOBID}_\${SLURM_NODEID}
         mkdir -p \$MIOPEN_USER_DB_PATH
         # 为每个节点分配独立的MIOPEN缓存
         export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH
         torchrun \
             --nnodes=${SLURM_NNODES} \
             --nproc_per_node=4 \
             --node_rank=\${SLURM_NODEID} \
             --master_port=$MASTER_PORT \
             --rdzv_id=$SLURM_JOBID \
             --rdzv_backend=c10d \
             --rdzv_endpoint=$head_node_ip:$MASTER_PORT \
             --max_restarts=0 \
             PMST_net_test_7.py
     "
echo "Job finished at: `date`"