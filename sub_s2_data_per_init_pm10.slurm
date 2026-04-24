#!/bin/bash
#SBATCH -J s2_pm10_per_init
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH -D /public/home/putianshu/vis_mlp
# 日志目录须已存在，否则 Slurm 打不开 .out/.err，作业会直接失败且无日志。
#SBATCH -o /public/home/putianshu/vis_mlp/log/s2_per_init/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/log/s2_per_init/%j_%x.err
#
# --mem=0 在本集群表示使用该节点全部内存（与 sub_PMST_net_s2.slurm 一致）。
#
# 数据集构建（CPU/内存型，无需 DCU）。划分：按 valid 时间月末策略（见 s2_data_per_init_split_pm10.py）。
#
# 提交:
#   mkdir -p /public/home/putianshu/vis_mlp/log/s2_per_init   # 若仓库里尚无此目录
#   sbatch /public/home/putianshu/vis_mlp/train/sub_s2_data_per_init_pm10.slurm
#
# 换分区（例如交互里用过的 kshcexclu04）:
#   sbatch -p kshcexclu04 train/sub_s2_data_per_init_pm10.slurm
#
# 可选环境变量（sbatch 前 export 或 --export=ALL,VAR=val）:
#   RUNS_PER_BATCH=4  OUTPUT_DIR=/path/out  MAX_RUNS=100  PARTS_DIR=/tmp/${USER}_s2_parts
#   sbatch --export=ALL,RUNS_PER_BATCH=4,OUTPUT_DIR=/path/to/out train/sub_s2_data_per_init_pm10.slurm
#
set -euo pipefail

echo "JobId=${SLURM_JOB_ID}  Node=$(hostname)  Started=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd /public/home/putianshu/vis_mlp

export PYTHONUNBUFFERED=1

RUNS_PER_BATCH="${RUNS_PER_BATCH:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_RUNS="${MAX_RUNS:-}"
PARTS_DIR="${PARTS_DIR:-}"

args=(--runs-per-batch "${RUNS_PER_BATCH}")
[ -n "${OUTPUT_DIR}" ] && args+=(--output-dir "${OUTPUT_DIR}")
[ -n "${MAX_RUNS}" ] && args+=(--max-runs "${MAX_RUNS}")
[ -n "${PARTS_DIR}" ] && args+=(--parts-dir "${PARTS_DIR}")

echo "args: ${args[*]}"
python s2_data_per_init_split_pm10.py "${args[@]}"

echo "Finished at $(date)"
