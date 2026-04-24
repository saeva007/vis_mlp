#!/bin/bash
#SBATCH -J s2_diag_mist_dist
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 08:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/log/s2_diag/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/log/s2_diag/%j_%x.err
#
# CPU/内存型诊断：勿在登录节点跑（大 CSV + 千万行 y）。
# 提交前:
#   mkdir -p /public/home/putianshu/vis_mlp/log/s2_diag
#   sbatch /public/home/putianshu/vis_mlp/train/sub_diag_s2_mist_distribution.slurm
#
# 可选: sbatch --export=ALL,S2_DATA_DIR=/path/to/dataset ...
#
set -euo pipefail

echo "JobId=${SLURM_JOB_ID}  Node=$(hostname)  Started=$(date)"

LOG_ROOT="/public/home/putianshu/vis_mlp/log/s2_diag"
mkdir -p "${LOG_ROOT}"

# 与 sub_s2_data_per_init_pm10.slurm 相同 conda（勿在登录节点跑本脚本）
source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

export PYTHONUNBUFFERED=1

DATA_DIR="${S2_DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_fe_12h_48h_pm10_per_init_lead_split_uid27798}"

cd /public/home/putianshu/vis_mlp/train
python diag_s2_per_init_mist_distribution.py --data-dir "${DATA_DIR}" \
  2>&1 | tee "${LOG_ROOT}/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"

echo "Finished at $(date)  log: ${LOG_ROOT}/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"
