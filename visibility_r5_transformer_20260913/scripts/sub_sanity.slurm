#!/bin/bash
#SBATCH -J r5_sanity
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs/sanity_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs/sanity_%j.err
set -euo pipefail
ROOT=/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913
source "${ROOT}/scripts/runtime.sh"
export MIOPEN_USER_DB_PATH="/tmp/miopen_r5_sanity_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
HIP_VISIBLE_DEVICES=0 python "${ROOT}/scripts/r5_sanity.py"
