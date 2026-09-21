#!/bin/bash
#SBATCH -J vc_sanity
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_continuous_20260912/logs/%j_sanity.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_continuous_20260912/logs/%j_sanity.err
set -euo pipefail
ROOT=/public/home/putianshu/vis_mlp/visibility_continuous_20260912
source ${ROOT}/scripts/runtime.sh
test -f ${ROOT}/eval/R0/metrics.json || python -u ${ROOT}/scripts/viscast_continuous.py r0
python -u ${ROOT}/scripts/viscast_continuous.py sanity
