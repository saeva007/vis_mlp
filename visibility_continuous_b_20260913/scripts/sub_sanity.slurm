#!/bin/bash
#SBATCH -J vcb_sanity
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/sanity_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_continuous_b_20260913/logs/sanity_%j.err
set -euo pipefail
ROOT=/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913
source ${ROOT}/scripts/runtime.sh
python -u ${ROOT}/scripts/viscast_continuous_b.py sanity
