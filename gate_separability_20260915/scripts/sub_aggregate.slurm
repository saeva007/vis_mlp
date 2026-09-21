#!/bin/bash
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /public/home/putianshu/vis_mlp/gate_separability_20260915/logs/gate_aggregate_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/gate_separability_20260915/logs/gate_aggregate_%j.err

set -euo pipefail
ROOT=/public/home/putianshu/vis_mlp/gate_separability_20260915
source "${ROOT}/scripts/runtime.sh"
python "${ROOT}/scripts/aggregate.py"
