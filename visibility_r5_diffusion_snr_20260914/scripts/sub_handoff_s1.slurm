#!/bin/bash
#SBATCH -J r5snr_handoff_S1
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/logs/%x_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914/logs/%x_%j.err
set -euo pipefail
ROUTE=${1:?route required}
SEED=${2:?seed required}
ROOT=/public/home/putianshu/vis_mlp/visibility_r5_diffusion_snr_20260914
source "${ROOT}/scripts/runtime.sh"
unset R5_X_DIR R5_QUICK_X
python "${ROOT}/scripts/r5_pipeline.py" handoff-s1 --route "${ROUTE}" --seed "${SEED}" --submit-s2
