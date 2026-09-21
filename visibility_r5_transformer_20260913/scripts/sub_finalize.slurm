#!/bin/bash
#SBATCH -J r5_finalize
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs/%x_%j.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913/logs/%x_%j.err
set -euo pipefail
ROUTE=${1:?route required}
STAGE=${2:?stage required}
SEED=${3:?seed required}
ROOT=/public/home/putianshu/vis_mlp/visibility_r5_transformer_20260913
source "${ROOT}/scripts/runtime.sh"
python "${ROOT}/scripts/r5_pipeline.py" select --route "${ROUTE}" --stage "${STAGE}" --seed "${SEED}"
BEST=$(python "${ROOT}/scripts/r5_pipeline.py" best-path --route "${ROUTE}" --stage "${STAGE}" --seed "${SEED}")
if [ "${STAGE}" = "S1" ]; then
  sbatch --parsable --job-name="r5_${ROUTE#R5-}_S2_s${SEED}" "${ROOT}/scripts/sub_train.slurm" "${ROUTE}" S2 "${SEED}" "${BEST}"
else
  sbatch --parsable --job-name="r5_final_${ROUTE#R5-}_s${SEED}" "${ROOT}/scripts/sub_final_eval.slurm" "${ROUTE}" "${SEED}" "${BEST}"
fi
