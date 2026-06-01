#!/bin/bash
#SBATCH -J static_rnn_s1_need_month
#SBATCH -p kshdexclu01
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 240:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

# Submit this wrapper twice, once with ABLATION_MODE=s1_pretrained and once
# with ABLATION_MODE=scratch, to test whether S1 is useful on the same
# month-group Tianji S2 split.

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
ABLATION_MODE="${ABLATION_MODE:-s1_pretrained}"
MONTHGROUP_S2_DATA_DIR="${MONTHGROUP_S2_DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthgroup_train13579_val11_test24681012}"
S1_RUN_ID="${S1_RUN_ID:-}"
S1_CKPT="${S1_CKPT:-}"

if [ -z "${S1_CKPT}" ] && [ -n "${S1_RUN_ID}" ]; then
    S1_CKPT="/public/home/putianshu/vis_mlp/checkpoints/${S1_RUN_ID}_S1_best_score.pt"
fi

export LOWVIS_RNN_MODE="s2"
case "${ABLATION_MODE}" in
    s1_pretrained|pretrained|s1)
        if [ -z "${S1_CKPT}" ]; then
            echo "ERROR: set S1_CKPT=/public/home/putianshu/vis_mlp/checkpoints/<run_id>_S1_best_score.pt or S1_RUN_ID=<run_id>." >&2
            exit 1
        fi
        if [ ! -f "${S1_CKPT}" ]; then
            echo "ERROR: S1 checkpoint not found: ${S1_CKPT}" >&2
            exit 1
        fi
        export LOWVIS_RNN_RUN_ID="${LOWVIS_RNN_RUN_ID:-exp_${SLURM_JOB_ID}_static_mlp_gru_monthgroup_s1_pretrained}"
        export LOWVIS_RNN_PRETRAINED_CKPT="${S1_CKPT}"
        export LOWVIS_RNN_S2_A_STEPS="${LOWVIS_RNN_S2_A_STEPS:-8000}"
        export LOWVIS_RNN_S2_B_STEPS="${LOWVIS_RNN_S2_B_STEPS:-22000}"
        ;;
    scratch|direct|from_scratch)
        export LOWVIS_RNN_RUN_ID="${LOWVIS_RNN_RUN_ID:-exp_${SLURM_JOB_ID}_static_mlp_gru_monthgroup_scratch}"
        export LOWVIS_RNN_PRETRAINED_CKPT=""
        export LOWVIS_RNN_S2_A_STEPS="${LOWVIS_RNN_S2_A_STEPS:-0}"
        export LOWVIS_RNN_S2_B_STEPS="${LOWVIS_RNN_S2_B_STEPS:-30000}"
        ;;
    *)
        echo "ERROR: unknown ABLATION_MODE=${ABLATION_MODE}; use s1_pretrained or scratch." >&2
        exit 1
        ;;
esac

export LOWVIS_RNN_EXTRA_ARGS="${LOWVIS_RNN_EXTRA_ARGS:-} --s2-data-dir ${MONTHGROUP_S2_DATA_DIR}"

echo "S1 necessity month-group experiment"
echo "ABLATION_MODE=${ABLATION_MODE}"
echo "RUN_ID=${LOWVIS_RNN_RUN_ID}"
echo "MONTHGROUP_S2_DATA_DIR=${MONTHGROUP_S2_DATA_DIR}"
echo "S1_CKPT=${LOWVIS_RNN_PRETRAINED_CKPT:-none}"
echo "S2_A_STEPS=${LOWVIS_RNN_S2_A_STEPS}"
echo "S2_B_STEPS=${LOWVIS_RNN_S2_B_STEPS}"
echo "LOWVIS_RNN_EXTRA_ARGS=${LOWVIS_RNN_EXTRA_ARGS}"

exec bash "${SCRIPT_DIR}/sub_static_rnn_lowvis_main.slurm"
