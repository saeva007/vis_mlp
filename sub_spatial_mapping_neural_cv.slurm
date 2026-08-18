#!/bin/bash
#SBATCH -J spcv_neural
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=0
#SBATCH -t 120:00:00
#SBATCH --array=0-9%2
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%A_%a_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%A_%a_%x.err
#SBATCH --exclusive

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
DATA_DIR="${DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
RESULT_ROOT="${RESULT_ROOT:?set RESULT_ROOT}"
BUNDLE_ID="${BUNDLE_ID:?set BUNDLE_ID}"
MODEL_SEED="${MODEL_SEED:-20260815}"
MAPPING_DECISION_RULE="${MAPPING_DECISION_RULE:-argmax}"
CV_PREFIX="${CV_PREFIX:-spcv}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
MAPPING_NEURAL_MODEL="${MAPPING_NEURAL_MODEL:-}"

if [ -n "${MAPPING_NEURAL_MODEL}" ]; then
    case "${MAPPING_NEURAL_MODEL}" in
        mlp|gru) ;;
        *) echo "[preflight] invalid MAPPING_NEURAL_MODEL=${MAPPING_NEURAL_MODEL}" >&2; exit 2 ;;
    esac
    if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -gt 4 ]; then
        echo "[preflight] single-model array task must be in [0,4], got ${TASK_ID}" >&2
        exit 2
    fi
    MODEL="${MAPPING_NEURAL_MODEL}"
    FOLD="${TASK_ID}"
elif [ "${TASK_ID}" -lt 5 ]; then
    MODEL=mlp
    FOLD="${TASK_ID}"
elif [ "${TASK_ID}" -lt 10 ]; then
    MODEL=gru
    FOLD="$((TASK_ID - 5))"
else
    echo "[preflight] combined neural array task must be in [0,9], got ${TASK_ID}" >&2
    exit 2
fi

RUN_ID="${CV_PREFIX}_${BUNDLE_ID}_${MODEL}_fold${FOLD}"
OUTPUT_DIR="${RESULT_ROOT}/results/${MODEL}/fold_${FOLD}"
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
FOLD_DIR="${RESULT_ROOT}/folds/fold_${FOLD}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"

case "${MAPPING_DECISION_RULE}" in
    argmax|val_search) ;;
    *) echo "[preflight] invalid MAPPING_DECISION_RULE=${MAPPING_DECISION_RULE}" >&2; exit 2 ;;
esac

export LOWVIS_RNN_LOCAL_CACHE_ID="${LOWVIS_RNN_LOCAL_CACHE_ID:-mapping_cv_s2_shared_data}"
export LOWVIS_RNN_LOCAL_CACHE_DIR="${LOWVIS_RNN_LOCAL_CACHE_DIR:-/tmp}"
export LOWVIS_RNN_CLEAN_LOCAL_CACHE="${LOWVIS_RNN_CLEAN_LOCAL_CACHE:-1}"
export LOWVIS_RNN_REQUIRE_LOCAL_CACHE="${LOWVIS_RNN_REQUIRE_LOCAL_CACHE:-1}"

export LOWVIS_DIFFUSION_USE_DCU=1
source "${REPO_ROOT}/activate_lowvis_diffusion_runtime.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

mkdir -p "${REPO_ROOT}/logs" "${CKPT_DIR}"
cd "${REPO_ROOT}"

test -s "${FOLD_DIR}/s2_train_indices.npy"
test -s "${FOLD_DIR}/s2_val_indices.npy"
test -s "${FOLD_DIR}/s2_test_indices.npy"
echo "[preflight] model=${MODEL} fold=${FOLD} decision=${MAPPING_DECISION_RULE} run_id=${RUN_ID} indices=OK"

echo "[preflight] preparing node-local Static-RNN cache"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 \
    /bin/bash "${REPO_ROOT}/prepare_static_rnn_local_cache.sh" \
    --cache-dir "${LOWVIS_RNN_LOCAL_CACHE_DIR}" \
    --cache-id "${LOWVIS_RNN_LOCAL_CACHE_ID}" \
    --clean "${LOWVIS_RNN_CLEAN_LOCAL_CACHE}" \
    --require-local "${LOWVIS_RNN_REQUIRE_LOCAL_CACHE}" \
    "${DATA_DIR}/X_train.npy" \
    "${DATA_DIR}/y_train.npy" \
    "${DATA_DIR}/X_val.npy" \
    "${DATA_DIR}/y_val.npy"

srun --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" bash -c "
    set -euo pipefail
    cd \"${REPO_ROOT}\"
    export MIOPEN_USER_DB_PATH=/tmp/miopen-\${USER}-\${SLURM_JOB_ID}
    mkdir -p \"\${MIOPEN_USER_DB_PATH}\"
    torchrun --standalone --nproc_per_node=4 --max_restarts=0 \
        \"${REPO_ROOT}/train_static_rnn_lowvis.py\" \
        --mode s2 \
        --s2-from-scratch \
        --encoder \"${MODEL}\" \
        --seed \"${MODEL_SEED}\" \
        --run-id \"${RUN_ID}\" \
        --local-cache-id \"${LOWVIS_RNN_LOCAL_CACHE_ID}\" \
        --s2-data-dir \"${DATA_DIR}\" \
        --row-index-dir \"${FOLD_DIR}\" \
        --ckpt-dir \"${CKPT_DIR}\" \
        --no-fe \
        --pooling mean \
        --rnn-layers 1 \
        --hidden-dim 256 \
        --static-hidden-dim 96 \
        --fusion-hidden-dim 256 \
        --dropout 0.2 \
        --loss-mode ce \
        --sampler-mode stratified_balanced \
        --fog-ratio-s2 0.18 \
        --mist-ratio-s2 0.22 \
        --s2-phase-a-steps 0 \
        --s2-phase-b-steps \"${NEURAL_STEPS:-22000}\" \
        --s2-phase-c-steps 0 \
        --s2-phase-d-steps 0 \
        --s2-from-scratch-lr \"${NEURAL_LR:-2e-4}\" \
        --val-interval \"${VAL_INTERVAL:-500}\" \
        --batch-size \"${BATCH_SIZE:-512}\" \
        --grad-accum \"${GRAD_ACCUM:-2}\" \
        --num-workers 0 \
        --patience \"${PATIENCE:-10}\" \
        --selection-metric recall_csi \
        --threshold-mode \"${MAPPING_DECISION_RULE}\" \
        ${NEURAL_EXTRA_ARGS:-}
"

CHECKPOINT="${CKPT_DIR}/${RUN_ID}_S2_FromScratch_best_score.pt"
RUN_CONFIG="${CKPT_DIR}/${RUN_ID}_static_rnn_config.json"
SCALER=$(find "${CKPT_DIR}" -maxdepth 1 -type f -name "robust_scaler_${RUN_ID}_s2_*.pkl" -print -quit)
test -s "${CHECKPOINT}"
test -s "${RUN_CONFIG}"
test -n "${SCALER}"
test -s "${SCALER}"
echo "[preflight] checkpoint/scaler/config=OK; starting frozen-test evaluation"

python -u "${REPO_ROOT}/spatial_mapping_cv.py" evaluate-neural \
    --model "${MODEL}" \
    --fold "${FOLD}" \
    --data-dir "${DATA_DIR}" \
    --fold-dir "${FOLD_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --run-config "${RUN_CONFIG}" \
    --scaler "${SCALER}" \
    --decision-rule "${MAPPING_DECISION_RULE}" \
    --batch-size 4096 \
    --device cuda:0

echo "[done] model=${MODEL} fold=${FOLD} at $(date)"
