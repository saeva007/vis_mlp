#!/bin/bash
set -euo pipefail

# Formal IFS-input transfer experiment:
#   - balanced spatial blocked five-fold CV
#   - calendar-blocked temporal five-fold CV with embargo
#   - mainline Static-MLP + GRU only
#   - argmax for checkpoint selection and frozen-test classification
#   - native IFS diagnostic visibility retained as the matched reference

# Keep this dedicated IFS chain anchored to the checkout that contains this
# launcher.  A leaked REPO_ROOT (for example from an older frozen checkout)
# must not redirect the submitted jobs to stale workflow scripts.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${IFS_CV_REPO_ROOT:-${SCRIPT_DIR}}"
DATA_DIR="${DATA_DIR:-/public/home/putianshu/vis_mlp/ifs_baseline/ml_dataset_overlap_ifs_12h_pm10_pm25_source_full}"
RUN_ID="${RUN_ID:-mapping_cv_ifs_gru_argmax_$(date +%Y%m%d_%H%M%S)}"
SPATIAL_BUNDLE_ID="${SPATIAL_BUNDLE_ID:-${RUN_ID}_spatial}"
TEMPORAL_BUNDLE_ID="${TEMPORAL_BUNDLE_ID:-${RUN_ID}_temporal}"
SPATIAL_RESULT_ROOT="${SPATIAL_RESULT_ROOT:-/public/home/putianshu/vis_mlp/spatial_mapping_cv_ifs/${SPATIAL_BUNDLE_ID}}"
TEMPORAL_RESULT_ROOT="${TEMPORAL_RESULT_ROOT:-/public/home/putianshu/vis_mlp/temporal_mapping_cv_ifs/${TEMPORAL_BUNDLE_ID}}"
IFS_PER_SAMPLE_CSV="${IFS_PER_SAMPLE_CSV:-/public/home/putianshu/vis_mlp/static_rnn_eval_results/p13_seed_mean_timefix_20260719_130856_paper_figures/exp_20260718_232510_p13_sampling_calibration_manual_retry_p13_seed42_2_proposed_rare_event_focal/per_sample_eval.csv}"

export REPO_ROOT DATA_DIR RUN_ID SPATIAL_BUNDLE_ID TEMPORAL_BUNDLE_ID
export SPATIAL_RESULT_ROOT TEMPORAL_RESULT_ROOT IFS_PER_SAMPLE_CSV
export MAPPING_MODELS=gru
export MAPPING_DECISION_RULE=argmax
export MAPPING_P13="${MAPPING_P13:-0}"
export MAPPING_CV_SEEDS="${MAPPING_CV_SEEDS:-}"
export MAPPING_LOGISTIC_CONCURRENCY="${MAPPING_LOGISTIC_CONCURRENCY:-1}"
export MAPPING_NEURAL_CONCURRENCY="${MAPPING_NEURAL_CONCURRENCY:-5}"
export LOWVIS_RNN_LOCAL_CACHE_DIR="${LOWVIS_RNN_LOCAL_CACHE_DIR:-/dev/shm}"
export LOWVIS_RNN_LOCAL_CACHE_ID="${LOWVIS_RNN_LOCAL_CACHE_ID:-mapping_cv_ifs_source_full}"
export LOWVIS_RNN_CLEAN_LOCAL_CACHE="${LOWVIS_RNN_CLEAN_LOCAL_CACHE:-1}"
export LOWVIS_RNN_REQUIRE_LOCAL_CACHE="${LOWVIS_RNN_REQUIRE_LOCAL_CACHE:-1}"
export FIGURE_STEM="${FIGURE_STEM:-ifs_gru_spatiotemporal_cv}"
export MAPPING_GRU_LABEL="${MAPPING_GRU_LABEL:-IFS-input Static-MLP + GRU (12 h)}"

echo "[ifs-gru] data=${DATA_DIR}"
echo "[ifs-gru] models=${MAPPING_MODELS} decision=${MAPPING_DECISION_RULE}"
echo "[ifs-gru] p13=${MAPPING_P13} cv_seeds=${MAPPING_CV_SEEDS:-single_seed}"
echo "[ifs-gru] spatial_result=${SPATIAL_RESULT_ROOT}"
echo "[ifs-gru] temporal_result=${TEMPORAL_RESULT_ROOT}"
echo "[ifs-gru] local_cache=${LOWVIS_RNN_LOCAL_CACHE_DIR}/${LOWVIS_RNN_LOCAL_CACHE_ID}"
echo "[ifs-gru] native_ifs_baseline=${IFS_PER_SAMPLE_CSV}"

# Submit the complete DAG in this process.  The shared dispatcher already
# assigns afterok dependencies (prepare -> fold array -> aggregate -> plot),
# so there is no need for a detached local worker to submit later stages.
exec bash "${REPO_ROOT}/submit_mapping_cv_argmax_full_chain.sh" --worker "$@"
