#!/bin/bash
#SBATCH -J audit_lowvis_data
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
BASE="${LOWVIS_AUDIT_BASE:-/public/home/putianshu/vis_mlp}"
OUT_DIR="${LOWVIS_AUDIT_OUT_DIR:-${BASE}/data_audits/lowvis_data_quality_${SLURM_JOB_ID}}"

DATASET_SPECS="${LOWVIS_AUDIT_DATASETS:-main_s1=${BASE}/ml_dataset_pmst_v5_aligned_12h_pm10_pm25;main_s2=${BASE}/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2;monthgroup_s2=${BASE}/ml_dataset_s2_tianji_12h_pm10_pm25_monthgroup_train13579_val11_test24681012;vera_s1=${BASE}/ml_dataset_pmst_v5_aligned_12h_pm10_pm25_vera;vera_s2=${BASE}/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2_vera;eval_48h=${BASE}/ml_dataset_fe_12h_48h_pm10_pm25_testonly_leadtime;trajectory=${BASE}/ml_dataset_s2_tianji_trajectory_1_48h_pm10_pm25_v1}"
RAW_VISIBILITY_SPECS="${LOWVIS_AUDIT_RAW_VISIBILITY:-cma_s1=${BASE}/CMA_visibility_2021_2023_GeoCoords_1.nc;tianji_s2=${BASE}/tianji_auto_station/merged_final_all_vars.nc}"
RAW_PM_SPECS="${LOWVIS_AUDIT_RAW_PM:-pm10_s2=${BASE}/pm10_station/pm10_station_s2_2025.nc;pm25_s2=${BASE}/pm2.5_station/pm2p5_station_s2_2025.nc}"

mkdir -p "${SCRIPT_DIR}/logs" "${OUT_DIR}"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/activate_lowvis_data_runtime.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-32}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"

extra_args=()
if [[ "${LOWVIS_AUDIT_REQUIRE_ALL:-0}" == "1" ]]; then
    extra_args+=(--require-all)
fi
if [[ "${LOWVIS_AUDIT_STRICT:-0}" == "1" ]]; then
    extra_args+=(--strict)
fi
if [[ -n "${LOWVIS_AUDIT_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    user_extra=( ${LOWVIS_AUDIT_EXTRA_ARGS} )
    extra_args+=("${user_extra[@]}")
fi

echo "[audit] output=${OUT_DIR}"
echo "[audit] datasets=${DATASET_SPECS}"
echo "[audit] raw_visibility=${RAW_VISIBILITY_SPECS}"
echo "[audit] raw_pm=${RAW_PM_SPECS}"

python "${SCRIPT_DIR}/audit_lowvis_data_quality.py" \
    --dataset-specs "${DATASET_SPECS}" \
    --raw-visibility-specs "${RAW_VISIBILITY_SPECS}" \
    --raw-pm-specs "${RAW_PM_SPECS}" \
    --trajectory-raw-tag "${LOWVIS_AUDIT_TRAJECTORY_RAW_TAG:-tianji_s2}" \
    --pmst-common-dir "${PMST_COMMON_DIR:-${BASE}/ifs_baseline}" \
    --out-dir "${OUT_DIR}" \
    --chunk-rows "${LOWVIS_AUDIT_CHUNK_ROWS:-4096}" \
    --quantile-rows "${LOWVIS_AUDIT_QUANTILE_ROWS:-5000}" \
    --metadata-chunksize "${LOWVIS_AUDIT_METADATA_CHUNK:-20000}" \
    --netcdf-time-chunk "${LOWVIS_AUDIT_NETCDF_TIME_CHUNK:-168}" \
    --max-trajectory-raw-rows "${LOWVIS_AUDIT_MAX_TRAJECTORY_RAW_ROWS:-0}" \
    "${extra_args[@]}"

echo "[audit] finished=$(date --iso-8601=seconds) output=${OUT_DIR}"
