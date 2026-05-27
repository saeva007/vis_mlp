#!/bin/bash
#SBATCH -J static_rnn_airport_data
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 120:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
OUT_DIR="${AIRPORT_RNN_DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_static_rnn_airport_metar_2025_12h}"

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Queue=${SLURM_JOB_PARTITION:-kshcexclu04} CPUs=${SLURM_CPUS_PER_TASK:-32}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "OUT_DIR=${OUT_DIR}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd "${SCRIPT_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

python build_static_rnn_airport_metar_dataset.py \
    --visibility-file "${AIRPORT_VISIBILITY_FILE:-/public/home/putianshu/vis_mlp/metar_visibility_2025_fixed.nc}" \
    --weather-file "${AIRPORT_WEATHER_FILE:-/public/home/putianshu/vis_mlp/merged_station_weather_data_20250101_20251231.nc}" \
    --station-file "${AIRPORT_STATION_FILE:-/public/home/putianshu/vis_diffusion/test_data/178_stations_renumbered.csv}" \
    --vegetation-file "${AIRPORT_VEG_FILE:-/public/home/putianshu/vis_cnn/data_vegtype.nc}" \
    --orography-file "${AIRPORT_ORO_FILE:-/public/home/putianshu/vis_cnn/data_orography.nc}" \
    --output-dir "${OUT_DIR}" \
    --window-size "${AIRPORT_WINDOW_SIZE:-12}" \
    --step-size "${AIRPORT_STEP_SIZE:-1}" \
    --time-chunk-wins "${AIRPORT_TIME_CHUNK_WINS:-96}" \
    --local-time-offset-hours "${AIRPORT_LOCAL_TIME_OFFSET_HOURS:-8}" \
    --weather-time-shift-hours "${AIRPORT_WEATHER_TIME_SHIFT_HOURS:-0}" \
    --visibility-time-shift-hours "${AIRPORT_VISIBILITY_TIME_SHIFT_HOURS:-0}" \
    --train-ratio "${AIRPORT_TRAIN_RATIO:-0.80}" \
    --val-ratio "${AIRPORT_VAL_RATIO:-0.20}" \
    --test-ratio "${AIRPORT_TEST_RATIO:-0.0}" \
    --gap-hours "${AIRPORT_GAP_HOURS:-24}"

echo "Finished at $(date)"
