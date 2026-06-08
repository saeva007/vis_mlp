#!/bin/bash
#SBATCH -J airport_raw_qc
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
OUT_DIR="${AIRPORT_QC_OUT_DIR:-/public/home/putianshu/vis_mlp/airport_raw_qc_$(date +%Y%m%d_%H%M%S)}"

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Queue=${SLURM_JOB_PARTITION:-kshcexclu04} CPUs=${SLURM_CPUS_PER_TASK:-16}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "OUT_DIR=${OUT_DIR}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd "${SCRIPT_DIR}"
mkdir -p logs "${OUT_DIR}"

args=(--output-dir "${OUT_DIR}")

if [ -n "${AIRPORT_WEATHER_FILE:-}" ]; then
    args+=(--weather-file "${AIRPORT_WEATHER_FILE}")
fi
if [ -n "${AIRPORT_VISIBILITY_FILE:-}" ]; then
    args+=(--visibility-file "${AIRPORT_VISIBILITY_FILE}")
fi
if [ -n "${AIRPORT_STATION_FILE:-}" ]; then
    args+=(--station-file "${AIRPORT_STATION_FILE}")
fi
if [ -n "${AIRPORT_VISIBILITY_VAR:-}" ]; then
    args+=(--visibility-var "${AIRPORT_VISIBILITY_VAR}")
fi

python check_airport_metar_raw_data.py "${args[@]}"

echo "Finished at $(date)"
