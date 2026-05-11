#!/bin/bash
#SBATCH -J s2_data_airport
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

echo "JobId=${SLURM_JOB_ID} Node=$(hostname) Started=$(date)"
echo "Queue=${SLURM_JOB_PARTITION:-kshcexclu04} CPUs=${SLURM_CPUS_PER_TASK:-32}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd /public/home/putianshu/vis_mlp/train
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "Running airport METAR S2 data build..."
python s2_data_airport_metar.py \
    --visibility-file /public/home/putianshu/vis_mlp/metar_visibility_2025_fixed.nc \
    --weather-file /public/home/putianshu/vis_mlp/merged_station_weather_data_20250101_20251231.nc \
    --station-file /public/home/putianshu/vis_diffusion/test_data/178_stations_renumbered.csv \
    --vegetation-file /public/home/putianshu/vis_cnn/data_vegtype.nc \
    --orography-file /public/home/putianshu/vis_cnn/data_orography.nc \
    --output-dir /public/home/putianshu/vis_mlp/ml_dataset_airport_metar_2025_12h \
    --window-size 12 \
    --step-size 1 \
    --time-chunk-wins 96 \
    --local-time-offset-hours 8

echo "Finished at $(date)"
