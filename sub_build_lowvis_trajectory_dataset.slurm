#!/bin/bash
#SBATCH -J build_lowvis_traj
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err

set -euo pipefail
SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
BASE="${LOWVIS_TRAJ_BASE:-/public/home/putianshu/vis_mlp}"
DATA_DIR="${LOWVIS_TRAJ_DATA_DIR:-${BASE}/ml_dataset_s2_tianji_trajectory_0_48h_pm10_pm25_v1}"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/activate_lowvis_diffusion_runtime.sh"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-32}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

python "${SCRIPT_DIR}/build_lowvis_trajectory_dataset.py" \
    --base "${BASE}" \
    --current-48h-dir "${CURRENT_48H_DIR:-${BASE}/tianji_auto_station/current_48h}" \
    --visibility-nc "${VIS_SOURCE_NC:-${BASE}/tianji_auto_station/merged_final_all_vars.nc}" \
    --pm10-file "${PM10_S2_FILE:-${BASE}/pm10_station/pm10_station_s2_2025.nc}" \
    --pm25-file "${PM25_S2_FILE:-${BASE}/pm2.5_station/pm2p5_station_s2_2025.nc}" \
    --pmst-common-dir "${PMST_COMMON_DIR:-${BASE}/ifs_baseline}" \
    --out-dir "${DATA_DIR}" \
    ${LOWVIS_TRAJ_BUILD_EXTRA_ARGS:-}

