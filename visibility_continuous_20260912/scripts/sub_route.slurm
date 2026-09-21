#!/bin/bash
#SBATCH -J vc_route
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --mem=64G
#SBATCH -t 240:00:00
#SBATCH -o /public/home/putianshu/vis_mlp/visibility_continuous_20260912/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/visibility_continuous_20260912/logs/%j_%x.err
set -euo pipefail
ROUTE=${1:?R1, R2, R3, or R4 required}
ROOT=/public/home/putianshu/vis_mlp/visibility_continuous_20260912
source ${ROOT}/scripts/runtime.sh
SOURCE_DATA=/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2
LOCAL_X=/tmp/viscast_continuous_${SLURM_JOB_ID}
test "${LOCAL_X}" = "/tmp/viscast_continuous_${SLURM_JOB_ID}"
mkdir -p ${LOCAL_X}
trap 'test "${LOCAL_X}" = "/tmp/viscast_continuous_${SLURM_JOB_ID}" && rm -rf -- "${LOCAL_X}"' EXIT
for split in train val test; do
  cp ${SOURCE_DATA}/X_${split}.npy ${LOCAL_X}/X_${split}.npy
done
export VISCAST_X_DIR=${LOCAL_X}
python -u ${ROOT}/scripts/viscast_continuous.py train --route ${ROUTE}
python -u ${ROOT}/scripts/viscast_continuous.py eval --route ${ROUTE}
