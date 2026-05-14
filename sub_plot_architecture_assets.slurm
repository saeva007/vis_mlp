#!/bin/bash
#SBATCH -J arch_map_assets
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 08:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err
#SBATCH --exclusive
#
# CPU/内存型绘图任务：读取 S2 test set + China shp，生成架构图用站点地图堆叠 PNG。
#
# 提交:
#   mkdir -p /public/home/putianshu/vis_mlp/train/logs
#   sbatch /public/home/putianshu/vis_mlp/train/sub_plot_architecture_assets.slurm
#
# 可选环境变量（sbatch 前 export 或 --export=ALL,VAR=val）:
#   DATA_DIR=/path/to/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2
#   OUT_DIR=/path/to/architecture_map_assets
#   DYNAMIC_VARS=RH2M,T2M,PM10,PM2.5
#   FE_VARS=near_saturation_proxy,night_clear_sky_cooling,fog_potential_index,ventilation_proxy
#   VALID_TIME="2025-10-31 15:00"
#   OUTPUT_TIMES=4
#   EVAL_CSV=/public/home/putianshu/vis_mlp/paper_eval_results_pm10_pm25_journal/per_sample_eval.csv
#   MAX_POINTS=0
#
# 示例:
#   sbatch --export=ALL,DYNAMIC_VARS=RH2M,T2M,PM10,PM2.5,OUTPUT_TIMES=6 \
#     /public/home/putianshu/vis_mlp/train/sub_plot_architecture_assets.slurm
#
set -euo pipefail

echo "JobId=${SLURM_JOB_ID}  Node=$(hostname)  Started=$(date)"
echo "Queue=${SLURM_JOB_PARTITION:-kshcexclu04} CPUs=${SLURM_CPUS_PER_TASK:-32}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

source /public/home/putianshu/miniconda3/bin/activate
conda activate torch

cd /public/home/putianshu/vis_mlp/train

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

DATA_DIR="${DATA_DIR:-/public/home/putianshu/vis_mlp/ml_dataset_s2_tianji_12h_pm10_pm25_monthtail_2}"
SHP_PATH="${SHP_PATH:-/public/home/putianshu/中华人民共和国/中华人民共和国.shp}"
ORO_PATH="${ORO_PATH:-/public/home/putianshu/vis_cnn/data_orography.nc}"
OUT_DIR="${OUT_DIR:-/public/home/putianshu/vis_mlp/architecture_map_assets}"
DYNAMIC_VARS="${DYNAMIC_VARS:-RH2M,T2M,PM10}"
FE_VARS="${FE_VARS:-near_saturation_proxy,night_clear_sky_cooling,fog_potential_index,ventilation_proxy}"
VALID_TIME="${VALID_TIME:-}"
OUTPUT_TIMES="${OUTPUT_TIMES:-4}"
EVAL_CSV="${EVAL_CSV:-}"
MAX_POINTS="${MAX_POINTS:-0}"
DPI="${DPI:-220}"

args=(
  --data_dir "${DATA_DIR}"
  --shp "${SHP_PATH}"
  --oro "${ORO_PATH}"
  --out_dir "${OUT_DIR}"
  --dynamic_vars "${DYNAMIC_VARS}"
  --fe_vars "${FE_VARS}"
  --output_times "${OUTPUT_TIMES}"
  --max_points "${MAX_POINTS}"
  --dpi "${DPI}"
)

[ -n "${VALID_TIME}" ] && args+=(--time "${VALID_TIME}")
[ -n "${EVAL_CSV}" ] && args+=(--eval_csv "${EVAL_CSV}")

echo "Running architecture map asset generation..."
echo "args: ${args[*]}"
python plot_architecture_station_map_assets.py "${args[@]}"

echo "Finished at $(date)"
echo "Output: ${OUT_DIR}"
