#!/bin/bash
#SBATCH -J pm10_IDW
#SBATCH -p kshdexclu01
#SBATCH -N 1
#SBATCH -n 1


source /public/home/putianshu/miniconda3/bin/activate
conda activate torch
#cd /public/home/putianshu/vis_mlp

python /public/home/putianshu/vis_mlp/pm10_station_IDW.py --both