#!/bin/bash

# 检查参数
if [ "$#" -ne 2 ]; then
    echo "Usage: bash submit.sh <model_type> <ablation_type>"
    echo "Example: bash submit.sh mlp window"
    exit 1
fi

MODEL=$1
ABLATION=$2

# 自动拼接作业名称，例如: mlp_window
JOB_NAME="${MODEL}_${ABLATION}"

# 提交任务
# -J 覆盖作业名
# -o 和 -e 可以让日志文件名也带上实验名，方便查找！
echo "Submitting job: ${JOB_NAME}..."

sbatch \
    -J ${JOB_NAME} \
    -o logs/${JOB_NAME}_%j.out \
    -e logs/${JOB_NAME}_%j.err \
    sub_Ablation.sh ${MODEL} ${ABLATION} 12