#!/bin/bash
#SBATCH -J pangu_deps
#SBATCH --partition=dcu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -D /data2/share/chenxi/PuTS/mlp
#SBATCH -o /data2/share/chenxi/PuTS/mlp/logs/%j_%x.out
#SBATCH -e /data2/share/chenxi/PuTS/mlp/logs/%j_%x.err

set -euo pipefail

source /lustre/chenxi/yliang/miniforge3/bin/activate
module purge
module load compiler/dtk/23.10
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
source /lustre/chenxi/yliang/work/cast3/env.sh

WORK_DIR="${WORK_DIR:-/data2/share/chenxi/PuTS/mlp}"
PYTHON_PACKAGES="${PYTHON_PACKAGES:-${WORK_DIR}/python_packages}"
mkdir -p "${PYTHON_PACKAGES}"

echo "[setup] python=$(command -v python)"
echo "[setup] target=${PYTHON_PACKAGES}"

# Never let pip replace the DTK/Cast3 builds of torch or torchvision.
python - <<'PY'
import sys
import torch
import torchvision

print(f"[setup] python={sys.executable}")
print(f"[setup] torch={torch.__version__}")
print(f"[setup] torchvision={torchvision.__version__}")
print(f"[setup] torch.cuda.is_available={torch.cuda.is_available()}")
print(f"[setup] torch.cuda.device_count={torch.cuda.device_count()}")
PY

python -m pip install \
    --upgrade \
    --target "${PYTHON_PACKAGES}" \
    --no-deps \
    --only-binary=:all: \
    onnx==1.14.1 \
    onnx2torch==1.5.15

PYTHONPATH="${PYTHON_PACKAGES}:${PYTHONPATH:-}" python - <<'PY'
import onnx
import onnx2torch
import torch
import torchvision

print(f"[setup-ok] onnx={onnx.__version__}")
print(f"[setup-ok] onnx2torch={getattr(onnx2torch, '__version__', 'unknown')}")
print(f"[setup-ok] torch={torch.__version__}")
print(f"[setup-ok] torchvision={torchvision.__version__}")
PY

echo "[setup-ok] Pangu Python dependencies installed under ${PYTHON_PACKAGES}"
