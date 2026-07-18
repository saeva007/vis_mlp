#!/bin/bash
#SBATCH -J lowvis_loss_watch
#SBATCH -p kshcexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 240:00:00
#SBATCH -D /public/home/putianshu/vis_mlp/train
#SBATCH -o /public/home/putianshu/vis_mlp/train/logs/%j_%x.out
#SBATCH -e /public/home/putianshu/vis_mlp/train/logs/%j_%x.err

set -euo pipefail

SCRIPT_DIR="${REPO_ROOT:-/public/home/putianshu/vis_mlp/train}"
PYTHON_BIN="${WATCH_PYTHON:-}"

cd "${SCRIPT_DIR}"
mkdir -p logs

if [ -z "${WATCH_MANIFEST:-}" ] && [ -z "${WATCH_MANIFESTS:-}" ]; then
    echo "ERROR: set WATCH_MANIFEST=<manifest.tsv> or WATCH_MANIFESTS=<manifest1:manifest2>." >&2
    exit 2
fi

if [ -z "${PYTHON_BIN}" ]; then
    WATCH_CONDA_ACTIVATE="${WATCH_CONDA_ACTIVATE:-/public/home/putianshu/miniconda3/bin/activate}"
    WATCH_CONDA_ENV="${WATCH_CONDA_ENV:-torch}"
    if [ ! -f "${WATCH_CONDA_ACTIVATE}" ]; then
        echo "ERROR: conda activation script not found: ${WATCH_CONDA_ACTIVATE}" >&2
        echo "Set WATCH_PYTHON to a Python >=3.10 executable." >&2
        exit 2
    fi
    source "${WATCH_CONDA_ACTIVATE}"
    conda activate "${WATCH_CONDA_ENV}"
    PYTHON_BIN="$(command -v python)"
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: watchdog Python not found: ${PYTHON_BIN}" >&2
    exit 2
fi

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

export PYTHONUNBUFFERED=1

echo "Static-RNN loss watchdog started at: $(date)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "WATCH_MANIFEST=${WATCH_MANIFEST:-}"
echo "WATCH_MANIFESTS=${WATCH_MANIFESTS:-}"
echo "WATCH_COMBINED_RESOLVED_MANIFEST=${WATCH_COMBINED_RESOLVED_MANIFEST:-}"
echo "WATCH_ONLY_ROWS=${WATCH_ONLY_ROWS:-}"
echo "WATCH_AUTO_RETRY=${WATCH_AUTO_RETRY:-0}"
echo "WATCH_POLL_SECONDS=${WATCH_POLL_SECONDS:-300}"
echo "WATCH_STARTUP_STALE_MINUTES=${WATCH_STARTUP_STALE_MINUTES:-120}"
echo "WATCH_DATA_STALE_MINUTES=${WATCH_DATA_STALE_MINUTES:-180}"
echo "WATCH_TRAIN_STALE_MINUTES=${WATCH_TRAIN_STALE_MINUTES:-180}"
echo "WATCH_VALIDATION_STALE_MINUTES=${WATCH_VALIDATION_STALE_MINUTES:-360}"
echo "WATCH_CONFIRMATIONS=${WATCH_CONFIRMATIONS:-2}"
echo "WATCH_MAX_RETRIES=${WATCH_MAX_RETRIES:-2}"
echo "WATCH_CHECKPOINT_GRACE_MINUTES=${WATCH_CHECKPOINT_GRACE_MINUTES:-30}"
echo "WATCH_LOCAL_CACHE_DIR=${WATCH_LOCAL_CACHE_DIR:-/tmp}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/watch_static_rnn_loss_chain.py"
