#!/bin/bash
#
# Submit only the plain focal-loss ablation as S1 -> S2 dependent jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export LOWVIS_RNN_EXPERIMENTS="${LOWVIS_RNN_EXPERIMENTS:-3}"
export LOWVIS_RNN_RUN_PREFIX="${LOWVIS_RNN_RUN_PREFIX:-exp_$(date +%Y%m%d_%H%M%S)_static_rnn_plain_focal}"

if [ "${LOWVIS_RNN_EXPERIMENTS}" != "3" ]; then
    echo "ERROR: this launcher is intentionally scoped to LOWVIS_RNN_EXPERIMENTS=3." >&2
    echo "Use submit_static_rnn_lowvis_loss_matrix_chain.sh for multi-experiment runs." >&2
    exit 2
fi

echo "Submitting only the plain focal-loss experiment"
echo "LOWVIS_RNN_RUN_PREFIX=${LOWVIS_RNN_RUN_PREFIX}"
echo "LOWVIS_RNN_EXPERIMENTS=${LOWVIS_RNN_EXPERIMENTS}"

bash submit_static_rnn_lowvis_loss_matrix_chain.sh
