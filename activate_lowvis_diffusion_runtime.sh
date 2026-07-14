#!/bin/bash
# Directly activate the tracked low-vis trajectory runtime without invoking the
# cluster Conda plugin loader.  Set LOWVIS_DIFFUSION_USE_DCU=1 for training.

set -euo pipefail

if [[ "${LOWVIS_DIFFUSION_USE_DCU:-0}" == "1" ]]; then
    module purge
    module load compiler/devtoolset/7.3.1
    module load mpi/hpcx/2.11.0/gcc-7.3.1
    source /public/home/xichen/ncydata/dtk/dtk-24.04.1/env.sh
    export LD_LIBRARY_PATH="/public/home/xichen/ncydata/dtk/dtk-24.04.1/.hyhal/lib:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="/public/home/xichen/ncydata/panpy_test_liud/hipnn/lib/release:${LD_LIBRARY_PATH}"
fi

# The Jarvis runtime is the validated environment for both CPU dataset builds
# and DCU training.  Keep the OpenSSL 1.1 compatibility directory available on
# CPU nodes too; torch imports before any DCU-specific code is reached.
TORCH_ENV="${LOWVIS_DIFFUSION_TORCH_ENV:-/public/home/jarvis226/miniconda3/envs/torch}"
OPENSSL_COMPAT_LIB="${LOWVIS_DIFFUSION_OPENSSL_LIB:-/public/home/xichen/.conda/envs/py310_ppy/openssl/lib}"
if [[ ! -x "${TORCH_ENV}/bin/python" ]]; then
    echo "ERROR: trajectory runtime Python is missing: ${TORCH_ENV}/bin/python" >&2
    return 2 2>/dev/null || exit 2
fi
if [[ -d "${OPENSSL_COMPAT_LIB}" ]]; then
    export LD_LIBRARY_PATH="${OPENSSL_COMPAT_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PATH="${TORCH_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${TORCH_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CONDA_PREFIX="${TORCH_ENV}"
export CONDA_DEFAULT_ENV="torch"
export CONDA_NO_PLUGINS="true"
export CONDA_SOLVER="classic"

actual_python="$(command -v python)"
if [[ "${actual_python}" != "${TORCH_ENV}/bin/python" ]]; then
    echo "ERROR: expected ${TORCH_ENV}/bin/python, got ${actual_python}" >&2
    return 2 2>/dev/null || exit 2
fi
python -c "import ssl, sys, torch; print('[trajectory-env] python=' + sys.executable); print('[trajectory-env] torch=' + torch.__version__); print('[trajectory-env] openssl=' + ssl.OPENSSL_VERSION)"
