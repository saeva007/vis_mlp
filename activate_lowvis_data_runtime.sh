#!/bin/bash
# Torch-free runtime for CPU trajectory dataset construction.

set -euo pipefail

DATA_ENV="${LOWVIS_TRAJ_DATA_ENV:-/public/home/putianshu/miniconda3/envs/torch}"
OPENSSL_COMPAT_LIB="${LOWVIS_TRAJ_DATA_OPENSSL_LIB:-/public/home/xichen/.conda/envs/py310_ppy/openssl/lib}"
if [[ ! -x "${DATA_ENV}/bin/python" ]]; then
    echo "ERROR: trajectory data Python is missing: ${DATA_ENV}/bin/python" >&2
    return 2 2>/dev/null || exit 2
fi
if [[ ! -e "${OPENSSL_COMPAT_LIB}/libssl.so.1.1" ]]; then
    echo "ERROR: required OpenSSL compatibility library is missing: ${OPENSSL_COMPAT_LIB}/libssl.so.1.1" >&2
    return 2 2>/dev/null || exit 2
fi

# A data build needs NumPy/xarray/NetCDF only.  Discard inherited DTK/HIP/MKL
# paths so a CPU job can never bind Torch or accelerator libraries by accident.
unset LD_PRELOAD
export PATH="${DATA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${DATA_ENV}/lib:${OPENSSL_COMPAT_LIB}"
export CONDA_PREFIX="${DATA_ENV}"
export CONDA_DEFAULT_ENV="torch"
export CONDA_NO_PLUGINS="true"
export CONDA_SOLVER="classic"

actual_python="$(command -v python)"
if [[ "${actual_python}" != "${DATA_ENV}/bin/python" ]]; then
    echo "ERROR: expected ${DATA_ENV}/bin/python, got ${actual_python}" >&2
    return 2 2>/dev/null || exit 2
fi
python -c "import ssl, sys; import netCDF4, numpy, pandas, pvlib, xarray; print('[trajectory-data-env] python=' + sys.executable); print('[trajectory-data-env] openssl=' + ssl.OPENSSL_VERSION); print('[trajectory-data-env] imports=ok; torch_not_imported=' + str('torch' not in sys.modules))"
