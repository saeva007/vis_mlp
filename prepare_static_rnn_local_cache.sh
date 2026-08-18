#!/bin/bash

# Prepare the node-local data cache used by train_static_rnn_lowvis.py.
#
# The cleanup is deliberately narrow: only user-owned cache files with names
# produced by copy_to_local() are candidates. Valid cache files for the current
# cache id and source paths are retained, so repeated folds/jobs can reuse them.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  prepare_static_rnn_local_cache.sh \
    --cache-dir DIR --cache-id ID [--clean 0|1] [--require-local 0|1] \
    [--reserve-bytes N] SOURCE.npy [SOURCE.npy ...]
EOF
}

die() {
    echo "[Local-Cache] ERROR: $*" >&2
    exit 2
}

cache_dir="${LOWVIS_RNN_LOCAL_CACHE_DIR:-/tmp}"
cache_id="${LOWVIS_RNN_LOCAL_CACHE_ID:-}"
clean_cache="${LOWVIS_RNN_CLEAN_LOCAL_CACHE:-1}"
require_local="${LOWVIS_RNN_REQUIRE_LOCAL_CACHE:-0}"
reserve_bytes="${LOWVIS_RNN_CACHE_RESERVE_BYTES:-524288000}"
source_paths=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cache-dir)
            [ "$#" -ge 2 ] || die "--cache-dir requires a value"
            cache_dir="$2"
            shift 2
            ;;
        --cache-id)
            [ "$#" -ge 2 ] || die "--cache-id requires a value"
            cache_id="$2"
            shift 2
            ;;
        --clean)
            [ "$#" -ge 2 ] || die "--clean requires 0 or 1"
            clean_cache="$2"
            shift 2
            ;;
        --require-local)
            [ "$#" -ge 2 ] || die "--require-local requires 0 or 1"
            require_local="$2"
            shift 2
            ;;
        --reserve-bytes)
            [ "$#" -ge 2 ] || die "--reserve-bytes requires an integer"
            reserve_bytes="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                source_paths+=("$1")
                shift
            done
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            source_paths+=("$1")
            shift
            ;;
    esac
done

[ -n "${cache_id}" ] || die "cache id must not be empty"
[ "${clean_cache}" = "0" ] || [ "${clean_cache}" = "1" ] || die "--clean must be 0 or 1"
[ "${require_local}" = "0" ] || [ "${require_local}" = "1" ] || die "--require-local must be 0 or 1"
case "${reserve_bytes}" in
    ''|*[!0-9]*) die "--reserve-bytes must be a non-negative integer" ;;
esac
[ "${#source_paths[@]}" -gt 0 ] || die "at least one source .npy file is required"

command -v realpath >/dev/null 2>&1 || die "realpath is required"
command -v md5sum >/dev/null 2>&1 || die "md5sum is required"

resolved_cache_dir="$(realpath -m -- "${cache_dir}")"
[ -n "${resolved_cache_dir}" ] || die "cache directory resolved to an empty path"
[ "${resolved_cache_dir}" != "/" ] || die "refusing to operate on filesystem root"
cache_owner="${USER:-$(id -un)}"
[ -n "${cache_owner}" ] || die "could not determine the current user"

mkdir -p -- "${resolved_cache_dir}"
[ -d "${resolved_cache_dir}" ] || die "cache directory is unavailable: ${resolved_cache_dir}"
[ -w "${resolved_cache_dir}" ] || die "cache directory is not writable: ${resolved_cache_dir}"

echo "[Local-Cache] host=$(hostname) dir=${resolved_cache_dir} id=${cache_id} clean=${clean_cache} require_local=${require_local}"
df -h "${resolved_cache_dir}" | tail -1 || true
df -i "${resolved_cache_dir}" | tail -1 || true

preserve_paths=()
missing_bytes=0

for source_path in "${source_paths[@]}"; do
    [ -f "${source_path}" ] || die "missing source file: ${source_path}"
    # copy_to_local() hashes os.path.abspath(src_path), which normalizes '..'
    # without resolving symlinks. GNU realpath -ms has the same behavior.
    absolute_source="$(realpath -ms -- "${source_path}")"
    source_size="$(stat -c '%s' -- "${source_path}")"
    source_base="$(basename -- "${source_path}")"
    source_stem="${source_base%.*}"
    source_ext=".${source_base##*.}"
    file_hash="$(printf '%s' "${cache_id}_${absolute_source}" | md5sum | awk '{print substr($1, 1, 8)}')"
    target_path="${resolved_cache_dir}/${source_stem}_${file_hash}${source_ext}"

    if [ -f "${target_path}" ] && [ "$(stat -c '%s' -- "${target_path}")" = "${source_size}" ]; then
        preserve_paths+=("${target_path}")
        echo "[Local-Cache] keep valid cache: ${target_path}"
    else
        missing_bytes=$((missing_bytes + source_size))
    fi
done

is_preserved_path() {
    local candidate="$1"
    local keep
    if [ "${#preserve_paths[@]}" -eq 0 ]; then
        return 1
    fi
    for keep in "${preserve_paths[@]}"; do
        if [ "${candidate}" = "${keep}" ]; then
            return 0
        fi
    done
    return 1
}

deleted_count=0
if [ "${clean_cache}" = "1" ]; then
    while IFS= read -r -d '' candidate; do
        if is_preserved_path "${candidate}"; then
            continue
        fi
        echo "[Local-Cache] remove stale user-owned cache: ${candidate}"
        rm -f -- "${candidate}"
        deleted_count=$((deleted_count + 1))
    done < <(
        find "${resolved_cache_dir}" -maxdepth 1 -user "${cache_owner}" -type f \
            \( -name 'X_train_*.npy' -o -name 'X_val_*.npy' \
               -o -name 'y_train_*.npy' -o -name 'y_val_*.npy' \
               -o -name 'X_train_*.npy.tmp' -o -name 'X_val_*.npy.tmp' \
               -o -name 'y_train_*.npy.tmp' -o -name 'y_val_*.npy.tmp' \
               -o -name '*.nfs_fallback' \) -print0
    )
fi

# Recompute missing bytes after cleanup so invalid expected targets are counted
# correctly even when cleanup was disabled or could not remove them.
missing_bytes=0
for source_path in "${source_paths[@]}"; do
    absolute_source="$(realpath -ms -- "${source_path}")"
    source_size="$(stat -c '%s' -- "${source_path}")"
    source_base="$(basename -- "${source_path}")"
    source_stem="${source_base%.*}"
    source_ext=".${source_base##*.}"
    file_hash="$(printf '%s' "${cache_id}_${absolute_source}" | md5sum | awk '{print substr($1, 1, 8)}')"
    target_path="${resolved_cache_dir}/${source_stem}_${file_hash}${source_ext}"
    if [ ! -f "${target_path}" ] || [ "$(stat -c '%s' -- "${target_path}")" != "${source_size}" ]; then
        missing_bytes=$((missing_bytes + source_size))
    fi
done

available_bytes="$(df -PB1 "${resolved_cache_dir}" | awk 'NR == 2 {print $4}')"
case "${available_bytes}" in
    ''|*[!0-9]*) die "could not determine available bytes for ${resolved_cache_dir}" ;;
esac

if [ "${missing_bytes}" -gt 0 ]; then
    required_bytes=$((missing_bytes + reserve_bytes))
else
    required_bytes=0
fi

echo "[Local-Cache] deleted=${deleted_count} missing_bytes=${missing_bytes} reserve_bytes=${reserve_bytes} available_bytes=${available_bytes}"
df -h "${resolved_cache_dir}" | tail -1 || true
df -i "${resolved_cache_dir}" | tail -1 || true

if [ "${require_local}" = "1" ] && [ "${available_bytes}" -lt "${required_bytes}" ]; then
    echo "[Local-Cache] ERROR: insufficient node-local space after cleanup; need=${required_bytes} available=${available_bytes}. Refusing slow NFS fallback." >&2
    exit 75
fi

echo "[Local-Cache] preflight=OK"
