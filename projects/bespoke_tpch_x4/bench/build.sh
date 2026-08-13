#!/usr/bin/env bash
# Build an engine directory into a standalone benchmark binary.
#
# Usage: bench/build.sh <engine_dir> [<out_binary>]
#
# Compiles harness/main.cpp + the engine's loader/builder/query translation
# units with the paper's optimization flags (-O3 -flto) and links against
# Apache Arrow/Parquet. An engine directory may provide an optional
# `extra_flags.txt` file (one flag per line) with additional compiler flags
# (e.g. -march=native -fopenmp) used by optimized variants.

set -euo pipefail

ENGINE_DIR=$(realpath "$1")
ROOT=$(realpath "$(dirname "$0")/..")
OUT=${2:-"$ENGINE_DIR/db"}

EXTRA_FLAGS=()
if [[ -f "$ENGINE_DIR/extra_flags.txt" ]]; then
    while IFS= read -r line; do
        [[ -n "$line" ]] && EXTRA_FLAGS+=("$line")
    done < "$ENGINE_DIR/extra_flags.txt"
fi

g++ -g -std=c++20 -O3 -flto "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
    -I"$ENGINE_DIR" \
    "$ROOT/harness/main.cpp" \
    "$ENGINE_DIR/loader_impl.cpp" \
    "$ENGINE_DIR/loader_utils.cpp" \
    "$ENGINE_DIR/builder_impl.cpp" \
    "$ENGINE_DIR/query_impl.cpp" \
    -larrow -lparquet \
    -o "$OUT"

echo "built $OUT"
