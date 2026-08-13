#!/usr/bin/env bash
# Build the optimized engine with -DBESPOKE_AUDIT so every fast-path /
# fallback branch prints an "AUDIT <marker>" line to stderr. Mirrors
# bench/build.sh exactly (same flags, same translation units) plus the
# audit define. Also builds the pristine baseline engine for differential
# comparison.
#
# Usage: tests/build_audit.sh [<out_dir>]   (default out_dir: tests/bin)
set -euo pipefail

HERE=$(realpath "$(dirname "$0")")
ROOT=$(realpath "$HERE/..")
OUT_DIR=${1:-"$HERE/bin"}
mkdir -p "$OUT_DIR"

build_one() {
    local engine_dir=$1 out=$2; shift 2
    local extra=()
    if [[ -f "$engine_dir/extra_flags.txt" ]]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && extra+=("$line")
        done < "$engine_dir/extra_flags.txt"
    fi
    g++ -g -std=c++20 -O3 -flto "${extra[@]+"${extra[@]}"}" "$@" \
        -I"$engine_dir" \
        "$ROOT/harness/main.cpp" \
        "$engine_dir/loader_impl.cpp" \
        "$engine_dir/loader_utils.cpp" \
        "$engine_dir/builder_impl.cpp" \
        "$engine_dir/query_impl.cpp" \
        -larrow -lparquet \
        -o "$out"
    echo "built $out"
}

build_one "$ROOT/engine" "$OUT_DIR/db_audit" -DBESPOKE_AUDIT
build_one "$ROOT/engine_baseline" "$OUT_DIR/db_baseline"
