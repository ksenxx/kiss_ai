#!/usr/bin/env bash
# Builds kvstore_bench from the fixed harness + YOUR engine in baseline/*.cc.
# Never uses parent-directory paths. Only include/ is on the include path.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
shopt -s nullglob
SRC=( "$HERE"/baseline/*.cc "$HERE"/baseline/*.cpp )
[ ${#SRC[@]} -gt 0 ] || { echo "ERROR: put your engine (defining create_kvstore()) in baseline/*.cc"; exit 1; }
g++ -O3 -march=native -DNDEBUG -flto=auto -std=c++17 -pthread \
    -I"$HERE/include" \
    "$HERE/harness/benchmark_harness.cc" "${SRC[@]}" \
    -o "$HERE/kvstore_bench"
echo "built: $HERE/kvstore_bench"
