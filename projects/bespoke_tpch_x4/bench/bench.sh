#!/usr/bin/env bash
# One-shot benchmark: build an engine dir, run it, validate, print JSON.
#
# Usage: bench/bench.sh <engine_dir> <sf_label> [<seed>] [<reps>] [extra run_bench args...]
#
# The parquet data is expected at $TPCH_DATA_DIR/sf<sf_label>/ (default
# TPCH_DATA_DIR: <repo_root>/tmp/data/tpch_parquet).

set -euo pipefail

ENGINE_DIR=$1
SF=$2
SEED=${3:-42}
REPS=${4:-3}
shift $(( $# > 4 ? 4 : $# ))

HERE=$(realpath "$(dirname "$0")")
DATA_DIR=${TPCH_DATA_DIR:-"$(realpath "$HERE/../../..")/tmp/data/tpch_parquet"}

"$HERE/build.sh" "$ENGINE_DIR" >&2
uv run --no-project --with duckdb --with pandas python3 \
    "$HERE/run_bench.py" "$ENGINE_DIR/db" "$DATA_DIR/sf$SF" "$SF" "$SEED" \
    --reps "$REPS" "$@"
