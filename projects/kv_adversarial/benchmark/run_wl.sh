#!/usr/bin/env bash
# run_wl.sh <workload_id> <tag> [extra env pairs...] — one full benchmark run
# with the exact scored protocol (user cgroup 14GiB/no-swap, numactl node0,
# 16 threads, 8GiB budget, AUDIT=1), log to wl_<tag>.log.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WL="$1"; TAG="$2"; shift 2
BIN="$HERE/kvstore_bench"
LOAD="${KV_LOAD:-$HERE/traces/load_250M.dat}"; RUN="${KV_RUN:-$HERE/traces/run_250M.dat}"
SPILL="${KV_SPILL:-/mnt/ssd/ksen/kv_spill_wl}"
BUDGET="${KV_BUDGET:-8589934592}"
CGMAX=15032385536
TMO=1800
THREADS="${KV_THREADS:-16}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
mkdir -p "$SPILL"; rm -rf "$SPILL"/* 2>/dev/null
sync
log="$HERE/wl_${TAG}.log"
systemd-run --user --scope -q -p MemoryMax=$CGMAX -p MemorySwapMax=0 \
  env -i PATH=/usr/bin:/bin \
    KVSTORE_VALUE_SIZE=100 KVSTORE_ASYNC_EVAL=1 KVSTORE_PIPELINE_DEPTH=256 \
    KVSTORE_COMPLETE_PENDING_INTERVAL=64 KVSTORE_AUDIT=1 KVSTORE_VALIDATE_LOAD_KEYS=0 \
    KVSTORE_BIMODAL_VALUES=0 KVSTORE_BENCH_FAST_EXIT=0 "$@" \
    timeout -k 15 $TMO numactl --cpunodebind=0 --membind=0 \
    "$BIN" "$WL" "$THREADS" "$LOAD" "$RUN" $BUDGET "$SPILL" >"$log" 2>&1
rc=$?
m=$(grep -oE "Total: [0-9.]+ Mops/s" "$log" | tail -1 | grep -oE "[0-9.]+")
rss_kb=$(grep -oE "StoreRSS:[[:space:]]*[0-9]+" "$log" | grep -oE "[0-9]+$" | tail -1)
echo "WL=$WL TAG=$TAG rc=$rc Mops=${m:-none} StoreRSS_kB=${rss_kb:-?}"
grep -iE "integrity|audit|fail|error|corrupt|mismatch|lost|WARN" "$log" | head -20
