#!/usr/bin/env bash
# CORRECTNESS run (untimed, separate from scoring): every-key stride-1 read-back + integrity.
# USER-level cgroup (no sudo). Exit 0 = correct; 2 = value/load mismatch; 3 = in-run integrity failure.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
[ -x "$HERE/kvstore_bench" ] || { echo "ERROR: build first (./build.sh)"; exit 1; }
mkdir -p "${KV_SPILL:-/mnt/ssd/kv_spill}" 2>/dev/null || true
rm -rf "${KV_SPILL:-/mnt/ssd/kv_spill}"/* 2>/dev/null
sync; { echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; } || true
systemd-run --user --scope -q -p MemoryMax=15032385536 -p MemorySwapMax=0 \
  env -i PATH=/usr/bin:/bin \
    KVSTORE_VALUE_SIZE=100 KVSTORE_AUDIT=1 KVSTORE_VALIDATE_LOAD_KEYS=1 KVSTORE_VALIDATE_LOAD_STRIDE=1 \
    timeout -k 15 3600 numactl --cpunodebind=0 --membind=0 \
    "$HERE/kvstore_bench" 0 16 "$HERE/traces/load_250M.dat" "$HERE/traces/run_250M.dat" 8589934592 "${KV_SPILL:-/mnt/ssd/kv_spill}"
echo "exit=$?  (0=correct, 2=value/load mismatch, 3=integrity/hack detected)"
