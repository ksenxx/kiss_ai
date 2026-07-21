#!/usr/bin/env bash
# SCORED run: median of 3 x (best-effort drop_caches -> USER cgroup 14GiB/no-swap -> numactl node0 -> 30s window).
# Engine data budget = 8 GiB (arg) + ENFORCED StoreRSS <= 8 GiB (over-budget => score 0).
# Uses a USER-level systemd scope (no sudo needed). timeout wraps the binary (hang => score 0).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/kvstore_bench"; LOAD="$HERE/traces/load_250M.dat"; RUN="$HERE/traces/run_250M.dat"; SPILL="${KV_SPILL:-/mnt/ssd/kv_spill}"
BUDGET=8589934592          # 8 GiB engine data budget (ENFORCED via StoreRSS below)
mkdir -p "$SPILL"
CGMAX=15032385536          # 14 GiB cgroup ceiling; no swap
TMO=1800                   # per-run wall limit (s); true hang => score 0
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
[ -x "$BIN" ] || { echo "ERROR: build first (./build.sh)"; exit 1; }
systemctl --user show-environment >/dev/null 2>&1 || { echo "ERROR: user systemd bus down (need: loginctl enable-linger \$USER; systemctl --user status)"; exit 1; }
res=()
for i in 1 2 3; do
  rm -rf "$SPILL"/* 2>/dev/null
  sync; { echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; } || true   # best-effort (needs root; skipped for non-root)
  log="$HERE/run_$i.log"
  systemd-run --user --scope -q -p MemoryMax=$CGMAX -p MemorySwapMax=0 \
    env -i PATH=/usr/bin:/bin \
      KVSTORE_VALUE_SIZE=100 KVSTORE_ASYNC_EVAL=1 KVSTORE_PIPELINE_DEPTH=256 \
      KVSTORE_COMPLETE_PENDING_INTERVAL=64 KVSTORE_AUDIT=1 KVSTORE_VALIDATE_LOAD_KEYS=0 \
      KVSTORE_BIMODAL_VALUES=0 KVSTORE_BENCH_FAST_EXIT=0 \
      timeout -k 15 $TMO numactl --cpunodebind=0 --membind=0 \
      "$BIN" 0 16 "$LOAD" "$RUN" $BUDGET "$SPILL" >"$log" 2>&1
  rc=$?
  m=$(grep -oE "Total: [0-9.]+ Mops/s" "$log" | tail -1 | grep -oE "[0-9.]+")
  rss_kb=$(grep -oE "StoreRSS:[[:space:]]*[0-9]+" "$log" | grep -oE "[0-9]+$" | tail -1)
  status="ok"
  if [ "$rc" -ne 0 ]; then m=0; status="FAILED rc=$rc (nonzero/timeout/OOM)"; fi
  if [ -n "${rss_kb:-}" ] && [ $(( rss_kb * 1024 )) -gt "$BUDGET" ]; then
    m=0; status="REJECTED over-budget StoreRSS=$(( rss_kb/1048576 ))GiB > 8GiB"
  fi
  echo "run $i: ${m:-0} Mops/s   (StoreRSS=${rss_kb:-?} kB; $status)"
  res+=("${m:-0}")
done
echo "MEDIAN: $(printf "%s\n" "${res[@]}" | sort -n | sed -n "2p") Mops/s   (target: beat FASTER 0.93; StoreRSS<=8GiB enforced)"
