#!/bin/bash
# Build + run the full HydraKV e2e test matrix (opt / buffered-fd / no-uring /
# ASAN+UBSAN / TSAN / coverage). Exit prints per-phase status lines.
# Defaults match this directory's layout (hydra.cc + kvstore_interface.h here).
# On the benchmark server's task/tests/ layout, override with:
#   HYDRA_INCLUDE_DIR=../include HYDRA_ENGINE=../baseline/hydra.cc HYDRA_TEST_DIR=<fast-disk-dir> ./run_all_tests.sh
cd "$(dirname "$0")" || exit 1
set -o pipefail
I=${HYDRA_INCLUDE_DIR:-.}; E=${HYDRA_ENGINE:-hydra.cc}; T=test_hydra.cc
P=${HYDRA_TEST_DIR:-${TMPDIR:-/tmp}/hydra_tests}
S=${HYDRA_SHM_DIR:-/dev/shm/hydra_tests}
mkdir -p "$P" || exit 1
RC=0

echo "=== BUILD sanitizer/coverage binaries"
rm -f test_opt test_asan test_tsan test_cov   # never run stale binaries
g++ -O2 -g -std=c++17 -pthread -I$I $E $T -o test_opt 2>&1 | grep -v Wunused-result
g++ -O1 -g -std=c++17 -pthread -fsanitize=address,undefined -fno-omit-frame-pointer -I$I $E $T -o test_asan 2>&1 | grep -v Wunused-result
g++ -O2 -g -std=c++17 -pthread -fsanitize=thread -I$I $E $T -o test_tsan 2>&1 | grep -v Wunused-result
g++ -O1 -g -std=c++17 -pthread --coverage -fprofile-update=atomic -I$I $E $T -o test_cov 2>&1 | grep -v Wunused-result
ls -la test_opt test_asan test_tsan test_cov || exit 1

echo "=== PHASE opt scale1"
./test_opt $P "" 1 > phase_opt.log 2>&1 && echo "PHASE_OPT PASS" || { echo "PHASE_OPT FAIL"; RC=1; }
echo "=== PHASE buffered-fd tmpfs scale4"
./test_opt $S "" 4 > phase_shm.log 2>&1 && echo "PHASE_SHM PASS" || { echo "PHASE_SHM FAIL"; RC=1; }
rm -rf $S
echo "=== PHASE no-uring scale4"
HYDRA_NO_URING=1 ./test_opt $P "" 4 > phase_nouring.log 2>&1 && echo "PHASE_NOURING PASS" || { echo "PHASE_NOURING FAIL"; RC=1; }
echo "=== PHASE asan+ubsan scale4"
ASAN_OPTIONS=detect_leaks=1 ./test_asan $P "" 4 > phase_asan.log 2>&1 && echo "PHASE_ASAN PASS" || { echo "PHASE_ASAN FAIL"; RC=1; }
echo "=== PHASE tsan scale8 (subset)"
# setarch -R: disable ASLR — this kernel's vm.mmap_rnd_bits=32 exceeds what
# gcc's TSAN runtime can map ("FATAL: unexpected memory mapping" otherwise).
TSAN_FAIL=0
for t in audit rywrite ckpt oversized tiny alias twostores delete updel recover compact oversizebound; do
  TSAN_OPTIONS="halt_on_error=0 history_size=7" setarch "$(uname -m)" -R \
    ./test_tsan $P $t 8 > phase_tsan_$t.log 2>&1 || TSAN_FAIL=1
done
grep -l "WARNING: ThreadSanitizer" phase_tsan_*.log > tsan_warn_files.txt
[ $TSAN_FAIL -eq 0 ] && [ ! -s tsan_warn_files.txt ] && echo "PHASE_TSAN PASS" || { echo "PHASE_TSAN FAIL"; RC=1; }
echo "=== PHASE coverage"
rm -f *.gcda
./test_cov $P "" 2 > phase_cov.log 2>&1 && \
HYDRA_NO_URING=1 ./test_cov $P audit 8 >> phase_cov.log 2>&1 && \
HYDRA_NO_URING=1 ./test_cov $P readfault 1 >> phase_cov.log 2>&1 && \
HYDRA_NO_URING=1 ./test_cov $P recover 2 >> phase_cov.log 2>&1 && \
./test_cov $P compactcold 1 >> phase_cov.log 2>&1 && \
./test_cov $S basic 4 >> phase_cov.log 2>&1 && echo "PHASE_COV PASS" || { echo "PHASE_COV FAIL"; RC=1; }
rm -rf $S
gcov -b test_cov-hydra > gcov_summary.txt 2>&1 || gcov -b hydra >> gcov_summary.txt 2>&1
echo "=== gcov:"
grep -A3 "File.*hydra.cc" gcov_summary.txt | head -8
echo "ALL_PHASES_DONE"
exit $RC
