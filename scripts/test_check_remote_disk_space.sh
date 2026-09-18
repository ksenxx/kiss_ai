#!/bin/bash
# End-to-end test for scripts/check-remote-disk-space.sh.
# Run: bash scripts/test_check_remote_disk_space.sh
#
# The script is run the way ./rsorcar runs it -- fed to ``bash -s`` with
# NEED_BYTES, DB_BYTES and TARGET in the environment -- against a fake HOME.
# The room it sees comes from a ``df`` stub on PATH that answers with a fixed
# table (a full 10 GB root disk, a 9.7 TB data disk, tmpfs, an EFI partition,
# an NFS export), so every branch is reached on any machine.  Checked:
#   1. enough room: exit 0, one INFO line, nothing else touched;
#   2. not enough: exit 1, the message names the free and needed amounts, the
#      database's share, the largest entries of HOME, the data disk with its
#      free room and the exact move-home-to-disk.sh command with TARGET and
#      the mount point in it, and never a tmpfs, the EFI partition or the NFS
#      export as a place to move to;
#   3. not enough and no other disk: the grow-the-disk advice instead;
#   4. a stale sorcar.db.incoming and .outgoing in ~/.kiss are removed (and
#      counted in the INFO line) before the room is measured, and the live
#      database beside them is left alone;
#   5. NEED_BYTES missing or not a number is a usage error;
#   6. the real df: with NEED_BYTES=0 the check passes on this machine.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/check-remote-disk-space.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

mkdir -p "$WORK/home/.kiss" "$WORK/home/kiss" "$WORK/bin"
echo live > "$WORK/home/.kiss/sorcar.db"
head -c 300000 /dev/zero > "$WORK/home/kiss/big"

# df stub: $FAKE_DF names the table to answer with.
cat > "$WORK/bin/df" <<'EOF'
#!/bin/bash
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
case "${FAKE_DF:-full}" in
    full)
        echo "/dev/nvme0n1p1 10000000 10000000 0 100% /"
        echo "/dev/nvme0n1p15 126000 9000 117000 8% /boot/efi"
        echo "tmpfs 165000000 0 165000000 0% /tmp"
        echo "/dev/nvme0n3 10400000000 2000 10399998000 1% /data"
        echo "fileserver:/export 20000000000 0 20000000000 0% /nfs"
        ;;
    full-alone)
        echo "/dev/sda1 10000000 9000000 1000000 90% /"
        echo "tmpfs 165000000 0 165000000 0% /tmp"
        ;;
    roomy)
        echo "/dev/sda1 100000000 10000000 90000000 10% /"
        ;;
    home-partition)     # the home's own row first: df <dir> prints only that one
        echo "/dev/sda2 2000000 1900000 100000 95% /home"
        echo "/dev/sda1 100000000 10000000 90000000 10% /"
        ;;
esac
EOF
chmod +x "$WORK/bin/df"

# $1: FAKE_DF table, $2: NEED_BYTES, $3: DB_BYTES; prints combined output,
# returns the script's exit status.
run() {
    HOME="$WORK/home" PATH="$WORK/bin:$PATH" FAKE_DF="$1" \
    NEED_BYTES="$2" DB_BYTES="$3" TARGET="ksen@203.0.113.7" \
        bash -s < "$SCRIPT" 2>&1
}

# --- 1. enough room -----------------------------------------------------------
OUT="$(run roomy 5000000000 4000000000)" || fail "enough room reported as a failure: $OUT"
[ "$(printf '%s\n' "$OUT" | wc -l | tr -d ' ')" = 1 ] || fail "more than one line for a passing check: $OUT"
echo "$OUT" | grep -q "INFO.*85.8 GiB free.*needs about 4.7 GiB" || fail "the INFO line is off: $OUT"
pass "enough room is one INFO line and exit 0"

# --- 2. not enough, a data disk beside ---------------------------------------
if OUT="$(run full 9000000000 6000000000)"; then fail "a full disk passed: $OUT"; fi
echo "$OUT" | grep -q "Not enough room on .*: 0 B free in $WORK/home (the 9.5 GiB filesystem at /)" \
    || fail "the shortfall line is off: $OUT"
echo "$OUT" | grep -q "needs about 8.4 GiB: 5.6 GiB for the task database .* plus 2.8 GiB for the checkout" \
    || fail "the breakdown is off: $OUT"
echo "$OUT" | grep -q "What is in $WORK/home" || fail "the largest entries are not listed: $OUT"
echo "$OUT" | grep -q "$WORK/home/kiss" || fail "the biggest entry (kiss) is not listed: $OUT"
echo "$OUT" | grep -q "The filesystem at /data on .* has 9.7 TiB free" || fail "/data is not offered: $OUT"
echo "$OUT" | grep -qF "ssh ksen@203.0.113.7 'sudo -n bash -s -- /data' < scripts/move-home-to-disk.sh" \
    || fail "the move command is off: $OUT"
echo "$OUT" | grep -q "filesystem at /tmp" && fail "tmpfs offered as a disk: $OUT"
echo "$OUT" | grep -q "filesystem at /boot/efi" && fail "the EFI partition offered: $OUT"
echo "$OUT" | grep -q "filesystem at /nfs" && fail "an NFS export offered: $OUT"
echo "$OUT" | grep -q "Or grow the disk behind /" || fail "the grow-the-disk alternative is missing: $OUT"
echo "$OUT" | grep -q "SORCAR_DISK_HEADROOM_GB.*SORCAR_SKIP_DISK_CHECK=1" || fail "the overrides are not named: $OUT"
pass "a full disk is refused with the data disk and the move command named"

# --- 3. not enough, no other disk ---------------------------------------------
if OUT="$(run full-alone 9000000000 6000000000)"; then fail "a nearly full disk passed: $OUT"; fi
echo "$OUT" | grep -q "976.6 MiB free" || fail "the free room is off: $OUT"
echo "$OUT" | grep -q "^.*ERR.*Grow the disk behind /" || fail "growing the disk is not the first advice: $OUT"
echo "$OUT" | grep -q "The filesystem at" && fail "a disk was offered where there is none: $OUT"
echo "$OUT" | grep -q "attach a larger disk" || fail "attaching a disk is not suggested: $OUT"
pass "without another disk the advice is to grow or attach one"

# --- 3b. a small /home partition beside a roomy root: root is never offered ---
if OUT="$(run home-partition 9000000000 6000000000)"; then fail "a full /home partition passed: $OUT"; fi
echo "$OUT" | grep -q "The filesystem at / on" && fail "the root filesystem was offered as a disk to move to: $OUT"
echo "$OUT" | grep -q "Grow the disk behind /home" || fail "the advice does not name /home: $OUT"
pass "the root filesystem is never offered as the disk to move to"

# --- 4. stale scratch files are removed first ----------------------------------
head -c 2000000 /dev/zero > "$WORK/home/.kiss/sorcar.db.incoming"
head -c 1000 /dev/zero > "$WORK/home/.kiss/sorcar.db.outgoing"
OUT="$(run roomy 1000 0)" || fail "the check failed after removing scratch files: $OUT"
echo "$OUT" | grep -q "Removed ~/.kiss/sorcar.db.incoming (1.9 MiB) on .*interrupted task-database sync" \
    || fail "the incoming file's removal is not reported: $OUT"
echo "$OUT" | grep -q "Removed ~/.kiss/sorcar.db.outgoing (1000 B)" || fail "the outgoing file's removal is not reported: $OUT"
[ ! -e "$WORK/home/.kiss/sorcar.db.incoming" ] || fail "sorcar.db.incoming is still there"
[ ! -e "$WORK/home/.kiss/sorcar.db.outgoing" ] || fail "sorcar.db.outgoing is still there"
[ "$(cat "$WORK/home/.kiss/sorcar.db")" = live ] || fail "the live database was touched"
pass "stale scratch files of an interrupted sync are removed, the database kept"

# --- 5. usage errors ------------------------------------------------------------
if OUT="$(HOME="$WORK/home" bash -s < "$SCRIPT" 2>&1)"; then fail "no NEED_BYTES accepted: $OUT"; fi
echo "$OUT" | grep -q "NEED_BYTES must be a number of bytes, got 'nothing'" || fail "the usage error is off: $OUT"
if OUT="$(HOME="$WORK/home" NEED_BYTES=lots bash -s < "$SCRIPT" 2>&1)"; then fail "NEED_BYTES=lots accepted: $OUT"; fi
echo "$OUT" | grep -q "got 'lots'" || fail "the usage error does not quote the value: $OUT"
pass "a missing or non-numeric NEED_BYTES is a usage error"

# --- 6. the real df ---------------------------------------------------------------
OUT="$(HOME="$WORK/home" NEED_BYTES=0 bash -s < "$SCRIPT" 2>&1)" || fail "the real df failed the check: $OUT"
echo "$OUT" | grep -q "free in $WORK/home on" || fail "the real df's INFO line is off: $OUT"
pass "the real df is read"

echo "ALL TESTS PASSED"
