#!/bin/bash
# End-to-end test for scripts/move-home-to-disk.sh.
# Run: bash scripts/test_move_home_to_disk.sh
#
# A bind mount is a Linux operation and needs root, so every case runs inside
# one privileged debian:13 container (docker is required; the test is skipped
# without it).  The "bigger disk" is a tmpfs mounted at /data.  Checked:
#   1. a home with files, a symlink, a hard link, a socket and a hidden dir
#      moves onto the disk, comes back bind-mounted at the same path with the
#      same owner and mode, the fstab line is written, the original is gone;
#   2. a second run is a no-op with exit 0 and no second fstab line;
#   3. a process of the user with its working directory under the home stops
#      the move (and the home is untouched), FORCE=1 lets it through;
#   4. a process of another user under the home does not count;
#   5. an fstab line that mounts something else on the home stops the move;
#   6. a non-empty <disk>/home/<user> stops the move;
#   7. a copy that fails leaves the home as it was (the disk is too small);
#   8. the argument checks: no argument, not root, unknown user, root itself,
#      a directory that is not a mount point, the home's own filesystem, a
#      whitespace path;
#   9. a disk that is mounted but not in /etc/fstab is refused (it would be
#      gone after a reboot, and the home with it), with the line to add;
#  10. a filesystem mounted inside the home stops the move;
#  11. a copy that fails after something was written into the emptied home
#      (a login's fresh authorized_keys) still puts the home back in place,
#      the intruding file inside it, not the home inside the intruder's dir;
#  12. after a boot on which the disk was missing (fstab line present, home
#      not mounted, a login wrote into the empty home), a rerun mounts the
#      copy again and carries the new files into it without replacing any.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if ! docker info >/dev/null 2>&1; then
    echo "SKIP: docker is not available; scripts/move-home-to-disk.sh needs a Linux root to test."
    exit 0
fi

docker run --rm -i --privileged -v "$REPO_ROOT/scripts/move-home-to-disk.sh:/move.sh:ro" \
    debian:13 bash -s <<'IN_CONTAINER'
set -e
fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }
MOVE="bash /move.sh"

useradd -m -s /bin/bash alice
useradd -m -s /bin/bash bob
mkdir -p /home/alice/.ssh /home/alice/.kiss/deep
echo key > /home/alice/.ssh/authorized_keys; chmod 700 /home/alice/.ssh
# 711, so that another user's process can sit in a directory under it (case 4).
mkdir -p /home/alice/shared && chmod 755 /home/alice/shared && chmod 711 /home/alice
head -c 3000000 /dev/urandom > /home/alice/.kiss/sorcar.db
ln -s .kiss/sorcar.db /home/alice/link
ln /home/alice/.kiss/sorcar.db /home/alice/hard
echo deep > /home/alice/.kiss/deep/file
python3 -c 'import socket; socket.socket(socket.AF_UNIX).bind("/home/alice/.kiss/sorcar.sock")' 2>/dev/null \
    || perl -e 'use Socket; socket(S, PF_UNIX, SOCK_STREAM, 0); bind(S, sockaddr_un("/home/alice/.kiss/sorcar.sock"))' 2>/dev/null \
    || echo "(no socket could be created; skipping that part)"
chown -R alice:alice /home/alice
BEFORE="$(cd /home/alice && find . ! -type s | sort | xargs -I{} stat -c '%n %U %a' {} )"
mkdir /data && mount -t tmpfs -o size=200m tmpfs /data

# --- 9. a disk that is not in fstab is refused ---------------------------------
OUT="$(MOVE_USER=alice $MOVE /data 2>&1)" && fail "a disk missing from fstab accepted"
echo "$OUT" | grep -q "has no line for it, so it would not be there after a reboot" || fail "the fstab advice is missing: $OUT"
echo "$OUT" | grep -q "UUID=.* /data tmpfs defaults,nofail 0 2" || fail "the fstab line to add is not shown: $OUT"
[ -d /home/alice/.kiss/deep ] && ! mountpoint -q /home/alice || fail "a refused run touched the home"
pass "a disk that would not come back after a reboot is refused, with the fstab line to add"
echo "tmpfs /data tmpfs defaults 0 0" >> /etc/fstab

# --- 10. a filesystem mounted inside the home stops the move -------------------
mkdir -p /home/alice/mnt && mount -t tmpfs tmpfs /home/alice/mnt
OUT="$(MOVE_USER=alice $MOVE /data 2>&1)" && fail "a nested mount accepted"
echo "$OUT" | grep -q "Something is mounted inside /home/alice.*/home/alice/mnt" || fail "the nested mount is not named: $OUT"
umount /home/alice/mnt && rmdir /home/alice/mnt
pass "a filesystem mounted inside the home stops the move"

# --- 8. argument checks (nothing is moved) ---------------------------------
$MOVE >/dev/null 2>&1 && fail "no argument accepted"
MOVE_USER=alice $MOVE /nowhere >/dev/null 2>&1 && fail "a missing directory accepted"
mkdir /notamount
MOVE_USER=alice $MOVE /notamount 2>&1 | grep -q "not a mounted filesystem" || fail "a plain directory accepted as the disk"
MOVE_USER=nobody-here $MOVE /data 2>&1 | grep -q "no user named" || fail "an unknown user accepted"
MOVE_USER=root $MOVE /data 2>&1 | grep -q "Not moving root" || fail "root accepted"
$MOVE /data 2>&1 | grep -q "Whose home directory" || fail "a run without sudo and MOVE_USER accepted"
su -s /bin/bash alice -c "bash /move.sh /data" 2>&1 | grep -q "must run as root" || fail "a non-root run accepted"
mkdir "/data/with space" && mount -t tmpfs tmpfs "/data/with space"
MOVE_USER=alice $MOVE "/data/with space" 2>&1 | grep -q "whitespace" || fail "a path with a space accepted"
umount "/data/with space"; rmdir "/data/with space"
MOVE_USER=alice $MOVE / 2>&1 | grep -q "already on the filesystem\|root filesystem" || fail "the root filesystem accepted"
[ -d /home/alice/.kiss/deep ] || fail "a rejected run moved something"
mountpoint -q /home/alice && fail "a rejected run mounted something"
pass "argument checks refuse without touching anything"

# --- 5. fstab already mounts something else there ---------------------------
echo "/dev/sdz1 /home/alice ext4 defaults 0 0" >> /etc/fstab
MOVE_USER=alice $MOVE /data 2>&1 | grep -q "already mounts /dev/sdz1" || fail "a conflicting fstab line ignored"
sed -i '/sdz1/d' /etc/fstab
pass "a conflicting fstab line stops the move"

# --- 6. the destination already holds something -----------------------------
mkdir -p /data/home/alice/old && MOVE_USER=alice $MOVE /data 2>&1 | grep -q "not empty" || fail "a non-empty destination accepted"
rm -rf /data/home
pass "a non-empty destination stops the move"

# --- 3. a process of the user under the home stops the move -----------------
setpriv --reuid=alice --regid=alice --clear-groups sh -c "cd /home/alice/.kiss && exec sleep 300" &
BUSY_PID=$!
sleep 0.5
OUT="$(MOVE_USER=alice $MOVE /data 2>&1)" && fail "moved under a process of the user"
echo "$OUT" | grep -q "sleep 300" || fail "the busy process was not named: $OUT"
mountpoint -q /home/alice && fail "the refused move mounted something"
[ -f /home/alice/.kiss/deep/file ] || fail "the refused move lost files"
[ -e /data/home/alice ] && fail "the refused move left a copy on the disk"
pass "a process working under the home stops the move"

# --- 4. another user's process under the home does not count ----------------
kill "$BUSY_PID" 2>/dev/null; wait "$BUSY_PID" 2>/dev/null || true
setpriv --reuid=bob --regid=bob --clear-groups sh -c "cd /home/alice/shared && exec sleep 300" &
BOB_PID=$!
sleep 0.5
# --- 7. a disk too small leaves the home as it was --------------------------
mkdir /small && mount -t tmpfs -o size=1m tmpfs /small && echo "tmpfs /small tmpfs defaults 0 0" >> /etc/fstab
OUT="$(MOVE_USER=alice $MOVE /small 2>&1)" && fail "moved onto a disk without room"
echo "$OUT" | grep -q "free, and /home/alice holds" || fail "the room check did not speak: $OUT"
# Lie about the room so the copy itself fails, and check the restore.
mkdir /fakebin && printf '#!/bin/bash\nif [ "$1" = -Pk ]; then echo "Filesystem 1024-blocks Used Available Capacity Mounted on"; echo "tmpfs 999999999 0 999999999 0%% /small"; else exec /usr/bin/df "$@"; fi\n' > /fakebin/df && chmod +x /fakebin/df
OUT="$(PATH=/fakebin:$PATH MOVE_USER=alice $MOVE /small 2>&1)" && fail "a failed copy reported success"
echo "$OUT" | grep -q "as it was" || fail "the failed copy did not say the home was restored: $OUT"
mountpoint -q /home/alice && fail "a failed copy left a mount"
[ -f /home/alice/.kiss/deep/file ] && [ -f /home/alice/hard ] || fail "a failed copy lost files"
ls /home | grep -q moved && fail "a failed copy left the renamed original behind"
[ -e /small/home/alice ] && fail "a failed copy left its partial copy behind"
umount /small
pass "a copy that fails leaves the home as it was"

# --- 11. a copy that fails after a login wrote into the emptied home -------------
# The cp stub fails the copy itself (its first call), after dropping a fresh
# authorized_keys into the (new, empty) home the way the cloud guest agent
# does; the restore's own cp calls are the real cp.
printf '#!/bin/bash\nif [ ! -e /tmp/cp-failed-once ]; then touch /tmp/cp-failed-once; mkdir -p /home/alice/.ssh && echo fresh-key > /home/alice/.ssh/authorized_keys && echo intruder > /home/alice/newcomer; exit 1; fi\nexec /bin/cp "$@"\n' > /fakebin/cp && chmod +x /fakebin/cp
OUT="$(PATH=/fakebin:$PATH MOVE_USER=alice $MOVE /data 2>&1)" && fail "a failed copy (with an intruder) reported success"
rm /fakebin/cp
[ -f /home/alice/.kiss/deep/file ] && [ -f /home/alice/hard ] || fail "the original did not come back to /home/alice"
ls -d /home/alice/*moved* /home/alice/.*moved* 2>/dev/null && fail "the original was moved INTO the emptied home: $(ls -A /home/alice)"
[ "$(cat /home/alice/.ssh/authorized_keys)" = key ] || fail "the intruder replaced the original authorized_keys"
[ "$(cat /home/alice/newcomer)" = intruder ] || fail "what was written into the emptied home was lost"
rm /home/alice/newcomer
ls /home | grep -q moved && fail "a renamed original was left behind"
[ -e /data/home/alice ] && fail "the partial copy was left on the disk"
mountpoint -q /home/alice && fail "a failed copy left a mount"
pass "a failed copy puts the home back under its own name, keeping what a login wrote meanwhile"

# --- 1. the move -------------------------------------------------------------
OUT="$(MOVE_USER=alice $MOVE /data 2>&1)" || fail "the move failed: $OUT"
kill "$BOB_PID" 2>/dev/null; wait "$BOB_PID" 2>/dev/null || true
mountpoint -q /home/alice || fail "the home is not a mount point"
[ "$(stat -c %d:%i /home/alice)" = "$(stat -c %d:%i /data/home/alice)" ] || fail "the home is not the copy on the disk"
[ "$(stat -c '%U %a' /home/alice)" = "alice 711" ] || fail "owner or mode changed: $(stat -c '%U %a' /home/alice)"
AFTER="$(cd /home/alice && find . ! -type s | sort | xargs -I{} stat -c '%n %U %a' {} )"
[ "$BEFORE" = "$AFTER" ] || fail "the tree differs after the move:
$BEFORE
---
$AFTER"
[ "$(readlink /home/alice/link)" = .kiss/sorcar.db ] || fail "the symlink changed"
[ "$(stat -c %h /home/alice/hard)" = 2 ] || fail "the hard link was not kept"
cmp -s /home/alice/hard /home/alice/.kiss/sorcar.db || fail "file content differs"
grep -qxF "/data/home/alice /home/alice none bind,nofail,x-systemd.requires-mounts-for=/data 0 0" /etc/fstab \
    || fail "fstab line missing: $(cat /etc/fstab)"
ls /home | grep -q moved && fail "the renamed original was not removed"
[ -e /data/home/alice/.kiss/deep/file ] || fail "the copy is incomplete"
su -s /bin/sh alice -c 'cd && test -w . && echo ok > written' || fail "alice cannot write in her home"
[ -f /data/home/alice/written ] || fail "a write in the home did not land on the disk"
pass "the home moves onto the disk and comes back at the same path"

# --- 2. a second run is a no-op ------------------------------------------------
OUT="$(MOVE_USER=alice $MOVE /data 2>&1)" || fail "the second run failed: $OUT"
echo "$OUT" | grep -q "nothing to do" || fail "the second run did not say it had nothing to do"
[ "$(grep -c '/home/alice' /etc/fstab)" = 1 ] || fail "a second fstab line was written"
pass "a second run changes nothing"

# --- 3b. FORCE=1 moves in spite of a process ----------------------------------
useradd -m -s /bin/bash carol
setpriv --reuid=carol --regid=carol --clear-groups sh -c "cd /home/carol && exec sleep 300" &
CAROL_PID=$!
sleep 0.5
OUT="$(FORCE=1 MOVE_USER=carol $MOVE /data 2>&1)" || fail "FORCE=1 did not move: $OUT"
kill "$CAROL_PID" 2>/dev/null; wait "$CAROL_PID" 2>/dev/null || true
mountpoint -q /home/carol || fail "FORCE=1 did not mount"
pass "FORCE=1 moves in spite of a process under the home"

# --- 12. the disk was missing at boot: fstab line present, home not mounted ------
umount /home/carol
echo "boot-key" > /home/carol/.boot-written     # a login wrote into the empty underlay
mkdir -p /home/carol/.ssh && echo "new-key" > /home/carol/.ssh/authorized_keys
mkdir -p /data/home/carol/.ssh && echo "disk-key" > /data/home/carol/.ssh/authorized_keys
OUT="$(MOVE_USER=carol $MOVE /data 2>&1)" || fail "the rerun after a failed boot mount failed: $OUT"
echo "$OUT" | grep -q "already mounts /data/home/carol on /home/carol, but it is not mounted; mounting it" || fail "the remount was not announced: $OUT"
mountpoint -q /home/carol || fail "the home was not mounted again"
[ "$(cat /home/carol/.boot-written)" = boot-key ] || fail "what the login wrote was not carried into the copy"
[ "$(cat /home/carol/.ssh/authorized_keys)" = disk-key ] || fail "a file of the copy was replaced by the underlay's"
[ "$(grep -c '/home/carol' /etc/fstab)" = 1 ] || fail "a second fstab line was written on the remount"
pass "after a boot without the disk, a rerun mounts the copy again and carries new files into it"

echo "ALL TESTS PASSED"
IN_CONTAINER
