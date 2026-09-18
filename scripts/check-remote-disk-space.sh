#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Is there room in this machine's home directory for a deploy?
#
# Usage:  ssh user@host 'NEED_BYTES=<n> DB_BYTES=<n> TARGET=user@host bash -s' \
#             < scripts/check-remote-disk-space.sh
#         NEED_BYTES=<n> scripts/check-remote-disk-space.sh     (on the machine itself)
#
#   NEED_BYTES  what the deploy is about to write under $HOME (required)
#   DB_BYTES    the part of that which is the task database (for the message)
#   TARGET      how the deploying machine addresses this one, so the commands
#               this script suggests can be pasted as they are
#
# ./rsorcar runs this before anything of size travels (step 1c).  A deploy
# writes the whole task database into ~/.kiss, the checkout into ~/<project>,
# a Python environment into its .venv, uv's cache into ~/.cache and code-server
# into ~/.local -- all under $HOME.  A cloud VM with a 10 GB boot disk and a
# multi-terabyte data disk mounted beside it holds none of that on the boot
# disk: the upload died with "gzip: stdout: No space left on device", the
# deploy carried on, and every later step failed in its own way on a full
# disk.  Checking first turns that into one clear stop that says what would
# fit where.
#
# Two scratch files of an interrupted sync (scripts/sync-task-db.sh) are
# removed first: sorcar.db.incoming is an upload that never verified, and
# sorcar.db.outgoing a snapshot that was already read back or never was.
# Neither is ever the live database, and either can be as large as it.
set -euo pipefail

info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()  { printf '\033[0;31m[ERR]\033[0m  %s\n' "$*" >&2; }
die()  { err "$@"; exit 1; }

human() {
    awk -v b="$1" 'BEGIN {
        split("B KiB MiB GiB TiB PiB", unit, " "); i = 1
        while (b >= 1024 && i < 6) { b /= 1024; i++ }
        printf (i == 1 ? "%d %s" : "%.1f %s"), b, unit[i] }'
}

HOST="$(hostname -s 2>/dev/null || hostname)"
NEED_BYTES="${NEED_BYTES:-}"
[[ "$NEED_BYTES" =~ ^[0-9]+$ ]] || die "NEED_BYTES must be a number of bytes, got '${NEED_BYTES:-nothing}'."
DB_BYTES="${DB_BYTES:-0}"
[[ "$DB_BYTES" =~ ^[0-9]+$ ]] || DB_BYTES=0
TARGET="${TARGET:-user@$HOST}"

# --- Scratch files a sync that did not finish may have left ------------------
for scratch in "$HOME/.kiss/sorcar.db.incoming" "$HOME/.kiss/sorcar.db.outgoing"; do
    [ -f "$scratch" ] || continue
    size="$(wc -c < "$scratch" | tr -d ' ')"
    rm -f "$scratch"
    info "Removed ~/.kiss/${scratch##*/} ($(human "$size")) on $HOST, left by an interrupted task-database sync."
done

# --- The filesystem $HOME is on ----------------------------------------------
# df -P is the portable form: one line per filesystem, 1024-byte blocks,
# fields: source, size, used, available, capacity, mount point.
HOME_FS="$(df -Pk "$HOME" | awk 'NR == 2')"
[ -n "$HOME_FS" ] || die "Cannot read how much room $HOME has on $HOST (df failed)."
FREE_BYTES="$(printf '%s\n' "$HOME_FS" | awk '{ print $4 * 1024 }')"
SIZE_BYTES="$(printf '%s\n' "$HOME_FS" | awk '{ print $2 * 1024 }')"
MOUNT="$(printf '%s\n' "$HOME_FS" | awk '{ print $6 }')"

if [ "$FREE_BYTES" -ge "$NEED_BYTES" ]; then
    info "$(human "$FREE_BYTES") free in $HOME on $HOST (the filesystem at $MOUNT); the deploy needs about $(human "$NEED_BYTES")."
    exit 0
fi

# --- Not enough: say what would fit where -----------------------------------
HEADROOM_BYTES=$((NEED_BYTES > DB_BYTES ? NEED_BYTES - DB_BYTES : 0))
err "Not enough room on $HOST: $(human "$FREE_BYTES") free in $HOME (the $(human "$SIZE_BYTES") filesystem at $MOUNT)," \
    "and the deploy needs about $(human "$NEED_BYTES"):" \
    "$(human "$DB_BYTES") for the task database (~/.kiss/sorcar.db) plus $(human "$HEADROOM_BYTES") for the checkout," \
    "its Python environment (.venv, ~/.cache/uv) and the tools install.sh brings (code-server, Node.js)."

# The five largest things in the home directory, so the reader knows whether
# there is anything to remove.  -x keeps du on this one filesystem.
LARGEST="$(du -xsk "$HOME"/* "$HOME"/.[!.]* 2>/dev/null | sort -rn | head -5 \
    | awk '{ printf "         %s\t%s\n", $1, $2 }' || true)"
if [ -n "$LARGEST" ]; then
    err "What is in $HOME (KiB):"
    while IFS= read -r line; do err "$line"; done <<< "$LARGEST"
fi

# Other local filesystems with room for the whole deploy.  A source that is a
# path ("/dev/sdb1", or a bind mount's) is a disk; "tmpfs", "udev" and the
# like are memory, and "server:/export" is somebody else's disk.  The root
# filesystem is never offered: move-home-to-disk.sh refuses it, and a home on
# a /home partition smaller than / is a partitioning choice, not a disk to
# move to.
CANDIDATES="$(df -Pk | awk -v need="$NEED_BYTES" -v home_mount="$MOUNT" '
    NR > 1 && $1 ~ /^\// && $6 != home_mount && $6 != "/" && $6 !~ /^\/boot/ && $4 * 1024 >= need {
        print $6 "\t" $4 * 1024 }' || true)"
if [ -n "$CANDIDATES" ]; then
    while IFS=$'\t' read -r mount free; do
        [ -n "$mount" ] || continue
        err "The filesystem at $mount on $HOST has $(human "$free") free. Move the home directory there" \
            "(everything under $HOME, ~/.kiss and the checkout included; it stays reachable as $HOME) with:"
        err "         ssh $TARGET 'sudo -n bash -s -- $mount' < scripts/move-home-to-disk.sh"
    done <<< "$CANDIDATES"
    err "Or grow the disk behind $MOUNT in your cloud console (the root filesystem of a cloud image grows" \
        "into it on the next boot), or free room there (sudo journalctl --vacuum-size=200M; sudo apt-get clean)," \
        "then deploy again."
else
    err "Grow the disk behind $MOUNT in your cloud console (the root filesystem of a cloud image grows" \
        "into it on the next boot), or attach a larger disk, mount it and move the home directory onto it" \
        "(ssh $TARGET 'sudo -n bash -s -- /mount/point' < scripts/move-home-to-disk.sh), or free room there" \
        "(sudo journalctl --vacuum-size=200M; sudo apt-get clean), then deploy again."
fi
err "SORCAR_DISK_HEADROOM_GB sets how much room is asked for beyond the database; SORCAR_SKIP_DISK_CHECK=1 deploys anyway."
exit 1
