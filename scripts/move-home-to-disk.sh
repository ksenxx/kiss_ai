#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Move a user's home directory onto a bigger disk, and bind-mount it back
# where it was -- so $HOME keeps its path and everything that names it (the
# passwd entry, the systemd user service, the checkout's recorded work
# directories, sshd's authorized_keys lookup) keeps working, while the bytes
# live on the disk that has room for them.
#
# Usage:  ssh user@host 'sudo -n bash -s -- /data' < scripts/move-home-to-disk.sh
#         sudo bash scripts/move-home-to-disk.sh /data      (on the machine itself)
#
#   $1          the mount point of the disk to move onto (/data)
#   MOVE_USER   whose home directory (default: the user who ran sudo)
#   FORCE=1     go ahead although processes of that user hold files under the
#               home directory open (see below); through sudo, which resets
#               the environment: sudo -n env FORCE=1 bash -s -- /data
#
# This is the fix scripts/check-remote-disk-space.sh names when a deploy does
# not fit: a cloud VM is often a 10 GB boot disk with a large data disk
# mounted beside it, and a deploy writes everything under $HOME.
#
# What it does, in order, and what it undoes when a step fails:
#   1. Checks: root, a real user, the disk is a mounted filesystem other than
#      the one the home is on and is listed in /etc/fstab (a disk mounted by
#      hand is not there after a reboot, and the home would come back empty),
#      it has room, nothing is mounted inside the home, <disk>/home/<user>
#      does not already hold something, /etc/fstab does not already mount
#      something else there.  A home directory that is already this bind
#      mount is left alone (exit 0); one whose fstab line exists but is not
#      mounted (the disk was missing at boot) is mounted, after what was
#      written into the empty home meanwhile is carried into the copy.
#   2. Stops the user's kiss-web service, and refuses if any other process of
#      the user has its working directory, its executable or an open file
#      under the home directory: such a process would go on writing into the
#      copy that step 5 deletes.  FORCE=1 overrides.
#   3. Renames the home directory aside, creates an empty one in its place
#      with the same owner and mode, and copies the renamed one onto the disk
#      (cp -a: owners, modes, times, links).  A copy that fails is removed and
#      the renamed directory is put back.
#   4. Bind-mounts the copy over the (empty) home directory, and records the
#      mount in /etc/fstab -- nofail, so a disk that does not come up cannot
#      stop the boot, and ordered after the disk's own mount.  A mount that
#      fails is undone the same way as a failed copy.
#   5. Removes the renamed original and starts kiss-web again if it was
#      running.
#
# Between steps 3 and 4 the home directory is empty for the seconds the copy
# takes: a login attempted right then is refused (no authorized_keys) and
# succeeds a moment later.
set -euo pipefail

info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()  { printf '\033[0;31m[ERR]\033[0m  %s\n' "$*" >&2; exit 1; }

human() {
    awk -v b="$1" 'BEGIN {
        split("B KiB MiB GiB TiB PiB", unit, " "); i = 1
        while (b >= 1024 && i < 6) { b /= 1024; i++ }
        printf (i == 1 ? "%d %s" : "%.1f %s"), b, unit[i] }'
}

# --- 1. Checks -----------------------------------------------------------------
DISK="${1:-}"
[ -n "$DISK" ] || die "Usage: sudo bash $0 /mount/point/of/the/bigger/disk"
[ "$(id -u)" = 0 ] || die "This must run as root: ssh user@host 'sudo -n bash -s -- $DISK' < scripts/move-home-to-disk.sh"
[ -d /proc/self ] || die "This script moves a home directory with a bind mount, which is a Linux operation."
for tool in getent mountpoint mount cp find awk; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is not installed."
done

USER_NAME="${MOVE_USER:-${SUDO_USER:-}}"
[ -n "$USER_NAME" ] || die "Whose home directory? Run this through sudo, or set MOVE_USER=<user>."
[ "$USER_NAME" != root ] || die "Not moving root's home directory."
ENTRY="$(getent passwd "$USER_NAME")" || die "There is no user named '$USER_NAME' on this machine."
USER_UID="$(printf '%s' "$ENTRY" | cut -d: -f3)"
USER_GID="$(printf '%s' "$ENTRY" | cut -d: -f4)"
HOME_DIR="$(printf '%s' "$ENTRY" | cut -d: -f6)"
case "$HOME_DIR" in
    ""|/|/root|/nonexistent) die "'$HOME_DIR' is not a home directory this script moves." ;;
esac
[ -d "$HOME_DIR" ] || die "$USER_NAME's home directory $HOME_DIR does not exist."
[ -d "$DISK" ] || die "$DISK is not a directory."
DISK="$(cd "$DISK" && pwd -P)"
case "$DISK$HOME_DIR" in
    *[[:space:]]*) die "A path with whitespace in it cannot be written to /etc/fstab: '$DISK', '$HOME_DIR'." ;;
esac
mountpoint -q "$DISK" || die "$DISK is not a mounted filesystem (see: df -h; lsblk)."
[ "$DISK" != / ] || die "$DISK is the root filesystem, not a bigger disk."
NEW="$DISK/home/$USER_NAME"
FSTAB=/etc/fstab
FSTAB_LINE="$NEW $HOME_DIR none bind,nofail,x-systemd.requires-mounts-for=$DISK 0 0"
# Carry what is in directory $1 into directory $2 without replacing anything
# already there -- neither a file nor the owner and mode of a directory that
# exists on both sides (cp -a would set the destination directory's
# attributes to the source's).  For what a login or the cloud guest agent
# wrote into an emptied home directory while the copy of the real one was
# elsewhere.
carry_into() {
    local src="$1" dst="$2" path rel
    # find lists a directory before what is in it, so a new directory exists
    # by the time its files are reached.
    while IFS= read -r -d '' path; do
        rel="${path#"$src"/}"
        if [ -e "$dst/$rel" ] || [ -L "$dst/$rel" ]; then continue; fi
        if [ -d "$path" ] && [ ! -L "$path" ]; then
            mkdir "$dst/$rel" && chown --reference="$path" "$dst/$rel" && chmod --reference="$path" "$dst/$rel"
        else
            cp -a "$path" "$dst/$rel"
        fi
    done < <(find "$src" -mindepth 1 -print0)
}

# The second field of an fstab line, for a mount point: the disk's own line
# and the home's.
fstab_source_of() {
    awk -v target="$1" '$1 !~ /^#/ && $2 == target { print $1 }' "$FSTAB" 2>/dev/null || true
}

# Already this bind mount?  Then the home directory and the copy are one
# directory: same device, same inode.
if mountpoint -q "$HOME_DIR"; then
    if [ -d "$NEW" ] \
        && [ "$(stat -c %d:%i "$HOME_DIR")" = "$(stat -c %d:%i "$NEW")" ]; then
        info "$HOME_DIR is already $NEW, bind-mounted; nothing to do."
        exit 0
    fi
    die "$HOME_DIR is a mount point of its own ($(findmnt -n -o SOURCE --target "$HOME_DIR" 2>/dev/null || echo unknown source)); not touching it."
fi
[ "$(stat -c %d "$HOME_DIR")" != "$(stat -c %d "$DISK")" ] \
    || die "$HOME_DIR is already on the filesystem at $DISK."
EXISTING_SOURCE="$(fstab_source_of "$HOME_DIR")"
if [ -n "$EXISTING_SOURCE" ] && [ "$EXISTING_SOURCE" != "$NEW" ]; then
    die "$FSTAB already mounts $EXISTING_SOURCE on $HOME_DIR; remove that line first."
fi
# The move was done before, and the bind mount did not come up at the last
# boot (the disk was missing, and nofail let the boot go on).  The copy on
# the disk is the home; what a login wrote into the empty directory since (a
# fresh authorized_keys, say) is carried over without replacing anything, and
# the fstab line is mounted.
if [ "$EXISTING_SOURCE" = "$NEW" ] && [ -d "$NEW" ]; then
    info "$FSTAB already mounts $NEW on $HOME_DIR, but it is not mounted; mounting it."
    carry_into "$HOME_DIR" "$NEW" \
        || warn "Could not carry everything written into $HOME_DIR since the last boot into $NEW."
    mount "$HOME_DIR" || die "Could not mount $NEW on $HOME_DIR from $FSTAB."
    info "$HOME_DIR is $NEW again."
    exit 0
fi
# The disk itself has to come back after a reboot, or the home would not.
[ -n "$(fstab_source_of "$DISK")" ] \
    || die "$DISK is mounted, but $FSTAB has no line for it, so it would not be there after a reboot" \
           "and $HOME_DIR would come back empty. Add it first, for example:" \
           "UUID=$(blkid -s UUID -o value "$(findmnt -n -o SOURCE --target "$DISK" 2>/dev/null)" 2>/dev/null || echo '<uuid>')" \
           "$DISK $(findmnt -n -o FSTYPE --target "$DISK" 2>/dev/null || echo '<fstype>') defaults,nofail 0 2"
# A filesystem mounted inside the home would be copied in full, and the copy
# would still be there when the mount is not.
NESTED="$(findmnt -rn -o TARGET 2>/dev/null | awk -v home="$HOME_DIR/" 'index($0, home) == 1' || true)"
[ -z "$NESTED" ] || die "Something is mounted inside $HOME_DIR; unmount it first:" "$NESTED"
if [ -e "$NEW" ]; then
    [ -d "$NEW" ] || die "$NEW exists and is not a directory."
    [ -z "$(find "$NEW" -mindepth 1 -maxdepth 1 -print -quit)" ] \
        || die "$NEW already exists and is not empty; move it out of the way first."
fi

USED_K="$(du -sxk "$HOME_DIR" | cut -f1)"
FREE_K="$(df -Pk "$DISK" | awk 'NR == 2 { print $4 }')"
[ "$FREE_K" -gt "$USED_K" ] \
    || die "$DISK has $(human $((FREE_K * 1024))) free, and $HOME_DIR holds $(human $((USED_K * 1024)))."

# --- 2. Nothing may be writing under the home directory -----------------------
# The user's own systemd instance is asked as the user; its bus lives in
# /run/user/<uid>, which only exists while the user has a session or lingers.
user_systemctl() {
    [ -d "/run/user/$USER_UID" ] && command -v systemctl >/dev/null 2>&1 || return 1
    su -s /bin/sh "$USER_NAME" -c "XDG_RUNTIME_DIR=/run/user/$USER_UID systemctl --user $*" 2>/dev/null
}
WEB_APP_WAS_ACTIVE=0
if user_systemctl is-active --quiet kiss-web.service; then
    WEB_APP_WAS_ACTIVE=1
    info "Stopping $USER_NAME's kiss-web service for the move ..."
    user_systemctl stop kiss-web.service || warn "Could not stop kiss-web.service."
    sleep 1
fi

# The processes this very command runs in: the user's login shell that
# invoked sudo has its working directory in the home, and must not count.
ANCESTORS=" $$ "
pid=$$
while [ "$pid" -gt 1 ]; do
    pid="$(awk '/^PPid:/ { print $2 }' "/proc/$pid/status" 2>/dev/null || echo 1)"
    ANCESTORS="$ANCESTORS$pid "
done
under_home() {
    case "$1/" in "$HOME_DIR"/*) return 0 ;; esac
    return 1
}
uses_home() {
    local pid="$1" fd
    under_home "$(readlink "/proc/$pid/cwd" 2>/dev/null || true)" && return 0
    under_home "$(readlink "/proc/$pid/exe" 2>/dev/null || true)" && return 0
    for fd in /proc/"$pid"/fd/*; do
        under_home "$(readlink "$fd" 2>/dev/null || true)" && return 0
    done
    return 1
}
BUSY=""
for status in /proc/[0-9]*/status; do
    pid="${status#/proc/}"; pid="${pid%/status}"
    [ "$(awk '/^Uid:/ { print $2 }' "$status" 2>/dev/null)" = "$USER_UID" ] || continue
    case "$ANCESTORS" in *" $pid "*) continue ;; esac
    uses_home "$pid" || continue
    BUSY="$BUSY
         $pid  $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)"
done
if [ -n "$BUSY" ] && [ "${FORCE:-}" != 1 ]; then
    die "These processes of $USER_NAME are working under $HOME_DIR, and would go on writing into the" \
        "copy this move deletes:$BUSY
       Stop them and run this again, or move anyway with: sudo -n env FORCE=1 bash -s -- $DISK"
fi

# --- 3. Copy --------------------------------------------------------------------
OLD="$HOME_DIR.moved-$(date -u +%Y%m%dT%H%M%SZ)"
info "Moving $HOME_DIR ($(human $((USED_K * 1024)))) to $NEW ..."
mkdir -p "$DISK/home"
mkdir -p "$NEW"
chown "$USER_UID:$USER_GID" "$NEW"
chmod --reference="$HOME_DIR" "$NEW"
mv "$HOME_DIR" "$OLD"
mkdir "$HOME_DIR"
chown "$USER_UID:$USER_GID" "$HOME_DIR"
chmod --reference="$OLD" "$HOME_DIR"

# Undo a move that did not complete: the original goes back under its own
# name.  Something may have written into the empty home meanwhile (a login,
# the cloud guest agent refreshing authorized_keys); that is carried into the
# original without replacing anything of it, so the empty directory can go
# and the rename is a rename -- not a move of the original *into* it.
put_back() {
    if mountpoint -q "$HOME_DIR"; then
        umount "$HOME_DIR" || { warn "Could not unmount $HOME_DIR; the original is still at $OLD."; return; }
    fi
    carry_into "$HOME_DIR" "$OLD" || true
    rm -rf "$HOME_DIR"
    mv -T "$OLD" "$HOME_DIR"
    rm -rf "$NEW"
}
if ! cp -a "$OLD/." "$NEW/"; then
    put_back
    die "Copying $HOME_DIR to $NEW failed; $HOME_DIR is as it was."
fi
# Sockets describe processes, not data; cp recreates them where it can and
# they are not worth a failed move where it cannot.
OLD_COUNT="$(find "$OLD" -mindepth 1 ! -type s | wc -l | tr -d ' ')"
NEW_COUNT="$(find "$NEW" -mindepth 1 ! -type s | wc -l | tr -d ' ')"
if [ "$OLD_COUNT" != "$NEW_COUNT" ]; then
    put_back
    die "The copy holds $NEW_COUNT entries, the original $OLD_COUNT; $HOME_DIR is as it was."
fi

# --- 4. Mount, and keep it mounted across reboots ------------------------------
if ! mount --bind "$NEW" "$HOME_DIR"; then
    put_back
    die "Could not bind-mount $NEW on $HOME_DIR; $HOME_DIR is as it was."
fi
if [ "$EXISTING_SOURCE" != "$NEW" ]; then
    printf '%s\n' "$FSTAB_LINE" >> "$FSTAB"
fi
if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    systemctl daemon-reload 2>/dev/null || true
fi

# --- 5. The original is no longer needed ---------------------------------------
rm -rf "$OLD"
if [ "$WEB_APP_WAS_ACTIVE" = 1 ]; then
    user_systemctl start kiss-web.service && info "kiss-web service started again." \
        || warn "Could not start kiss-web.service again; start it by hand."
fi
info "$HOME_DIR now lives at $NEW ($(df -Pk "$HOME_DIR" | awk 'NR == 2 { printf "%.1f GiB", $4 / 1048576 }') free), mounted from $FSTAB."
