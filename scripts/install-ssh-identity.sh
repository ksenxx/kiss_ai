#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Receive the ssh identity a deploy sends and put it into ~/.ssh/ without
# destroying anything that is already there.
#
# Usage:  tar -cf - -C ~/.ssh id_rsa id_rsa.pub config ... \
#             | ssh user@host 'SSH_BACKUP=$HOME/.kiss/ssh-replaced-<time> bash -s' \
#         -- except that ./rsorcar copies this file to ~/.kiss/ first and runs
#         it from there, because standard input carries the tar stream.
#
# ./rsorcar used to do this with ``rsync --backup``, which needs rsync on the
# remote, and a fresh cloud image (Debian 13, for one) does not have it: the
# deploy stopped at "rsync: command not found".  tar is on every Linux
# machine, so the files now travel as a tar stream over the ssh connection
# itself and this script does on the remote what rsync's options did:
#
#   * every file in the stream ends up in ~/.ssh/ under its relative path;
#   * a file of the remote's own that a file in the stream would replace,
#     and whose content differs, is moved into $SSH_BACKUP/ (under the same
#     relative path) rather than overwritten: a private key cannot be
#     reconstructed.  The directory is created only if there was something
#     to put there, and it is outside ~/.ssh so ssh never reads it;
#   * a file whose content is already identical is replaced silently;
#   * what is not in the stream is never touched — in particular
#     ``authorized_keys``, which the sender never puts in it: that file is
#     what lets the sender SSH *into* this machine;
#   * ssh refuses to use a key that others can read, so the permissions of
#     ~/.ssh/ are fixed afterwards: 700 on the directory, 600 on every file,
#     644 on public keys and known_hosts.
#
# Environment:
#   SSH_BACKUP   where the replaced files go (required, absolute path)
set -euo pipefail

[ -n "${SSH_BACKUP:-}" ] \
    || { echo "install-ssh-identity.sh: SSH_BACKUP is not set." >&2; exit 1; }
command -v tar >/dev/null 2>&1 \
    || { echo "install-ssh-identity.sh: tar is not installed on $(hostname -s 2>/dev/null || echo this machine)." >&2; exit 1; }

SSH_DIR="$HOME/.ssh"
mkdir -p "$SSH_DIR" "$HOME/.kiss"
chmod 700 "$SSH_DIR" "$HOME/.kiss"

# The stream is unpacked beside its destination first, so a transfer that
# fails on the way (tar reports a file cut short) leaves ~/.ssh/ as it was:
# nothing is moved into place until tar is done.  --no-same-owner: a deploy
# run as root would otherwise keep the sender's numeric owner on the files,
# and ssh refuses a key that belongs to someone else.
STAGE="$(mktemp -d "$HOME/.kiss/ssh-incoming.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
tar -xf - --no-same-owner -C "$STAGE"

cd "$STAGE"
find . -mindepth 1 \( -type f -o -type l \) -print0 | while IFS= read -r -d '' rel; do
    rel="${rel#./}"
    dest="$SSH_DIR/$rel"
    if [ -e "$dest" ] || [ -L "$dest" ]; then
        if [ -L "$dest" ] && [ -L "$rel" ] \
           && [ "$(readlink "$rel")" = "$(readlink "$dest")" ]; then
            : # the same link is already there
        elif [ -f "$dest" ] && [ ! -L "$dest" ] && [ -f "$rel" ] && [ ! -L "$rel" ] \
           && cmp -s "$rel" "$dest"; then
            : # the same content is already there: nothing worth keeping
        else
            mkdir -p "$SSH_BACKUP/$(dirname "$rel")"
            mv -f "$dest" "$SSH_BACKUP/$rel"
        fi
    fi
    mkdir -p "$(dirname "$dest")"
    mv -f "$rel" "$dest"
done

chmod 700 "$SSH_DIR"
find "$SSH_DIR" -type f -exec chmod 600 {} +
find "$SSH_DIR" -type f \( -name '*.pub' -o -name 'known_hosts*' \) -exec chmod 644 {} +
if [ -d "$SSH_BACKUP" ]; then
    chmod -R go-rwx "$SSH_BACKUP"
    # Shown with ~ for the home directory.  Not ${SSH_BACKUP/#$HOME/~}: in
    # bash 5 the replacement undergoes tilde expansion, which puts $HOME back.
    case "$SSH_BACKUP" in
        "$HOME"/*) SHOWN="~${SSH_BACKUP#"$HOME"}" ;;
        *) SHOWN="$SSH_BACKUP" ;;
    esac
    echo "[$(hostname -s)] kept the ssh files this copy replaced in $SHOWN:"
    find "$SSH_BACKUP" \( -type f -o -type l \) | sed "s|^$SSH_BACKUP/|         |"
fi
