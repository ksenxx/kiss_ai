#!/bin/sh
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# KISS Sorcar @VERSION@ for Linux: the installer scripts/build-linux-run.sh
# builds.  Run it in a terminal:
#
#     bash KISS-Sorcar-Linux.run
#
# Everything after the last line of this script is a gzip-compressed tar
# archive, a copy of the KISS Sorcar sources: the payload the Mac package
# and the Windows setup program carry too.  This script unpacks it into
# ~/.kiss/kiss_pkg and runs the install.sh in it, the one installer every
# other way of installing KISS Sorcar ends in.  Its arguments go to
# install.sh (--non-interactive for an automated caller).
#
# It installs for the user who runs it and needs no root: everything goes
# into the home directory.  The machine must be online: install.sh
# downloads uv, Python, the Python packages and Chromium, about 1 GB.
#
# install.sh makes the kiss-web daemon serve the directory it is run
# from, so the directory this file is run from becomes the one KISS
# Sorcar works in.
#
# Written for /bin/sh: ``sh KISS-Sorcar-Linux.run`` works where sh is not
# bash (dash on Debian and Ubuntu).

set -eu

# The line of this file the archive starts at; the builder fills it in.
PAYLOAD_LINE=@PAYLOAD_LINE@
PAYLOAD_DIR="$HOME/.kiss/kiss_pkg"
LOCK="$HOME/.kiss/.update.lock"

die() {
    echo "KISS Sorcar: $*" >&2
    exit 1
}

for tool in bash perl tar gzip tail; do
    command -v "$tool" >/dev/null 2>&1 \
        || die "'$tool' is needed and was not found. Install it and run this file again."
done
# gzip checks the archive's length and checksum: a download that was cut
# short or damaged stops here, before anything is replaced.  So does a
# file piped into a shell, which has no file to read the archive from.
tail -n "+$PAYLOAD_LINE" "$0" 2>/dev/null | gzip -t 2>/dev/null \
    || die "the archive in this file cannot be read: the download is incomplete or damaged, or the file was piped into a shell. Download it again and run: bash KISS-Sorcar-Linux.run"
# Whatever installs or updates KISS Sorcar holds the update lock while it
# works (an earlier run of this file, the app's Update button, the curl
# installer): a flock(2) on the lock file, which the kernel releases
# when the last process that has the descriptor ends.  This file takes
# it before it replaces the payload, as replacing the payload under a
# running install.sh would break that install, and keeps it: install.sh
# below inherits descriptor 9 and is told that it runs under the lock
# (KISS_UPDATE_LOCK_HELD, as scripts/install.sh tells it), so nobody
# replaces the payload under this install either.
mkdir -p "$HOME/.kiss"
exec 9>> "$LOCK"
if ! perl -e 'use Fcntl qw(:flock);
              open(my $f, ">&=", 9) or exit 2;
              exit(flock($f, LOCK_EX | LOCK_NB) ? 0 : 1)'; then
    die "an install or update is running right now. Run this file again when it has finished; its progress is in ~/.kiss/install.log."
fi
echo "$$" > "$LOCK"
export KISS_UPDATE_LOCK_HELD=1

echo "KISS Sorcar @VERSION@: unpacking into $PAYLOAD_DIR"
# Unpacked beside the old payload and swapped in whole, so that a file
# the new version no longer ships is not installed from the old copy.
# ``-o``: the files belong to the user who runs this, also for root.
rm -rf "$PAYLOAD_DIR.new"
mkdir -p "$PAYLOAD_DIR.new"
tail -n "+$PAYLOAD_LINE" "$0" | gzip -dc | tar -xof - -C "$PAYLOAD_DIR.new"
rm -rf "$PAYLOAD_DIR"
mv "$PAYLOAD_DIR.new" "$PAYLOAD_DIR"

exec bash "$PAYLOAD_DIR/install.sh" "$@"
# Not reached: exec replaced the shell.  A shell that goes on all the
# same must not read the archive below as commands.
exit 1
