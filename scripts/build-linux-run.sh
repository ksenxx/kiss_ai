#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Build the Linux installer of KISS Sorcar.
#
#   scripts/build-linux-run.sh [-o RUN]
#
#   -o, --output RUN     the installer to write
#                        (default: dist/KISS-Sorcar-Linux-<version>.run)
#   -h, --help           print this text
#
# The installer is what a Linux user downloads and runs in a terminal,
# ``bash KISS-Sorcar-Linux.run``, to install KISS Sorcar on a machine
# that has nothing set up: no git, no Python, no curl.  It is one file
# for every distribution and for x86-64 and ARM64: it holds sources only,
# and what depends on the machine is downloaded when it runs.
#
# It installs for the user who runs it and nobody else: everything goes
# into the home directory and it asks for no root.
#
# The file is a shell script (scripts/linux-run/header.sh) followed by a
# gzip-compressed tar archive of the payload the Mac package and the
# Windows setup program carry (scripts/installer-payload.sh): a copy of
# the sources of this checkout, the files install.sh installs and
# install.sh itself with the scripts it needs.  The script unpacks the
# archive into ~/.kiss/kiss_pkg and runs that install.sh, which downloads
# uv, Python, the Python packages and Chromium (about 1 GB, so the
# machine must be online), builds the app in ~/.kiss/kiss, starts the
# kiss-web daemon and opens the app's window.
#
# It builds on macOS and on Linux, with tar and gzip alone.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HEADER="$PROJECT_DIR/scripts/linux-run/header.sh"

die() {
    echo "build-linux-run.sh: $*" >&2
    exit 1
}

usage() {
    # Print the usage: this file's header comment from "Build" to the
    # option list.
    sed -n '6,12p' "$0" | sed 's/^# \{0,1\}//'
}

# project_version and stage_payload: shared with the Mac package and the
# Windows setup program.
# shellcheck source=scripts/installer-payload.sh
. "$PROJECT_DIR/scripts/installer-payload.sh"

write_installer() {
    # Write the installer of version $1 at $2: the header, with the
    # version and the line its archive starts at filled in, then the
    # archive of the payload staged in directory $3.
    local lines
    lines="$(wc -l < "$HEADER" | tr -d ' ')"
    sed -e "s/@VERSION@/$1/g" -e "s/@PAYLOAD_LINE@/$((lines + 1))/" "$HEADER" > "$2"
    # COPYFILE_DISABLE: no "._name" entries from the tar of macOS.
    # ustar: the format every tar reads without a warning.
    (cd "$3" && COPYFILE_DISABLE=1 tar -cf - --format ustar .) | gzip -9 >> "$2"
    chmod 755 "$2"
}

main() {
    local output="" version
    while [ $# -gt 0 ]; do
        case "$1" in
            -o|--output) [ $# -ge 2 ] || die "$1 needs a value"; output="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) usage >&2; die "unknown argument: $1" ;;
        esac
    done
    version="$(project_version)"
    [ -n "$version" ] || die "no version found in $PROJECT_DIR/src/kiss/core/_version.py"
    [ -n "$output" ] || output="$PROJECT_DIR/dist/KISS-Sorcar-Linux-$version.run"
    mkdir -p "$(dirname "$output")"

    # Global: the trap runs after main has returned.
    WORK_DIR="$(mktemp -d)"
    trap 'rm -rf "$WORK_DIR"' EXIT
    stage_payload "$WORK_DIR/payload"
    write_installer "$version" "$output" "$WORK_DIR/payload"
    echo "Built $output (KISS Sorcar $version, $(du -h "$output" | cut -f1 | tr -d ' '))"
    echo "To install with it: bash $(basename "$output")"
}

main "$@"
