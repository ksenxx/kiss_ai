#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Build the Windows setup program of KISS Sorcar.
#
#   scripts/build-windows-exe.sh [-o EXE]
#
#   -o, --output EXE     the setup program to write
#                        (default: dist/KISS-Sorcar-Setup-<version>.exe)
#   -h, --help           print this text
#
# The setup program is what a Windows user double-clicks to install KISS
# Sorcar on a PC that has nothing set up: no git, no Python, no bash.  It
# installs on 64-bit Windows 10 and later (x64, and ARM through its x64
# emulation).
#
# It installs for the user who runs it and nobody else: everything goes
# into the user's profile and it asks for no administrator rights.
#
# Its payload is the Mac package's (scripts/installer-payload.sh): a copy
# of the sources of this checkout, the files install.sh installs and
# install.sh itself with the scripts it needs, unpacked into
# %USERPROFILE%\.kiss\kiss_pkg.  scripts/windows-exe/setup.ps1 then runs
# that install.sh with Git Bash, after downloading Git for Windows into
# the profile when the PC has none: it downloads uv, Python, the Python
# packages and Chromium (about 1 GB, so the PC must be online), builds
# the app in %USERPROFILE%\.kiss\kiss, starts the kiss-web daemon and
# opens the app's window.  "KISS-Sorcar-Setup.exe /S" installs without a
# window.  The uninstaller it registers (Settings > Apps) removes the app
# and keeps the user's settings and history.
#
# The program is built with NSIS (scripts/windows-exe/installer.nsi) on
# macOS, Linux or Windows: ``brew install makensis``, ``apt install
# nsis``.  It is not signed: Windows SmartScreen shows "Windows protected
# your PC" for a downloaded copy until the user picks "More info", "Run
# anyway".
#
# scripts/windows-exe/kiss-sorcar.ico is the app's icon
# (src/kiss/webapp/media/kiss-icon.png) as kiss.server.standalone_app's
# write_ico converts it.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCES="$PROJECT_DIR/scripts/windows-exe"

die() {
    echo "build-windows-exe.sh: $*" >&2
    exit 1
}

usage() {
    # Print the usage: this file's header comment from "Build" to the
    # option list.
    sed -n '6,12p' "$0" | sed 's/^# \{0,1\}//'
}

# project_version and stage_payload: shared with the Mac package.
# shellcheck source=scripts/installer-payload.sh
. "$PROJECT_DIR/scripts/installer-payload.sh"

main() {
    local output="" version
    while [ $# -gt 0 ]; do
        case "$1" in
            -o|--output) [ $# -ge 2 ] || die "$1 needs a value"; output="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) usage >&2; die "unknown argument: $1" ;;
        esac
    done
    command -v makensis >/dev/null 2>&1 \
        || die "makensis not found: install NSIS ('brew install makensis', 'apt install nsis')."
    version="$(project_version)"
    [ -n "$version" ] || die "no version found in $PROJECT_DIR/src/kiss/core/_version.py"
    [ -n "$output" ] || output="$PROJECT_DIR/dist/KISS-Sorcar-Setup-$version.exe"
    mkdir -p "$(dirname "$output")"
    # makensis resolves OutFile against the script's directory.
    output="$(cd "$(dirname "$output")" && pwd)/$(basename "$output")"

    WORK_DIR="$(mktemp -d)"
    trap 'rm -rf "$WORK_DIR"' EXIT
    stage_payload "$WORK_DIR/payload"
    # ``-D``, not ``/D``: the only spelling makensis reads on every
    # platform (a ``/D...`` argument is a file path off Windows).
    makensis -V2 -DVERSION="$version" -DPAYLOAD="$WORK_DIR/payload" \
        -DRESOURCES="$RESOURCES" -DOUTPUT="$output" "$RESOURCES/installer.nsi" \
        || die "makensis failed"
    echo "Built $output (KISS Sorcar $version, $(du -h "$output" | cut -f1 | tr -d ' '))"
    echo "It is not signed: Windows SmartScreen asks before it runs a downloaded copy"
    echo "(\"More info\", then \"Run anyway\")."
    echo "To install it without a window: $(basename "$output") /S"
}

main "$@"
