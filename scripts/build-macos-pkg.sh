#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Build the Mac installation package of KISS Sorcar.
#
#   scripts/build-macos-pkg.sh [-o PKG] [--sign IDENTITY [--keychain FILE]] [--notarize PROFILE]
#
#   -o, --output PKG     the package to write
#                        (default: dist/KISS-Sorcar-<version>.pkg)
#   --sign IDENTITY      sign the package with this certificate of the
#                        keychain, a "Developer ID Installer: ..." one
#   --keychain FILE      the keychain that holds the certificate, when it
#                        is not one of the default ones
#   --notarize PROFILE   have Apple notarize the signed package and staple
#                        the ticket to it; PROFILE names credentials saved
#                        with ``xcrun notarytool store-credentials``
#   -h, --help           print this text
#
# The package is what a Mac user double-clicks to install KISS Sorcar on a
# Mac that has nothing set up: no git, no Homebrew, no Python and none of
# Apple's developer tools (neither it nor the app ever brings up the
# "install the command line developer tools" dialog).  It installs on
# macOS 14 and later, on Apple silicon and on Intel.
#
# It installs for the user who opens it and nobody else: everything goes
# into the home directory, the Installer asks for no administrator
# password and runs the package's scripts as that user.
#
# Its payload is a copy of the sources of this checkout, the files
# install.sh installs and install.sh itself with the scripts it needs,
# which the Installer writes to ~/.kiss/kiss_pkg.  The package's
# postinstall script (scripts/macos-pkg/postinstall) then runs that
# install.sh, the one installer every other way of installing KISS Sorcar
# ends in: it downloads uv, Python, the Python packages and Chromium
# (about 1 GB, so the Mac must be online), builds the app in ~/.kiss/kiss,
# starts the kiss-web daemon and opens the app's window.  The installed
# app is the one the curl installer installs, and its Update button
# updates it the same way.
#
# An unsigned package installs only after the user has allowed it under
# System Settings > Privacy & Security > "Open Anyway".  A package for
# other people to download should be signed and notarized: --sign and
# --notarize.
#
# macOS 14 is the oldest system Playwright has a Chromium for, the browser
# of the agent's browser tools and of the app's window ("Playwright does
# not support chromium on mac13"); uv and Python run on older ones.
# scripts/check-macos-pkg-vm.sh checks the package on a clean Mac of any
# of these systems.

set -euo pipefail

IDENTIFIER="org.kiss-sorcar.pkg"
# Under the home directory: where the Installer writes the payload.
INSTALL_LOCATION="/.kiss/kiss_pkg"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCES="$PROJECT_DIR/scripts/macos-pkg"

die() {
    echo "build-macos-pkg.sh: $*" >&2
    exit 1
}

usage() {
    # Print the usage: this file's header comment from "Build" to the
    # option list.
    sed -n '6,19p' "$0" | sed 's/^# \{0,1\}//'
}

# project_version and stage_payload: shared with the Windows setup program.
# shellcheck source=scripts/installer-payload.sh
. "$PROJECT_DIR/scripts/installer-payload.sh"

stage_scripts() {
    # Copy the package's preinstall and postinstall scripts into directory $1.
    mkdir -p "$1"
    cp "$RESOURCES/preinstall" "$RESOURCES/postinstall" "$RESOURCES/alert.sh" "$1/"
    chmod 755 "$1/preinstall" "$1/postinstall"
}

strip_extended_attributes() {
    # Remove the extended attributes of the files from component package
    # $1.  pkgbuild archives them as a "._name" entry next to each file,
    # and macOS marks every file created from a downloaded app (a terminal
    # other than Apple's, an editor) with one that nobody can remove,
    # com.apple.provenance: a package built there held such an entry for
    # each of its files.  They mean nothing on another Mac.
    local dir="$1.expanded" count
    pkgutil --expand "$1" "$dir"
    # tar reads the entries of the cpio archive (@) and writes them again.
    tar -c --format odc --exclude '._*' -f - "@$dir/Payload" | gzip > "$dir/Payload.new"
    mv "$dir/Payload.new" "$dir/Payload"
    lsbom "$dir/Bom" | grep -v '/\._' | mkbom -i /dev/stdin "$dir/Bom.new"
    mv "$dir/Bom.new" "$dir/Bom"
    count="$(lsbom -s "$dir/Bom" | wc -l | tr -d ' ')"
    sed -i '' "s/numberOfFiles=\"[0-9]*\"/numberOfFiles=\"$count\"/" "$dir/PackageInfo"
    find "$dir/Scripts" -name '._*' -delete
    rm "$1"
    pkgutil --flatten "$dir" "$1"
    rm -rf "$dir"
}

build_component() {
    # Build the component package of version $1, the payload with the
    # scripts, at $2 in the work directory $3.
    stage_payload "$3/payload"
    stage_scripts "$3/scripts"
    mkdir -p "$(dirname "$2")"
    pkgbuild --quiet --root "$3/payload" --install-location "$INSTALL_LOCATION" \
        --identifier "$IDENTIFIER" --version "$1" --scripts "$3/scripts" "$2"
    strip_extended_attributes "$2"
}

build_product() {
    # Build the package the user opens, of version $1 at $2, from the
    # component packages and the Installer's texts in the work directory
    # $3, signed with identity $4 (of keychain $5) unless it is empty.
    local sign=()
    mkdir -p "$3/resources" "$(dirname "$2")"
    cp "$RESOURCES/welcome.txt" "$RESOURCES/conclusion.txt" "$3/resources/"
    sed "s/@VERSION@/$1/g" "$RESOURCES/distribution.xml" > "$3/distribution.xml"
    if [ -n "$4" ]; then
        sign=(--sign "$4")
        [ -z "$5" ] || sign+=(--keychain "$5")
    fi
    # ${sign[@]+...}: an empty array is "unbound" for the bash 3.2 of macOS.
    productbuild --quiet --distribution "$3/distribution.xml" \
        --package-path "$3/packages" --resources "$3/resources" \
        ${sign[@]+"${sign[@]}"} "$2"
}

notarize_package() {
    # Have Apple notarize package $1 with the notarytool keychain profile
    # $2 and staple the ticket to it, so that it opens on a Mac that is
    # offline too.
    xcrun notarytool submit "$1" --keychain-profile "$2" --wait
    xcrun stapler staple "$1"
}

main() {
    local output="" identity="" keychain="" profile="" version
    while [ $# -gt 0 ]; do
        case "$1" in
            -o|--output) [ $# -ge 2 ] || die "$1 needs a value"; output="$2"; shift 2 ;;
            --sign) [ $# -ge 2 ] || die "$1 needs a value"; identity="$2"; shift 2 ;;
            --keychain) [ $# -ge 2 ] || die "$1 needs a value"; keychain="$2"; shift 2 ;;
            --notarize) [ $# -ge 2 ] || die "$1 needs a value"; profile="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) usage >&2; die "unknown argument: $1" ;;
        esac
    done
    [ "$(uname -s)" = Darwin ] || die "a Mac package is built on macOS (pkgbuild, productbuild)."
    [ -z "$profile" ] || [ -n "$identity" ] || die "--notarize needs --sign: Apple notarizes signed packages only."
    [ -z "$keychain" ] || [ -n "$identity" ] || die "--keychain needs --sign."
    version="$(project_version)"
    [ -n "$version" ] || die "no version found in $PROJECT_DIR/src/kiss/core/_version.py"
    [ -n "$output" ] || output="$PROJECT_DIR/dist/KISS-Sorcar-$version.pkg"

    # Global: the trap runs after main has returned.
    WORK_DIR="$(mktemp -d)"
    trap 'rm -rf "$WORK_DIR"' EXIT
    build_component "$version" "$WORK_DIR/packages/kiss-sorcar.pkg" "$WORK_DIR"
    build_product "$version" "$output" "$WORK_DIR" "$identity" "$keychain"
    [ -z "$profile" ] || notarize_package "$output" "$profile"

    echo "Built $output (KISS Sorcar $version, $(du -h "$output" | cut -f1 | tr -d ' '))"
    if [ -z "$identity" ]; then
        echo "It is not signed: a Mac that downloads it opens it only after the user allows it"
        echo "under System Settings > Privacy & Security > \"Open Anyway\" (see --sign, --notarize)."
    elif [ -z "$profile" ]; then
        echo "It is signed but not notarized: Gatekeeper still stops it on other Macs (see --notarize)."
    fi
    echo "To install it from a terminal: installer -pkg '$output' -target CurrentUserHomeDirectory"
}

main "$@"
