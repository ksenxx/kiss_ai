#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Check the Mac package on a clean Mac of another macOS version: boot a
# virtual Mac, install the package in it and run a task there.
#
#   scripts/check-macos-pkg-vm.sh [--intel] <image> <key-name> [<package>]
#
#   <image>     a macOS image for tart with the user admin (password
#               admin), e.g. ghcr.io/cirruslabs/macos-sonoma-vanilla:latest;
#               tart downloads it on first use (about 20 GB)
#   <key-name>  the API key the virtual Mac may use: the name of the
#               environment variable that holds it here (ANTHROPIC_API_KEY)
#   <package>   the package to check (default: installers/KISS-Sorcar.pkg)
#   --intel     install as an Intel Mac does: the Intel builds of uv, Python,
#               the libraries and cloudflared, which Rosetta runs here (the
#               browser stays the Apple silicon one: Playwright looks at the
#               processor).  The closest this Mac comes to an Intel Mac.
#
# The virtual Mac gets what a new user's Mac has: no Homebrew, no git and
# none of Apple's developer tools.  The package, marked as a download from
# the web, is installed for the user admin from the command line, the key
# is saved as the app's Settings save it, and the
# scripts/check-macos-install.sh the package ships runs the tasks.
#
# Needs an Apple silicon Mac with tart (https://tart.run).  The virtual Mac
# is deleted at the end; KEEP_VM=1 keeps it running for a look inside.
# KISS_VM_DNS (8.8.8.8, say) makes it ask that DNS server instead of this
# Mac's, whose resolver for virtual machines does not know every name this
# Mac knows (none that /etc/hosts lists).

set -euo pipefail

VM="kiss-pkg-check-$$"
IP=""  # the virtual Mac's address, once it runs
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
SSH_OPTIONS=(-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o PubkeyAuthentication=no -o ConnectTimeout=10 -o ServerAliveInterval=15)
# ssh asks this program for the password instead of the terminal.
export SSH_ASKPASS="$WORK/askpass" SSH_ASKPASS_REQUIRE=force

die() {
    echo "check-macos-pkg-vm.sh: $*" >&2
    exit 1
}

usage() {
    # Print the usage: this file's header comment from "Check" on.
    sed -n '6,32p' "$0" | sed 's/^# \{0,1\}//'
}

vm_run() {
    # Run the shell command line $1 in the virtual Mac.
    ssh "${SSH_OPTIONS[@]}" "admin@$IP" "$1"
}

remove_vm() {
    # Runs when the script ends, also when the virtual Mac was never made
    # or never ran.
    rm -rf "$WORK"
    if [ -n "${KEEP_VM:-}" ]; then
        echo "The virtual Mac $VM is kept: ssh admin@$IP (password admin);" \
            "tart stop $VM && tart delete $VM removes it."
    else
        tart stop "$VM" > /dev/null 2>&1 || true
        tart delete "$VM" > /dev/null 2>&1 || true
    fi
}

wait_for_install() {
    # Wait until the install the package started has ended (it can outlive
    # the Installer by minutes on a slow connection) and fail unless
    # install.sh succeeded.
    local status=""
    SECONDS=0
    while [ -z "$status" ] && [ "$SECONDS" -lt 1800 ]; do
        sleep 5
        status="$(vm_run 'cat ~/.kiss/.pkg-install-status 2>/dev/null' || true)"
    done
    [ "$(echo "$status" | head -1)" = 0 ] \
        || die "the install failed or did not end in 30 minutes: ${status:-no status}"
}

main() {
    local arch=arm64 installer=installer image key_name package python_arch
    if [ "${1:-}" = --intel ]; then
        arch=x86_64
        # The Installer's scripts run as it does: install.sh sees an Intel Mac.
        installer="arch -x86_64 installer"
        shift
    fi
    if [ $# -lt 2 ] || [ $# -gt 3 ]; then
        usage >&2
        exit 2
    fi
    image="$1"
    key_name="$2"
    package="${3:-$(cd "$(dirname "$0")/.." && pwd)/installers/KISS-Sorcar.pkg}"
    command -v tart > /dev/null || die "tart is not installed (https://tart.run)"
    # ${!key_name}: the value of the variable that $key_name names.
    [ -n "${!key_name:-}" ] || die "the environment variable $key_name holds no API key"
    [ -f "$package" ] || die "no such package: $package"
    printf '#!/bin/sh\necho admin\n' > "$SSH_ASKPASS"
    chmod 700 "$SSH_ASKPASS"

    # Before the clone: one that fails halfway can leave the virtual Mac behind.
    trap remove_vm EXIT
    tart clone "$image" "$VM"
    tart run --no-graphics "$VM" > /dev/null 2>&1 < /dev/null &
    IP="$(tart ip --wait 300 "$VM")"
    until vm_run true; do sleep 5; done
    echo "The virtual Mac runs macOS $(vm_run 'sw_vers -productVersion') at $IP"
    if [ -n "${KISS_VM_DNS:-}" ]; then
        vm_run "echo admin | sudo -S networksetup -setdnsservers Ethernet $KISS_VM_DNS"
    fi

    # A new user's Mac: the image comes with the command line developer tools.
    vm_run 'echo admin | sudo -S rm -rf /Library/Developer/CommandLineTools'
    if [ "$arch" = x86_64 ]; then
        # On an Intel Mac install.sh downloads the Intel uv, and that uv the
        # Intel Python and libraries.  Here uv's installer would see through
        # Rosetta and fetch the Apple silicon uv, so the Intel one is put
        # where install.sh looks for a uv first.
        vm_run 'echo admin | sudo -S softwareupdate --install-rosetta --agree-to-license > /dev/null 2>&1 &&
            mkdir -p ~/.local/bin &&
            curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin.tar.gz |
                LC_ALL=C tar -xz --strip-components 1 -C ~/.local/bin' \
            || die "could not install Rosetta and the Intel uv"
    fi

    scp "${SSH_OPTIONS[@]}" "$package" "admin@$IP:KISS-Sorcar.pkg"
    # Marked as Safari marks a download, which is what Gatekeeper judges.
    vm_run 'xattr -w com.apple.quarantine "0083;$(printf %x "$(date +%s)");Safari;" KISS-Sorcar.pkg'
    vm_run "$installer -pkg KISS-Sorcar.pkg -target CurrentUserHomeDirectory"
    wait_for_install
    python_arch="$(vm_run 'cd ~/.kiss/kiss && .venv/bin/python -c "import platform; print(platform.machine())"')"
    [ "$python_arch" = "$arch" ] || die "the app's Python is an $python_arch one, not $arch"

    # The key is saved with the function the app's Settings save it with:
    # into the key store and, for new Terminal windows, through a line in
    # the login shell's rc file.  It travels as NAME=value on standard
    # input, not on a command line.  The daemon, which Settings would tell,
    # reads the store when it starts.
    # shellcheck disable=SC2088  # the virtual Mac's shell expands the tilde
    vm_run '~/.local/bin/uv run --quiet --directory ~/.kiss/kiss python -c "
import sys
from kiss.core.user_config import save_api_key
save_api_key(*sys.stdin.read().strip().split(\"=\", 1))"' <<< "$key_name=${!key_name}"
    vm_run 'launchctl kickstart -k "gui/$(id -u)/com.kiss.web-server"'
    vm_run 'curl -fs -o /dev/null --retry 30 --retry-delay 2 --retry-connrefused http://127.0.0.1:8787/' \
        || die "the kiss-web daemon did not come back after its restart"
    vm_run "KISS_CHECK_MODEL='${KISS_CHECK_MODEL:-}' bash ~/.kiss/kiss_pkg/scripts/check-macos-install.sh"
}

main "$@"
