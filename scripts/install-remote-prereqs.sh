#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Install the command-line tools a deploy needs on this machine before the
# checkout is here: git, curl, tar, python3 and the ssh client.
#
# Usage:  ssh user@host 'bash -s' < scripts/install-remote-prereqs.sh
#         scripts/install-remote-prereqs.sh            (on the machine itself)
#
# ./rsorcar runs this as its first action on the remote (step 1a).  A fresh
# cloud image -- Debian's on Google Compute Engine, for one -- ships without
# git, and the sync that follows (scripts/sync-repo.sh) is git from its first
# command.  install.sh knows how to install git, but install.sh runs out of
# the checkout, and the checkout is what the sync creates: so the host's
# package manager is asked here, before either.
#
# Nothing is installed when every tool is already there, which is every run
# after the first.  Root installs directly; anyone else through ``sudo -n``,
# which never prompts (there is no terminal on the other end of ``bash -s``),
# so a host whose sudo wants a password stops with the exact command to run by
# hand rather than hanging.  apt is told to wait for the dpkg lock instead of
# failing when unattended-upgrades holds it in the minutes after a boot.
set -euo pipefail

info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()  { printf '\033[0;31m[ERR]\033[0m  %s\n' "$*" >&2; exit 1; }

HOST="$(hostname -s 2>/dev/null || hostname)"
NEEDED="git curl tar python3 ssh"

MISSING=""
for tool in $NEEDED; do
    command -v "$tool" >/dev/null 2>&1 || MISSING="$MISSING $tool"
done
MISSING="${MISSING# }"
if [ -z "$MISSING" ]; then
    info "The tools the deploy needs are all installed on $HOST ($NEEDED)."
    exit 0
fi

# The package that provides a command differs between distributions only for
# two of the five: the ssh client and python3.
if command -v apt-get >/dev/null 2>&1; then
    PM=apt-get
elif command -v dnf >/dev/null 2>&1; then
    PM=dnf
elif command -v yum >/dev/null 2>&1; then
    PM=yum
elif command -v zypper >/dev/null 2>&1; then
    PM=zypper
elif command -v pacman >/dev/null 2>&1; then
    PM=pacman
elif command -v apk >/dev/null 2>&1; then
    PM=apk
else
    die "$HOST is missing $MISSING and has no package manager this script knows" \
        "(apt-get, dnf, yum, zypper, pacman, apk). Install them by hand, then deploy again."
fi

package_for() {
    case "$1" in
        ssh)
            case "$PM" in
                apt-get|apk) echo openssh-client ;;
                dnf|yum|zypper) echo openssh-clients ;;
                pacman) echo openssh ;;
            esac ;;
        python3)
            case "$PM" in
                pacman) echo python ;;
                *) echo python3 ;;
            esac ;;
        *) echo "$1" ;;
    esac
}
PACKAGES=""
for tool in $MISSING; do
    PACKAGES="$PACKAGES $(package_for "$tool")"
done
PACKAGES="${PACKAGES# }"

install_command() {
    case "$PM" in
        apt-get) echo "apt-get -o DPkg::Lock::Timeout=300 install -y -q $PACKAGES" ;;
        dnf)     echo "dnf install -y -q $PACKAGES" ;;
        yum)     echo "yum install -y -q $PACKAGES" ;;
        zypper)  echo "zypper --non-interactive install $PACKAGES" ;;
        pacman)  echo "pacman -S --noconfirm --needed $PACKAGES" ;;
        apk)     echo "apk add $PACKAGES" ;;
    esac
}

# The command to run by hand when this script cannot: $1 is the prefix each
# command gets ("sudo" for a user, nothing for root).  A fresh Debian image
# has no package lists, so on apt the install alone would end in "Unable to
# locate package git"; the refresh comes first there, as it does below.
manual_command() {
    local prefix="${1:+$1 }"
    case "$PM" in
        apt-get) echo "${prefix}apt-get update && ${prefix}$(install_command)" ;;
        *)       echo "${prefix}$(install_command)" ;;
    esac
}

if [ "$(id -u)" = 0 ]; then
    SUDO=""
elif ! command -v sudo >/dev/null 2>&1; then
    die "$HOST is missing $MISSING, and installing them needs root, but sudo is" \
        "not installed. Run this on $HOST as root, then deploy again:" \
        "$(manual_command)"
elif sudo -n true 2>/dev/null; then
    SUDO="sudo -n"
else
    die "$HOST is missing $MISSING, and installing them needs root, but sudo wants" \
        "a password. Run this on $HOST, then deploy again: $(manual_command sudo)"
fi

info "Installing $MISSING on $HOST with $PM ($PACKAGES) ..."
# sudo resets the environment, so the variable rides on ``env`` through it.
# The word splitting of $SUDO and of the command is intended (SC2046): the
# pieces are option words and package names, none with a space or a glob.
case "$PM" in
    apt-get)
        # A fresh image has no package lists.  A failed refresh (one dead
        # third-party repository is enough) is not fatal by itself: the
        # install below tells whether the lists that are there suffice.
        $SUDO env DEBIAN_FRONTEND=noninteractive \
            apt-get -o DPkg::Lock::Timeout=300 update -q >/dev/null 2>&1 \
            || warn "apt-get update failed on $HOST; trying the install with the package lists it has."
        # shellcheck disable=SC2046
        $SUDO env DEBIAN_FRONTEND=noninteractive $(install_command) >/dev/null \
            || die "apt-get could not install $PACKAGES on $HOST."
        ;;
    *)
        # shellcheck disable=SC2046
        $SUDO $(install_command) >/dev/null \
            || die "$PM could not install $PACKAGES on $HOST."
        ;;
esac

STILL=""
for tool in $MISSING; do
    command -v "$tool" >/dev/null 2>&1 || STILL="$STILL $tool"
done
[ -z "$STILL" ] || die "$PM reported success on $HOST, yet${STILL} still cannot be found on PATH."
info "Installed $MISSING on $HOST."
