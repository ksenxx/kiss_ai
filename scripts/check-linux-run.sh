#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Check the Linux installer on a clean Linux: start a container of a
# distribution, install in it as a new user does and check the app.
#
#   scripts/check-linux-run.sh <image> [<key-name> [<installer>]]
#
#   <image>      a Docker image of the distribution: ubuntu:26.04,
#                debian:13, fedora:45, opensuse/tumbleweed, archlinux
#   <key-name>   the API key the container may use to run two tasks: the
#                name of the environment variable that holds it here
#                (ANTHROPIC_API_KEY); without it, or with "", no task runs
#   <installer>  the installer to check
#                (default: installers/KISS-Sorcar-Linux.run)
#
# The container gets what a desktop install of the distribution has and
# its image lacks (scripts/linux-run/desktop-setup.sh): perl, lsof, the
# libraries Chromium needs and a download tool, which on Debian and Ubuntu
# is wget alone (their desktops ship no curl and no git).  The user
# "tester" has no sudo.  The installer runs for that user at a terminal,
# from the home directory, as typed into a new terminal window; then
# scripts/check-linux-install.sh, which the installer ships, checks the
# app.  The installer runs a second time over the first install, and the
# app is checked again, with a key by running two tasks.
#
# A container has no systemd, so kiss-web runs as a detached process
# here; scripts/check-linux-vm.sh checks the installer in a virtual
# machine, where the daemon is the user's systemd service.
#
# KISS_CHECK_MODEL names the model of the tasks (default: the best model
# for the key).  KEEP_CONTAINER=1 leaves the container running for a look
# inside.

set -euo pipefail

NAME="kiss-linux-check-$$"
SETUP="$(cd "$(dirname "$0")" && pwd)/linux-run/desktop-setup.sh"
# What a terminal of the user gives a command.
AS_TESTER=(-u tester -w /home/tester -e USER=tester -e LOGNAME=tester)
CHECK=/home/tester/.kiss/kiss_pkg/scripts/check-linux-install.sh

die() {
    echo "check-linux-run.sh: $*" >&2
    exit 1
}

usage() {
    # Print the usage: this file's header comment from "Check" on.
    sed -n '6,35p' "$0" | sed 's/^# \{0,1\}//'
}

remove_container() {
    if [ -n "${KEEP_CONTAINER:-}" ]; then
        echo "The container $NAME is left running; remove it with: docker rm -f $NAME"
    else
        docker rm -f "$NAME" > /dev/null 2>&1 || true
    fi
}

run_installer() {
    # Run the installer as the user does: at a terminal (-t), from the
    # home directory.
    docker exec -t "${AS_TESTER[@]}" "$NAME" bash /home/tester/Downloads/KISS-Sorcar-Linux.run \
        || die "the installer failed"
}

check_install() {
    # Run the check the installer ships, with the arguments given.
    docker exec -i "${AS_TESTER[@]}" -e KISS_CHECK_MODEL="${KISS_CHECK_MODEL:-}" "$NAME" \
        bash "$CHECK" "$@" || die "the check of the installed app failed"
}

save_key_and_stop_daemon() {
    # Save the API key named $1 the way the app's Settings do (it travels
    # as NAME=value on standard input, not on a command line) and stop
    # the daemon, which reads the key store when it starts: the second
    # run of the installer starts it again.
    docker exec -i "${AS_TESTER[@]}" "$NAME" /home/tester/.kiss/kiss/.venv/bin/python -c '
import json, os, pathlib, signal, sys
from kiss.core.user_config import save_api_key
save_api_key(*sys.stdin.read().strip().split("=", 1))
os.kill(json.loads((pathlib.Path.home() / ".kiss/sorcar-local.json").read_text())["pid"], signal.SIGTERM)
' <<< "$1=${!1}"
}

main() {
    local image key_name installer
    case "${1:-}" in
        -h|--help) usage; exit 0 ;;
        "") usage >&2; exit 2 ;;
    esac
    image="$1"
    key_name="${2:-}"
    installer="${3:-$(cd "$(dirname "$0")/.." && pwd)/installers/KISS-Sorcar-Linux.run}"
    command -v docker > /dev/null || die "docker is not installed"
    # ${!key_name}: the value of the variable that $key_name names.
    [ -z "$key_name" ] || [ -n "${!key_name:-}" ] || die "the environment variable $key_name holds no API key"
    [ -f "$installer" ] || die "no such installer: $installer"

    # Before the start: one that fails halfway can leave the container behind.
    trap remove_container EXIT
    docker run -d --name "$NAME" "$image" sleep infinity > /dev/null
    # Make the container the machine of a new desktop user.
    docker exec -i "$NAME" sh -s < "$SETUP"
    docker cp -q "$installer" "$NAME:/home/tester/Downloads/KISS-Sorcar-Linux.run"

    echo "== the installer on a new machine"
    run_installer
    check_install
    [ -z "$key_name" ] || save_key_and_stop_daemon "$key_name"
    echo "== the installer again, over the first install"
    run_installer
    check_install ${key_name:+--tasks}
    echo "PASS: the installer works on $image ($(docker exec "$NAME" uname -m))"
}

main "$@"
