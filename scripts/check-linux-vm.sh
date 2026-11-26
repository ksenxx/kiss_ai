#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Check the Linux installer on a real Linux machine: boot a virtual machine
# of a distribution, install in it as a new desktop user does and check
# the app.
#
#   scripts/check-linux-vm.sh <template> [<key-name> [<installer>]]
#
#   <template>   a Linux template of lima (limactl start --list-templates):
#                ubuntu-26.04, ubuntu-26.10, debian-13, fedora-45,
#                experimental/opensuse-tumbleweed; lima downloads the
#                distribution's cloud image on first use
#   <key-name>   the API key the machine may use to run two tasks: the
#                name of the environment variable that holds it here
#                (ANTHROPIC_API_KEY); without it, or with "", no task runs
#   <installer>  the installer to check
#                (default: installers/KISS-Sorcar-Linux.run)
#
# Unlike the container of scripts/check-linux-run.sh, the machine boots the
# distribution's own kernel and systemd, so the kiss-web daemon is the
# user's systemd service.  The machine gets what a desktop install has and
# a cloud image lacks (scripts/linux-run/desktop-setup.sh): on Debian and
# Ubuntu wget and no curl, the libraries Chromium needs, a file manager.
# The user "tester" has no sudo.  Then:
#
#   1. the installer runs for tester at a terminal over ssh, from the home
#      directory, and scripts/check-linux-install.sh, which the installer
#      ships, checks the app in a new session;
#   2. the machine restarts, and the daemon must serve the app again while
#      tester is not logged in;
#   3. the installer runs again on a display (Xvfb), as typed into a
#      terminal window of the desktop: the app must open its window and
#      install a valid launcher; the app is checked again, with a key by
#      running two tasks.
#
# Needs lima (https://lima-vm.io: brew install lima).  The machine is
# deleted at the end; KEEP_VM=1 keeps it running for a look inside.
# KISS_VM_DNS (8.8.8.8, say) makes it ask that DNS server instead of this
# machine's resolver.  SHOTS names a directory for a picture of the display
# with the app's window.  KISS_CHECK_MODEL names the model of the tasks
# (default: the best model for the key).

set -euo pipefail

VM="kiss-linux-check-$$"
SETUP="$(cd "$(dirname "$0")" && pwd)/linux-run/desktop-setup.sh"
# In tester's home directory, where an ssh command starts.
CHECK=.kiss/kiss_pkg/scripts/check-linux-install.sh
SSH_OPTIONS=()  # how ssh reaches the machine, once it runs

die() {
    echo "check-linux-vm.sh: $*" >&2
    exit 1
}

usage() {
    # Print the usage: this file's header comment from "Check" on.
    sed -n '6,44p' "$0" | sed 's/^# \{0,1\}//'
}

admin() {
    # Run the shell command line $1 in the machine as its administrator.
    ssh "${SSH_OPTIONS[@]}" "lima-$VM" "$1"
}

tester() {
    # Run the shell command line $1 in the machine in a new login session
    # of the user tester.
    ssh "${SSH_OPTIONS[@]}" -l tester "lima-$VM" "$1"
}

at_terminal() {
    # Run the command line $1 as tester at a terminal, as typed into one.
    ssh "${SSH_OPTIONS[@]}" -tt -l tester "lima-$VM" "$1"
}

remove_vm() {
    # Runs when the script ends, also when the machine was never made.
    if [ -n "${KEEP_VM:-}" ]; then
        echo "The machine $VM is kept: limactl shell $VM opens a shell in it;" \
            "limactl delete -f $VM removes it."
    else
        limactl delete -f "$VM" > /dev/null 2>&1 || true
    fi
}

start_machine() {
    # Boot a machine of the lima template $1 and make it the machine of a
    # new desktop user, with tester's login.
    # --plain: nothing of this machine in the virtual one, no folder of it
    # and none of its ports (the app has the same port on both).
    limactl --log-level warn start --name "$VM" --tty=false --plain \
        --cpus 4 --memory 4 --disk 20 ${KISS_VM_DNS:+--dns "$KISS_VM_DNS"} "template:$1" \
        || die "lima could not start a machine of the template $1"
    # ssh as lima set it up for the machine, without the connection lima
    # keeps open as the administrator: a login as tester must not reuse it.
    SSH_OPTIONS=(-F "$(limactl list "$VM" --format '{{.SSHConfigFile}}')"
        -o ControlMaster=no -o ControlPath=none -o LogLevel=ERROR)
    admin 'sudo sh -s display' < "$SETUP"
    # tester logs in with the key lima gave the administrator (restorecon:
    # the label SELinux wants on the file, where the machine has SELinux).
    admin 'sudo install -d -m 700 -o tester -g tester /home/tester/.ssh &&
        sudo install -m 600 -o tester -g tester ~/.ssh/authorized_keys /home/tester/.ssh/ &&
        { ! command -v restorecon > /dev/null || sudo restorecon -R /home/tester/.ssh; }'
}

restart_machine() {
    # Restart the machine and wait until the daemon serves the app again,
    # which it must do while tester is not logged in.
    local old new="" classes
    old="$(admin 'cat /proc/sys/kernel/random/boot_id')"
    admin 'sudo systemctl reboot' || true  # ssh can end with the machine
    SECONDS=0
    until [ -n "$new" ] && [ "$new" != "$old" ] \
        && admin 'wget -qO- http://127.0.0.1:8787/ | grep -q "KISS Sorcar"'; do
        [ "$SECONDS" -lt 300 ] || die "the daemon does not serve the app after the restart of the machine"
        sleep 5
        new="$(admin 'cat /proc/sys/kernel/random/boot_id' 2> /dev/null || true)"
    done
    # A login is a session of the class "user".  Since systemd 256 the
    # service manager of a user who stays without a login has a session
    # too, of the class "manager".
    classes="$(admin 'for session in $(loginctl show-user tester --value -p Sessions); do
        loginctl show-session "$session" --value -p Class; done')"
    if grep -q '^user' <<< "$classes"; then
        die "tester is logged in after the restart of the machine"
    fi
}

save_key() {
    # Save the API key named $1 the way the app's Settings do (it travels
    # as NAME=value on standard input, not on a command line).  The daemon
    # reads the key store when it starts: the next run of the installer
    # restarts it.
    # shellcheck disable=SC2088  # the machine's shell expands the tilde
    tester '~/.kiss/kiss/.venv/bin/python -c "
import sys
from kiss.core.user_config import save_api_key
save_api_key(*sys.stdin.read().strip().split(\"=\", 1))"' <<< "$1=${!1}"
}

install_on_display() {
    # Run the installer again as typed into a terminal window of the
    # desktop, which has a display and nothing of ssh (by which install.sh
    # tells a remote machine), and check the app's window and launcher.
    tester 'nohup Xvfb :99 -screen 0 1440x900x24 > /dev/null 2>&1 &'
    at_terminal 'env -u SSH_CONNECTION -u SSH_CLIENT -u SSH_TTY DISPLAY=:99 bash ~/Downloads/KISS-Sorcar-Linux.run' \
        || die "the installer failed on the display"
    # The window's class is the one the launcher names (--class).
    SECONDS=0
    until tester 'DISPLAY=:99 xwininfo -root -tree | grep -q kiss-sorcar'; do
        [ "$SECONDS" -lt 30 ] || die "the app did not open its window on the display"
        sleep 2
    done
    tester 'desktop-file-validate ~/.local/share/applications/kiss-sorcar.desktop' \
        || die "the launcher of the app is not a valid desktop entry"
    if [ -n "${SHOTS:-}" ]; then
        mkdir -p "$SHOTS"
        tester 'DISPLAY=:99 import -window root png:-' > "$SHOTS/$(basename "$1").png"
    fi
}

main() {
    local template key_name installer
    case "${1:-}" in
        -h|--help) usage; exit 0 ;;
        "") usage >&2; exit 2 ;;
    esac
    template="$1"
    key_name="${2:-}"
    installer="${3:-$(cd "$(dirname "$0")/.." && pwd)/installers/KISS-Sorcar-Linux.run}"
    command -v limactl > /dev/null || die "lima is not installed (https://lima-vm.io)"
    # ${!key_name}: the value of the variable that $key_name names.
    [ -z "$key_name" ] || [ -n "${!key_name:-}" ] || die "the environment variable $key_name holds no API key"
    [ -f "$installer" ] || die "no such installer: $installer"

    # Before the start: one that fails halfway can leave the machine behind.
    trap remove_vm EXIT
    start_machine "$template"
    # tester's own file, in the folder a browser downloads into.
    tester 'cat > ~/Downloads/KISS-Sorcar-Linux.run' < "$installer"

    echo "== the installer on a new machine, at a terminal over ssh"
    at_terminal 'bash ~/Downloads/KISS-Sorcar-Linux.run' || die "the installer failed"
    tester "bash $CHECK" || die "the check of the installed app failed"
    echo "== the machine restarts"
    restart_machine
    echo "The daemon serves the app again, and tester is not logged in"
    [ -z "$key_name" ] || save_key "$key_name"
    echo "== the installer again, on a display"
    install_on_display "$template"
    tester "KISS_CHECK_MODEL='${KISS_CHECK_MODEL:-}' bash $CHECK ${key_name:+--tasks}" \
        || die "the check of the installed app failed after the second install"
    echo "PASS: the installer works in a virtual machine of $template ($(admin 'uname -m'))"
}

main "$@"
