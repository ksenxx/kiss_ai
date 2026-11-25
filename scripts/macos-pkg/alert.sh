# shellcheck shell=bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Sourced by the preinstall and postinstall scripts of the KISS Sorcar
# Mac package.

alert() {
    # Show message $1 with the detail $2 in a dialog, without waiting for
    # the user to close it.  Not during a command-line install
    # (``installer -pkg``, which sets COMMAND_LINE_INSTALL): its messages
    # go to the terminal, and there may be no screen to show a dialog on.
    # The dialog gets no standard output of ours: the Installer waits for
    # everything that holds a script's output open.
    [ -z "${COMMAND_LINE_INSTALL:-}" ] || return 0
    osascript -e 'on run argv' \
        -e 'activate' \
        -e 'display alert (item 1 of argv) message (item 2 of argv) as critical' \
        -e 'end run' "$1" "$2" < /dev/null > /dev/null 2>&1 &
}
