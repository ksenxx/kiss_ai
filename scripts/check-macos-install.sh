#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Check that the KISS Sorcar installed on this Mac can run a task.
#
#   scripts/check-macos-install.sh
#
# Run it on a Mac where the Mac package (scripts/build-macos-pkg.sh) or
# install.sh has finished and the API key of a model provider has been
# entered in the app's Settings.  It checks that the install recorded no
# problem, that the app is in ~/Applications and the daemon registered with
# launchd, then runs two small tasks and checks the files each leaves behind:
#
#   1. with the sorcar command in a new login shell, as typed into a new
#      Terminal window (the agent runs without the daemon): the output of a
#      shell command;
#   2. on the kiss-web daemon, as the app does: the same, and the title of
#      a web page the agent opens in its browser.
#
# A task may spend at most 2 USD.  KISS_CHECK_MODEL names the model to use
# (default: the best model for the configured API keys).
#
# The package ships this file (~/.kiss/kiss_pkg/scripts), and
# scripts/check-macos-pkg-vm.sh runs it on a clean virtual Mac.

set -euo pipefail

UV="$HOME/.local/bin/uv"
APP_DIR="$HOME/.kiss/kiss"
SHELL_TASK='Run the shell command "sw_vers -productVersion" and write exactly its output to the file version.txt in the current directory.'
BROWSER_TASK='Then open https://example.com in the browser and write exactly the title of the page to the file title.txt in the current directory.'
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

expect_file() {
    # Fail unless file $1 holds the text $2.
    [ "$(cat "$1" 2>/dev/null)" = "$2" ] || fail "$1 holds '$(cat "$1" 2>/dev/null)', not '$2'"
}

expect_no_developer_tools_dialog() {
    # Fail if macOS is offering to install Apple's command line developer
    # tools: a command the install or a task ran needs them, and the offer
    # stays on the screen until the user answers it.
    ! pgrep -f "Install Command Line Developer Tools" > /dev/null \
        || fail "macOS offers to install the command line developer tools after $1"
}

run_on_daemon() {
    # Run task $1 in directory $2 on the kiss-web daemon and print its result.
    "$UV" run --quiet --directory "$APP_DIR" python - "$1" "$2" <<'EOF'
import os
import sys

from kiss.server import sorcar

result = sorcar.run(
    sys.argv[1],
    work_dir=sys.argv[2],
    model=os.environ.get("KISS_CHECK_MODEL", ""),
    max_budget=2.0,
    timeout=900,
)
print(result.text)
sys.exit(0 if result.success else 1)
EOF
}

# install.sh records there what it could not install (the browser, say)
# while the install as a whole succeeds.
[ ! -s "$HOME/.kiss/.install-problems" ] \
    || fail "the install recorded problems:"$'\n'"$(cat "$HOME/.kiss/.install-problems")"
[ -d "$HOME/Applications/KISS Sorcar.app" ] || fail "the KISS Sorcar app is not in ~/Applications"
launchctl print "gui/$(id -u)/com.kiss.web-server" > /dev/null 2>&1 \
    || fail "the kiss-web daemon is not registered with launchd"
expect_no_developer_tools_dialog "the installation"
installed="$("$UV" run --quiet --directory "$APP_DIR" python -c \
    'import platform, kiss; print(kiss.__version__, platform.machine())')"
macos="$(sw_vers -productVersion)"
echo "KISS Sorcar ${installed% *} (${installed#* } Python) on macOS $macos"

mkdir "$WORK/command" "$WORK/daemon"
echo "== task 1: the sorcar command in a new login shell"
# The shell reads its rc file, as in a new Terminal window: that is where
# install.sh put ~/.local/bin on the PATH and where saving a key in the
# app's Settings made the keys visible to commands.
"${SHELL:-/bin/zsh}" -lic 'sorcar "$@"' sorcar -t "$SHELL_TASK" -m "${KISS_CHECK_MODEL:-}" -b 2 \
    --work-dir "$WORK/command" || fail "the sorcar command could not run a task"
expect_file "$WORK/command/version.txt" "$macos"

echo "== task 2: the kiss-web daemon, with the browser"
run_on_daemon "$SHELL_TASK $BROWSER_TASK" "$WORK/daemon" || fail "the daemon could not run a task"
expect_file "$WORK/daemon/version.txt" "$macos"
expect_file "$WORK/daemon/title.txt" "Example Domain"
expect_no_developer_tools_dialog "the tasks"

echo "PASS: KISS Sorcar ${installed% *} ran a task with the sorcar command and one on the daemon" \
    "(macOS $macos, ${installed#* } Python)"
