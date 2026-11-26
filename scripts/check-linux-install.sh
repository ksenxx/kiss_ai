#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Check the KISS Sorcar installed on this Linux machine.
#
#   scripts/check-linux-install.sh [--tasks]
#
# Run it as the user who installed KISS Sorcar, after the Linux installer
# (scripts/build-linux-run.sh) or install.sh has finished.  It checks that
# the install recorded no problem and left the update lock free, that the
# installed app has the version of the sources this file is part of, that
# the kiss-web daemon serves the app (as the user's systemd service, where
# the user has a systemd session), that a new login shell finds the sorcar
# command and that Chromium loads a web page.
#
# With --tasks it then runs two small tasks and checks the files each
# leaves behind.  The API key of a model provider must have been entered
# in the app's Settings before:
#
#   1. with the sorcar command in a new login shell, as typed into a new
#      terminal window (the agent runs without the daemon): the output of a
#      shell command;
#   2. on the kiss-web daemon, as the app does: the same, and the title of
#      a web page the agent opens in its browser.
#
# A task may spend at most 2 USD.  KISS_CHECK_MODEL names the model to use
# (default: the best model for the configured API keys).
#
# The Linux installer ships this file (~/.kiss/kiss_pkg/scripts), and
# scripts/check-linux-run.sh runs it in a container of a distribution.

set -euo pipefail

SOURCES="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$HOME/.kiss/kiss/.venv/bin/python"
SHELL_TASK='Run the shell command "uname -m" and write exactly its output to the file machine.txt in the current directory.'
BROWSER_TASK='Then open https://example.com in the browser and write exactly the title of the page to the file title.txt in the current directory.'

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

usage() {
    # Print the usage: this file's header comment from "Check" on.
    sed -n '6,32p' "$0" | sed 's/^# \{0,1\}//'
}

expect_file() {
    # Fail unless file $1 holds the text $2.
    [ "$(cat "$1" 2>/dev/null)" = "$2" ] || fail "$1 holds '$(cat "$1" 2>/dev/null)', not '$2'"
}

check_install() {
    # Check what the install left on this machine and print its version.
    local expected installed
    # install.sh records there what it could not install (the browser, say)
    # while the install as a whole succeeds.
    [ ! -s "$HOME/.kiss/.install-problems" ] \
        || fail "the install recorded problems:"$'\n'"$(cat "$HOME/.kiss/.install-problems")"
    # The install held the update lock, a flock(2) on this file.  Nothing
    # it started (the daemon, the app's window) may have kept it: no
    # update could run then.
    perl -e 'use Fcntl qw(:flock);
             open(my $f, ">>", $ARGV[0]) or die "FAIL: the update lock $ARGV[0] cannot be opened: $!\n";
             flock($f, LOCK_EX | LOCK_NB)
                 or die "FAIL: something the install started still holds the update lock\n"' \
        "$HOME/.kiss/.update.lock" || exit 1
    expected="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$SOURCES/src/kiss/core/_version.py")"
    installed="$("$PYTHON" -c 'import kiss; print(kiss.__version__)')"
    [ "$installed" = "$expected" ] || fail "the installed app is version $installed, not $expected"
    # install.sh makes the daemon the user's service under the same condition.
    if systemctl --user show-environment > /dev/null 2>&1; then
        systemctl --user is-active --quiet kiss-web.service \
            || fail "the kiss-web service of the user is not active"
    fi
    "$PYTHON" - <<'EOF' || fail "the kiss-web daemon does not serve the app"
import json
import pathlib
import urllib.request

url = json.loads((pathlib.Path.home() / ".kiss/remote-url.json").read_text())["loopback"]
page = urllib.request.urlopen(url, timeout=30).read().decode()
assert "KISS Sorcar" in page, page[:200]
EOF
    # The shell reads its rc file, as in a new terminal window: that is where
    # install.sh put ~/.local/bin on the PATH.
    "${SHELL:-/bin/bash}" -lic 'sorcar --help' > /dev/null 2>&1 \
        || fail "a new login shell does not find the sorcar command"
    "$PYTHON" - <<'EOF' || fail "Chromium did not load a web page"
from playwright.sync_api import sync_playwright

with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto("https://example.com")
    assert page.title() == "Example Domain", page.title()
    browser.close()
EOF
    # shellcheck source=/dev/null
    echo "KISS Sorcar $installed on $(. /etc/os-release && echo "$PRETTY_NAME") ($(uname -m)):" \
        "the daemon serves the app, sorcar is on the PATH, Chromium loads a page"
}

run_tasks() {
    # Run a task with the sorcar command and one on the daemon and check
    # the files each leaves behind.
    # Global: the trap runs after this function has returned.
    WORK="$(mktemp -d)"
    trap 'rm -rf "$WORK"' EXIT
    mkdir "$WORK/command" "$WORK/daemon"
    echo "== task 1: the sorcar command in a new login shell"
    # The rc file the shell reads is also where saving a key in the app's
    # Settings made the keys visible to commands.
    "${SHELL:-/bin/bash}" -lic 'sorcar "$@"' sorcar -t "$SHELL_TASK" -m "${KISS_CHECK_MODEL:-}" -b 2 \
        --work-dir "$WORK/command" < /dev/null || fail "the sorcar command could not run a task"
    expect_file "$WORK/command/machine.txt" "$(uname -m)"
    echo "== task 2: the kiss-web daemon, with the browser"
    "$PYTHON" - "$SHELL_TASK $BROWSER_TASK" "$WORK/daemon" <<'EOF' || fail "the daemon could not run a task"
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
    expect_file "$WORK/daemon/machine.txt" "$(uname -m)"
    expect_file "$WORK/daemon/title.txt" "Example Domain"
    echo "PASS: the sorcar command and the daemon each ran a task"
}

case "${1:-}" in
    "") check_install ;;
    --tasks) check_install; run_tasks ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
