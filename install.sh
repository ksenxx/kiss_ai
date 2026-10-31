#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Install KISS Sorcar: the kiss-web daemon, its webapp and the sorcar CLI.
#
# What ends up on the machine:
#
#   ~/.kiss/kiss/                 the app directory: $KISS_HOME/<its name without
#                                 the dot> (~/.s10s/s10s for a brand home)
#     pyproject.toml, uv.lock     the project uv builds the venv from
#     src/kiss/**/*.pyc           the KISS Sorcar library, byte-compiled;
#                                 the ``.py`` sources are not installed
#     src/kiss/agents/seas/**.py  the bundled SEAs, kept as Python scripts:
#                                 ``run_agent``/``sea`` load them by path
#     src/kiss/**/*_sea.py        channel and cron SEAs, same reason
#     (and the few other modules COMPILE_KEEP_SOURCE names: the /cron
#     agent, _version.py, the files sync_db and the memory sync ship
#     to a remote)
#     src/kiss/webapp/media/  the webapp frontend kiss-web serves
#     .venv/                      the Python environment (``uv sync``)
#   ~/.local/bin/sorcar           the CLI (``uv run --directory ~/.kiss/kiss sorcar``)
#   ~/.local/bin/cloudflared      the public tunnel of the webapp
#
# Steps:
#   [1/5] Copy the project files into ~/.kiss/kiss (the library, the
#         SEAs and the webapp's ``media`` frontend; no tests).
#   [2/5] Build the Python environment with uv (installed when missing).
#   [3/5] Byte-compile the library with the venv's own interpreter and
#         delete the compiled sources (see COMPILE_KEEP_SOURCE below).
#   [4/5] Install the ``sorcar`` CLI, Playwright's Chromium and cloudflared.
#   [5/5] Refresh ~/.kiss/MODEL_INFO.json, run the post-install hooks
#         (each brand hook's git checkout is first reset to its upstream,
#         so the branding steps run the brand's latest code) and
#         set the remote-access password; repeat steps 1-5 into every
#         other brand home on the machine (a ``~/.<brand>`` with
#         post-install hooks, or the home the existing service runs
#         from); start the kiss-web daemon -- the brand's app, not the
#         stock one, when there is a brand home: the one the service
#         runs from, else the first one installed -- then open the webapp as
#         a standalone app (kiss-app: a desktop launcher plus a Chromium
#         app-mode window; the browser when that is not possible) and,
#         for a VS Code user, retire the VS Code extension:
#         uninstall it, open the notice that VS Code is no longer
#         supported in their VS Code window, print it, and have the web
#         app show it (see "Retire the VS Code extension" below).
#
# The daemon serves the directory this script is run from, except that a
# run from inside ~/.kiss (the curl bootstrap's clone, the app) serves
# $HOME (see daemon_work_dir).
#
# Usage:
#   ./install.sh                  interactive: asks before installing Homebrew
#   ./install.sh --non-interactive  every question takes its default answer
#   ./install.sh --restart-only   install nothing; only (re)start kiss-web from
#                                 the app already in $KISS_HOME, serving the
#                                 directory it is run from -- how kiss-web
#                                 finishes a restart a run deferred
#
# Environment:
#   KISS_HOME            state directory (default ~/.kiss); the app goes in
#                        $KISS_HOME/<name without the leading dot> (~/.s10s/s10s),
#                        $KISS_HOME/kiss when the name has no leading dot
#   KISS_NONINTERACTIVE  same as --non-interactive
#   KISS_SKIP_LAUNCH     do not start kiss-web or open the webapp
#   KISS_WEB_PORT        port of the kiss-web daemon (default 8787)
#   KISS_CODE_CLI        the one editor CLI to retire the VS Code extension
#                        with (default: code, code-insiders, cursor and
#                        code-server on PATH); empty for none
#   _KISS_BRAND_PASS     set by this script on the nested run it makes
#                        into another brand home (see step [5/5])
#
# The first run sets a random remote-access password (stored as
# ``remote_password`` in $KISS_HOME/config.json, where the Settings page
# shows it); every later run keeps it, so an update never invalidates
# the password your devices know.  It is printed in a box of ``*`` at
# the end of every run, on the terminal only, never in the log.  For a
# new one, run ``$KISS_HOME/kiss/.venv/bin/python -m
# kiss.core.remote_password --rotate``.
#
# Re-running the script updates the app in place: the app directory is
# rebuilt from the checkout this script lives in, the venv is re-synced
# and the daemon is restarted -- never while it reports tasks in flight:
# the restart is then deferred, and the daemon finishes it by itself once
# its tasks are done (record_deferred_restart).
# The daemon's Update button and the curl bootstrap (scripts/install.sh)
# both run this script from the ``~/.kiss/kiss_ai`` checkout.

# The remote webapp's Terminal tab is a shell the kiss-web daemon forks
# (src/kiss/server/terminal_tab.py); so are the daemon's Update button
# and a Sorcar task's shell tool.  On Linux the daemon is the systemd
# user service ``kiss-web.service``, and a service's children stay in
# its control group however they detach: ``setsid`` (the perl re-exec
# below) changes the session, never the cgroup.  ``systemctl --user stop
# kiss-web.service`` — which the runtime setup at the end of this script
# runs to bring up the rebuilt daemon — SIGTERMs *every* process of that
# group (systemd's ``KillMode=control-group``), so the script killed
# itself: "Terminated" twice, then ``handle_interrupt`` reading the
# second signal as a confirmed abort.
#
# ``systemd-run --user --scope`` moves a process into a transient scope
# unit of its own: systemd-run registers the scope for its own pid and
# ``exec``s the payload in place, so the pid, the controlling terminal,
# stdio and every open fd stay as they are — the same trick
# web_server.py uses to keep cloudflared alive across a restart.
# ``_KISS_HOST_SERVICE`` names the service the script escaped from; the
# re-exec'd copy sees it and does not loop, and the runtime setup uses
# it to warn that the terminal closes with the daemon.  Any failure (no
# systemd, no user manager) falls through to the previous behaviour.
#
# scripts/install.sh, the curl bootstrap that holds the update lock and
# waits for this script, carries the identical block (a test keeps the
# two the same): were it left behind in the group, the stop below would
# hang until systemd gave up on it and SIGKILLed it, dropping the lock
# mid-install.  Piped in by ``curl ... | bash`` it has no ``$0`` file
# to re-run, hence the ``bash -s`` branch.
# BEGIN: kiss-service-cgroup-escape  (tests extract this block verbatim)
systemd_service_cgroup() {
    # Print the ``*.service`` unit owning this process's control group
    # per cgroup file $1 (default /proc/self/cgroup), or nothing.  Only
    # the last path component counts, and ``user@UID.service`` never
    # (every user process sits under it; a cgroup v1 controller line can
    # end there); a transient scope ends in ``.scope``.
    sed -n '\|/user@[^/]*\.service$|d; s|.*/\([^/]*\.service\)$|\1|p' \
        "${1:-/proc/self/cgroup}" 2>/dev/null | head -n 1
}
if [ -z "${_KISS_HOST_SERVICE:-}" ] && command -v systemd-run >/dev/null 2>&1 \
    && { [ -f "$0" ] || [ ! -t 0 ]; }; then
    _kiss_host_service="$(systemd_service_cgroup)"
    if [ -n "$_kiss_host_service" ] \
        && systemd-run --user --scope --quiet --collect -- true 2>/dev/null; then
        export _KISS_HOST_SERVICE="$_kiss_host_service"
        if [ -f "$0" ]; then
            exec systemd-run --user --scope --quiet --collect \
                --description="install.sh (moved out of $_kiss_host_service)" \
                -- bash "$0" "$@"
        fi
        # The script is stdin (``curl ... | bash``).  Bash reads a pipe
        # one byte at a time and seeks a file back to where it stopped
        # before running any command, so everything after this block is
        # still unread there: the new bash carries on right after it.
        exec systemd-run --user --scope --quiet --collect \
            --description="install.sh (moved out of $_kiss_host_service)" \
            -- bash -s "$@"
    fi
    unset _kiss_host_service
fi
# END: kiss-service-cgroup-escape
#
# ---------------------------------------------------------------------------
# Interactive mode (the default at a terminal)
# ---------------------------------------------------------------------------
# A human running ``./install.sh`` at a terminal is asked ``[Y/n]``
# before Homebrew is installed (see ``confirm`` below); "no" skips it and
# carries on.  The tool the install cannot proceed without (uv) is still
# installed without a question when missing, and an already-installed
# tool is used as-is — this script never upgrades third-party software.
#
# ``_KISS_INTERACTIVE`` is 0 instead when
#
# * ``--non-interactive`` is passed or ``KISS_NONINTERACTIVE`` is set —
#   what an automated caller does; or
# * ``/dev/tty`` cannot be opened, i.e. there is no terminal to ask on
#   (cron, CI, a daemon), including inside the detached re-exec below.
#
# Non-interactive runs behave as before: every question takes its default
# answer (Yes).
# BEGIN: kiss-interactive-mode
_KISS_INTERACTIVE=1
if [ -n "${KISS_NONINTERACTIVE:-}" ]; then
    _KISS_INTERACTIVE=0
fi
for _kiss_arg in "$@"; do
    if [ "$_kiss_arg" = "--non-interactive" ]; then
        _KISS_INTERACTIVE=0
    fi
done
unset _kiss_arg
if [ "$_KISS_INTERACTIVE" = 1 ] && ! { : </dev/tty; } 2>/dev/null; then
    _KISS_INTERACTIVE=0
fi
# END: kiss-interactive-mode
#
# BEGIN: kiss-new-session-reexec  (tests extract this block verbatim)
if [ -z "${_KISS_NEW_SESSION:-}" ] && [ "${_KISS_INTERACTIVE:-0}" != 1 ] && command -v perl >/dev/null 2>&1; then
    # Probe POSIX::setsid availability before committing to the re-exec —
    # if perl is present but the POSIX module fails to load (custom
    # micro-perl builds), fall through to the trap-only path.
    if perl -e 'use POSIX qw(setsid); exit 0' >/dev/null 2>&1; then
        export _KISS_NEW_SESSION=1
        # ``exec`` replaces the current bash with perl so a stray SIGINT to
        # the original terminal's process group hits perl (which ignores it)
        # rather than this bash (which would default-terminate).  The
        # heredoc is the perl program; ``$0`` and ``$@`` are passed as
        # positional args so the child can re-exec ``bash <script> <args>``.
        exec /usr/bin/env perl - "$0" "$@" <<'KISS_PERL_REEXEC'
use strict;
use warnings;
use POSIX ();

my $script = shift @ARGV;
my $pid = fork();
die "kiss-install: fork failed: $!\n" unless defined $pid;

if ($pid == 0) {
    # Child: create a brand-new session with no controlling terminal so
    # the kernel cannot deliver terminal-driven signals (SIGINT from
    # ``\x03``, SIGHUP from PTY close) to this process or any of its
    # descendants.  POSIX::setsid only fails with EPERM for a process
    # group leader; we just forked so we are not the leader.
    POSIX::setsid() or die "kiss-install: setsid failed: $!\n";
    # Reopen STDIN from /dev/null.  The detached session has no
    # controlling TTY anyway, but explicit /dev/null prevents any
    # accidental read() blocking on the dead inherited FD.  STDOUT and
    # STDERR are inherited unchanged so the user still sees progress
    # in the original terminal.
    open(STDIN, "<", "/dev/null") or die "kiss-install: reopen stdin: $!\n";
    exec { "bash" } "bash", $script, @ARGV
        or die "kiss-install: exec bash failed: $!\n";
}

# Parent: ignore every terminal-driven signal so that even if the
# original PTY injects ``\x03`` (SIGINT) or closes (SIGHUP), or
# something kills our pgrp with SIGTERM, this waitpid loop continues
# undisturbed until the install child finishes.
$SIG{INT}  = "IGNORE";
$SIG{TERM} = "IGNORE";
$SIG{HUP}  = "IGNORE";
$SIG{QUIT} = "IGNORE";

my $status;
while (1) {
    my $w = waitpid($pid, 0);
    if ($w == $pid) { $status = $?; last; }
    # waitpid returns -1 with EINTR if a signal interrupted it even
    # though we asked the kernel to ignore those signals (very rare —
    # only on some platforms for SIGCHLD races).  Just retry.
    next if $w == -1 && $!{EINTR};
    # ECHILD = the child already reaped (shouldn't happen given we did
    # not set $SIG{CHLD} = "IGNORE", but be defensive).
    if ($w == -1) { $status = 0; last; }
}

if (($status & 0xff) == 0) {
    # Normal exit — forward exit code.
    exit($status >> 8);
} else {
    # Killed by signal — surface as 128+signum so callers can tell.
    exit(128 + ($status & 0x7f));
}
KISS_PERL_REEXEC
    fi
fi
# END: kiss-new-session-reexec

# `pipefail` is required so any internal pipeline whose tail is `tee` (or
# any always-zero command, like the ``tar | tar`` copy below) propagates a
# non-zero exit from its body instead of returning the tail's status.
set -eo pipefail

# ``--restart-only``: kiss-web finishing the restart an earlier run
# deferred because tasks were in flight (record_deferred_restart).  The
# app is installed already; the run only (re)starts the daemon from it.
# Read before the update lock below, which such a run waits for.
_KISS_RESTART_ONLY=0
for _kiss_arg in "$@"; do
    if [ "$_kiss_arg" = "--restart-only" ]; then
        _KISS_RESTART_ONLY=1
    fi
done
unset _kiss_arg

# ---------------------------------------------------------------------------
# Cross-process update lock — the same lock as install.sh and
# scripts/install.sh.
#
# Two installers at once (this script and install.sh, or two runs of this
# script) would race each other's copy into ~/.kiss/kiss, uv sync and
# daemon restart.  scripts/install.sh holds the lock for its whole
# lifetime and exports KISS_UPDATE_LOCK_HELD=1 before handing over to an
# installer; callers that run this script directly get the same
# protection here.
#
# The lock is a kernel advisory lock (flock(2)) on $HOME/.kiss/.update.lock
# -- $HOME, not $KISS_HOME, because the resources it protects (the
# checkout under ~/.kiss/kiss_ai, ~/.local/bin/sorcar) follow $HOME.  Bash keeps the file open on fd 9 for its whole lifetime; perl
# (already required by the re-exec above; flock(1) is not on macOS) locks
# that very open file description, so the lock persists after perl exits
# and the kernel drops it when this process dies, however it dies -- no
# EXIT trap, no stale lock to break.  The pid in the file only feeds the
# refusal message.  fd 9 must not leak into long-lived children (the
# kiss-web daemon, anything a step leaves behind): every launch line
# below closes it with ``9>&-``.
#
# Placement matters: this block sits AFTER the new-session re-exec above, so
# the lock is taken (and its pid recorded) by the detached bash that does
# the work — the parent ``exec``s perl and never reaches this line — and
# after ``set -eo pipefail`` because the tests that exercise the re-exec
# block paste everything up to that line into a harness run under the
# real ``$HOME``, which must never take a real lock.
# ---------------------------------------------------------------------------
# BEGIN: kiss-update-lock
_kiss_lock_file="$HOME/.kiss/.update.lock"

acquire_update_lock() {
    local holder attempt
    mkdir -p "$HOME/.kiss"
    exec 9>>"$_kiss_lock_file"
    if ! perl -e 'use Fcntl qw(:flock); open(my $f, ">&=", 9) or exit 2; exit(flock($f, LOCK_EX | LOCK_NB) ? 0 : 1)'; then
        # The winner writes its pid right after locking; give it a moment.
        # Until then the file still names the previous run's (dead) holder:
        # the kernel released that lock on exit but nothing clears the pid.
        for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
            holder=$(cat "$_kiss_lock_file" 2>/dev/null || true)
            [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null && break
            sleep 0.05
        done
        if [ "$_KISS_RESTART_ONLY" != 1 ]; then
            echo "another KISS update is already running (pid ${holder:-unknown}); exiting." >&2
            exit 1
        fi
        # A deferred restart waits instead: the run that deferred it may
        # still be finishing.  Any run holding the lock leaves an app on
        # disk to restart onto, and a daemon it restarted meanwhile is left
        # running (restart_kiss_web_locked compares fingerprints).
        echo "   Waiting for the KISS update running as pid ${holder:-unknown} to finish..."
        perl -e 'use Fcntl qw(:flock); open(my $f, ">&=", 9) or exit 2; exit(flock($f, LOCK_EX) ? 0 : 1)' \
            || exit 1
    fi
    echo "$$" > "$_kiss_lock_file"
    export KISS_UPDATE_LOCK_HELD=1
}

if [ -z "${KISS_UPDATE_LOCK_HELD:-}" ]; then
    acquire_update_lock
fi
# The marker's only consumer is the lock decision above: this script hands
# over to no further installer, so drop it now.  Left exported it would
# leak into every long-lived child (the kiss-web daemon), whose OWN later
# update runs would then skip acquire_update_lock entirely — with the
# original lock long gone.
unset KISS_UPDATE_LOCK_HELD
# END: kiss-update-lock

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

BIN_DIR="$HOME/.local/bin"
LOG_DIR="$HOME/.kiss"
LOG_FILE="$LOG_DIR/install.log"
# Name of the state directory under $HOME the build being installed uses
# when $KISS_HOME is unset: ``home_dir`` of the brand overlay
# (.brand/brand.json, see copy_brand_overlay below) when there is one,
# else of the checkout's own media/brand.json -- ``.kiss`` for stock KISS
# Sorcar.  The same key drives kiss.core.config.kiss_home(), so the
# installer puts the app, the progress file and MODEL_INFO.json where the
# installed product reads them.  Anything but a single path component
# falls back to .kiss.
brand_field() {
    # Print string field $1 of brand.json $2, or without $2 of the
    # checkout's brand.json (the .brand/ overlay first); nothing when no
    # file defines it.
    local file files=("$PROJECT_DIR/.brand/brand.json" "$PROJECT_DIR/src/kiss/webapp/media/brand.json")
    [ -z "${2:-}" ] || files=("$2")
    for file in "${files[@]}"; do
        [ -f "$file" ] || continue
        tr -d '\n\r' < "$file" | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p"
        return 0
    done
}
brand_home_dir_name() {
    local name
    name="$(brand_field home_dir)"
    case "$name" in
        ""|.|..|*/*|*\\*) echo ".kiss" ;;
        *) echo "$name" ;;
    esac
}
BRAND_HOME_DIR_NAME="$(brand_home_dir_name)"
BRAND_PRODUCT_NAME="$(brand_field product_name)"
BRAND_PRODUCT_NAME="${BRAND_PRODUCT_NAME:-KISS Sorcar}"
# The state directory ($KISS_HOME, else ~/<home_dir>).  The app and the
# progress file below must land here, not in a hard-coded $HOME/.kiss, or
# a custom-KISS_HOME install never sees them.
KISS_HOME_DIR="${KISS_HOME:-$HOME/$BRAND_HOME_DIR_NAME}"
# The directory the kiss-web daemon serves (its Sorcar tasks default
# their work dir to it): where this script was run from, unless that is
# inside a state directory -- the daemon's own update runs the curl
# bootstrap from its clone, ~/.kiss/kiss_ai, and the retired VS Code
# extension's Update button did the same, and neither should turn the
# clone into the user's work dir.  Those runs serve $HOME instead.
daemon_work_dir() {
    # Physical paths: a symlinked $HOME (macOS /var -> /private/var) must
    # still match.
    local here stock state
    here="$(pwd -P)"
    stock="$(cd "$HOME/.kiss" 2>/dev/null && pwd -P)" || stock="$HOME/.kiss"
    state="$(cd "$KISS_HOME_DIR" 2>/dev/null && pwd -P)" || state="$KISS_HOME_DIR"
    case "$here/" in
        "$stock/"*|"$state/"*) echo "$HOME" ;;
        *) echo "$PWD" ;;
    esac
}
USER_PWD="$(daemon_work_dir)"
# Current install step, for anything watching the install (the daemon's
# update log tails this script; ``cat`` the file to see where a detached
# run is).  Line 1 is this script's pid, line 2 the step text.  The EXIT
# trap below removes the file when the install ends.
PROGRESS_FILE="$KISS_HOME_DIR/.install-progress"
# Where the app is installed: the directory of the state directory named
# after it without the leading dot, ~/.kiss/kiss for stock KISS Sorcar
# and ~/.s10s/s10s for a brand whose home is ~/.s10s (a state directory
# not named that way, KISS_HOME=/opt/sorcar, keeps the stock name,
# /opt/sorcar/kiss).  Everything the runtime needs lives here (project
# files, byte-compiled library, SEAs, frontend, venv); ``~/.kiss/kiss_ai``
# next to it is the checkout the curl bootstrap clones and is not touched.
app_dir_of() {
    # Print the app directory of state directory $1.
    local name="${1%/}"
    name="${name##*/}"
    case "$name" in
        ..) echo "$1/kiss" ;;
        .?*) echo "$1/${name#.}" ;;
        *) echo "$1/kiss" ;;
    esac
}
APP_DIR="$(app_dir_of "$KISS_HOME_DIR")"
# Every failure of this run, one line each (``warn``, ``fail`` and the
# EXIT trap), for the daemon that runs afterwards: kiss-web shows the
# file's lines as a notification in the web app (see
# kiss/server/install_problems.py).  A run starts afresh; a run with no
# failure leaves no file, so a previous run's problems are not reported
# again once an update succeeds.  A ``--restart-only`` run keeps them:
# they are the deferred run's, for the daemon it restarts to report.
PROBLEMS_FILE="$KISS_HOME_DIR/.install-problems"

mkdir -p "$BIN_DIR" "$LOG_DIR" "$KISS_HOME_DIR"
[ "$_KISS_RESTART_ONLY" = 1 ] || rm -f "$PROBLEMS_FILE"
export PATH="$BIN_DIR:$PATH"

# BEGIN: kiss-install-problems  (tests extract this block verbatim)
record_problem() {
    # Append failure $1 to PROBLEMS_FILE (best effort: an unwritable state
    # directory must not fail the install).
    [ -n "${PROBLEMS_FILE:-}" ] || return 0
    printf '%s\n' "$1" >> "$PROBLEMS_FILE" 2>/dev/null || true
}
warn() {
    # Print "   WARNING: $*" and record it; the install carries on.
    echo "   WARNING: $*"
    record_problem "WARNING: $*"
}
fail() {
    # Print "   ERROR: $*", record it and stop the install.
    echo "   ERROR: $*"
    record_problem "ERROR: $*"
    FAILED=1
    exit 1
}
# END: kiss-install-problems

# ---------------------------------------------------------------------------
# Signal handling (as in install.sh)
#
# ``uv sync`` and the Playwright download can sit silent for tens of
# seconds — long enough for a stray signal (a backgrounded shell, a
# sleeping laptop, a closed terminal tab) to kill the script just as it
# was about to make progress.  Trap SIGINT/SIGTERM at the bash level so a
# single stray signal prints a diagnostic instead of silently
# terminating; a *second* signal within 3 s is honored as a real abort.
#
# The trap alone is not enough: the signal is delivered to the entire
# foreground process group, so a wrapped child that does not trap SIGINT
# itself dies at once, which makes the ``wait`` in ``run_with_heartbeat``
# return non-zero and ``set -e`` abort the install.  Hence
# ``run_with_heartbeat`` spawns the wrapped command inside a subshell that
# sets ``trap '' INT TERM`` and then ``exec``s the binary: a signal
# *ignored* at exec time stays ignored in the new process.  A confirmed
# double-Ctrl+C in ``handle_interrupt`` kills the tracked child explicitly.
# ---------------------------------------------------------------------------
LAST_SIGNAL_TS=0
# PID of the wrapped command currently running under ``run_with_heartbeat``
# — used by ``handle_interrupt`` to forcibly stop it on a confirmed
# double-interrupt (since the child ignores SIGINT by design).
CURRENT_CMD_PID=""
# PID of that command's heartbeat subshell, stopped alongside it on abort
# so it does not outlive the script by up to one HEARTBEAT_INTERVAL.
CURRENT_HB_PID=""
# The ``confirm`` question currently waiting for an answer, if any.  A
# single Ctrl-C at a question runs ``handle_interrupt`` but, on bash 5,
# leaves the ``read`` waiting; re-printing the question after the notice
# tells the user the script still expects an answer.
CONFIRM_PENDING=""
# The notice goes to stderr: the trap runs after whatever foreground
# command the signal hit, and when that command's stdout was a file
# (``list_app_files > "$list"``) the notice would otherwise land in it.
handle_interrupt() {
    local now
    now=$(date +%s)
    if [ $((now - LAST_SIGNAL_TS)) -lt 3 ]; then
        echo ""
        echo "   Second interrupt received — aborting install."
        if [ -n "$CURRENT_CMD_PID" ]; then
            # The wrapped command ignores SIGINT (``trap '' INT TERM`` in
            # its subshell), so SIGINT alone would do nothing.  Send
            # SIGTERM, give it a moment to clean up, then SIGKILL.
            kill -TERM "$CURRENT_CMD_PID" 2>/dev/null || true
            sleep 1
            kill -KILL "$CURRENT_CMD_PID" 2>/dev/null || true
        fi
        if [ -n "$CURRENT_HB_PID" ]; then
            kill "$CURRENT_HB_PID" 2>/dev/null || true
        fi
        echo "   Re-run 'bash $0' to resume; the Python environment is preserved."
        exit 130
    fi
    LAST_SIGNAL_TS=$now
    echo ""
    echo "   ⚠ Interrupt received but ignored — uv and Playwright can sit"
    echo "      silent for 30-60 s while they download or extract.  Press"
    echo "      Ctrl+C again within 3 s to really abort."
    if [ -n "$CONFIRM_PENDING" ]; then
        printf '   %s [Y/n] ' "$CONFIRM_PENDING"
    fi
} >&2

# Re-route stdout/stderr to the log file when the controlling terminal
# closes (SIGHUP).  This matters when the script runs in a terminal the
# kiss-web daemon hosts (the webapp's Terminal tab): restarting the daemon
# in step [5/5] closes that terminal, first writing ``\x03`` (Ctrl+C) to
# the PTY (caught by ``handle_interrupt`` above) and then closing the PTY
# (SIGHUP).  Without this trap the SIGHUP kills bash mid-step.
# Every fd still on the dead PTY is moved off it: stdin to /dev/null and
# the saved terminal (fd 3, the password box's) to /dev/null as well,
# because once the terminal's shell has exited the PTY is revoked and
# any use of it fails with EBADF (macOS) or EIO: a child started with
# that stdin dies at once (``kiss-app`` in step [5/5]: ``Fatal Python
# error: init_sys_streams``), and ``echo >&3`` fails, which ``set -e``
# turns into a failed install.  ``TERMINAL_GONE`` tells the end of the
# script that the password box had nowhere to go.  ``|| true`` keeps
# ``set -e`` from killing the script if the re-route itself fails.
# BEGIN: kiss-hup-trap  (tests extract this block verbatim)
TERMINAL_GONE=0
handle_hup() {
    exec </dev/null >>"$LOG_FILE" 2>&1 3>/dev/null || true
    if [ "$TERMINAL_GONE" = 1 ]; then
        return 0
    fi
    TERMINAL_GONE=1
    echo ""
    echo "   ⚠ Controlling terminal closed (SIGHUP) — continuing with"
    echo "      output redirected to $LOG_FILE only."
}
trap handle_hup HUP
# END: kiss-hup-trap
trap handle_interrupt INT TERM

# Print a step banner and publish the step to PROGRESS_FILE (see there).
# Publishing is best-effort: an unwritable $KISS_HOME must not fail the
# install.
CURRENT_STEP="starting"
report_step() {
    echo ">>> $1"
    CURRENT_STEP="$1"
    publish_step "$1"
}
publish_step() {
    # Write, then rename: a poll must never read a truncated file, which
    # would look like the install had ended.
    { printf '%s\n%s\n' "$$" "$1" > "$PROGRESS_FILE.tmp" \
        && mv -f "$PROGRESS_FILE.tmp" "$PROGRESS_FILE"; } 2>/dev/null || true
}
# KISS_WEB_RESTART_LOCK_HELD names the kiss-web restart lock this script
# holds, if any (acquire_kiss_web_restart_lock below); released here so an
# aborted run does not leave the daemon locked out of restarting.
KISS_WEB_RESTART_LOCK_HELD=""
# BEGIN: kiss-exit-trap  (tests extract this block verbatim)
on_exit() {
    # The EXIT trap: a run that stops with a status (``set -e`` on an
    # unexpected failure, an abort) that ``fail`` did not already record
    # is recorded with the step it was in.
    local rc=$1
    if [ "$rc" -ne 0 ] && [ -z "${FAILED:-}" ]; then
        record_problem "ERROR: the install stopped with status $rc during: $CURRENT_STEP"
    fi
    rm -f "$PROGRESS_FILE" "$PROGRESS_FILE.tmp"
    [ -z "$KISS_WEB_RESTART_LOCK_HELD" ] || release_kiss_web_restart_lock "$KISS_WEB_RESTART_LOCK_HELD"
}
trap 'on_exit $?' EXIT
# END: kiss-exit-trap

# Run "$@" while printing a heartbeat every HEARTBEAT_INTERVAL seconds so
# the user can tell the install is still working.  Without this ``uv sync``
# can sit silent for ~1 min and look hung.  Exit code is forwarded from
# the wrapped command.
HEARTBEAT_INTERVAL="${KISS_HEARTBEAT_INTERVAL:-15}"
run_with_heartbeat() {
    local label="$1"
    shift
    local start
    start=$(date +%s)
    # Run the command inside a subshell that ignores SIGINT/SIGTERM, then
    # ``exec`` the real binary.  POSIX says SIG_IGN survives exec, so the
    # command and every descendant inherit "ignore" for INT/TERM — a stray
    # signal delivered to this script's terminal process group can no
    # longer kill them.  The trap above remains the only way to actually
    # stop the install (double-Ctrl+C).
    ( trap '' INT TERM; exec "$@" ) 9>&- &
    local cmd_pid=$!
    CURRENT_CMD_PID=$cmd_pid
    # Heartbeat loop runs in its own subshell so a failing ``sleep`` (rare)
    # cannot abort the parent script under ``set -e``.  The parent's
    # cleanup at end-of-function stops it with SIGTERM; the TERM trap
    # takes the current ``sleep`` down with it.  Without the trap bash
    # would die but leave that ``sleep`` (a foreground child blocked for
    # up to HEARTBEAT_INTERVAL seconds) orphaned after every wrapped
    # command.  A stray SIGINT cannot reach this subshell: bash starts
    # background jobs with SIGINT ignored, and the wrapped command stays
    # alive via its own SIG_IGN above.
    (
        set +e
        hb_sleep_pid=""
        trap 'kill "$hb_sleep_pid" 2>/dev/null; wait "$hb_sleep_pid" 2>/dev/null; exit 0' TERM
        while kill -0 "$cmd_pid" 2>/dev/null; do
            sleep "$HEARTBEAT_INTERVAL" &
            hb_sleep_pid=$!
            wait "$hb_sleep_pid"
            if kill -0 "$cmd_pid" 2>/dev/null; then
                local elapsed=$(( $(date +%s) - start ))
                printf "   … %s still running (%ds elapsed)\n" "$label" "$elapsed"
            fi
        done
    ) 9>&- &
    local hb_pid=$!
    CURRENT_HB_PID=$hb_pid
    # Use ``+e`` so a non-zero exit from the wrapped command is returned to
    # the caller instead of aborting the whole script — callers need to
    # inspect the exit code.  ``wait`` itself
    # can also return early under signal delivery; loop until the child
    # is actually gone so a stray signal during this exact instant cannot
    # leave the caller seeing a bogus non-zero rc while the child keeps
    # running.
    set +e
    local rc
    while :; do
        wait "$cmd_pid"
        rc=$?
        # ``wait`` returns >128 when interrupted by a trapped signal but
        # the child is still alive.  Detect that case and keep waiting.
        if [ $rc -gt 128 ] && kill -0 "$cmd_pid" 2>/dev/null; then
            continue
        fi
        break
    done
    set -e
    CURRENT_CMD_PID=""
    CURRENT_HB_PID=""
    kill "$hb_pid" 2>/dev/null || true
    wait "$hb_pid" 2>/dev/null || true
    return $rc
}

# Ask the user a yes/no question; returns 0 for "yes" and 1 for "no".
#
# When ``_KISS_INTERACTIVE`` is 0 (see the "Interactive mode" block at the top)
# this asks nothing and answers "yes", which is the historical behaviour
# of every caller.  Otherwise the question goes to stdout (so it is logged
# and stays in order with the surrounding output) and the answer is read
# from ``/dev/tty`` rather than stdin, so it works for ``curl ... | bash``
# too.
#
# Guards, each of which once crashed this script under ``set -e``:
#
# * ``read`` runs in an ``||`` list so a non-zero status can never trip
#   ``set -e``; EOF (Ctrl-D) and a ``/dev/tty`` that can no longer be
#   opened (the terminal went away after the startup probe) both take the
#   default "yes" instead of dying;
# * the single Ctrl-C that ``handle_interrupt`` deliberately ignores is not
#   mistaken for an answer: bash 5 keeps the ``read`` waiting after the
#   trap (the trap re-prints the question via ``CONFIRM_PENDING``), and a
#   bash whose ``read`` gives up with status > 128 simply reads again.
#
# ``read -s`` turns off the terminal's own echo and the answer is printed
# back through stdout instead, so question and answer travel the same
# ``tee`` pipe and land in the log complete and in order (a direct append
# to the log file could overtake ``tee``).
confirm() {
    local question="$1" answer rc
    if [ "$_KISS_INTERACTIVE" != 1 ]; then
        return 0
    fi
    CONFIRM_PENDING="$question"
    printf '   %s [Y/n] ' "$question"
    while :; do
        rc=0
        IFS= read -rs answer </dev/tty || rc=$?
        if [ "$rc" -gt 128 ]; then
            continue
        fi
        if [ "$rc" -ne 0 ]; then
            CONFIRM_PENDING=""
            echo "(no answer from the terminal; assuming yes)"
            return 0
        fi
        printf '%s\n' "${answer:-yes}"
        case "$answer" in
            ""|[Yy]|[Yy][Ee][Ss]) CONFIRM_PENDING=""; return 0 ;;
            [Nn]|[Nn][Oo]) CONFIRM_PENDING=""; return 1 ;;
            *)
                echo "   Please answer y or n."
                printf '   %s [Y/n] ' "$question"
                ;;
        esac
    done
}

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Darwin|Linux) ;;
    *)  fail "Unsupported OS: $OS" ;;
esac

case "$ARCH" in
    x86_64|aarch64|arm64) ;;
    *)  fail "Unsupported architecture: $ARCH" ;;
esac

if ! command -v curl &>/dev/null; then
    fail "curl is required but not found. Please install curl first."
fi

# Make Homebrew visible even when this script runs detached from a login
# shell.  The webapp's update button spawns install.sh from the kiss-web
# daemon, whose launchd/systemd environment has a minimal PATH without
# /opt/homebrew/bin (or /usr/local/bin on Intel Macs).  Without this,
# `command -v brew` failed even though Homebrew was installed, so
# `ensure_homebrew` tried to re-install it.
if [ "$OS" = "Darwin" ] && ! command -v brew &>/dev/null; then
    if [ -x /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
fi

ensure_homebrew() {
    [ "$OS" = "Darwin" ] || return 0

    if command -v brew &>/dev/null; then
        echo "   Homebrew already installed at $(command -v brew)"
        return 0
    fi

    if [ -n "${KISS_NO_BREW:-}" ]; then
        echo "   KISS_NO_BREW set — skipping Homebrew install. KISS Sorcar may not"
        echo "   be able to install some tools on demand without it."
        return 0
    fi

    echo ""
    echo "   Homebrew (https://brew.sh) is not installed."
    echo "   Installing it enables KISS Sorcar to install necessary tools on demand"
    echo "   (e.g. git, cloudflared, and other runtime dependencies)."
    echo "   Set KISS_NO_BREW=1 to skip this step."
    echo ""
    if ! confirm "Install Homebrew now?"; then
        echo "   Skipping the Homebrew install; KISS Sorcar may not be able to"
        echo "   install some tools on demand without it."
        return 0
    fi
    echo "   Installing Homebrew..."
    # `|| true`: a failed Homebrew bootstrap (no sudo, no network)
    # must not abort the install — the check below prints a warning
    # and the script continues without brew.
    NONINTERACTIVE=1 /bin/bash -c \
        "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh || true)" || true
    # Make brew available in the current shell session.
    if [ -x /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    if command -v brew &>/dev/null; then
        echo "   Homebrew installed at $(command -v brew)"
    else
        warn "Homebrew install did not complete; continuing without it."
    fi
}

# ---------------------------------------------------------------------------
# Runtime setup (as in install.sh, with the app directory as the project)
# ---------------------------------------------------------------------------
# The daemon is restarted under the same rules as install.sh: never while
# the running daemon reports tasks in flight, and not at all when it is
# healthy and already runs this exact runtime (that keeps its public
# tunnel URL).
# BEGIN: kiss-runtime-setup
# kiss-web always listens on 8787; the override exists for the tests.
KISS_WEB_PORT="${KISS_WEB_PORT:-8787}"

ensure_uv() {
    # Print the path of uv, installing it into ~/.local/bin when missing.
    local candidate
    for candidate in "$(command -v uv 2>/dev/null || true)" \
                     "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    echo "   Installing uv (https://astral.sh/uv)..." >&2
    curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh >&2 || return 1
    [ -x "$HOME/.local/bin/uv" ] || return 1
    echo "$HOME/.local/bin/uv"
}

add_local_bin_to_shell_rc() {
    # Put ~/.local/bin (uv, sorcar, cloudflared) on PATH in the login
    # shell's rc file, once.
    local rc line
    case "${SHELL:-}" in
        */zsh|*/zsh-5) rc="$HOME/.zshrc" ;;
        */fish) rc="$HOME/.config/fish/config.fish" ;;
        *) rc="$HOME/.bashrc" ;;
    esac
    if [ "$rc" = "$HOME/.config/fish/config.fish" ]; then
        line='fish_add_path "$HOME/.local/bin"'
        grep -Eqs 'fish_add_path.*(\$HOME|~)/\.local/bin' "$rc" && return 0
    else
        line='export PATH="$HOME/.local/bin:$PATH"'
        grep -Eqs 'PATH.*(\$HOME|~)/\.local/bin' "$rc" && return 0
    fi
    if [ -s "$rc" ] && [ -n "$(tail -c1 "$rc")" ]; then
        line=$'\n'"$line"
    fi
    if mkdir -p "$(dirname "$rc")" && echo "$line" >> "$rc"; then
        echo "   Added ~/.local/bin to PATH in $rc"
    else
        warn "could not add ~/.local/bin to PATH in $rc"
    fi
}

install_cloudflared() {
    # cloudflared serves the public tunnel URL of the web app.  Best
    # effort: Homebrew on macOS, else the latest GitHub release.
    local arch bin_dir="$HOME/.local/bin" url
    command -v cloudflared &>/dev/null && return 0
    [ -x "$bin_dir/cloudflared" ] && return 0
    case "$(uname -m)" in
        x86_64|amd64) arch=amd64 ;;
        arm64|aarch64) arch=arm64 ;;
        *) echo "   cloudflared: unsupported architecture $(uname -m); skipped"; return 0 ;;
    esac
    echo "   Installing cloudflared..."
    mkdir -p "$bin_dir"
    if [ "$(uname -s)" = "Darwin" ]; then
        if command -v brew &>/dev/null && brew install cloudflared; then
            return 0
        fi
        url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-$arch.tgz"
        curl -fsSL "$url" | tar xzf - -C "$bin_dir" || warn "cloudflared download failed; the web app will have no public URL"
    else
        url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$arch"
        curl -fsSL -o "$bin_dir/cloudflared" "$url" \
            || { rm -f "$bin_dir/cloudflared"; warn "cloudflared download failed; the web app will have no public URL"; }
    fi
    [ -f "$bin_dir/cloudflared" ] && chmod 755 "$bin_dir/cloudflared"
    return 0
}

kiss_web_port_open() {
    (exec 3<>"/dev/tcp/127.0.0.1/$KISS_WEB_PORT") 2>/dev/null
}

kiss_web_fingerprint() {
    # Digest of the kiss-web launcher bytes, the work dir, and the newest
    # mtime of the bundled modules — the ``.pyc`` files too, which every
    # install writes afresh, so an updated library never matches the
    # running daemon's fingerprint.
    "$1/.venv/bin/python" - "$1" "$2" <<'EOF'
import hashlib, os, sys
project, workdir = sys.argv[1], sys.argv[2]
digest = hashlib.sha256()
with open(os.path.join(project, ".venv", "bin", "kiss-web"), "rb") as f:
    digest.update(f.read())
digest.update(workdir.encode())
latest = 0


def walk(directory):
    global latest
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return
    for entry in entries:
        if entry.name in ("__pycache__", "tests"):
            continue
        if entry.is_dir(follow_symlinks=False):
            walk(entry.path)
        elif entry.is_file(follow_symlinks=False) and entry.name.endswith((".py", ".pyc")):
            try:
                latest = max(latest, entry.stat().st_mtime_ns)
            except OSError:
                pass


walk(os.path.join(project, "src", "kiss"))
digest.update(str(latest).encode())
print(digest.hexdigest())
EOF
}

kiss_web_is_idle() {
    # True when no daemon reachable through any of the endpoint files $@
    # reports tasks in flight (scripts/check-kiss-web-active-tasks.py:
    # exit 0 = absent, refused or idle; 1 = busy, unknown, or a daemon
    # too old to answer).  A busy daemon that finishes a deferred restart
    # by itself names the file it reads it from (the probe prints it):
    # KISS_WEB_RESTART_PENDING, empty otherwise.
    local python="$1" endpoint
    shift
    KISS_WEB_RESTART_PENDING=""
    for endpoint in "$@"; do
        KISS_WEB_RESTART_PENDING="$(KISS_SORCAR_LOCAL="$endpoint" KISS_ACTIVE_TASKS_STRICT=1 \
            "$python" "$PROJECT_DIR/scripts/check-kiss-web-active-tasks.py")" || return 1
    done
}

record_deferred_restart() {
    # Record in file $1, the busy daemon's (KISS_WEB_RESTART_PENDING), the
    # restart deferred for state dir $3 and work dir $4: once no task is
    # in flight, kiss-web runs this script with --restart-only, KISS_HOME
    # $3, from $4 (kiss.server.deferred_restart).  Python $2 writes the
    # JSON; the rename keeps the daemon from reading half a file.
    "$2" -c 'import json, sys; print(json.dumps(dict(installer=sys.argv[1], home=sys.argv[2], workdir=sys.argv[3])))' \
        "$PROJECT_DIR/install.sh" "$3" "$4" > "$1.tmp" && mv -f "$1.tmp" "$1"
}

service_kiss_home() {
    # Print the KISS_HOME the existing kiss-web unit/plist sets, if any.
    local unit="$HOME/.config/systemd/user/kiss-web.service"
    local plist="$HOME/Library/LaunchAgents/com.kiss.web-server.plist"
    [ -f "$unit" ] && sed -n 's/^Environment=KISS_HOME=//p' "$unit" | unit_unescape
    [ -f "$plist" ] && sed -n '/<key>KISS_HOME<\/key>/{n;s/.*<string>\(.*\)<\/string>.*/\1/p;}' "$plist" | xml_unescape
    return 0
}

stored_remote_password() {
    # Print the ``remote_password`` of the state directory $1 as the app
    # at $2 reads it (empty when none is stored or it cannot be read);
    # never stores one.
    KISS_HOME="$1" "$2/.venv/bin/python" -m kiss.core.remote_password --show 2>/dev/null 9>&- || true
}

home_relative() {
    # Print path $1 with a leading $HOME shortened to ``~``.  Not a
    # pattern substitution: bash 3.2 (macOS) prints its ``\~`` replacement
    # literally and bash 5 tilde-expands a bare ``~`` back to $HOME.
    local home="${HOME%/}"
    case "$1" in
        "$home") printf '~\n' ;;
        "$home"/*) printf '~%s\n' "${1#"$home"}" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

other_brand_homes() {
    # Print, one per line, every state directory on this machine other
    # than $KISS_HOME_DIR that a white-label brand keeps: a ``$HOME/.<name>``
    # whose post-install.d/ holds an executable hook, and the home the
    # existing service runs from (service_kiss_home) when it exists.
    local dir hook seen=$'\n'"${KISS_HOME_DIR%/}"$'\n'
    {
        for dir in "$HOME"/.*/; do
            dir="${dir%/}"
            case "${dir##*/}" in .|..) continue ;; esac
            for hook in "$dir"/post-install.d/*; do
                if [ -f "$hook" ] && [ -x "$hook" ]; then
                    echo "$dir"
                    break
                fi
            done
        done
        service_kiss_home
    } | while IFS= read -r dir; do
        dir="${dir%/}"
        [ -n "$dir" ] && [ -d "$dir" ] || continue
        case "$seen" in *$'\n'"$dir"$'\n'*) continue ;; esac
        seen="$seen$dir"$'\n'
        echo "$dir"
    done
}

hook_checkout_dir() {
    # Print the git checkout a post-install hook runs its brand from, or
    # nothing.  A brand's installer writes its hook as a one-line
    # ``exec <script> <args>`` stub (s10s: ``exec /path/s10s/scripts/
    # install.sh --from-kiss-install``); the checkout is the work tree
    # that script sits in.  A hook that execs nothing, or a script
    # outside any git work tree (a copied file), gives nothing.  The
    # script word is split off with the shell's own rules (a brand
    # installer writes it with ``printf %q``, so a path with a space is
    # backslash-escaped or quoted): the hook is executed in full a
    # moment later, so evaluating its command line here adds nothing.
    local hook="$1" line script dir
    line="$(grep -m 1 '^exec[[:space:]]' "$hook" 2>/dev/null || true)"
    [ -n "$line" ] || return 0
    script="$(set -f; eval "set -- ${line#exec}" 2>/dev/null && printf '%s' "$1")"
    case "$script" in
        /*) ;;
        *) return 0 ;;
    esac
    dir="$(dirname "$script")"
    [ -d "$dir" ] || return 0
    git -C "$dir" rev-parse --show-toplevel 2>/dev/null || true
}

update_brand_checkout() {
    # Bring the git checkout a post-install hook runs from up to date
    # before the hook runs, so every install carries the brand's latest
    # code.  The upstream is fetched and the checkout rebased onto it,
    # local edits stashed around the rebase (git's autostash), so the
    # brand's unpushed commits and work in progress ride along on top
    # of the latest code.  When that rebase cannot complete (a conflict),
    # the checkout is forced to the upstream anyway: the local commits
    # are kept on a ``brand-update-backup-<time>`` branch and the edits
    # in a stash entry, both named in the warning, and the work tree is
    # reset hard to the upstream.  A checkout with no upstream, or one
    # that cannot be fetched (offline, no remote), is left as it is with
    # a warning; the hook then runs the code already there.
    #
    # Replaying the local commits needs a committer identity, which an
    # end user's machine often lacks (no user.name/user.email, an empty
    # passwd name): git would then refuse the rebase and the commits
    # would land on the backup branch for no reason.  Such a machine
    # gets a placeholder identity for the rebase command alone; the
    # authorship of the commits is untouched.
    local hook="$1" dir upstream before after backup ident=""
    dir="$(hook_checkout_dir "$hook")"
    [ -n "$dir" ] || return 0
    upstream="$(git -C "$dir" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
    if [ -z "$upstream" ]; then
        warn "brand checkout $dir tracks no upstream branch; running its current code"
        return 0
    fi
    echo "   Updating brand checkout $dir from $upstream..."
    if ! git -C "$dir" fetch --force --prune "${upstream%%/*}" < /dev/null; then
        warn "could not fetch $upstream for brand checkout $dir; running its current code"
        return 0
    fi
    before="$(git -C "$dir" rev-parse --short HEAD 2>/dev/null || true)"
    git -C "$dir" var GIT_COMMITTER_IDENT > /dev/null 2>&1 \
        || ident="-c user.name=kiss-install -c user.email=kiss-install@localhost"
    # shellcheck disable=SC2086  # $ident holds separate -c options
    if ! git -C "$dir" $ident rebase --quiet --autostash '@{u}' < /dev/null; then
        git -C "$dir" rebase --abort < /dev/null 2>/dev/null || true
        backup="brand-update-backup-$(date +%Y%m%d-%H%M%S)"
        git -C "$dir" branch -f "$backup" HEAD < /dev/null
        if [ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]; then
            git -C "$dir" stash push --quiet --include-untracked \
                -m "install.sh: local edits before the forced update to $upstream" < /dev/null
        fi
        if ! git -C "$dir" reset --hard '@{u}' < /dev/null; then
            warn "could not reset brand checkout $dir to $upstream; running its current code"
            return 0
        fi
        warn "brand checkout $dir did not rebase onto $upstream cleanly; forced to $upstream, the local commits are on branch $backup and the local edits, if any, in the latest stash entry"
    elif [ -n "$(git -C "$dir" diff --name-only --diff-filter=U 2>/dev/null)" ]; then
        # The commits rebased, but re-applying the autostashed edits
        # conflicted: git still exits 0, leaves conflict markers in the
        # work tree and keeps the edits as a regular stash entry.  The
        # hook must not run a file with markers in it.
        git -C "$dir" reset --hard --quiet HEAD < /dev/null
        warn "brand checkout $dir rebased onto $upstream but its local edits did not re-apply cleanly; they are kept in the latest stash entry"
    fi
    after="$(git -C "$dir" rev-parse --short HEAD 2>/dev/null || true)"
    if [ "$before" = "$after" ]; then
        echo "   Brand checkout $dir is up to date ($after)."
    else
        echo "   Brand checkout $dir updated $before -> $after."
    fi
}

previous_kiss_web_homes() {
    # State directories of the daemon the existing service runs: the
    # KISS_HOME its unit/plist sets, and its runtime's own default home.
    # A busy daemon left by an earlier install under another home must
    # still be asked before it is replaced.
    local unit="$HOME/.config/systemd/user/kiss-web.service"
    local plist="$HOME/Library/LaunchAgents/com.kiss.web-server.plist" bin python
    service_kiss_home
    # The first word of ExecStart is the launcher (older units pass
    # ``--workdir``); its sibling python knows the runtime's own home.
    for bin in "$([ -f "$unit" ] && sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' "$unit" | unit_unescape)" \
               "$([ -f "$plist" ] && sed -n '/<key>ProgramArguments<\/key>/{n;n;s/.*<string>\(.*\)<\/string>.*/\1/p;}' "$plist" | xml_unescape)"; do
        python="${bin%/*}/python"
        [ -n "$bin" ] && [ -x "$python" ] \
            && env -u KISS_HOME "$python" -c 'from kiss.core.config import kiss_home; print(kiss_home())' 2>/dev/null
    done
    return 0
}

acquire_kiss_web_restart_lock() {
    # Take the restart lock $1 ({"pid":…,"token":…}, created exclusively)
    # so two installers (the daemon's Update button and a terminal run,
    # say) never restart the daemon at the same time; a lock whose owner
    # is dead is broken.  A lock without a readable owner is left alone
    # for 2 minutes (the file is created before the identity is written),
    # a live owner for 10 minutes.
    local owner
    if ( set -o noclobber; echo "{\"pid\":$$,\"token\":\"install-$$\"}" > "$1" ) 2>/dev/null; then
        KISS_WEB_RESTART_LOCK_HELD="$1"
        return 0
    fi
    owner="$(sed -n 's/.*"pid":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$1" 2>/dev/null)"
    if [ -z "$owner" ]; then
        [ -n "$(find "$1" -mmin +2 2>/dev/null)" ] || return 1
    elif kill -0 "$owner" 2>/dev/null; then
        [ -n "$(find "$1" -mmin +10 2>/dev/null)" ] || return 1
    fi
    rm -f "$1"
    ( set -o noclobber; echo "{\"pid\":$$,\"token\":\"install-$$\"}" > "$1" ) 2>/dev/null || return 1
    KISS_WEB_RESTART_LOCK_HELD="$1"
}

release_kiss_web_restart_lock() {
    # Remove lock $1 if this script holds it (also run from the EXIT trap,
    # so an interrupted restart never leaves the next installer locked out).
    grep -qs "\"token\":\"install-$$\"" "$1" && rm -f "$1"
    KISS_WEB_RESTART_LOCK_HELD=""
    return 0
}

stop_stray_kiss_web() {
    # A kiss-web outside the service (an old direct spawn) would keep the
    # port and make the new daemon crash-loop.  Best effort: needs lsof
    # or fuser, and only touches processes that are kiss-web.
    local pid pids=""
    if command -v lsof &>/dev/null; then
        pids="$(lsof -t -iTCP:"$KISS_WEB_PORT" -sTCP:LISTEN 2>/dev/null || true)"
    elif command -v fuser &>/dev/null; then
        pids="$(fuser "$KISS_WEB_PORT/tcp" 2>/dev/null || true)"
    fi
    for pid in $pids; do
        case "$(ps -o command= -p "$pid" 2>/dev/null)" in
            *kiss-web*|*kiss.server*) kill "$pid" 2>/dev/null || true ;;
        esac
    done
}

kiss_web_hosts_this_shell() {
    # True when the kiss-web daemon is an ancestor of this script: the
    # webapp's Terminal tab, a task's shell tool, the Update button.  On
    # Linux the cgroup escape at the top already named the host service;
    # elsewhere (macOS) the parent chain is walked, as launchd keeps no
    # such record.  Restarting the daemon closes that terminal.
    [ "${_KISS_HOST_SERVICE:-}" = "kiss-web.service" ] && return 0
    [ "$(uname -s)" = "Darwin" ] || return 1
    local pid=$$ hops=0
    while [ "$pid" -gt 1 ] 2>/dev/null && [ "$hops" -lt 32 ]; do
        case "$(ps -o command= -p "$pid" 2>/dev/null)" in
            *kiss-web*|*kiss.server*) return 0 ;;
        esac
        pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
        [ -n "$pid" ] || return 1
        hops=$((hops + 1))
    done
    return 1
}

restart_kiss_web_launch_agent() {
    # Boot service $1 (gui/<uid>/<label>) out of launchd, register plist
    # $3 in domain $2 afresh and start it.  ``launchctl bootout`` only asks
    # the daemon to exit and returns at once; until the old process is
    # gone the label stays registered, so an immediate ``bootstrap`` is
    # refused ("Bootstrap failed: 5: Input/output error") and the
    # ``kickstart`` after it finds no service.  Done naively, the daemon
    # ended up booted out and never re-registered, and the caller's
    # start wait ran its full ${KISS_WEB_START_TIMEOUT}s for a kiss-web
    # nobody had started — what looked like a hang after
    # "WARNING: launchctl kickstart failed".
    local service="$1" domain="$2" plist="$3" waited=0 attempt
    launchctl bootout "$service" 2>/dev/null || true
    # Wait for launchd to drop the registration (the old daemon exited).
    if launchctl print "$service" &>/dev/null; then
        echo "   Waiting for the old kiss-web to exit..."
        while launchctl print "$service" &>/dev/null; do
            if [ "$waited" -ge "${KISS_WEB_DRAIN_TIMEOUT:-30}" ]; then
                # Fail closed: keep the registration launchd still holds (the plist it read,
                # not the one just written) and restart its process.
                warn "the old kiss-web has not exited after ${waited}s; restarting it in place."
                stop_stray_kiss_web
                launchctl kickstart -k "$service" || warn "launchctl kickstart failed"
                KISS_WEB_OLD_RUNTIME=1
                return 0
            fi
            sleep 1
            waited=$((waited + 1))
        done
    fi
    stop_stray_kiss_web
    # ``launchctl disable`` / ``unload -w`` from an earlier life would make
    # every bootstrap fail with "Service is disabled".
    launchctl enable "$service" 2>/dev/null || true
    # launchd can refuse the first bootstrap right after a removal.
    for attempt in 1 2 3 4; do
        launchctl bootstrap "$domain" "$plist" 2>/dev/null && break
        if [ "$attempt" -lt 4 ]; then sleep 1; fi
    done
    if ! launchctl print "$service" &>/dev/null; then
        # The legacy loader registers (and re-enables) the plist too.
        launchctl load -w "$plist" 2>/dev/null || true
        if ! launchctl print "$service" &>/dev/null; then
            warn "launchctl could not register $plist; run 'launchctl bootstrap $domain $plist' by hand."
            return 0
        fi
    fi
    # RunAtLoad already started the service with the bootstrap; this
    # starts it if that did not happen and is a no-op when it is running.
    launchctl kickstart "$service" || warn "launchctl kickstart failed"
}

xml_escape() {
    # sed rather than ${s//&/&amp;}: bash 5.2 reads an unquoted & in a
    # replacement as the matched text.
    printf '%s' "$1" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g; s/'"'"'/\&apos;/g'
}

unit_escape() {
    # The trailing "x" survives $(…)'s newline stripping and is removed.
    local s
    s="$(printf '%sx' "$1" | sed 's/\\/\\\\/g; s/%/%%/g')"
    s="${s%x}"
    printf '%s' "${s//$'\n'/\\n}"
}

unit_unescape() {
    sed 's/%%/%/g; s/\\\\/\\/g'
}

xml_unescape() {
    sed 's/&lt;/</g; s/&gt;/>/g; s/&quot;/"/g; s/&apos;/'"'"'/g; s/&amp;/\&/g'
}

write_kiss_web_plist() {
    # Write LaunchAgent $1 running kiss-web $2 in work dir $3 with logs
    # under $4.  The daemon resolves
    # its state dir from $KISS_HOME, so a KISS_HOME set for this install
    # must reach the service too.
    local plist="$1" kiss_web="$2" workdir="$3" home="$4" kiss_home_entry=""
    if [ -n "${KISS_HOME:-}" ]; then
        kiss_home_entry=$'\n        <key>KISS_HOME</key>\n        <string>'"$(xml_escape "$KISS_HOME")"'</string>'
    fi
    mkdir -p "$(dirname "$plist")"
    cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kiss.web-server</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(xml_escape "$kiss_web")</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$(xml_escape "$workdir")</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>5</integer>
    <key>StandardOutPath</key>
    <string>$(xml_escape "$home")/kiss-web-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$(xml_escape "$home")/kiss-web-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$(xml_escape "/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")</string>$kiss_home_entry
    </dict>
</dict>
</plist>
EOF
}

write_kiss_web_unit() {
    # Write systemd user unit $1 running kiss-web $2 in work dir $3 with
    # logs under $4.
    local unit="$1" kiss_web="$2" workdir="$3" home="$4" kiss_home_line=""
    if [ -n "${KISS_HOME:-}" ]; then
        kiss_home_line="Environment=KISS_HOME=$(unit_escape "$KISS_HOME")"$'\n'
    fi
    mkdir -p "$(dirname "$unit")"
    cat > "$unit" <<EOF
[Unit]
Description=${BRAND_PRODUCT_NAME:-KISS Sorcar} Remote Web Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=$(unit_escape "$kiss_web")
WorkingDirectory=$(unit_escape "$workdir")
Restart=always
RestartSec=5
Environment=PATH=$(unit_escape "$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin")
${kiss_home_line}StandardOutput=append:$(unit_escape "$home")/kiss-web-stdout.log
StandardError=append:$(unit_escape "$home")/kiss-web-stderr.log

[Install]
WantedBy=default.target
EOF
}

# ---------------------------------------------------------------------------
# Retire the VS Code extension
# ---------------------------------------------------------------------------
# The product left VS Code: its extension is gone from the repository and
# will never be updated.  Its users still land here, through the old
# extension's Update button (a VS Code terminal running this script) or
# the daemon's own update, so this run tells them where the product lives
# now -- in the terminal, in a file opened in their VS Code window and as
# a notice in the web app -- and takes the stale extension out of the
# editor.
VSCODE_EXTENSION_ID="ksenxx.kiss-sorcar"
# Under the daemon's home; kiss.server.vscode_retired turns a fresh flag
# into the web app's notice.
VSCODE_RETIRED_FLAG=".vscode-retired"
VSCODE_NOTICE_FILE="VSCODE-EXTENSION-REMOVED.txt"

vscode_extension_dirs() {
    # Print every installed copy of the retired extension: VS Code and
    # VS Code Insiders, their remote servers, Cursor and code-server.
    local dir ext
    for dir in "$HOME/.vscode-server/extensions" "$HOME/.vscode-server-insiders/extensions" \
               "$HOME/.vscode/extensions" "$HOME/.vscode-insiders/extensions" \
               "$HOME/.cursor/extensions" "$HOME/.local/share/code-server/extensions"; do
        for ext in "$dir/$VSCODE_EXTENSION_ID"-*; do
            [ ! -d "$ext" ] || printf '%s\n' "$ext"
        done
    done
}

vscode_user_detected() {
    # True for a VS Code user: this shell runs in a VS Code terminal (the
    # retired extension's Update button ran this script there), or a copy
    # of that extension is installed.
    [ "${TERM_PROGRAM:-}" = vscode ] && return 0
    [ -n "${VSCODE_IPC_HOOK_CLI:-}${VSCODE_GIT_IPC_HANDLE:-}" ] && return 0
    [ -n "$(vscode_extension_dirs)" ]
}

flag_vscode_retired() {
    # Have the web app of the daemon whose home is $1 show its "VS Code
    # extension removed" notice: the daemon sends it to every page while
    # the flag is fresh.  Only an installed copy of the extension earns
    # it; a VS Code terminal alone is any developer's shell.
    [ -n "$(vscode_extension_dirs)" ] || return 0
    mkdir -p "$1"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$1/$VSCODE_RETIRED_FLAG"
}

find_code_clis() {
    # Print the editor CLIs that can uninstall the extension and open a
    # file in the editor: code, code-insiders, cursor and code-server on
    # PATH (a VS Code terminal puts the ``code`` of its own window there,
    # a Remote-SSH one included), and on macOS the CLI inside the app
    # bundle when ``code`` is not on PATH.  ``KISS_CODE_CLI`` names the
    # one CLI to use instead (tests point it at a stub).
    local name path
    if [ -n "${KISS_CODE_CLI+x}" ]; then
        [ -z "$KISS_CODE_CLI" ] || printf '%s\n' "$KISS_CODE_CLI"
        return 0
    fi
    for name in code code-insiders cursor code-server; do
        path="$(command -v "$name" 2>/dev/null || true)"
        [ -z "$path" ] || printf '%s\n' "$path"
    done
    if [ "$OS" = Darwin ] && ! command -v code >/dev/null 2>&1; then
        for path in "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
                    "$HOME/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"; do
            [ ! -x "$path" ] || printf '%s\n' "$path"
        done
    fi
}

vscode_window_reachable() {
    # Whether ``code --reuse-window FILE`` would land in a window the user
    # is looking at: this script runs in a VS Code terminal, or VS Code is
    # running on this machine.  Otherwise the CLI would launch the editor
    # just to show the notice.
    [ "${TERM_PROGRAM:-}" = vscode ] && return 0
    [ -n "${VSCODE_IPC_HOOK_CLI:-}" ] && return 0
    command -v pgrep >/dev/null 2>&1 || return 1
    case "$OS" in
        Darwin) pgrep -f "Visual Studio Code.app" >/dev/null 2>&1 ;;
        Linux) pgrep -x code >/dev/null 2>&1 || pgrep -f "/share/code/code" >/dev/null 2>&1 ;;
        *) return 1 ;;
    esac
}

retire_vscode_extension() {
    # Uninstall every installed copy of the extension through the editor
    # CLIs, open the notice as a file in the running VS Code window (the
    # old extension's Update terminal closes with this script, taking the
    # box below with it), and leave the ``.extension-updated`` marker that
    # extension reloads its window on, so the uninstall takes effect now
    # instead of at the next editor restart.  Prints the lines of the
    # terminal notice box.
    local cli notice_file="$KISS_HOME_DIR/$VSCODE_NOTICE_FILE" had_extension=0
    local url="http://127.0.0.1:${KISS_WEB_PORT:-8787}" product="$BRAND_PRODUCT_NAME"
    # The product the user runs is the brand's when this run restarted a
    # brand home's daemon (step [5/5]): its hook has branded that app, so
    # the notice names what the user saw in VS Code, as the web app's
    # notice from that daemon does.
    if [ -n "${RESTART_HOME:-}" ]; then
        product="$(brand_field product_name "$(app_dir_of "$RESTART_HOME")/src/kiss/webapp/media/brand.json")"
        product="${product:-$BRAND_PRODUCT_NAME}"
    fi
    [ -z "$(vscode_extension_dirs)" ] || had_extension=1
    if [ "$had_extension" = 1 ]; then
        while IFS= read -r cli; do
            [ -n "$cli" ] || continue
            "$cli" --uninstall-extension "$VSCODE_EXTENSION_ID" >/dev/null 2>&1 3>&- 9>&- < /dev/null || true
        done < <(find_code_clis)
    fi
    {
        echo "$product has left VS Code."
        echo ""
        echo "The $product VS Code extension has been removed and will never be updated."
        if [ "$had_extension" = 1 ] && [ -z "$(vscode_extension_dirs)" ]; then
            echo "This update uninstalled it; VS Code may ask to reload its window."
        elif [ "$had_extension" = 1 ]; then
            echo "Uninstall it from the Extensions view of VS Code ($VSCODE_EXTENSION_ID)."
        fi
        echo ""
        echo "Use the web app instead: $url"
        echo "It opens in its own window after every update (a launcher is in your"
        echo "applications menu), and the sorcar CLI is unchanged."
    } > "$notice_file"
    if vscode_window_reachable; then
        # The first CLI; no ``| head`` (its early exit would SIGPIPE the
        # producer under pipefail).
        cli="$(find_code_clis)"
        cli="${cli%%$'\n'*}"
        # Detached: a CLI with no window to talk to must not hold the
        # install; ``3>&-`` keeps the saved terminal out of the editor.
        [ -z "$cli" ] || (nohup "$cli" --reuse-window "$notice_file" >/dev/null 2>&1 3>&- 9>&- < /dev/null &)
    fi
    [ "$had_extension" = 0 ] || date -u '+%Y-%m-%dT%H:%M:%SZ' > "$KISS_HOME_DIR/.extension-updated"
    echo "VS Code is no longer supported by $product."
    echo "The VS Code extension has been removed and will never be updated."
    if [ "$had_extension" = 1 ] && [ -n "$(vscode_extension_dirs)" ]; then
        echo "Uninstall it from the Extensions view of VS Code ($VSCODE_EXTENSION_ID)."
    fi
    echo "Use the web app instead: $url"
}

print_vscode_notice() {
    # The VS Code extension is gone: retire it and tell its users where
    # the product lives now, on the terminal (fd 3) and in the log, and
    # as the last install step so an old extension's progress toast
    # shows it too.
    local line lines=()
    while IFS= read -r line; do lines+=("$line"); done < <(retire_vscode_extension)
    print_box "${lines[@]}" >&3
    print_box "${lines[@]}" >> "$LOG_FILE"
    publish_step "${lines[*]}"
}

# The app directory kiss-web was confirmed to run from by
# restart_kiss_web_locked (empty when the restart was deferred or failed,
# or when launchd restarted the old registration instead:
# KISS_WEB_OLD_RUNTIME, restart_kiss_web_launch_agent).
KISS_WEB_SERVES=""
KISS_WEB_OLD_RUNTIME=""

retire_legacy_app_dir() {
    # Remove the installer's files from $1/kiss, the app directory of
    # state directory $1 from before the app was named after its home
    # (~/.s10s/kiss, now ~/.s10s/s10s), once kiss-web runs from the new
    # one, $2.  Anything else in it (a task's .kiss.artifacts, say) is
    # left to the user.  Nothing to do for the stock ~/.kiss/kiss.
    local legacy="$1/kiss" app="$2"
    [ "$legacy" != "$app" ] && [ ! -L "$legacy" ] && [ -d "$legacy/src/kiss" ] \
        && [ ! -e "$legacy/.git" ] || return 0
    if [ "$KISS_WEB_SERVES" != "$app" ]; then
        echo "   The old app directory $legacy is removed once kiss-web runs from $app."
        return 0
    fi
    rm -rf "$legacy/src" "$legacy/.venv" "$legacy/pyproject.toml" "$legacy/uv.lock" \
        "$legacy/README.md" "$legacy/LICENSE" "$legacy/.python-version"
    if rmdir "$legacy" 2>/dev/null; then
        echo "   Removed the old app directory $legacy (the app is in $app now)."
    else
        echo "   Removed the old app from $legacy (the app is in $app now); what else it holds is yours to remove."
    fi
}

absorb_problems() {
    # Take over the problems a nested pass recorded in brand home $1,
    # each line prefixed with the home, so the daemon this run starts
    # reports them (that home's daemon may never run).
    local file="$1/.install-problems" line
    [ -s "$file" ] || return 0
    while IFS= read -r line; do
        [ -z "$line" ] || record_problem "$(home_relative "$1"): $line"
    done < "$file"
    rm -f "$file"
}

move_problems_to() {
    # Record this run's problems in state directory $1 from now on, the
    # ones so far included.
    local file="$1/.install-problems"
    [ "$file" = "$PROBLEMS_FILE" ] && return 0
    if [ -s "$PROBLEMS_FILE" ]; then
        { cat "$PROBLEMS_FILE" >> "$file" && rm -f "$PROBLEMS_FILE"; } 2>/dev/null || true
    fi
    PROBLEMS_FILE="$file"
}

start_kiss_web_daemon() {
    # Register kiss-web from kiss_project $1 as the user's service
    # (launchd on macOS, systemd on Linux, a detached process elsewhere)
    # serving work dir $2, and (re)start it when that is safe.
    local project="$1" workdir="$2" kiss_web="$1/.venv/bin/kiss-web"
    local python="$1/.venv/bin/python" home lock
    if [ ! -x "$kiss_web" ]; then
        echo "   kiss-web not built in $project; not starting the daemon."
        return 0
    fi
    home="$("$python" -c 'from kiss.core.config import kiss_home; print(kiss_home())')" || return 0
    mkdir -p "$home"
    lock="$home/.kiss-web.restart.lock"
    if ! acquire_kiss_web_restart_lock "$lock"; then
        echo "   Another installer is restarting kiss-web; leaving it to that."
        return 0
    fi
    restart_kiss_web_locked "$project" "$workdir" "$home"
    release_kiss_web_restart_lock "$lock"
}

restart_kiss_web_locked() {
    # The decision and restart of start_kiss_web_daemon, under the lock:
    # kiss_project $1, work dir $2, state dir $3.
    local project="$1" workdir="$2" home="$3" kiss_web="$1/.venv/bin/kiss-web"
    local python="$1/.venv/bin/python" endpoint fingerprint stamp waited=0 old
    endpoint="$home/sorcar-local.json"
    fingerprint="$(kiss_web_fingerprint "$project" "$workdir")" || return 0
    if [ "$(cat "$home/.kiss-web.fingerprint" 2>/dev/null)" = "$fingerprint" ] \
        && [ -f "$endpoint" ] && kiss_web_port_open; then
        echo "   kiss-web already serves this runtime; leaving it running."
        KISS_WEB_SERVES="$project"
        return 0
    fi
    # A brand switch or another KISS_HOME moves the state directory, so
    # the daemon owning the port may be reachable only through an old
    # one: probe every candidate before stopping anything.
    local candidates=("$endpoint" "$HOME/.kiss/sorcar-local.json" \
                      "${KISS_HOME:-$HOME/${BRAND_HOME_DIR_NAME:-.kiss}}/sorcar-local.json")
    while IFS= read -r old; do
        [ -n "$old" ] && candidates+=("$old/sorcar-local.json")
    done < <(previous_kiss_web_homes)
    if ! kiss_web_is_idle "$python" "${candidates[@]}"; then
        # Not from a --restart-only run: that is the daemon's own attempt,
        # and a record would have it retry at once, in a loop.
        if [ -n "$KISS_WEB_RESTART_PENDING" ] && [ "${_KISS_RESTART_ONLY:-0}" != 1 ] \
            && record_deferred_restart "$KISS_WEB_RESTART_PENDING" "$python" "$home" "$workdir"; then
            echo "   kiss-web has tasks in flight; restart deferred: kiss-web restarts itself onto this build once they finish."
        else
            echo "   kiss-web has tasks in flight; restart deferred (re-run this script once they finish)."
        fi
        return 0
    fi
    # "Up" below means the new daemon rewrote its endpoint file after this
    # stamp; the pause keeps that comparison valid on a bash whose ``-nt``
    # has whole-second resolution (macOS).
    stamp="$home/.kiss-web.restart-stamp"
    : > "$stamp"
    sleep 1
    if [ "$(uname -s)" = "Darwin" ]; then
        local uid plist="$HOME/Library/LaunchAgents/com.kiss.web-server.plist"
        uid="$(id -u)"
        write_kiss_web_plist "$plist" "$kiss_web" "$workdir" "$home"
        echo "   Restarting the kiss-web LaunchAgent..."
        if [ "${_KISS_INTERACTIVE:-}" = 1 ] && kiss_web_hosts_this_shell; then
            echo "   kiss-web hosts this terminal, so it closes with the daemon;"
            echo "   the install carries on and its remaining output is in $LOG_FILE."
        fi
        restart_kiss_web_launch_agent "gui/$uid/com.kiss.web-server" "gui/$uid" "$plist"
    elif [ "$(uname -s)" = "Linux" ] && systemctl --user show-environment &>/dev/null; then
        write_kiss_web_unit "$HOME/.config/systemd/user/kiss-web.service" "$kiss_web" "$workdir" "$home"
        echo "   Restarting the kiss-web systemd user service..."
        # A shell kiss-web hosts (the webapp's Terminal tab) dies with the
        # daemon; this script survives it (see kiss-service-cgroup-escape)
        # and ``handle_hup`` sends the rest of its output to the log.
        if [ "${_KISS_INTERACTIVE:-}" = 1 ] && kiss_web_hosts_this_shell; then
            echo "   kiss-web hosts this terminal, so it closes with the daemon;"
            echo "   the install carries on and its remaining output is in $LOG_FILE."
        fi
        systemctl --user daemon-reload || true
        systemctl --user enable kiss-web.service &>/dev/null || true
        # The dying daemon hangs up its terminal, whose shell forwards the
        # SIGHUP to this script's whole process group; ignoring it in the
        # stop client keeps the stop blocking until the old daemon is gone.
        (trap '' HUP; exec systemctl --user stop kiss-web.service 2>/dev/null) || true
        stop_stray_kiss_web
        systemctl --user start kiss-web.service || warn "systemctl start failed"
        loginctl enable-linger "$(id -un)" 2>/dev/null || true
    else
        echo "   Starting kiss-web as a detached background process..."
        stop_stray_kiss_web
        # ``exec`` so no shell lingers holding this script's output open;
        # ``3>&-`` keeps the saved terminal (the password box's fd) and
        # ``9>&-`` the update lock out of the daemon.  The closes are a
        # separate ``exec``: ``exec CMD 9>&-`` leaks bash 3.2's (macOS)
        # undo copy of fd 9 into CMD, which then holds the lock.
        (cd "$workdir" && exec 3>&- 9>&- && exec nohup "$kiss_web" >> "$home/kiss-web-stdout.log" 2>> "$home/kiss-web-stderr.log" < /dev/null) &
    fi
    while [ "$waited" -lt "${KISS_WEB_START_TIMEOUT:-90}" ]; do
        if [ "$endpoint" -nt "$stamp" ] && kiss_web_port_open; then
            echo "$fingerprint" > "$home/.kiss-web.fingerprint"
            rm -f "$stamp" "$home/.kiss-web.restart-pending"
            echo "   kiss-web is up (${waited}s): http://localhost:$KISS_WEB_PORT"
            [ -n "$KISS_WEB_OLD_RUNTIME" ] || KISS_WEB_SERVES="$project"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    rm -f "$stamp"
    warn "kiss-web did not come up within ${waited}s; see $home/kiss-web-stderr.log"
}
# END: kiss-runtime-setup

# ---------------------------------------------------------------------------
# The app directory
# ---------------------------------------------------------------------------
# BEGIN: kiss-app-files  (tests extract this block verbatim)
# Python files installed as scripts rather than byte-compiled: glob
# patterns (bash ``case`` syntax, ``*`` also matches ``/``) matched
# against the path relative to the app directory.  Every other
# ``src/kiss/**/*.py`` is replaced by its ``.pyc`` in step [3/5].
COMPILE_KEEP_SOURCE=(
    # The bundled SEAs: ``run_agent``, the ``sea`` CLI and the webapp
    # load them from their ``.py`` path (sea_commands.py, sea_lint.py).
    "src/kiss/agents/seas/*"
    # Channel SEAs (agents/third_party_agents/*/<channel>_sea.py) and the
    # cron prompt SEA (agents/sorcar/cron_prompt_sea.py): loaded by path too.
    "*_sea.py"
    # The ``/cron`` built-in command (BUILTIN_COMMANDS in sea_commands.py):
    # a SEA script like the others, located through its module name.
    "src/kiss/agents/sorcar/cron_agent.py"
    # Read as text by the webapp's version check (web_server._parse_version_py)
    # and by hatchling for the project version.
    "src/kiss/core/_version.py"
    # kiss.scripts.sync_db ships its own source (``__file__``) and these
    # two modules' to the remote machine it synchronizes the database with.
    "src/kiss/scripts/sync_db.py"
    "src/kiss/agents/sorcar/chat_summary.py"
    "src/kiss/agents/sorcar/history_columns.py"
    # The /sync_history_and_memory SEA ships this module's source with its
    # memory_delta.py to the remote (peer_sync.MERGE_PAGES_PATH), and
    # scripts/sync-memory.sh runs it as a script.
    "src/kiss/scripts/merge_memory_pages.py"
)

keep_source() {
    # True when the app-relative path $1 is in COMPILE_KEEP_SOURCE.
    local pattern
    for pattern in "${COMPILE_KEEP_SOURCE[@]}"; do
        # shellcheck disable=SC2254
        case "$1" in $pattern) return 0 ;; esac
    done
    return 1
}

list_app_files() {
    # Print, relative to the checkout, every file the app needs: the
    # project files uv builds the venv from, and the kiss package without
    # its tests and without the webapp's JS tooling (package.json, eslint,
    # the jsdom tests) — only the ``media`` frontend, which kiss-web serves.
    # A git checkout lists its tracked files; a tarball falls back to find.
    local path
    for path in pyproject.toml uv.lock README.md LICENSE; do
        [ -f "$PROJECT_DIR/$path" ] && echo "$path"
    done
    (
        cd "$PROJECT_DIR"
        if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            git ls-files src/kiss
        else
            find src/kiss \( -name __pycache__ -o -name node_modules -o -name out \) -prune \
                -o -type f ! -name '*.pyc' ! -name .DS_Store -print
        fi
    ) | while IFS= read -r path; do
        case "$path" in
            src/kiss/tests/*) ;;
            src/kiss/webapp/media/*) echo "$path" ;;
            src/kiss/webapp/*) ;;
            *) echo "$path" ;;
        esac
    done
}

copy_app_files() {
    # Rebuild the project files and ``src`` of $APP_DIR from the checkout.
    # The venv is kept (``uv sync`` reuses it); the rest is replaced so a
    # file removed upstream does not linger.
    local list="$KISS_HOME_DIR/.install-files" count
    mkdir -p "$APP_DIR"
    rm -rf "$APP_DIR/src" "$APP_DIR/pyproject.toml" "$APP_DIR/uv.lock" \
           "$APP_DIR/README.md" "$APP_DIR/LICENSE"
    list_app_files > "$list"
    count="$(wc -l < "$list" | tr -d ' ')"
    if [ "$count" -eq 0 ]; then
        echo "   ERROR: no files to install found under $PROJECT_DIR/src/kiss"
        record_problem "ERROR: no files to install found under $PROJECT_DIR/src/kiss"
        rm -f "$list"
        return 1
    fi
    (cd "$PROJECT_DIR" && tar cf - -T "$list") | (cd "$APP_DIR" && tar xf -) || {
        rm -f "$list"
        return 1
    }
    rm -f "$list"
    echo "   Copied $count files to $APP_DIR"
}

# A white-label brand keeps its own versions of these frontend files in
# the checkout's ``.brand/`` directory; the app gets them in place of the
# stock files.
BRAND_OVERLAY_FILES=(brand.json brand.css kiss-icon.svg kiss-icon.png thumbnail.jpeg welcome-logo.png welcome-logo-dark.png)
BRAND_MEDIA_REL="src/kiss/webapp/media"

copy_brand_overlay() {
    local overlay="$PROJECT_DIR/.brand" f
    [ -d "$overlay" ] || return 0
    echo "   Applying brand overlay from $overlay..."
    for f in "${BRAND_OVERLAY_FILES[@]}"; do
        [ -f "$overlay/$f" ] || continue
        cp -f "$overlay/$f" "$APP_DIR/$BRAND_MEDIA_REL/$f" || return 1
        echo "      $BRAND_MEDIA_REL/$f"
    done
}

build_app_venv() {
    # Build the Python environment of $APP_DIR with uv ($1).
    echo "   Installing Python dependencies in $APP_DIR..."
    run_with_heartbeat "uv sync" "$1" sync --directory "$APP_DIR" || return 1
    # Pin the interpreter line uv chose: the ``.pyc`` files written below
    # load only on that line, so a later ``uv run`` must not switch it.
    "$APP_DIR/.venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' \
        > "$APP_DIR/.python-version"
}

compile_app_library() {
    # Byte-compile the library with the venv's own interpreter (a ``.pyc``
    # loads only on the Python line that wrote it) and delete the compiled
    # sources.  ``-b`` writes ``foo.pyc`` beside ``foo.py``, where Python
    # imports it once ``foo.py`` is gone; ``unchecked-hash`` keeps Python
    # from looking for a source to validate it against.  The files of
    # COMPILE_KEEP_SOURCE stay as scripts and lose their ``.pyc`` instead.
    local python="$APP_DIR/.venv/bin/python" src="$APP_DIR/src" path rel
    local compiled=0 scripts=0
    "$python" -m compileall -q -f -b --invalidation-mode unchecked-hash "$src" || return 1
    find "$src" -name __pycache__ -type d -prune -exec rm -rf {} +
    while IFS= read -r path; do
        rel="${path#"$APP_DIR"/}"
        if keep_source "$rel"; then
            rm -f "${path%.py}.pyc"
            scripts=$((scripts + 1))
        else
            rm -f "$path"
            compiled=$((compiled + 1))
        fi
    done < <(find "$src" -name '*.py' -type f)
    echo "   Byte-compiled $compiled modules; $scripts Python scripts kept as source"
}
# END: kiss-app-files

install_sorcar_cli() {
    # The ``sorcar`` CLI wrapper, bound to $APP_DIR through uv ($1).
    if printf '%s\n' '#!/bin/bash' \
            "# Installed by ${BRAND_PRODUCT_NAME:-KISS Sorcar} install.sh" \
            'export KISS_WORKDIR="$PWD"' \
            "exec \"$1\" run --directory \"$APP_DIR\" sorcar \"\$@\"" > "$BIN_DIR/sorcar.tmp" \
            && chmod 755 "$BIN_DIR/sorcar.tmp" && mv -f "$BIN_DIR/sorcar.tmp" "$BIN_DIR/sorcar"; then
        echo "   Installed the sorcar CLI at $BIN_DIR/sorcar"
    else
        warn "could not install the sorcar CLI at $BIN_DIR/sorcar"
    fi
    add_local_bin_to_shell_rc
}

install_repo_script_launcher() {
    # A launcher in $BIN_DIR for the checkout's $1 script (rsorcar,
    # sorcar-docker): a wrapper, not a symlink or copy, so the script sees
    # its own directory as the checkout.
    local name="$1"
    local target="$PROJECT_DIR/$name"
    if [ ! -f "$target" ]; then
        warn "$target not found — skipping $name launcher."
        return 0
    fi
    {
        echo '#!/bin/bash'
        echo "# Installed by install.sh — launcher for $target"
        echo "# Wrapper (not a symlink/copy): the real script must see its own"
        echo "# directory as the KISS Sorcar checkout."
        echo "exec bash \"$target\" \"\$@\""
    } > "$BIN_DIR/$name"
    chmod +x "$BIN_DIR/$name"
    echo "   Installed $BIN_DIR/$name -> $target"
}

install_playwright_chromium() {
    # Playwright's Chromium for the browser tools, through uv ($1).
    echo "   Installing Playwright Chromium..."
    if ! run_with_heartbeat "playwright install" "$1" run --directory "$APP_DIR" python -m playwright install chromium; then
        warn "Playwright Chromium install failed; run '$APP_DIR/.venv/bin/python -m playwright install chromium' later."
    elif [ "$(uname -s)" = "Linux" ]; then
        # Chromium's system libraries need root (install-deps calls sudo
        # itself); only try when that cannot stall on a password prompt.
        if [ "$(id -u)" = 0 ] || sudo -n true 2>/dev/null; then
            (cd "$APP_DIR" && "$1" run python -m playwright install-deps chromium) \
                || warn "Chromium system libraries not installed (playwright install-deps failed)."
        else
            echo "   Chromium system libraries: run 'sudo $APP_DIR/.venv/bin/python -m playwright install-deps chromium' if the browser fails to start."
        fi
    fi
}

# Tee stdout+stderr to the install log AND the terminal.  A process
# substitution, not a ``{ … } | tee`` pipeline: bash resets trapped
# signals to their default disposition inside a pipeline's subshells, so
# the INT/TERM/HUP traps above would be dead for the whole install body;
# the script also keeps its own pid (the lock, the progress file and the
# daemon's updater all name it) and ``set -e`` keeps seeing the script's
# exit status.  The tee child ignores
# INT/TERM/HUP so a stray terminal signal cannot cut the log short, and
# closes the lock fd so it never keeps the update lock alive.
# fd 3 keeps the terminal: the remote-access password is printed there
# at the end, and must not land in the log.  A nested brand-home pass
# (``_KISS_BRAND_PASS``, step [5/5]) writes to the outer run's tee
# instead: its own tee would put every line in the log twice.
exec 3>&1
if [ -z "${_KISS_BRAND_PASS:-}" ]; then
    exec > >(trap '' INT TERM HUP; exec 9>&-; exec tee -a "$LOG_FILE") 2>&1
else
    exec 2>&1
fi

{
    echo "=== ${BRAND_PRODUCT_NAME:-KISS Sorcar} App Install ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Checkout: $PROJECT_DIR"
    echo "App directory: $APP_DIR"
    echo "OS: $OS ($ARCH)"
    if [ "$_KISS_INTERACTIVE" = 1 ]; then
        echo "Mode: interactive (asks before installing Homebrew; pass --non-interactive to skip the question)"
    else
        echo "Mode: non-interactive (every question takes its default answer)"
    fi
    echo ""

    if [ ! -d "$PROJECT_DIR/src/kiss" ]; then
        fail "$PROJECT_DIR/src/kiss not found; run this script from a KISS Sorcar checkout."
    fi

    # --restart-only: what the deferred run's step [5/5] left undone, the
    # daemon restart onto the app it installed; nothing else runs.
    if [ "$_KISS_RESTART_ONLY" = 1 ]; then
        report_step "Restarting kiss-web onto the app in $APP_DIR..."
        start_kiss_web_daemon "$APP_DIR" "$USER_PWD" \
            || warn "kiss-web was not restarted; start it with '$APP_DIR/.venv/bin/kiss-web'."
        retire_legacy_app_dir "$KISS_HOME_DIR" "$APP_DIR"
        exit 0
    fi

    if [ "$OS" = "Darwin" ]; then
        # Homebrew is optional: it is only the preferred way to install
        # cloudflared (install_cloudflared falls back to a GitHub download).
        report_step "Checking Homebrew..."
        ensure_homebrew
        echo ""
    fi

    report_step "[1/5] Copying ${BRAND_PRODUCT_NAME:-KISS Sorcar} to $APP_DIR..."
    copy_app_files
    copy_brand_overlay
    echo ""

    report_step "[2/5] Building the Python environment..."
    if ! UV="$(ensure_uv)"; then
        fail "uv is not available and could not be installed (https://astral.sh/uv)."
    fi
    echo "   Using uv at $UV"
    if ! build_app_venv "$UV"; then
        fail "uv sync failed in $APP_DIR"
    fi
    echo ""

    report_step "[3/5] Byte-compiling the library..."
    if ! compile_app_library; then
        fail "byte-compiling $APP_DIR/src failed"
    fi
    echo ""

    report_step "[4/5] Installing the sorcar CLI, Playwright Chromium and cloudflared..."
    install_sorcar_cli "$UV"
    install_repo_script_launcher rsorcar
    install_repo_script_launcher sorcar-docker
    install_playwright_chromium "$UV"
    install_cloudflared
    echo ""

    report_step "[5/5] Finishing the setup..."
    # An installed KISS Sorcar (a project root without .git, as $APP_DIR)
    # reads its model catalog from $KISS_HOME/MODEL_INFO.json
    # (kiss.core.models.model_info._select_catalog_path); refresh that copy
    # from the bundled file on every install.  User overrides live in
    # ~/.kiss/MY_MODELS.json and are never touched.
    MODEL_INFO_SRC="$PROJECT_DIR/src/kiss/core/models/MODEL_INFO.json"
    MODEL_INFO_DST="$KISS_HOME_DIR/MODEL_INFO.json"
    if [ -f "$MODEL_INFO_SRC" ]; then
        cp "$MODEL_INFO_SRC" "$MODEL_INFO_DST"
        echo "   Copied MODEL_INFO.json to $MODEL_INFO_DST"
    else
        warn "$MODEL_INFO_SRC not found; skipped the $KISS_HOME_DIR copy"
    fi

    # Post-install hooks: every executable in $KISS_HOME_DIR/post-install.d/
    # runs now, with the app on disk and before the daemon restart, so a
    # layer that patches the installed files (a white-label brand) is
    # applied before the daemon serves them.  Hooks run in name order
    # with KISS_HOME and KISS_APP_DIR set and no stdin; a failing hook is
    # reported and the install still completes.  A hook that runs a
    # brand's installer out of a git checkout gets that checkout pulled
    # to its upstream first (update_brand_checkout), so the brand steps
    # it performs are the brand's latest.
    for hook in "$KISS_HOME_DIR"/post-install.d/*; do
        [ -f "$hook" ] && [ -x "$hook" ] || continue
        update_brand_checkout "$hook"
        echo "   Running post-install hook $hook..."
        hook_rc=0
        KISS_HOME="$KISS_HOME_DIR" KISS_APP_DIR="$APP_DIR" "$hook" < /dev/null || hook_rc=$?
        if [ "$hook_rc" != 0 ]; then
            warn "post-install hook $hook exited with status $hook_rc"
        fi
    done

    # The remote-access password, stored as ``remote_password`` in
    # $KISS_HOME_DIR/config.json (the Settings page shows it); the daemon
    # checks it on every command a browser from another device sends.
    # kiss.core.remote_password keeps the stored one and draws a random
    # one only when none is stored yet, so an update never changes the
    # password.  It is also shown in the box at the end of this script.
    PREVIOUS_PASSWORD="$(stored_remote_password "$KISS_HOME_DIR" "$APP_DIR")"
    REMOTE_PASSWORD="$(KISS_HOME="$KISS_HOME_DIR" "$APP_DIR/.venv/bin/python" -m kiss.core.remote_password 9>&-)" \
        || { REMOTE_PASSWORD=""; warn "could not set the remote-access password."; }
    if [ -z "$REMOTE_PASSWORD" ]; then
        :
    elif [ "$REMOTE_PASSWORD" = "$PREVIOUS_PASSWORD" ] && [ -n "${_KISS_BRAND_PASS:-}" ]; then
        echo "   Kept the remote-access password of $KISS_HOME_DIR (shown in its Settings)."
    elif [ "$REMOTE_PASSWORD" = "$PREVIOUS_PASSWORD" ]; then
        echo "   Kept the existing remote-access password (shown at the end)."
    elif [ -n "${_KISS_BRAND_PASS:-}" ]; then
        echo "   Set a new remote-access password for $KISS_HOME_DIR (shown in its Settings)."
    else
        echo "   Set a new remote-access password (shown at the end)."
    fi

    # Other brand homes.  A white-label brand (Seamless Loop, say) keeps
    # its own state directory ~/.<brand> with its own app copy in
    # ~/.<brand>/<brand> and a post-install hook that re-brands it, and its
    # kiss-web service runs from there (KISS_HOME in the unit/plist).
    # Installing only into $KISS_HOME_DIR would leave that app stale and
    # the restart below would switch the daemon to this one: so this
    # checkout is installed into every other brand home as well, by a
    # nested non-interactive run of this script -- steps 1-5 into that
    # home's app directory, its MODEL_INFO.json, hooks and password.
    # ``_KISS_BRAND_PASS`` makes the nested run skip this discovery (no
    # recursion), the password box and the VS Code notice;
    # KISS_SKIP_LAUNCH keeps it from touching the daemon or a browser.  A
    # failed nested pass is a warning; this install still completes.
    #
    # The daemon is then the brand's, not this stock app: the home the
    # existing service runs from when that is one of those homes and its
    # app is there, else -- for an install into the stock ~/.kiss -- the
    # first brand home the nested pass installed into (its hook has
    # branded that app by now).  A brand on the machine is what the user
    # chose to run; the stock app stays installed next to it and comes
    # back once the brand's hook is removed (its own installer's revert)
    # and the service definition no longer names that home.  An install
    # into a brand's or a custom home starts its own daemon unless the
    # service runs from another brand home.
    RESTART_HOME=""
    if [ -z "${_KISS_BRAND_PASS:-}" ]; then
        first_brand_home=""
        while IFS= read -r brand_home; do
            [ -n "$brand_home" ] || continue
            report_step "Installing into brand home $(home_relative "$brand_home")..."
            if KISS_HOME="$brand_home" KISS_SKIP_LAUNCH=1 KISS_UPDATE_LOCK_HELD=1 _KISS_BRAND_PASS=1 \
                    bash "$PROJECT_DIR/install.sh" --non-interactive < /dev/null 9>&-; then
                echo "   Installed into brand home $brand_home"
                [ -n "$first_brand_home" ] || first_brand_home="$brand_home"
            else
                warn "the install into brand home $brand_home failed (status $?); this install carries on."
            fi
            absorb_problems "$brand_home"
        done < <(other_brand_homes)
        service_home="$(service_kiss_home | head -n 1)"
        service_home="${service_home%/}"
        if [ -n "$service_home" ] && [ "$service_home" != "${KISS_HOME_DIR%/}" ] \
            && [ -x "$(app_dir_of "$service_home")/.venv/bin/kiss-web" ]; then
            RESTART_HOME="$service_home"
            RESTART_WHY="The kiss-web service runs from brand home $RESTART_HOME; restarting that app."
        elif [ "${KISS_HOME_DIR%/}" = "$HOME/.kiss" ] && [ -n "$first_brand_home" ]; then
            RESTART_HOME="$first_brand_home"
            RESTART_WHY="Brand home $RESTART_HOME is installed and branded; starting the daemon from that app, not from $APP_DIR."
        fi
    fi

    # The web app's "VS Code extension removed" notice, from the home of
    # the daemon this run (re)starts; a nested brand pass starts none.
    [ -n "${_KISS_BRAND_PASS:-}" ] || flag_vscode_retired "${RESTART_HOME:-$KISS_HOME_DIR}"
    # So are this run's problems: recorded from here on, and the ones so
    # far moved, into that home, before its daemon comes up and reads them.
    [ -z "$RESTART_HOME" ] || move_problems_to "$RESTART_HOME"

    # The daemon serves the directory this script was run from (see
    # daemon_work_dir).
    if [ -z "${KISS_SKIP_LAUNCH:-}" ]; then
        if [ -n "$RESTART_HOME" ]; then
            echo "   $RESTART_WHY"
            restart_app="$(app_dir_of "$RESTART_HOME")"
            # That home's password is the one the restarted daemon asks for.
            brand_password="$(stored_remote_password "$RESTART_HOME" "$restart_app")"
            [ -z "$brand_password" ] || REMOTE_PASSWORD="$brand_password"
            KISS_HOME="$RESTART_HOME" start_kiss_web_daemon "$restart_app" "$USER_PWD" \
                || warn "kiss-web was not restarted; start it with '$restart_app/.venv/bin/kiss-web'."
            retire_legacy_app_dir "$RESTART_HOME" "$restart_app"
        else
            start_kiss_web_daemon "$APP_DIR" "$USER_PWD" \
                || warn "kiss-web was not restarted; start it with '$APP_DIR/.venv/bin/kiss-web'."
            retire_legacy_app_dir "$KISS_HOME_DIR" "$APP_DIR"
        fi
    else
        echo "   KISS_SKIP_LAUNCH set; not starting the kiss-web daemon."
    fi
    echo ""

    if [ -n "${_KISS_BRAND_PASS:-}" ]; then
        echo "=== Brand home $KISS_HOME_DIR: app install complete ==="
    else
        echo "=== App install complete ==="
        echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "App directory: $APP_DIR"
        echo "CLI: $BIN_DIR/sorcar"
        echo ""
        echo "API keys are set in the webapp's settings; the remote-access password is printed below."
    fi
}

print_box() {
    # Print the lines $@ framed in ``*``, as wide as the longest line.
    local line width=0 stars
    for line in "$@"; do
        [ ${#line} -le "$width" ] || width=${#line}
    done
    stars="$(printf '%*s' "$((width + 6))" '' | tr ' ' '*')"
    echo ""
    echo "$stars"
    for line in "$@"; do
        printf '*  %-*s  *\n' "$width" "$line"
    done
    echo "$stars"
}

print_password_box() {
    # The remote-access password ($1) framed in ``*``, on the terminal (fd 3,
    # the stdout saved before the log tee) so it never lands in the log.
    # With the terminal gone (``handle_hup``) there is nowhere to show it
    # but the log, which must not have it: say where to find it instead.
    if [ "$TERMINAL_GONE" = 1 ]; then
        echo ""
        echo "The remote-access password was not shown because the terminal closed;"
        echo "it is in the web app's Settings."
        return 0
    fi
    print_box "Remote access password:  $1" \
        "Type it in the web app on another device; it is also shown in Settings." >&3
}

if [ -z "${_KISS_BRAND_PASS:-}" ]; then
    echo ""
    echo "Log saved to $LOG_FILE"
fi

# BEGIN: kiss-browser-helpers  (tests extract this block verbatim)
machine_is_remote() {
    # An SSH session, or Linux without a display, has no browser to open.
    [ -n "${SSH_CONNECTION:-}${SSH_TTY:-}${SSH_CLIENT:-}" ] && return 0
    [ "$OS" = "Linux" ] && [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && return 0
    return 1
}

open_in_browser() {
    case "$OS" in
        Darwin)
            open "$1" >/dev/null 2>&1 3>&- 9>&-
            ;;
        Linux)
            command -v xdg-open >/dev/null 2>&1 || return 1
            # ``3>&-``: the saved terminal (the password box's fd) must not
            # stay open in a browser that outlives the installer.
            (nohup xdg-open "$1" >/dev/null 2>&1 3>&- 9>&- &)
            ;;
        *)
            return 1
            ;;
    esac
}
# END: kiss-browser-helpers

# ---------------------------------------------------------------------------
# Open the webapp
# ---------------------------------------------------------------------------
# Step [5/5] started the kiss-web daemon (or left a healthy one running).
# Wait for the daemon's URL file, then open the Local URL
# (http://127.0.0.1:PORT, plain HTTP on loopback) as a standalone app
# window: kiss-app (next to kiss-web in the app's venv) registers the
# desktop launcher, opens the webapp in a Chromium app-mode window and
# then shows the launcher's folder in the file manager, so the user
# sees where the app is.  Without kiss-app, or if it fails, the URL
# opens in the default browser.  On a remote machine (an SSH session, or Linux without a
# display) there is no window to open here, so the cloudflared URL is
# printed for the user to open on their own device.
# ``KISS_SKIP_LAUNCH`` skips this too.
[ -n "${KISS_SKIP_LAUNCH:-}" ] || report_step "Waiting for the kiss-web daemon to open the webapp..."
# BEGIN: kiss-open-webapp
KISS_WEBAPP_WAIT_SECS="${KISS_WEBAPP_WAIT_SECS:-120}"

find_kiss_web() {
    # kiss-web lives in the app's venv; a development checkout may carry
    # its own.  Any copy will do: the URLs come from remote-url.json.
    local candidate
    for candidate in \
        "$APP_DIR/.venv/bin/kiss-web" \
        "$(command -v kiss-web 2>/dev/null || true)" \
        "$PROJECT_DIR/.venv/bin/kiss-web"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

find_kiss_app() {
    # The kiss-app of the daemon serving state directory $1: the one in
    # that home's app venv (a brand's app when a brand daemon was
    # restarted), else the one next to kiss-web $2.
    local candidate
    for candidate in "$(app_dir_of "$1")/.venv/bin/kiss-app" "${2%/*}/kiss-app"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

url_file_field() {
    # Print the string value of key $2 in the daemon's remote-url.json ($1),
    # a flat object written with one ``"key": "value"`` pair per line.
    sed -n "s/^[[:space:]]*\"$2\":[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$1" | head -n 1
}

open_webapp() {
    # The state directory of the daemon that was (re)started: the brand
    # home when the daemon is a brand's (RESTART_HOME, step [5/5]).  Its
    # URL file names the webapp; its kiss-app, run with that home, owns
    # the window's profile and the desktop launcher.
    local app_home="${RESTART_HOME:-${KISS_HOME:-$HOME/${BRAND_HOME_DIR_NAME:-.kiss}}}"
    local url_file="$app_home/remote-url.json"
    local remote=0 kiss_web="" tunnel="" loopback="" waited=0
    machine_is_remote && remote=1
    echo ""
    echo "Waiting for the kiss-web daemon..."
    while :; do
        [ -n "$kiss_web" ] || kiss_web="$(find_kiss_web || true)"
        if [ -n "$kiss_web" ] && [ -s "$url_file" ]; then
            tunnel="$(url_file_field "$url_file" tunnel)"
            loopback="$(url_file_field "$url_file" loopback)"
            # A remote machine needs the cloudflared URL, which the daemon
            # adds to the file a little after its Local URL.
            if [ "$remote" = 0 ] || [ -n "$tunnel" ]; then
                break
            fi
        fi
        if [ "$waited" -ge "$KISS_WEBAPP_WAIT_SECS" ]; then
            echo "   The kiss-web daemon did not come up within ${KISS_WEBAPP_WAIT_SECS}s."
            echo "   Once it is running, print the webapp URL with:"
            # kiss-web is usually not on PATH: name the binary if found.
            echo "       '${kiss_web:-kiss-web}' --url"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 60)) -eq 0 ]; then
            echo "   ... still waiting (${waited}s)"
        fi
    done

    if [ "$remote" = 1 ]; then
        echo ""
        echo "This machine is remote; open the webapp on your own device at:"
        echo "    $tunnel"
        return 0
    fi
    if [ -z "$loopback" ]; then
        # Older daemons wrote only the localhost URL.
        loopback="$(url_file_field "$url_file" local)"
        loopback="${loopback/localhost/127.0.0.1}"
    fi
    echo ""
    local kiss_app
    kiss_app="$(find_kiss_app "$app_home" "$kiss_web" || true)"
    if [ -n "$kiss_app" ]; then
        echo "Opening the webapp in its own window at $loopback"
        local launcher=1
        KISS_HOME="$app_home" "$kiss_app" --install-launcher || {
            launcher=0
            warn "The desktop launcher was not installed; start the app with 'KISS_HOME=$app_home $kiss_app'."
        }
        # kiss-app starts the browser detached and returns once it is
        # up; non-zero means no Chromium-family browser, or one that
        # died at once (no display, say).
        if KISS_HOME="$app_home" "$kiss_app" --url "$loopback" --wait 0; then
            # After the window: the launcher's folder in the file
            # manager (Finder on macOS), so the user sees where the
            # app is and can pin it.
            [ "$launcher" = 0 ] \
                || KISS_HOME="$app_home" "$kiss_app" --reveal-launcher \
                || warn "Could not show the desktop launcher's folder."
            return 0
        fi
        warn "The standalone window did not open; using the browser instead."
    fi
    echo "Opening the webapp at $loopback"
    if ! open_in_browser "$loopback"; then
        echo "   Could not open a browser; open the URL above yourself."
    fi
}

if [ -z "${KISS_SKIP_LAUNCH:-}" ]; then
    open_webapp || true
fi
# END: kiss-open-webapp

# Last of all, so it is the first thing the user sees looking back.  A
# nested brand-home pass prints neither: the outer run shows its own
# password box (the restarted daemon's) and the notice once.
if [ -z "${_KISS_BRAND_PASS:-}" ]; then
    [ -z "${REMOTE_PASSWORD:-}" ] || print_password_box "$REMOTE_PASSWORD"
    ! vscode_user_detected || print_vscode_notice
fi
