#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Install KISS Sorcar from source.
#
# Bootstrap the tools needed to build and install the VS Code extension from
# a cloned checkout, finish the runtime setup the extension would otherwise
# do on its first activation (bundled Python environment, ``sorcar`` CLI,
# Playwright, cloudflared, the kiss-web daemon — see "Runtime setup without
# VS Code" below), then launch VS Code (the desktop app, and VS Code in the
# browser via ``code serve-web``) and open the webapp.  The extension's
# DependencyInstaller re-checks all of it on activation, so installing the
# VSIX directly takes the same path, just later; it also owns the
# interactive parts (API keys, the remote-access password).
#
# Usage: ./install.sh [--non-interactive]
#
#   Run from a terminal, the script asks ``[Y/n]`` before installing
#   Homebrew.  ``--non-interactive`` (same as
#   ``KISS_NONINTERACTIVE=1``) answers every question with its default (Yes)
#   and never touches the terminal; it is also what happens automatically
#   when there is no terminal to ask on.  See "Interactive mode" below.
#
# Log saved to ~/.kiss/install.log
#
# ---------------------------------------------------------------------------
# Bulletproof terminal-signal immunity via new-session detachment
# ---------------------------------------------------------------------------
#
# Failure mode this block cures
# -----------------------------
# A user clicked the VS Code "Update" button (settings panel), which calls
# ``runUpdate()`` in ``SorcarSidebarView.ts``.  That method used to open a
# VS Code integrated terminal and ``terminal.sendText`` a compound command
# ending in ``bash '/Users/ksen/.kiss/kiss_ai/install.sh'`` (current builds
# run the installer as the terminal process instead, but extensions in the
# wild still inject the command as prompt text).  The install ran through Xcode
# CLT, Homebrew, git, node and VS Code CLI, then died
# right in the middle of the TypeScript compile::
#
#     >>> [4/5] Building VS Code extension...
#        Compiling extension TypeScript...
#
#     > kiss-sorcar@2026.6.38 compile
#     > tsc -p ./
#
#     ^C
#        ⚠ Interrupt received but ignored — long npm/git steps can sit
#           silent for 30-60 s while they download or extract.  Press
#           Ctrl+C again within 3 s to really abort.
#     ksen@Mac kiss_ai %
#
# The user explicitly says they did NOT press Ctrl-C — something delivered
# SIGINT (or ``\x03`` into the PTY) during ``tsc``.  install.sh's outer
# ``handle_interrupt`` trap fired (the diagnostic printed) but the script
# STILL exited (the shell prompt returned).
#
# Why the existing trap defences are not enough
# ---------------------------------------------
# 1. SIGINT delivered to a terminal foreground process group is delivered to
#    EVERY process in that group simultaneously — including ``npm``, ``node``,
#    and ``tsc``.  install.sh's own SIGINT trap only protects install.sh's
#    own bash process.
# 2. ``run_with_heartbeat`` wraps its child in ``( trap '' INT TERM; exec ... )``
#    so the child inherits SIG_IGN across exec.  POSIX says SIG_IGN survives
#    exec, BUT Node.js installs its own SIGINT handling in some configurations
#    and may not respect inherited SIG_IGN — so ``tsc`` (which runs on Node)
#    can still die on a stray SIGINT, npm returns non-zero, and ``set -e``
#    aborts install.sh.
# 3. Some child processes (e.g. ``"$CODE_CLI" --install-extension``) are
#    NOT wrapped in ``run_with_heartbeat`` and therefore are NOT protected
#    by the SIG_IGN subshell at all.
#
# Why ``setsid`` (a new session with no controlling TTY) is the bulletproof
# answer
# -----------------------------------------------------------------------
# Terminal-driven signals (Ctrl-C / Ctrl-Z / hangup on ``\x03``-and-close
# from a PTY teardown) are delivered by the kernel ONLY to the process
# group(s) of the controlling terminal's session.  A session with NO
# controlling terminal literally cannot receive ``SIGINT`` from any
# terminal — the kernel has nowhere to deliver them from.  Once the install
# body runs inside a fresh session created with ``setsid(2)``, no amount of
# ``\x03`` injected into the original VS Code PTY can reach it.
#
# Why we fork via perl instead of ``exec setsid`` directly
# --------------------------------------------------------
# Running install.sh from bash makes install.sh the leader of its own
# process group (typically also of its session, depending on how it was
# launched).  ``setsid(2)`` refuses with EPERM when called by a process
# group leader — so a direct ``exec setsid bash install.sh`` would fail
# immediately.  We must fork FIRST: the child (not the leader) can then
# successfully call ``setsid`` and exec a fresh ``bash`` on this script.
# ``perl`` is available at ``/usr/bin/perl`` on every macOS release and
# every standard Linux distro the install supports, and ``POSIX::setsid``
# is part of the core POSIX module that ships with perl itself — no CPAN
# dependencies.
#
# The parent perl IGNOREs INT/TERM/HUP, then ``waitpid``s the child and
# forwards its exit code.  Ignoring those three signals in the parent is
# important too: a stray ``\x03`` from the original terminal can still hit
# the parent's process group, and if the parent died the user would see
# the same "shell prompt returned, install aborted" symptom even though
# the install child is happily continuing in its detached session.
#
# Defense in depth
# ----------------
# The existing ``handle_interrupt``/``handle_hup`` traps below, the
# ``run_with_heartbeat`` SIG_IGN subshell, and the
# ``exec > >(tee -a "$LOG_FILE") 2>&1`` redirect remain unchanged — they
# stay as belt-and-braces defence in depth (and keep the existing
# regression tests passing).  The new-session detachment is now the
# PRIMARY defence.
#
# Sentinel: ``_KISS_NEW_SESSION=1`` is exported before the re-exec so the
# re-exec'd child does NOT fork again (no infinite loop).
#
# Graceful fallback: if ``perl`` is unavailable (extremely unlikely on
# macOS / mainstream Linux), the script simply continues without
# detachment, preserving the previous trap-only behaviour.  Interactive
# mode (below) skips the detachment deliberately, for the same trap-only
# behaviour: its ``[Y/n]`` questions and ``sudo``'s password prompt need
# the controlling terminal that ``setsid`` would take away.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# Interactive mode (the default at a terminal)
# ---------------------------------------------------------------------------
# A human running ``./install.sh`` (or the ``curl ... | bash`` one-liner,
# which still has a controlling terminal) is asked ``[Y/n]`` before
# Homebrew is installed (see ``confirm`` below); "no" skips it and
# carries on.  Tools the install cannot proceed without (git, Node.js,
# VS Code) are still installed without a question when missing, and an
# already-installed tool is used as-is — this script never upgrades
# third-party software.
#
# ``_KISS_INTERACTIVE`` is 0 instead when
#
# * ``--non-interactive`` is passed or ``KISS_NONINTERACTIVE`` is set —
#   what the automated callers do (the VS Code Update button, the kiss-web
#   daemon's update endpoint, the Docker entrypoint); or
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
    # in the original VS Code terminal.
    open(STDIN, "<", "/dev/null") or die "kiss-install: reopen stdin: $!\n";
    exec { "bash" } "bash", $script, @ARGV
        or die "kiss-install: exec bash failed: $!\n";
}

# Parent: ignore every terminal-driven signal so that even if the
# original VS Code PTY injects ``\x03`` (SIGINT) or closes (SIGHUP), or
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
# any always-zero command) propagates a non-zero exit from its body
# (e.g. a failed `npm run package`) instead of returning `tee`'s
# always-zero status.  Without it, a broken VSIX build was silently
# masked and the container ended up shipping the stale committed VSIX.
set -eo pipefail

# ---------------------------------------------------------------------------
# Cross-process update lock — the same lock as scripts/install.sh.
#
# Two installers on one checkout (the kiss-web daemon's update endpoint runs
# this script directly with --non-interactive, the VS Code Update button in
# another window runs it too) race each other's git reset, npm build and
# extension install.  scripts/install.sh already holds the lock for its
# whole lifetime and exports KISS_UPDATE_LOCK_HELD=1 before handing over to
# this script; callers that run this script directly get the same
# protection here.
#
# The lock is a kernel advisory lock (flock(2)) on $HOME/.kiss/.update.lock
# -- $HOME, not $KISS_HOME, because the resources it protects (this
# checkout under ~/.kiss/kiss_ai, the global extension install) follow
# $HOME.  Bash keeps the file open on fd 9 for its whole lifetime; perl
# (already required by the re-exec above; flock(1) is not on macOS) locks
# that very open file description, so the lock persists after perl exits
# and the kernel drops it when this process dies, however it dies -- no
# EXIT trap, no stale lock to break.  The pid in the file only feeds the
# refusal message.  fd 9 must not leak into long-lived children (VS Code,
# anything a build step leaves behind): every launch line below closes it
# with ``9>&-``.
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
        echo "another KISS update is already running (pid ${holder:-unknown}); exiting." >&2
        exit 1
    fi
    echo "$$" > "$_kiss_lock_file"
    export KISS_UPDATE_LOCK_HELD=1
}

if [ -z "${KISS_UPDATE_LOCK_HELD:-}" ]; then
    acquire_update_lock
fi
# The marker's only consumer is the lock decision above: this script hands
# over to no further installer, so drop it now.  Left exported it would
# leak into every long-lived child (the launched VS Code, the daemon the
# extension restarts), whose OWN later update runs would then skip
# acquire_update_lock entirely — with the original lock long gone.
unset KISS_UPDATE_LOCK_HELD
# END: kiss-update-lock

# Capture the user's working directory *before* any `cd` so that VS Code can
# later be launched with this directory as the workspace root.  The agents
# spawned inside VS Code default their PWD to the workspace root (see
# ``kiss.server.server`` — ``os.getcwd()`` is the fallback when
# ``KISS_WORKDIR`` is unset), so opening the workspace here makes the
# agents' PWD match the user's original shell PWD.
USER_PWD="$PWD"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

BIN_DIR="$HOME/.local/bin"
LOG_DIR="$HOME/.kiss"
LOG_FILE="$LOG_DIR/install.log"
# Name of the state directory under $HOME the build being installed uses
# when $KISS_HOME is unset: ``home_dir`` of the brand overlay
# (.brand/brand.json, see apply_brand_overlay below) when there is one,
# else of the checkout's own media/brand.json -- ``.kiss`` for stock KISS
# Sorcar.  The same key drives kiss.core.config.kiss_home() and the
# extension's kissHomeDir() (src/kissHome.js), so the installer writes its
# marker, progress file and MODEL_INFO.json where the installed product
# reads them.  Anything but a single path component falls back to .kiss.
brand_field() {
    # Print string field $1 of the checkout's brand.json (the .brand/
    # overlay first), or nothing when neither file defines it.
    local file
    for file in "$PROJECT_DIR/.brand/brand.json" \
                "$PROJECT_DIR/src/kiss/agents/vscode/media/brand.json"; do
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
# The extension's state directory (kissHomeDir() in userAssets.ts honours
# $KISS_HOME).  The update marker and the progress file below must land
# here, not in a hard-coded $HOME/.kiss, or a custom-KISS_HOME install
# never sees them.
KISS_HOME_DIR="${KISS_HOME:-$HOME/$BRAND_HOME_DIR_NAME}"
# Current install step, mirrored by the VS Code extension as a live,
# non-blocking progress notification (src/installProgress.ts).  Line 1 is
# this script's pid (the toast is dropped when that process is gone, so a
# killed install cannot leave a spinner behind), line 2 the step text.
# The EXIT trap below removes the file when the install ends, which closes
# the notification.
PROGRESS_FILE="$KISS_HOME_DIR/.install-progress"
# Node.js release installed when the machine has none.  Keep this at the
# newest release of the 22.x line (https://nodejs.org/dist/latest-v22.x/):
# releases before v22.23.2 carry the HIGH-severity CVEs fixed in the
# July 2026 security release.
NODE_VERSION="v22.23.3"

mkdir -p "$BIN_DIR" "$LOG_DIR" "$KISS_HOME_DIR"
export PATH="$BIN_DIR:$PATH"

# ---------------------------------------------------------------------------
# Signal handling
#
# A previous regression looked like::
#
#     >>> [4/5] Building VS Code extension...
#     npm warn deprecated prebuild-install@7.1.3: No longer maintained. ...
#     ^C
#
# i.e. the install aborted right after npm ci's first deprecation warning.
# `npm ci` can sit silent for tens of seconds between log lines while it
# fetches/extracts tarballs — long enough for the user (or a stray signal
# from a backgrounded shell / sleeping laptop / closed terminal tab) to
# kill the script just as it was about to make progress.
#
# Trap SIGINT/SIGTERM at the bash level so a single stray signal prints a
# diagnostic instead of silently terminating, and so the user can see how
# far they got.  A *second* signal within 3 s is honored as a real abort.
#
# CRITICAL: install.sh ignoring SIGINT in its own trap is not enough.  The
# signal is delivered to the entire foreground process group, so any
# wrapped child (``npm ci``, ``bash copy-kiss.sh``, ``git ls-files``…)
# that does NOT trap SIGINT itself dies immediately — which made the
# subsequent ``wait`` in ``run_with_heartbeat`` return non-zero and
# triggered ``set -e``, aborting the install at e.g.
# "Copying source files..." even though install.sh's own trap had run.
# The fix below: ``run_with_heartbeat`` spawns the wrapped command inside
# a subshell that sets ``trap '' INT TERM`` and then ``exec``s the binary.
# POSIX guarantees that a signal *ignored* at exec time stays ignored in
# the new process, so npm and its descendants survive a single stray
# signal too.  A confirmed double-Ctrl+C in ``handle_interrupt`` kills the
# tracked child explicitly to give the user a real escape hatch.
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
        echo "   Re-run 'bash $0' to resume; the build cache is preserved."
        exit 130
    fi
    LAST_SIGNAL_TS=$now
    echo ""
    echo "   ⚠ Interrupt received but ignored — long npm/git steps can sit"
    echo "      silent for 30-60 s while they download or extract.  Press"
    echo "      Ctrl+C again within 3 s to really abort."
    if [ -n "$CONFIRM_PENDING" ]; then
        printf '   %s [Y/n] ' "$CONFIRM_PENDING"
    fi
}

# Re-route stdout/stderr to the log file when the controlling terminal
# closes (SIGHUP).  This matters when the VS Code "Update" button runs
# ``install.sh`` in an integrated terminal: VS Code disposes that
# terminal when the extension is deactivated, which is exactly what
# ``code --install-extension --force`` triggers inside step [5/5] —
# VS Code's extension manager detects the on-disk update, deactivates
# the running extension, and the documented behavior is to "dispose the
# terminal and exit the underlying process".  Terminal disposal first
# writes ``\x03`` (Ctrl+C) to the PTY (caught by ``handle_interrupt``
# above) and then closes the PTY (SIGHUP).  Without this trap the SIGHUP
# kills bash mid-step, leaving the ``.extension-updated`` marker
# unwritten — exactly the symptom users see: an unexplained ``^C`` in
# step [5/5] with the install aborted before the marker write.
# ``2>/dev/null`` swallows EBADF/ENXIO from the closed PTY; ``|| true``
# keeps ``set -e`` from killing the script if the re-route itself fails
# (the script then continues writing into the dead PTY, which is no
# worse than the pre-fix behavior).
handle_hup() {
    exec >>"$LOG_FILE" 2>&1 || true
    echo ""
    echo "   ⚠ Controlling terminal closed (SIGHUP) — continuing with"
    echo "      output redirected to $LOG_FILE only."
}
trap handle_interrupt INT TERM
trap handle_hup HUP

# Print a step banner and publish the step to PROGRESS_FILE (see there).
# Every ">>> ..." banner of the install goes through here so the VS Code
# notification always names the step the terminal is on.  Publishing is
# best-effort: an unwritable $KISS_HOME must not fail the install.
report_step() {
    echo ">>> $1"
    # Write, then rename: a poll must never read a truncated file, which
    # would look like the install had ended.
    { printf '%s\n%s\n' "$$" "$1" > "$PROGRESS_FILE.tmp" \
        && mv -f "$PROGRESS_FILE.tmp" "$PROGRESS_FILE"; } 2>/dev/null || true
}
# KISS_WEB_RESTART_LOCK_HELD names the kiss-web restart lock this script
# holds, if any (acquire_kiss_web_restart_lock below); released here so an
# aborted run does not leave the extension locked out of restarting.
KISS_WEB_RESTART_LOCK_HELD=""
trap 'rm -f "$PROGRESS_FILE" "$PROGRESS_FILE.tmp"; [ -z "$KISS_WEB_RESTART_LOCK_HELD" ] || release_kiss_web_restart_lock "$KISS_WEB_RESTART_LOCK_HELD"' EXIT

# Run "$@" while printing a heartbeat every HEARTBEAT_INTERVAL seconds so
# the user can tell the install is still working.  Without this the npm ci
# step can sit silent for ~1 min and look hung.  Exit code is forwarded
# from the wrapped command.
HEARTBEAT_INTERVAL="${KISS_HEARTBEAT_INTERVAL:-15}"
run_with_heartbeat() {
    local label="$1"
    shift
    local start
    start=$(date +%s)
    # Run the command inside a subshell that ignores SIGINT/SIGTERM, then
    # ``exec`` the real binary.  POSIX says SIG_IGN survives exec, so npm
    # and every descendant inherit "ignore" for INT/TERM — a stray signal
    # delivered to install.sh's terminal process group can no longer kill
    # them, which was the actual root cause of the
    # "Copying source files..." abort.  The install.sh-level trap above
    # remains the only way to actually stop the build (double-Ctrl+C).
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
    # the caller instead of aborting the whole script — callers (e.g. the
    # npm ci retry loop) need to inspect the exit code.  ``wait`` itself
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
    *)  echo "ERROR: Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64|aarch64|arm64) ;;
    *)  echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

if ! command -v curl &>/dev/null; then
    echo "ERROR: curl is required but not found. Please install curl first."
    exit 1
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

ensure_xcode_clt() {
    [ "$OS" = "Darwin" ] || return 0

    if xcode-select -p &>/dev/null && [ -e "$(xcode-select -p)/usr/bin/git" ]; then
        echo "   Xcode Command Line Tools already installed at $(xcode-select -p)"
        return 0
    fi

    echo "   Xcode Command Line Tools not found — attempting non-interactive install..."

    local SENTINEL=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
    sudo touch "$SENTINEL" 2>/dev/null || true
    local PROD
    # `|| true` keeps a failing `softwareupdate` (no network, managed Macs)
    # from killing the script via `set -eo pipefail`.
    PROD="$(softwareupdate -l 2>/dev/null \
        | awk '/^[[:space:]]*\*.*Command Line Tools/ {
                 sub(/^[[:space:]]*\*[[:space:]]*(Label:[[:space:]]*)?/, "");
                 print
             }' \
        | tail -n1 || true)"
    if [ -n "$PROD" ]; then
        echo "   Installing: $PROD"
        sudo softwareupdate -i "$PROD" --verbose 2>&1 || true
    else
        echo "   No Command Line Tools package found in softwareupdate catalog."
    fi
    sudo rm -f "$SENTINEL" 2>/dev/null || true

    if xcode-select -p &>/dev/null && [ -e "$(xcode-select -p)/usr/bin/git" ]; then
        echo "   Xcode Command Line Tools installed at $(xcode-select -p)"
        return 0
    fi

    echo "   Non-interactive install did not complete. Triggering GUI installer..."
    xcode-select --install 2>&1 || true

    if xcode-select -p &>/dev/null && [ -e "$(xcode-select -p)/usr/bin/git" ]; then
        echo "   Xcode Command Line Tools installed at $(xcode-select -p)"
    else
        # The GUI install can take many minutes and the non-interactive
        # runs (detached, see the kiss-new-session-reexec block above)
        # cannot wait for keyboard input, so exit-and-rerun is the one
        # behaviour that works for every launch path; it matches the
        # ``install_git`` fallback.
        echo ""
        echo "   A dialog has appeared to install the Xcode Command Line Tools."
        echo "   Complete the installation in that dialog, then re-run this script."
        exit 1
    fi
}

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
        echo "   WARNING: Homebrew install did not complete; continuing without it."
    fi
}

install_git() {
    case "$OS" in
        Darwin)
            if command -v brew &>/dev/null; then
                echo "   Installing git via Homebrew..."
                brew install git
            else
                echo "   Triggering Xcode Command Line Tools (provides git)..."
                xcode-select --install 2>&1 || true
                echo "   NOTE: Complete the Xcode CLT dialog, then re-run this script."
                exit 1
            fi
            ;;
        Linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -y && sudo apt-get install -y git
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y git
            elif command -v yum &>/dev/null; then
                sudo yum install -y git
            elif command -v pacman &>/dev/null; then
                sudo pacman -S --noconfirm git
            elif command -v apk &>/dev/null; then
                sudo apk add git
            else
                echo "   ERROR: No supported package manager found. Install git from https://git-scm.com"
                exit 1
            fi
            ;;
    esac
}

install_node() {
    echo "   Downloading Node.js $NODE_VERSION ..."
    local OS_NODE ARCH_NODE
    OS_NODE="$(echo "$OS" | tr '[:upper:]' '[:lower:]')"
    case "$ARCH" in
        x86_64)         ARCH_NODE="x64" ;;
        aarch64|arm64)  ARCH_NODE="arm64" ;;
    esac
    local URL="https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-${OS_NODE}-${ARCH_NODE}.tar.gz"
    mkdir -p "$HOME/.local"
    if curl -fsSL "$URL" | tar xz -C "$HOME/.local" --strip-components=1; then
        echo "   Node.js $NODE_VERSION installed to ~/.local/"
    else
        echo "   ERROR: Failed to download Node.js from $URL"
        return 1
    fi
}

install_code_cli() {
    case "$OS" in
        Darwin)
            local VSCODE_APP="/Applications/Visual Studio Code.app"
            if [ ! -d "$VSCODE_APP" ]; then
                echo "   Downloading VS Code for macOS..."
                local ARCH_VS
                case "$ARCH" in
                    aarch64|arm64) ARCH_VS="darwin-arm64" ;;
                    x86_64)        ARCH_VS="darwin" ;;
                esac
                local TMP_ZIP
                TMP_ZIP="$(mktemp /tmp/vscode-XXXXXX.zip)"
                if curl -fsSL "https://update.code.visualstudio.com/latest/${ARCH_VS}/stable" -o "$TMP_ZIP"; then
                    unzip -q "$TMP_ZIP" -d /Applications/
                    rm -f "$TMP_ZIP"
                    echo "   VS Code installed to /Applications/"
                else
                    rm -f "$TMP_ZIP"
                    echo "   ERROR: Failed to download VS Code"
                    return 1
                fi
            fi
            local CODE_BIN="$VSCODE_APP/Contents/Resources/app/bin/code"
            if [ -x "$CODE_BIN" ]; then
                ln -sf "$CODE_BIN" "$BIN_DIR/code"
                echo "   Linked VS Code CLI to $BIN_DIR/code"
            fi
            ;;
        Linux)
            if command -v snap &>/dev/null; then
                sudo snap install --classic code 2>&1 || true
            elif command -v apt-get &>/dev/null; then
                curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
                    | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg 2>&1
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/code stable main" \
                    | sudo tee /etc/apt/sources.list.d/vscode.list >/dev/null 2>&1
                sudo apt-get update -y && sudo apt-get install -y code 2>&1
            elif command -v dnf &>/dev/null; then
                sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc 2>&1
                sudo tee /etc/yum.repos.d/vscode.repo >/dev/null <<'REPO'
[code]
name=Visual Studio Code
baseurl=https://packages.microsoft.com/yumrepos/vscode
enabled=1
gpgcheck=1
gpgkey=https://packages.microsoft.com/keys/microsoft.asc
REPO
                sudo dnf install -y code 2>&1
            else
                echo "   Please install VS Code from https://code.visualstudio.com"
                return 1
            fi
            ;;
    esac
}

find_code_cli() {
    CODE_CLI=""
    # Honor an explicit override so callers running inside a specific editor
    # distribution can force its CLI.  The Docker/code-server entrypoint sets
    # KISS_CODE_CLI=code-server so the extension is installed into
    # code-server's extensions directory
    # (~/.local/share/code-server/extensions) — the one the browser IDE
    # actually reads — instead of a separately apt-installed Microsoft VS Code
    # (~/.vscode/extensions), which code-server never loads.
    if [ -n "${KISS_CODE_CLI:-}" ]; then
        local override
        override="$(command -v "$KISS_CODE_CLI" 2>/dev/null || true)"
        if [ -n "$override" ] && [ -x "$override" ]; then
            CODE_CLI="$override"
            return 0
        fi
    fi
    for candidate in \
        "$(command -v code 2>/dev/null || true)" \
        "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" \
        "$BIN_DIR/code" \
        "/usr/local/bin/code" \
        "/usr/bin/code" \
        "/snap/bin/code"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            CODE_CLI="$candidate"
            return 0
        fi
    done
    return 1
}

# Remove VS Code's on-disk caches before the extension is built and
# installed, so the freshly installed build never loads through stale cached
# state (a corrupted ``CachedExtensionVSIXs`` entry or stale ``CachedData``
# V8 snapshots can make ``--install-extension`` appear to succeed while the
# reloaded window keeps running old code).  Only cache directories are
# touched — VS Code recreates each of them on the next launch:
#
#     Cache, CachedData, CachedExtensions, CachedExtensionVSIXs,
#     "Code Cache", GPUCache
#
# swept under every user-data root this installer can target:
#
#     macOS:        ~/Library/Application Support/Code
#     Linux:        ~/.config/Code
#     code-server:  ${XDG_DATA_HOME:-~/.local/share}/code-server
#                   (the Docker entrypoint installs into code-server via
#                   KISS_CODE_CLI — see find_code_cli)
#
# User state next to the caches (``User/``, ``extensions/``,
# ``Local Storage``, ``Session Storage``, ``Workspaces``, ``Backups``) is
# deliberately NOT touched.  Every removal is best-effort (``|| true``): a
# cache directory being rewritten by a running VS Code can make ``rm -rf``
# fail (ENOTEMPTY/EACCES), and a failed cache sweep must never abort an
# otherwise healthy install under ``set -e``.
clear_vscode_cache() {
    local data_dir cache_subdir cleared
    cleared=0
    for data_dir in \
        "$HOME/Library/Application Support/Code" \
        "$HOME/.config/Code" \
        "${XDG_DATA_HOME:-$HOME/.local/share}/code-server"; do
        [ -d "$data_dir" ] || continue
        for cache_subdir in \
            "Cache" \
            "CachedData" \
            "CachedExtensions" \
            "CachedExtensionVSIXs" \
            "Code Cache" \
            "GPUCache"; do
            if [ -e "$data_dir/$cache_subdir" ]; then
                rm -rf "$data_dir/$cache_subdir" 2>/dev/null || true
                cleared=1
                # Report honestly: a cache directory being rewritten by a
                # running VS Code can survive the sweep (ENOTEMPTY/EACCES).
                if [ -e "$data_dir/$cache_subdir" ]; then
                    echo "   WARNING: could not fully clear $data_dir/$cache_subdir (in use?); continuing."
                else
                    echo "   Cleared $data_dir/$cache_subdir"
                fi
            fi
        done
    done
    if [ "$cleared" = 0 ]; then
        echo "   No VS Code caches found to clear."
    fi
}

launch_vscode() {
    # ``$USER_PWD`` is captured at the top of this script before any ``cd``.
    # Passing it to VS Code makes it the workspace root so that agents
    # spawned inside the editor inherit it as their PWD.
    case "$OS" in
        Darwin)
            if open -a "Visual Studio Code" "$USER_PWD" >/dev/null 2>&1 9>&-; then
                echo "Launched VS Code via 'open -a' with workspace $USER_PWD."
                return 0
            fi
            if [ -d "/Applications/Visual Studio Code.app" ] && open -a "/Applications/Visual Studio Code.app" "$USER_PWD" >/dev/null 2>&1 9>&-; then
                echo "Launched VS Code from /Applications with workspace $USER_PWD."
                return 0
            fi
            ;;
        Linux)
            for candidate in \
                "$(command -v code 2>/dev/null || true)" \
                "$BIN_DIR/code" \
                "/usr/local/bin/code" \
                "/usr/bin/code" \
                "/snap/bin/code" \
                "/usr/share/code/code"; do
                if [ -n "$candidate" ] && [ -x "$candidate" ]; then
                    (nohup "$candidate" "$USER_PWD" >/dev/null 2>&1 9>&- &)
                    echo "Launched VS Code from $candidate with workspace $USER_PWD."
                    return 0
                fi
            done
            ;;
    esac

    if find_code_cli && [ -n "$CODE_CLI" ]; then
        (nohup "$CODE_CLI" "$USER_PWD" >/dev/null 2>&1 9>&- &)
        echo "Launched VS Code from $CODE_CLI with workspace $USER_PWD."
        return 0
    fi

    echo "Could not launch VS Code automatically. Open VS Code manually to finish setup."
    return 1
}

# Return 0 (true) when a VS Code window is already running.  Used to skip the
# explicit ``launch_vscode`` at the end of the install: when the editor is
# already open, the extension's own file watchers detect the reinstall (the
# overwritten ``out/extension.js`` and the freshly written
# ``~/.kiss/.extension-updated`` marker) and fire
# ``workbench.action.reloadWindow``.  That reload already brings the user back
# into a working window, so a second ``open``/launch here would only spawn a
# redundant duplicate window.
vscode_is_running() {
    case "$OS" in
        Darwin)
            # AppleScript reliably reports whether the app is running.
            if command -v osascript &>/dev/null; then
                local running
                running="$(osascript -e 'application "Visual Studio Code" is running' 2>/dev/null)"
                [ "$running" = "true" ] && return 0
            fi
            # Fallback: match the app's main process by its bundle path.
            command -v pgrep &>/dev/null && pgrep -f "Visual Studio Code.app" &>/dev/null && return 0
            ;;
        Linux)
            command -v pgrep &>/dev/null || return 1
            # The Electron main process is named "code"; also match common
            # absolute-path invocations in case the name is shortened.
            pgrep -x code &>/dev/null && return 0
            pgrep -f "/usr/share/code/code" &>/dev/null && return 0
            pgrep -f "/snap/code/" &>/dev/null && return 0
            ;;
    esac
    return 1
}

# ---------------------------------------------------------------------------
# CLI launcher helpers
# ---------------------------------------------------------------------------

# Install a launcher for a repo-root script (e.g. ./rsorcar, ./sorcar-docker)
# into $BIN_DIR so it can be run from anywhere, mirroring how the ``sorcar``
# CLI itself is installed into ~/.local/bin (see ``installCliScript`` in
# DependencyInstaller.ts).
#
# The launcher is a thin wrapper that ``exec``s the real script inside
# $PROJECT_DIR — deliberately NOT a symlink and NOT a copy.  Both scripts
# locate their own directory (``dirname "$0"`` / ``BASH_SOURCE``) and treat
# it as the KISS Sorcar checkout: ``rsorcar`` deploys the folder in which the
# script is actually present, and ``sorcar-docker`` builds the Docker image
# from that folder.  A symlink or copy in ~/.local/bin would make them
# resolve ~/.local/bin instead of the checkout and fail.
install_repo_script_launcher() {
    local name="$1"
    local target="$PROJECT_DIR/$name"
    if [ ! -f "$target" ]; then
        echo "   WARNING: $target not found — skipping $name launcher."
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

# Keep the freshly built ``kiss-sorcar.vsix`` from dirtying git, in both
# kinds of checkout this script runs in ($1 = repo root):
#
# * Public ``kiss_ai`` clones: every release commit deliberately SHIPS the
#   prebuilt VSIX as a tracked file (``tree_with_vsix`` in scripts/release.sh)
#   so docker/code-server installs work without npm.  The build step above
#   just overwrote that tracked file, so without countermeasures ``git
#   status`` reports it modified and the auto-commit / worktree flows would
#   commit the multi-MB binary on every task.  Remedy: put HEAD's copy back
#   into BOTH the index and the working tree (``git checkout HEAD --``).
#   By the time this guard runs the freshly built VSIX has already been
#   installed into VS Code, so replacing it on disk with the release copy
#   loses nothing — manual ``code --install-extension`` retries and
#   docker-startup.sh then use the release-shipped bytes, and the repo is
#   left byte-for-byte clean so the Update button's later ``git stash`` /
#   ``git reset --hard @{upstream}`` preflight and ``git worktree add``
#   never trip over a dirty or skip-worktree-pinned entry.  A single
#   checkout also self-heals every broken index state this file can get
#   into: a staged modification, a staged ``git rm --cached`` deletion
#   (the old error message told users to run exactly that), and unmerged
#   stages left by a conflicted ``git stash pop``.  Failures here only
#   leave the file dirty for the preflight stash to handle, so they warn
#   instead of aborting an otherwise finished install.
#
# * The development repo, where no commit contains the VSIX (it is matched
#   by ``*.vsix`` in .gitignore): the file being tracked can only mean an
#   accidental ``git add -f``, which the auto-commit flow would turn into a
#   committed binary.  That remains a hard error (exit 1) telling the user
#   how to untrack it.  The locally built VSIX stays untracked on disk for
#   ``code --install-extension`` retries and docker-startup.sh.
guard_vsix_tracking() {
    local project_dir="$1"
    local vsix_rel="src/kiss/agents/vscode/kiss-sorcar.vsix"
    if git -C "$project_dir" rev-parse --verify --quiet "HEAD:$vsix_rel" >/dev/null; then
        # Public kiss_ai clone: the release ships the VSIX tracked, by design.
        # Clear any stale skip-worktree pin first: a pinned entry whose blob
        # changes upstream makes ``git reset --hard @{upstream}`` fail with
        # "Entry ... not uptodate", bricking every later update.
        git -C "$project_dir" update-index --no-skip-worktree -- "$vsix_rel" 2>/dev/null || true
        # Rewrite the index entry to HEAD's mode+blob via --index-info: the
        # mode-0 line drops every stage of the path (a plain entry, a staged
        # modification or deletion, and unmerged stages 1-3 alike), then the
        # stage-0 line re-registers HEAD's blob.  ``git checkout HEAD --`` is
        # NOT equivalent: it silently skips an unmerged entry whose stage-2
        # blob already matches HEAD.
        local mode blob zero
        read -r mode _ blob _ < <(git -C "$project_dir" ls-tree HEAD -- "$vsix_rel") || true
        zero="${blob//?/0}"
        printf '0 %s\t%s\n%s %s 0\t%s\n' "$zero" "$vsix_rel" "$mode" "$blob" "$vsix_rel" |
            git -C "$project_dir" update-index --index-info 2>/dev/null || true
        # Write the release blob back to the working tree over the local
        # rebuild.
        git -C "$project_dir" checkout-index -q -f -- "$vsix_rel" 2>/dev/null || true
        # Verify the result instead of trusting the exit codes above: only a
        # path that is byte-for-byte clean keeps later ``git stash`` /
        # ``git reset --hard`` preflights and worktree flows working.  The
        # ``ls-files -v`` check must report a plain tracked entry ("H", not a
        # skip-worktree "S"): a surviving pin hides a dirty file from ``git
        # status`` while still bricking the next ``reset --hard``, and it
        # also makes this verification fail honestly when the status command
        # itself errors out (an empty capture alone would look clean).
        local dirty
        if dirty=$(git -C "$project_dir" status --porcelain -- "$vsix_rel" 2>/dev/null) &&
            [ -z "$dirty" ] &&
            [ "$(git -C "$project_dir" ls-files -v -- "$vsix_rel" 2>/dev/null)" = "H $vsix_rel" ]; then
            echo "   Restored release-shipped $vsix_rel in git index and working tree"
            echo "   (the freshly built VSIX was already installed into VS Code)."
        else
            echo "   WARNING: could not restore $vsix_rel from HEAD; the locally" >&2
            echo "   rebuilt VSIX may show up as a git modification." >&2
        fi
        return 0
    fi
    if ! git -C "$project_dir" ls-files --error-unmatch "$vsix_rel" &>/dev/null; then
        return 0  # untracked (development repo, healthy state) — nothing to do
    fi
    echo "   ERROR: $vsix_rel is tracked by git but must remain ignored." >&2
    echo "   Run: git -C \"$project_dir\" rm --cached \"$vsix_rel\"" >&2
    echo "   and ensure \`*.vsix\` stays in .gitignore." >&2
    return 1
}

# ---------------------------------------------------------------------------
# Brand overlay: ``$PROJECT_DIR/.brand/``
# ---------------------------------------------------------------------------
# A white-label distribution re-brands KISS Sorcar by replacing the data
# files under src/kiss/agents/vscode/media/ (brand.json, brand.css,
# kiss-icon.svg, kiss-icon.png, thumbnail.jpeg, welcome-logo.png; see kiss.core.brand and
# scripts/apply-brand.js).  Editing those tracked files in place would
# make the Update button's pre-flight (``git stash`` / ``git reset --hard
# @{upstream}`` / ``git stash pop``) conflict on every release, because
# the branded display strings in package.json sit right next to the
# version line every release bumps.  The customization therefore lives in
# the git-ignored ``.brand/`` directory next to the checkout:
#
#   * apply_brand_overlay saves the checkout's own copies of package.json
#     and of every media file ``.brand/`` overrides into a temporary
#     snapshot, then copies the overlay over media/ right before the
#     extension build (copy-kiss.sh then rewrites package.json's display
#     strings from media/brand.json and bundles the branded media into
#     kiss_project);
#   * restore_brand_overlay copies the snapshot back once the VSIX has
#     been packaged, or the build failed.  package.json gets its pre-build
#     content plus the version copy-kiss.sh synced, which is the only
#     change a stock build makes to the manifest.
#
# The snapshot (not ``git checkout``) is what goes back, so unrelated local
# edits survive and a plain directory copy works too.  The checkout ends
# up exactly as a stock build leaves it, the installed extension carries
# the brand, and every later update rebuilds with the same overlay.
# Without a ``.brand/`` directory both functions do nothing (stock KISS
# Sorcar), so a development checkout is never re-branded by accident.
BRAND_OVERLAY_FILES=(brand.json brand.css kiss-icon.svg kiss-icon.png thumbnail.jpeg welcome-logo.png welcome-logo-dark.png)
BRAND_MEDIA_REL="src/kiss/agents/vscode/media"
BRAND_MANIFEST_REL="src/kiss/agents/vscode/package.json"
# Snapshot directory while the overlay is applied; empty otherwise.
BRAND_OVERLAY_BACKUP=""

apply_brand_overlay() {
    local project_dir="$1"
    local overlay="$project_dir/.brand"
    local f
    [ -d "$overlay" ] || return 0
    echo "   Applying brand overlay from $overlay..."
    BRAND_OVERLAY_BACKUP="$(mktemp -d "${TMPDIR:-/tmp}/kiss-brand-backup.XXXXXX")" || return 1
    cp -f "$project_dir/$BRAND_MANIFEST_REL" "$BRAND_OVERLAY_BACKUP/package.json" || return 1
    for f in "${BRAND_OVERLAY_FILES[@]}"; do
        [ -f "$overlay/$f" ] || continue
        cp -f "$project_dir/$BRAND_MEDIA_REL/$f" "$BRAND_OVERLAY_BACKUP/$f" || return 1
        cp -f "$overlay/$f" "$project_dir/$BRAND_MEDIA_REL/$f" || return 1
        echo "      $BRAND_MEDIA_REL/$f"
    done
}

restore_brand_overlay() {
    local project_dir="$1"
    local f
    [ -n "$BRAND_OVERLAY_BACKUP" ] || return 0
    for f in "${BRAND_OVERLAY_FILES[@]}"; do
        [ -f "$BRAND_OVERLAY_BACKUP/$f" ] || continue
        cp -f "$BRAND_OVERLAY_BACKUP/$f" "$project_dir/$BRAND_MEDIA_REL/$f"
    done
    # No manifest snapshot means apply failed on its very first copy and
    # nothing was changed yet.
    if [ -f "$BRAND_OVERLAY_BACKUP/package.json" ] &&
        ! python3 - "$BRAND_OVERLAY_BACKUP/package.json" "$project_dir/$BRAND_MANIFEST_REL" <<'EOF'
import json, pathlib, sys
saved, manifest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
before = json.loads(saved.read_text())
version = json.loads(manifest.read_text()).get("version")
if before.get("version") == version:
    manifest.write_bytes(saved.read_bytes())
else:
    before["version"] = version
    manifest.write_text(json.dumps(before, indent=2) + "\n")
EOF
    then
        cp -f "$BRAND_OVERLAY_BACKUP/package.json" "$project_dir/$BRAND_MANIFEST_REL"
    fi
    rm -rf "$BRAND_OVERLAY_BACKUP"
    BRAND_OVERLAY_BACKUP=""
    echo "   Restored the checkout's own brand files (the built VSIX keeps the overlay)."
}

# ---------------------------------------------------------------------------
# Runtime setup without VS Code
# ---------------------------------------------------------------------------
# The VS Code extension's DependencyInstaller performs the same steps on
# its first activation and stays the owner of the interactive ones (API
# keys, the remote-access password).  Doing the non-interactive part here
# means an install with no VS Code window attached (a headless server, a
# Docker build, a white-label post-install hook that needs the runtime)
# still ends with the bundled Python environment, the ``sorcar`` CLI and
# a running kiss-web daemon; the extension's first activation then finds
# a matching fingerprint and takes its fast path.  Every step is
# best-effort: a failure is reported and the install completes, since
# the extension retries on activation.
#
# The daemon is restarted under the extension's own rules: never while
# the running daemon reports tasks in flight, and not at all when it is
# healthy and already runs this exact runtime (that keeps its public
# tunnel URL).
# BEGIN: kiss-runtime-setup  (tests extract this block verbatim)
# kiss-web always listens on 8787; the override exists for the tests.
KISS_WEB_PORT="${KISS_WEB_PORT:-8787}"

installed_kiss_projects() {
    # Print the kiss_project directory of every installed copy of
    # extension version $1, the copy the ``code`` CLI installs into first
    # (desktop / code-server) and the VS Code Server copy last.
    local version="$1" root dir
    [ -n "$version" ] || return 0
    for root in "$HOME/.vscode/extensions" \
                "$HOME/.vscode-insiders/extensions" \
                "$HOME/.vscode-oss/extensions" \
                "$HOME/.local/share/code-server/extensions" \
                "$HOME/.vscode-server/extensions" \
                "$HOME/.vscode-server-insiders/extensions"; do
        dir="$root/ksenxx.kiss-sorcar-$version/kiss_project"
        [ -f "$dir/pyproject.toml" ] && echo "$dir"
    done
    return 0
}

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
    # shell's rc file, once; same file choice and line as the extension.
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
        echo "   WARNING: could not add ~/.local/bin to PATH in $rc"
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
        curl -fsSL "$url" | tar xzf - -C "$bin_dir" || echo "   WARNING: cloudflared download failed; the web app will have no public URL"
    else
        url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$arch"
        curl -fsSL -o "$bin_dir/cloudflared" "$url" \
            || { rm -f "$bin_dir/cloudflared"; echo "   WARNING: cloudflared download failed; the web app will have no public URL"; }
    fi
    [ -f "$bin_dir/cloudflared" ] && chmod 755 "$bin_dir/cloudflared"
    return 0
}

setup_bundled_runtime() {
    # Build the Python environment of every installed copy of extension
    # version $1, then the ``sorcar`` CLI (bound to the first copy),
    # Playwright's Chromium and cloudflared.
    # Leaves the first copy in KISS_PRIMARY_PROJECT for the daemon start.
    local version="$1" uv project primary="" bin_dir="$HOME/.local/bin"
    local projects
    KISS_PRIMARY_PROJECT=""
    projects="$(installed_kiss_projects "$version")"
    if [ -z "$projects" ]; then
        echo "   No installed copy of the extension found; the extension sets up its runtime on first activation."
        return 0
    fi
    if ! uv="$(ensure_uv)"; then
        echo "   WARNING: uv is not available; the extension installs it on first activation."
        return 0
    fi
    while IFS= read -r project; do
        echo "   Installing Python dependencies in $project..."
        if (cd "$project" && "$uv" sync); then
            [ -n "$primary" ] || primary="$project"
        else
            echo "   WARNING: uv sync failed in $project; the extension retries on activation."
        fi
    done <<< "$projects"
    [ -n "$primary" ] || return 0
    KISS_PRIMARY_PROJECT="$primary"

    # The CLI wrapper, byte for byte what the extension writes.
    if mkdir -p "$bin_dir" && printf '%s\n' '#!/bin/bash' \
            "# Installed by ${BRAND_PRODUCT_NAME:-KISS Sorcar} VS Code extension" \
            'export KISS_WORKDIR="$PWD"' \
            "exec \"$uv\" run --directory \"$primary\" sorcar \"\$@\"" > "$bin_dir/sorcar.tmp" \
            && chmod 755 "$bin_dir/sorcar.tmp" && mv -f "$bin_dir/sorcar.tmp" "$bin_dir/sorcar"; then
        echo "   Installed the sorcar CLI at $bin_dir/sorcar"
    else
        echo "   WARNING: could not install the sorcar CLI at $bin_dir/sorcar; the extension retries on activation."
    fi
    add_local_bin_to_shell_rc

    echo "   Installing Playwright Chromium..."
    if ! (cd "$primary" && "$uv" run python -m playwright install chromium); then
        echo "   WARNING: Playwright Chromium install failed; the extension retries on activation."
    elif [ "$(uname -s)" = "Linux" ]; then
        # Chromium's system libraries need root (install-deps calls sudo
        # itself); only try when that cannot stall on a password prompt.
        if [ "$(id -u)" = 0 ] || sudo -n true 2>/dev/null; then
            (cd "$primary" && "$uv" run python -m playwright install-deps chromium) \
                || echo "   WARNING: Chromium system libraries not installed (playwright install-deps failed)."
        else
            echo "   Chromium system libraries: run 'sudo $primary/.venv/bin/python -m playwright install-deps chromium' if the browser fails to start."
        fi
    fi
    install_cloudflared
}

kiss_web_port_open() {
    (exec 3<>"/dev/tcp/127.0.0.1/$KISS_WEB_PORT") 2>/dev/null
}

kiss_web_fingerprint() {
    # Same digest as computeKissWebFingerprint in DependencyInstaller.ts:
    # the kiss-web launcher bytes, the work dir, and the newest mtime of
    # the bundled sources.  Matching it lets the extension skip its own
    # restart on first activation.
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
        elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".py"):
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
    # too old to answer).
    local python="$1" endpoint
    shift
    for endpoint in "$@"; do
        KISS_SORCAR_LOCAL="$endpoint" KISS_ACTIVE_TASKS_STRICT=1 \
            "$python" "$PROJECT_DIR/scripts/check-kiss-web-active-tasks.py" || return 1
    done
}

previous_kiss_web_homes() {
    # State directories of the daemon the existing service runs: the
    # KISS_HOME its unit/plist sets, and its runtime's own default home.
    # A busy daemon left by an earlier install under another home must
    # still be asked before it is replaced.
    local unit="$HOME/.config/systemd/user/kiss-web.service"
    local plist="$HOME/Library/LaunchAgents/com.kiss.web-server.plist" bin python
    [ -f "$unit" ] && sed -n 's/^Environment=KISS_HOME=//p' "$unit" | unit_unescape
    [ -f "$plist" ] && sed -n '/<key>KISS_HOME<\/key>/{n;s/.*<string>\(.*\)<\/string>.*/\1/p;}' "$plist" | xml_unescape
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
    # Take the extension's restart lock $1 ({"pid":…,"token":…}, created
    # exclusively) so a VS Code window and this script never restart the
    # daemon at the same time; a lock whose owner is dead is broken.
    # Same patience as the extension: a lock without a readable owner is
    # left alone for 2 minutes (the extension creates the file before it
    # writes its identity), a live owner for 10 minutes.
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
    # so an interrupted restart never leaves the extension locked out).
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
    # under $4; same content as the extension writes.  The daemon resolves
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
    # logs under $4; same content as the extension writes.
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

start_kiss_web_daemon() {
    # Register kiss-web from kiss_project $1 as the user's service
    # (launchd on macOS, systemd on Linux, a detached process elsewhere)
    # serving work dir $2, and (re)start it when that is safe.
    local project="$1" workdir="$2" kiss_web="$1/.venv/bin/kiss-web"
    local python="$1/.venv/bin/python" home lock
    if [ ! -x "$kiss_web" ]; then
        echo "   kiss-web not built in $project; the extension starts the daemon on first activation."
        return 0
    fi
    home="$("$python" -c 'from kiss.core.config import kiss_home; print(kiss_home())')" || return 0
    mkdir -p "$home"
    lock="$home/.kiss-web.restart.lock"
    if ! acquire_kiss_web_restart_lock "$lock"; then
        echo "   A VS Code window is restarting kiss-web; leaving it to that."
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
        echo "   kiss-web has tasks in flight; restart deferred (the extension restarts it, or re-run this script once they finish)."
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
        launchctl bootout "gui/$uid/com.kiss.web-server" 2>/dev/null || true
        stop_stray_kiss_web
        launchctl bootstrap "gui/$uid" "$plist" 2>/dev/null || true
        launchctl kickstart -k "gui/$uid/com.kiss.web-server" || echo "   WARNING: launchctl kickstart failed"
    elif [ "$(uname -s)" = "Linux" ] && systemctl --user show-environment &>/dev/null; then
        write_kiss_web_unit "$HOME/.config/systemd/user/kiss-web.service" "$kiss_web" "$workdir" "$home"
        echo "   Restarting the kiss-web systemd user service..."
        systemctl --user daemon-reload || true
        systemctl --user enable kiss-web.service &>/dev/null || true
        systemctl --user stop kiss-web.service 2>/dev/null || true
        stop_stray_kiss_web
        systemctl --user start kiss-web.service || echo "   WARNING: systemctl start failed"
        loginctl enable-linger "$(id -un)" 2>/dev/null || true
    else
        echo "   Starting kiss-web as a detached background process..."
        stop_stray_kiss_web
        # ``exec`` so no shell lingers holding this script's output open;
        # ``9>&-`` keeps the update lock out of the daemon.
        (cd "$workdir" && exec nohup "$kiss_web" >> "$home/kiss-web-stdout.log" 2>> "$home/kiss-web-stderr.log" < /dev/null 9>&-) &
    fi
    while [ "$waited" -lt "${KISS_WEB_START_TIMEOUT:-90}" ]; do
        if [ "$endpoint" -nt "$stamp" ] && kiss_web_port_open; then
            echo "$fingerprint" > "$home/.kiss-web.fingerprint"
            rm -f "$stamp" "$home/.kiss-web.restart-pending"
            echo "   kiss-web is up (${waited}s): http://localhost:$KISS_WEB_PORT"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    rm -f "$stamp"
    echo "   WARNING: kiss-web did not come up within ${waited}s; see $home/kiss-web-stderr.log"
}
# END: kiss-runtime-setup

# Tee stdout+stderr to the install log AND the terminal.  We use ``exec``
# process substitution rather than wrapping the install body in
# ``{ ... } 2>&1 | tee "$LOG_FILE"`` because the latter forks a subshell
# for the entire install body, and POSIX bash *resets* trapped signals
# back to their default disposition inside that subshell (see bash(1)
# "TRAPS / Trapped signals that are not being ignored are reset to their
# original values in a subshell or subshell environment when one is
# created").  In other words, the ``trap handle_interrupt INT TERM``
# above had no effect inside the pipeline subshell — a stray ``\x03``
# injected into the PTY by VS Code's terminal-disposal teardown killed
# the subshell instantly, manifesting as an unexplained ``^C`` in step
# [5/5] with the install aborted before the ``.extension-updated``
# marker write.
#
# ``exec > >(tee -a "$LOG_FILE") 2>&1`` keeps the install body running
# in the outer (trap-handled) shell while still streaming output to
# both the user's terminal AND the log file.  ``-a`` appends so a
# previous install's log is preserved when this run is itself a retry
# after an interrupted attempt.
#
# The ``trap '' INT TERM`` INSIDE the process substitution is load-
# bearing too: VS Code's terminal teardown signals the whole foreground
# process GROUP, so the same stray SIGINT that the outer trap absorbs
# also reaches the tee child.  With default disposition tee died, and
# the outer shell's very next write (the trap's own diagnostic!) hit a
# dead pipe — SIGPIPE, script killed with rc=141 and an empty log,
# defeating the trap fix above.  Ignored dispositions survive exec, so
# tee inherits SIG_IGN and keeps draining until bash exits and closes
# the pipe.  ``9>&-`` keeps the update-lock fd out of tee: tee outlives
# this shell by the few milliseconds it takes to see EOF and exit, and
# it closes its stdout first (coreutils ``close_stdout``), so a caller
# that saw our output end would otherwise find the lock still held.
exec > >(trap '' INT TERM; exec tee -a "$LOG_FILE" 9>&-) 2>&1

{
    echo "=== KISS Sorcar Source Install ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Directory: $PROJECT_DIR"
    echo "OS: $OS ($ARCH)"
    if [ "$_KISS_INTERACTIVE" = 1 ]; then
        echo "Mode: interactive (asks before installing Homebrew; pass --non-interactive to skip the question)"
    else
        echo "Mode: non-interactive (every question takes its default answer)"
    fi
    echo ""

    if [ "$OS" = "Darwin" ]; then
        report_step "Checking Xcode Command Line Tools..."
        ensure_xcode_clt
        echo ""

        report_step "Checking Homebrew..."
        ensure_homebrew
        echo ""
    fi

    report_step "[1/5] Checking git..."
    if ! command -v git &>/dev/null; then
        install_git
        hash -r
    fi
    if ! command -v git &>/dev/null; then
        echo "   ERROR: git is still not available after the install attempt."
        exit 1
    fi
    # `|| true`: under `pipefail` a git that prints no parseable version
    # would otherwise abort the script at this assignment.
    INSTALLED_GIT=$(git --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
    echo "   git $INSTALLED_GIT ready"
    echo ""

    report_step "Checking uv..."
    if command -v uv &>/dev/null; then
        INSTALLED_UV=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
        echo "   uv $INSTALLED_UV ready"
    else
        echo "   uv not found — will be installed with the bundled runtime in step [5/5]"
    fi
    echo ""

    report_step "[2/5] Checking Node.js..."
    if ! command -v node &>/dev/null || ! command -v npm &>/dev/null || ! command -v npx &>/dev/null; then
        install_node || true
    fi
    if command -v node &>/dev/null && command -v npm &>/dev/null && command -v npx &>/dev/null; then
        INSTALLED_NODE=$(node --version 2>/dev/null | sed 's/^v//' || true)
        echo "   node v$INSTALLED_NODE ready"
        echo "   npm $(npm --version) ready"
    else
        echo "   ERROR: Node.js, npm, and npx are required to build the extension."
        echo "   Install Node.js from https://nodejs.org and re-run this script."
        exit 1
    fi
    echo ""

    report_step "[3/5] Checking VS Code CLI..."
    if ! find_code_cli; then
        install_code_cli || true
        find_code_cli || true
    fi
    if [ -n "$CODE_CLI" ]; then
        INSTALLED_VSCODE=$("$CODE_CLI" --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1 || true)
        echo "   code CLI ready: $CODE_CLI (v$INSTALLED_VSCODE)"
    else
        echo "   ERROR: VS Code CLI not found — cannot install the extension."
        echo "   Install VS Code from https://code.visualstudio.com and re-run this script."
        exit 1
    fi
    echo ""

    # Clear VS Code's caches BEFORE the extension is built and installed so
    # the update in step [5/5] cannot be served from stale cached state (see
    # clear_vscode_cache above for what is swept and why it is best-effort).
    report_step "Clearing VS Code caches..."
    clear_vscode_cache
    echo ""

    report_step "[4/5] Building VS Code extension..."
    VSCODE_EXT_DIR="$PROJECT_DIR/src/kiss/agents/vscode"
    VSIX="$VSCODE_EXT_DIR/kiss-sorcar.vsix"
    cd "$VSCODE_EXT_DIR"
    # npm ci flags — chosen so a fresh OR repeat run cannot hang:
    #
    # --ignore-scripts: the lockfile's only packages with install scripts are
    #   `keytar` (an *optional*, lazily-imported dep of @vscode/vsce used
    #   solely for publish credentials — never by `vsce package`) and
    #   `@vscode/vsce-sign` (signing only).  keytar's install script runs
    #   `prebuild-install || node-gyp rebuild`, which downloads from the
    #   archived atom/node-keytar GitHub releases (or compiles natively) and
    #   can block forever with no output — hanging the Update button's
    #   install at "[4/5] Building VS Code extension..." right after npm's
    #   deprecation warnings.  Neither script is needed to compile and
    #   package the VSIX.
    #
    # --omit=optional: matches the release scripts.  Skips keytar entirely
    #   (it is an *optional* dep of @vscode/vsce), so even npm's
    #   "deprecated prebuild-install" warning — the last line many users
    #   saw before the script appeared to hang — is gone.
    #
    # --prefer-offline: re-runs reuse the npm cache populated by the
    #   previous attempt instead of re-downloading the whole dependency
    #   tree.  Critical for the Update button: when a user re-runs the
    #   script after an interrupted attempt, the second run is ~10× faster
    #   because every tarball is already in ~/.npm.
    #
    # --no-audit --no-fund skip more network round-trips and noise.
    NPM_CI_FLAGS=(--ignore-scripts --omit=optional --prefer-offline --no-audit --no-fund)
    echo "   Installing extension dependencies (npm ci)..."
    echo "   This typically takes 30–90 s the first time and ~10 s on re-runs."
    # Retry once on transient failure (network blip, mirror flake).  The
    # heartbeat wrapper makes sure the user sees elapsed-time output every
    # ~15 s, so a silent stretch of npm output no longer looks like a hang.
    if ! run_with_heartbeat "npm ci" npm ci "${NPM_CI_FLAGS[@]}"; then
        echo "   npm ci failed — retrying once with a clean node_modules..."
        rm -rf node_modules
        run_with_heartbeat "npm ci (retry)" npm ci "${NPM_CI_FLAGS[@]}"
    fi
    echo "   Compiling extension TypeScript..."
    run_with_heartbeat "tsc" npm run compile
    # The build runs between apply_brand_overlay and restore_brand_overlay,
    # so a failing step is recorded instead of exiting under ``set -e``
    # and the checkout is restored either way.
    build_rc=0
    apply_brand_overlay "$PROJECT_DIR" || build_rc=$?
    if [ "$build_rc" = 0 ]; then
        echo "   Copying bundled KISS runtime..."
        run_with_heartbeat "copy-kiss" npm run copy-kiss || build_rc=$?
    fi
    if [ "$build_rc" = 0 ]; then
        echo "   Packaging VSIX..."
        run_with_heartbeat "vsce package" npm run package || build_rc=$?
    fi
    restore_brand_overlay "$PROJECT_DIR"
    if [ "$build_rc" != 0 ]; then
        echo "   ERROR: extension build failed (exit $build_rc)"
        exit "$build_rc"
    fi
    cd "$PROJECT_DIR"
    if [ ! -f "$VSIX" ]; then
        echo "   ERROR: Failed to build VSIX"
        exit 1
    fi
    echo "   Built $VSIX"
    echo ""

    # The ``sorcar`` CLI itself is installed into ~/.local/bin once the
    # bundled runtime is built (setup_bundled_runtime, step [5/5]).  Install
    # the companion repo-root scripts the same way so they too can be run
    # from anywhere.
    report_step "Installing rsorcar and sorcar-docker launchers..."
    install_repo_script_launcher rsorcar
    install_repo_script_launcher sorcar-docker
    echo ""

    # The banner sits outside the block below because tests extract that
    # block verbatim and run it without report_step.
    report_step "[5/5] Installing VS Code extension..."
    # BEGIN: kiss-step-5-5-terminal-freeze  (tests extract this block verbatim)
    # Heads-up BEFORE the disruptive part of this step.  When install.sh
    # runs inside a VS Code integrated terminal (the Update button, or a
    # user-opened terminal), ``--install-extension --force`` below makes
    # VS Code detect the on-disk extension update and reload — and that
    # reload can dispose (or simply stop rendering) the very terminal this
    # script is printing to.  The output then appears to freeze with no
    # prompt ever returning, and users conclude the install hung.  In a
    # non-interactive run it did not: the new-session (setsid) detachment
    # at the top of this script keeps the install running, and the log
    # (see ``$LOG_FILE``) shows it completing a few seconds later.  An
    # interactive run skipped that detachment to keep its terminal, so a
    # disposed terminal can cut it short; the log tells which happened.
    # The only reliable channel to tell the user is this terminal, BEFORE
    # it can die — hence the notice below must stay ahead of the
    # ``--install-extension`` call.
    echo "   NOTE: VS Code may reload to pick up the update while this step runs."
    echo "         If this terminal stops updating (or never shows a prompt again),"
    if [ "${_KISS_INTERACTIVE:-0}" = 1 ]; then
        echo "         check the log; if the reload closed this terminal the install"
        echo "         may have been cut short, and re-running this script resumes it."
    else
        echo "         the install is NOT stuck — it keeps running detached and"
        echo "         finishes on its own."
    fi
    echo "         Follow progress with:"
    echo "             tail -f \"$LOG_FILE\""
    echo "         Completion is marked by the line: === Source bootstrap complete ==="
    if ! "$CODE_CLI" --install-extension "$VSIX" --force 2>&1 9>&-; then
        echo "   ERROR: '$CODE_CLI --install-extension' failed; the update was not applied."
        exit 1
    fi
    echo "   Extension installed into VS Code"
    # VS Code in the browser (``code serve-web``, started at the end of
    # this script) is a VS Code Server: it loads its extensions from
    # ~/.vscode-server/extensions, not from the desktop's directory, so
    # install the VSIX there as well.  This happens before the post-install
    # hooks below so a brand hook patches this copy too (and before
    # guard_vsix_tracking may put the release archive back in place of the
    # fresh build).  Not fatal: the desktop install above is the one the
    # rest of the setup depends on.
    if [ -z "${KISS_SKIP_LAUNCH:-}" ]; then
        if "$CODE_CLI" --install-extension "$VSIX" --extensions-dir "$HOME/.vscode-server/extensions" --force 2>&1 9>&-; then
            echo "   Extension installed into VS Code Server ($HOME/.vscode-server/extensions)"
        else
            echo "   WARNING: could not install the extension into $HOME/.vscode-server/extensions;"
            echo "            VS Code in the browser will run without KISS Sorcar."
        fi
    fi
    # Keep the freshly built VSIX from dirtying git (see guard_vsix_tracking
    # for the full rationale).  Public kiss_ai clones ship the VSIX as a
    # tracked file in every release commit, so "tracked" is a healthy state
    # there and the guard restores HEAD's copy instead of failing; in the
    # development repo a tracked VSIX means an accidental ``git add -f``
    # and remains a hard error.
    if ! guard_vsix_tracking "$PROJECT_DIR"; then
        exit 1
    fi

    # The kiss-web daemon is restarted only at the very end of this step
    # (start_kiss_web_daemon), under the extension's own rules: never
    # while the running daemon reports tasks in flight, in which case the
    # old daemon keeps serving and the extension's DependencyInstaller
    # restarts it later (``restartKissWebDaemon``, fingerprint mismatch
    # after the window reload).  Running this script therefore never
    # clobbers in-flight agent runs.

    # MODEL_INFO.json IS copied into the user's kiss home directory:
    # every install/update refreshes ``$KISS_HOME/MODEL_INFO.json`` from
    # the bundled ``src/kiss/core/models/MODEL_INFO.json``, and an
    # *installed* KISS Sorcar reads its model catalog from that copy at
    # runtime (``kiss.core.models.model_info._select_catalog_path``).
    # The settings panel's "Update Models" button then refreshes the
    # copy in place via ``kiss.scripts.update_models --model-info``.
    # Development checkouts (project roots with a .git marker) keep
    # reading the bundled file directly, so this copy never shadows a
    # checkout's own source of truth.
    #
    # User-curated model overrides / extensions live in
    # ``~/.kiss/MY_MODELS.json`` — auto-seeded on first import with a
    # short documentation block and one commented-out example entry —
    # matching the ``MY_INJECTION.md`` pattern.
    # The state directory is spelled out here and below rather than taken
    # from KISS_HOME_DIR: the tests paste these fragments into harnesses
    # that define neither KISS_HOME_DIR nor BRAND_HOME_DIR_NAME (there the
    # stock ``.kiss`` applies).
    MODEL_INFO_SRC="$PROJECT_DIR/src/kiss/core/models/MODEL_INFO.json"
    MODEL_INFO_DST="${KISS_HOME:-$HOME/${BRAND_HOME_DIR_NAME:-.kiss}}/MODEL_INFO.json"
    if [ -f "$MODEL_INFO_SRC" ]; then
        mkdir -p "$(dirname "$MODEL_INFO_DST")"
        cp "$MODEL_INFO_SRC" "$MODEL_INFO_DST"
        echo "   Copied MODEL_INFO.json to $MODEL_INFO_DST"
    else
        echo "   WARNING: $MODEL_INFO_SRC not found; skipped the ~/.kiss copy"
    fi

    # INJECTIONS.md is intentionally NOT copied into the user's kiss
    # home directory.  The bundled ``src/kiss/INJECTIONS.md`` is read
    # directly from the installed package at runtime by
    # ``kiss.server.tricks.read_tricks`` and ``getTricks`` in
    # ``SorcarTab.ts``, so every extension upgrade automatically
    # delivers the latest bundled tricks without clobbering user
    # edits.  User-curated tricks live in ``~/.kiss/MY_INJECTION.md``
    # — auto-seeded on first read with a single ``## Trick`` starter
    # ("Write end-to-end 100% coverage tests for the feature first.
    # Then implement the feature.").
    #
    # Re-introducing the copy here would mean a stale user-side
    # ``~/.kiss/INJECTIONS.md`` shadowing the freshly installed
    # bundled file forever after the first install.
    KISS_HOME_DIR="${KISS_HOME:-$HOME/${BRAND_HOME_DIR_NAME:-.kiss}}"
    mkdir -p "$KISS_HOME_DIR"

    # Finish the runtime setup the extension would otherwise do on its
    # first activation (see "Runtime setup without VS Code" above).  The
    # Python environment is built before the hooks so a hook that needs
    # it finds it; the daemon is (re)started after them, below, so it
    # serves the extension as the hooks left it.
    EXT_VERSION="$(node -p 'require(process.argv[1]).version' \
        "$PROJECT_DIR/src/kiss/agents/vscode/package.json" 2>/dev/null || true)"
    # ``|| …`` keeps ``set -e`` out of the function: every failure inside
    # is reported and the install goes on.
    setup_bundled_runtime "$EXT_VERSION" \
        || echo "   WARNING: the runtime setup did not complete; the extension retries on activation."

    # Post-install hooks: every executable in $KISS_HOME_DIR/post-install.d/
    # runs now, with the new extension directory on disk and before the
    # reload marker below, so a layer that patches the installed extension
    # (a white-label brand such as SeamlessLabs' s10s registers its
    # installer here) is applied before the window reloads.  Hooks run in
    # name order with KISS_HOME set and no stdin; a failing hook is
    # reported and the update still completes.
    for hook in "$KISS_HOME_DIR"/post-install.d/*; do
        [ -f "$hook" ] && [ -x "$hook" ] || continue
        echo "   Running post-install hook $hook..."
        hook_rc=0
        KISS_HOME="$KISS_HOME_DIR" "$hook" < /dev/null || hook_rc=$?
        if [ "$hook_rc" != 0 ]; then
            echo "   WARNING: post-install hook $hook exited with status $hook_rc"
        fi
    done

    # The marker must land in $KISS_HOME_DIR, not a hard-coded $HOME/.kiss:
    # the extension resolves its state dir through $KISS_HOME (kissHomeDir()
    # in userAssets.ts) and watches $KISS_HOME/.extension-updated to reload
    # the window (extension.ts) and finish the update (DependencyInstaller).
    # Writing the marker elsewhere leaves a custom-KISS_HOME install without
    # its reload signal — the update appears to never happen.
    date -u +%Y-%m-%dT%H:%M:%SZ > "$KISS_HOME_DIR/.extension-updated"
    # Remove any stale source-install marker from older versions of this
    # installer.  The extension now always runs against the kiss_project
    # bundled inside the VSIX, so the marker is no longer consulted and
    # leaving it around would only mislead troubleshooting.  Older
    # installers wrote it to the hard-coded ~/.kiss regardless of
    # KISS_HOME, so clean both locations.
    rm -f "$KISS_HOME_DIR/install_dir" "$HOME/.kiss/install_dir"
    echo ""

    # The daemon serves the directory this script was run from — the same
    # one VS Code opens below, and so the extension's own work dir.
    if [ -n "${KISS_PRIMARY_PROJECT:-}" ]; then
        start_kiss_web_daemon "$KISS_PRIMARY_PROJECT" "$USER_PWD" \
            || echo "   WARNING: kiss-web was not restarted; the extension restarts it on activation."
    fi
    echo ""

    echo "=== Source bootstrap complete ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Project: $PROJECT_DIR"
    echo ""
    echo "API keys and the remote-access password are set in VS Code (the extension"
    echo "asks on first activation) or in the web app's settings."
    # END: kiss-step-5-5-terminal-freeze
}

echo ""
echo "Log saved to $LOG_FILE"
# Only explicitly launch VS Code when it is not already running.  If a window
# is already open, the extension's watchers on ``out/extension.js`` and
# ``~/.kiss/.extension-updated`` (both touched in step [5/5]) trigger
# ``workbench.action.reloadWindow`` to pick up the update — launching here too
# would open a redundant second window.
if [ -n "${KISS_SKIP_LAUNCH:-}" ]; then
    # The caller (e.g. scripts/docker-startup.sh) owns launching the editor —
    # typically because it will start code-server itself right after this
    # script returns.  Launching here too would bind the same port and make
    # the caller's code-server fail with EADDRINUSE, crashing the container.
    echo "KISS_SKIP_LAUNCH set; skipping VS Code launch (caller will start the editor)."
elif vscode_is_running; then
    echo "VS Code is already running; the extension will reload to finish setup."
    echo "Skipping explicit launch to avoid opening a duplicate window."
else
    echo "Launching VS Code to finish setup..."
    launch_vscode || true
fi

# ---------------------------------------------------------------------------
# Browser helpers shared by the two blocks below
# ---------------------------------------------------------------------------
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
            open "$1" >/dev/null 2>&1 9>&-
            ;;
        Linux)
            command -v xdg-open >/dev/null 2>&1 || return 1
            (nohup xdg-open "$1" >/dev/null 2>&1 9>&- &)
            ;;
        *)
            return 1
            ;;
    esac
}
# END: kiss-browser-helpers

# ---------------------------------------------------------------------------
# Open VS Code in the browser
# ---------------------------------------------------------------------------
# ``code serve-web`` (the VS Code CLI) serves VS Code itself over HTTP on
# 127.0.0.1, guarded by a connection token in the URL.  The CLI announces
# the URL at once ("Web UI available at http://127.0.0.1:PORT?tkn=...")
# and downloads the VS Code Server on the first visit (the page shows a
# "downloading, please wait" notice and reloads itself).  The server loads
# extensions from ~/.vscode-server/extensions, where step [5/5] installed
# the VSIX, and the window opens on the same workspace as the desktop
# launch above.  The server is left running detached, like the editor; a
# re-run of this script reuses it when it is still alive.  On a remote
# machine the URL is printed with an ssh port-forward hint instead of
# opening a browser.  ``KISS_SKIP_LAUNCH`` (Docker) skips this too.
#
# The progress notification is updated here, outside the block, because
# tests extract the block verbatim and run it without report_step.
[ -n "${KISS_SKIP_LAUNCH:-}" ] || report_step "Starting VS Code in the browser..."
# BEGIN: kiss-open-vscode-web
# 0 lets the CLI pick a free port; the announced URL carries the real one.
KISS_VSCODE_WEB_PORT="${KISS_VSCODE_WEB_PORT:-0}"
KISS_VSCODE_WEB_WAIT_SECS="${KISS_VSCODE_WEB_WAIT_SECS:-60}"

vscode_web_url() {
    # Print the URL ``code serve-web`` announced in its log ($1), if any.
    sed -n 's/^Web UI available at \(http[^[:space:]]*\).*/\1/p' "$1" | head -n 1
}

url_port() {
    # Print the port of an http://host:PORT... URL ($1).
    local rest="${1#http://}"
    rest="${rest%%[/?]*}"
    printf '%s\n' "${rest##*:}"
}

port_is_open() {
    # True when something accepts TCP connections on 127.0.0.1:$1.
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

open_vscode_web() {
    local home_dir="${KISS_HOME:-$HOME/${BRAND_HOME_DIR_NAME:-.kiss}}"
    local log="$home_dir/vscode-web.log" pid_file="$home_dir/vscode-web.pid"
    local pid url port waited=0
    if [ -z "${CODE_CLI:-}" ] && ! find_code_cli; then
        echo "No VS Code CLI found; not starting VS Code in the browser."
        return 1
    fi
    echo ""
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    url="$(vscode_web_url "$log" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && [ -n "$url" ] && port_is_open "$(url_port "$url")"; then
        echo "VS Code is already served in the browser by 'code serve-web' (pid $pid)."
    else
        echo "Starting VS Code in the browser ('code serve-web')..."
        echo "   This accepts the VS Code Server license terms (https://aka.ms/vscode-server-license)."
        mkdir -p "$home_dir"
        : > "$log"
        (nohup "$CODE_CLI" serve-web --port "$KISS_VSCODE_WEB_PORT" --accept-server-license-terms \
            --default-folder "$USER_PWD" < /dev/null > "$log" 2>&1 9>&- &
         echo $! > "$pid_file")
        pid="$(cat "$pid_file")"
        while :; do
            url="$(vscode_web_url "$log")"
            [ -n "$url" ] && break
            if ! kill -0 "$pid" 2>/dev/null || [ "$waited" -ge "$KISS_VSCODE_WEB_WAIT_SECS" ]; then
                echo "   'code serve-web' did not announce a URL (log: $log)."
                sed 's/^/       /' "$log" | tail -n 5
                echo "   To serve VS Code in the browser yourself, run:"
                echo "       '$CODE_CLI' serve-web --accept-server-license-terms"
                return 1
            fi
            sleep 1
            waited=$((waited + 1))
        done
    fi

    if machine_is_remote; then
        port="$(url_port "$url")"
        echo ""
        echo "This machine is remote; forward the port from your own device with"
        echo "    ssh -L $port:127.0.0.1:$port ${USER:-$(id -un)}@$(hostname)"
        echo "and then open VS Code in your browser at:"
        echo "    $url"
        return 0
    fi
    echo ""
    echo "Opening VS Code in the browser at $url"
    echo "   (the first visit downloads the VS Code Server and reloads itself)"
    if ! open_in_browser "$url"; then
        echo "   Could not open a browser; open the URL above yourself."
    fi
}

if [ -z "${KISS_SKIP_LAUNCH:-}" ]; then
    open_vscode_web || true
fi
# END: kiss-open-vscode-web

# ---------------------------------------------------------------------------
# Open the webapp
# ---------------------------------------------------------------------------
# The kiss-web daemon is started by the extension once VS Code has finished
# the runtime setup (uv sync, cloudflared, ...), which takes a few minutes
# on a fresh install.  Wait for the daemon's URL file, trust its local
# certificate authority in this user's browsers (``kiss-web --trust-ca``,
# so the browser does not warn about the Local URL), then open the Local
# URL (https://127.0.0.1:PORT) in the default browser.  On a remote
# machine (an SSH session, or Linux without a display) there is no
# browser to open here, so the cloudflared URL is printed for the user to
# open on their own device.  ``KISS_SKIP_LAUNCH`` (Docker) skips this too.
#
# The progress notification is updated here, outside the block, because
# tests extract the block verbatim and run it without report_step.
[ -n "${KISS_SKIP_LAUNCH:-}" ] || report_step "Waiting for the kiss-web daemon to open the webapp..."
# BEGIN: kiss-open-webapp
KISS_WEBAPP_WAIT_SECS="${KISS_WEBAPP_WAIT_SECS:-900}"

find_kiss_web() {
    # kiss-web lives in the venv the extension builds inside its installed
    # copy of the project (``kiss_project/.venv``); a development checkout
    # may carry its own.  Any copy will do: ``--trust-ca`` reads the CA
    # from $KISS_HOME/tls and the URLs come from remote-url.json.
    local candidate
    for candidate in \
        "$(command -v kiss-web 2>/dev/null || true)" \
        "$PROJECT_DIR/.venv/bin/kiss-web" \
        "$HOME"/.vscode*/extensions/ksenxx.kiss-sorcar-*/kiss_project/.venv/bin/kiss-web \
        "$HOME"/.local/share/code-server/extensions/ksenxx.kiss-sorcar-*/kiss_project/.venv/bin/kiss-web; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
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
    local url_file="${KISS_HOME:-$HOME/${BRAND_HOME_DIR_NAME:-.kiss}}/remote-url.json"
    local remote=0 kiss_web="" tunnel="" loopback="" waited=0
    machine_is_remote && remote=1
    echo ""
    echo "Waiting for the kiss-web daemon (the extension starts it inside VS Code)..."
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
            echo "   Once VS Code has finished the setup, trust its certificate and"
            echo "   print the webapp URL with:"
            # kiss-web is usually not on PATH: name the binary if found.
            echo "       '${kiss_web:-kiss-web}' --trust-ca && '${kiss_web:-kiss-web}' --url"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 60)) -eq 0 ]; then
            echo "   ... still waiting (${waited}s)"
        fi
    done

    echo "   Trusting the webapp's certificate authority in this user's browsers..."
    # The CLI reads $KISS_HOME/tls: pin the home this install uses (the
    # binary may come from the checkout's venv, whose brand may differ).
    if ! KISS_HOME="${KISS_HOME:-$HOME/${BRAND_HOME_DIR_NAME:-.kiss}}" "$kiss_web" --trust-ca 9>&-; then
        echo "   WARNING: 'kiss-web --trust-ca' failed; browsers may warn about the Local URL."
    fi

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
    echo "Opening the webapp at $loopback"
    if ! open_in_browser "$loopback"; then
        echo "   Could not open a browser; open the URL above yourself."
    fi
}

if [ -z "${KISS_SKIP_LAUNCH:-}" ]; then
    open_webapp || true
fi
# END: kiss-open-webapp
