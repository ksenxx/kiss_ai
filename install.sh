#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# Install KISS Sorcar from source.
#
# This script's job is intentionally small: bootstrap only the tools needed to
# build and install the VS Code extension from a cloned checkout, then launch
# VS Code.  Runtime setup is owned by the extension's DependencyInstaller so
# users get the same installation path whether they run this script or install
# the VSIX directly.
#
# Usage: ./install.sh [--non-interactive]
#
#   Run from a terminal, the script asks ``[Y/n]`` before installing Homebrew
#   or upgrading git, uv, Node.js and VS Code.  ``--non-interactive`` (same as
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
# ``runUpdate()`` in ``SorcarSidebarView.ts``.  That method opens a VS Code
# integrated terminal and ``terminal.sendText``s a compound command ending in
# ``bash '/Users/ksen/.kiss/kiss_ai/install.sh'``.  The install ran through Xcode
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
# which still has a controlling terminal) gets a say before anything is
# installed or upgraded system-wide: installing Homebrew and upgrading git,
# uv, Node.js and VS Code are each a ``[Y/n]`` question (see ``confirm``
# below), and "no" keeps the installed version and carries on.
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
# answer (Yes), outdated tools are upgraded without asking, and a failed
# upgrade is a warning, never an abort.
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
        for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
            holder=$(cat "$_kiss_lock_file" 2>/dev/null || true)
            [ -n "$holder" ] && break
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
NODE_VERSION="v22.16.0"

# Required versions — extracted from the repo's source of truth so that
# install.sh stays in sync with DependencyInstaller.ts and package.json
# without hard-coding duplicates.
DEP_INSTALLER_TS="$PROJECT_DIR/src/kiss/agents/vscode/src/DependencyInstaller.ts"
VSCODE_PACKAGE_JSON="$PROJECT_DIR/src/kiss/agents/vscode/package.json"

# The trailing `|| true` matters: with `set -eo pipefail` an absent file or
# renamed constant makes `grep` exit non-zero, which would otherwise kill the
# whole script at these assignments before it printed anything.  An empty
# version simply skips the corresponding version check below.
REQUIRED_GIT_VERSION=$(grep "const GIT_VERSION" "$DEP_INSTALLER_TS" 2>/dev/null | head -1 | sed "s/.*= '//;s/'.*//" || true)
REQUIRED_UV_VERSION=$(grep "const UV_VERSION" "$DEP_INSTALLER_TS" 2>/dev/null | head -1 | sed "s/.*= '//;s/'.*//" || true)
REQUIRED_VSCODE_VERSION=$(grep '"vscode"' "$VSCODE_PACKAGE_JSON" 2>/dev/null | head -1 | sed 's/[^0-9.]//g' || true)
REQUIRED_NODE_VERSION="${NODE_VERSION#v}"

mkdir -p "$BIN_DIR" "$LOG_DIR"
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
    # cannot abort the parent script under ``set -e``.  We deliberately do
    # NOT trap INT/TERM here: the parent's cleanup at end-of-function uses
    # SIGTERM to stop the heartbeat, and a stray SIGINT killing the
    # heartbeat is harmless — at worst one elapsed-time message is lost;
    # the wrapped command itself stays alive via its own SIG_IGN above.
    (
        set +e
        while kill -0 "$cmd_pid" 2>/dev/null; do
            sleep "$HEARTBEAT_INTERVAL"
            if kill -0 "$cmd_pid" 2>/dev/null; then
                local elapsed=$(( $(date +%s) - start ))
                printf "   … %s still running (%ds elapsed)\n" "$label" "$elapsed"
            fi
        done
    ) 9>&- &
    local hb_pid=$!
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
# `ensure_homebrew` tried to re-install it and `upgrade_git` aborted the
# whole update with "Cannot upgrade git without Homebrew".
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
# Version helpers
# ---------------------------------------------------------------------------

# Compare two dotted version strings.  Returns 0 (true) when $1 >= $2.
version_gte() {
    local IFS=.
    # shellcheck disable=SC2206
    local i a=($1) b=($2)
    for ((i = 0; i < ${#b[@]}; i++)); do
        # Force base-10 so components with leading zeros (e.g. "08") are not
        # parsed as invalid octal, which would error out the arithmetic.
        local va=$((10#${a[i]:-0}))
        local vb=$((10#${b[i]:-0}))
        if ((va > vb)); then return 0; fi
        if ((va < vb)); then return 1; fi
    done
    return 0
}

# ---------------------------------------------------------------------------
# Upgrade helpers — invoked when the installed version is older than required
# ---------------------------------------------------------------------------

# Upgrade failures are deliberately non-fatal: a missing package manager or
# a flaky network must not abort the whole update (the previous behaviour —
# `exit 1` / unguarded commands under `set -e` — made the update button fail
# whenever the git-upgrade question fired in an environment without brew).
# The caller re-checks the installed version afterwards and warns if it is
# still too old.
upgrade_git() {
    echo "   Upgrading git..."
    case "$OS" in
        Darwin)
            if command -v brew &>/dev/null; then
                brew install git 2>/dev/null || brew upgrade git \
                    || echo "   WARNING: Homebrew could not upgrade git; continuing with the installed git."
            else
                echo "   WARNING: Cannot upgrade git without Homebrew; continuing with the installed git."
            fi
            ;;
        Linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -y && sudo apt-get install -y --only-upgrade git || true
            elif command -v dnf &>/dev/null; then
                sudo dnf upgrade -y git || true
            elif command -v yum &>/dev/null; then
                sudo yum update -y git || true
            elif command -v pacman &>/dev/null; then
                sudo pacman -Syu --noconfirm git || true
            elif command -v apk &>/dev/null; then
                sudo apk upgrade git || true
            else
                echo "   WARNING: No supported package manager found to upgrade git; continuing."
            fi
            ;;
    esac
    # A freshly installed git may live at a new path (e.g. /opt/homebrew/bin)
    # that bash's command hash still shadows with the old binary.
    hash -r
}

upgrade_uv() {
    echo "   Upgrading uv to $REQUIRED_UV_VERSION..."
    curl -LsSf "https://astral.sh/uv/${REQUIRED_UV_VERSION}/install.sh" | sh \
        || echo "   WARNING: uv upgrade failed; the VS Code extension will retry during setup."
    export PATH="$HOME/.local/bin:$PATH"
    hash -r
}

upgrade_node() {
    echo "   Upgrading Node.js to $NODE_VERSION..."
    install_node || echo "   WARNING: Node.js upgrade failed; continuing with the installed version."
    hash -r
}

upgrade_vscode() {
    echo "   Upgrading VS Code..."
    case "$OS" in
        Darwin)
            local ARCH_VS
            case "$ARCH" in
                aarch64|arm64) ARCH_VS="darwin-arm64" ;;
                x86_64)        ARCH_VS="darwin" ;;
            esac
            local TMP_ZIP TMP_APP_DIR
            TMP_ZIP="$(mktemp /tmp/vscode-XXXXXX.zip)"
            osascript -e 'quit app "Visual Studio Code"' 2>/dev/null || true
            sleep 2
            # Unpack into a temp dir FIRST and only then swap the app: the
            # old code removed /Applications/Visual Studio Code.app before
            # unzip, so a corrupt download crashed the script (`set -e`)
            # AND left the user with no VS Code at all.
            if curl -fsSL "https://update.code.visualstudio.com/latest/${ARCH_VS}/stable" -o "$TMP_ZIP"; then
                TMP_APP_DIR="$(mktemp -d /tmp/vscode-app-XXXXXX)"
                if unzip -q "$TMP_ZIP" -d "$TMP_APP_DIR" \
                        && [ -d "$TMP_APP_DIR/Visual Studio Code.app" ]; then
                    # Guarded like every other upgrade: a permission error
                    # in /Applications must warn, not abort under ``set -e``.
                    if rm -rf "/Applications/Visual Studio Code.app" \
                            && mv "$TMP_APP_DIR/Visual Studio Code.app" /Applications/; then
                        echo "   VS Code upgraded in /Applications/"
                        local CODE_BIN="/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"
                        if [ -x "$CODE_BIN" ]; then
                            ln -sf "$CODE_BIN" "$BIN_DIR/code" || true
                        fi
                    else
                        echo "   WARNING: Could not replace /Applications/Visual Studio Code.app; continuing with the installed version."
                    fi
                else
                    echo "   WARNING: Failed to unpack VS Code; continuing with the installed version."
                fi
                rm -rf "$TMP_ZIP" "$TMP_APP_DIR"
            else
                rm -f "$TMP_ZIP"
                echo "   WARNING: Failed to download VS Code; continuing with the installed version."
            fi
            ;;
        Linux)
            if command -v snap &>/dev/null; then
                sudo snap refresh code 2>&1 || true
            elif command -v apt-get &>/dev/null; then
                sudo apt-get update -y && sudo apt-get install -y --only-upgrade code 2>&1 || true
            elif command -v dnf &>/dev/null; then
                sudo dnf upgrade -y code 2>&1 || true
            else
                echo "   WARNING: Cannot upgrade VS Code automatically."
                echo "   Please upgrade from https://code.visualstudio.com if problems occur."
            fi
            ;;
    esac
    find_code_cli || true
}

upgrade_brew() {
    echo "   Updating Homebrew..."
    brew update
}

# ---------------------------------------------------------------------------
# Repo update helpers — stash local changes, pull latest, then restore them.
# ---------------------------------------------------------------------------

# Set to 1 once ``update_repo`` has stashed the working tree so that
# ``restore_stashed_changes`` knows whether there is anything to pop.
STASHED_CHANGES=0

restore_stashed_changes() {
    # Pop the stash created by ``update_repo`` so the working tree is left
    # exactly as we found it.  Wired to the EXIT trap so the unstash runs
    # "finally" — even if the install aborts midway under ``set -e``.
    if [ "$STASHED_CHANGES" = "1" ]; then
        echo ">>> Restoring stashed local changes..."
        if ! git -C "$PROJECT_DIR" stash pop; then
            # A stash made by an OLDER install.sh can carry a stale rebuilt
            # VSIX; popping it over the release copy guard_vsix_tracking
            # just restored conflicts, leaving unmerged stages that make the
            # NEXT update's ``git stash push`` fail with "needs merge" and
            # skip the pull — bricking the Update button.  When the VSIX is
            # the only conflict, heal it (the guard drops unmerged stages
            # and restores HEAD's copy — the stale rebuild is regenerated by
            # step [4/5] anyway).  The stash itself is always KEPT: a
            # failed pop can also mean untracked stashed files could not
            # be restored (a same-named file appeared meanwhile), and
            # dropping it would lose them for good — "only the VSIX is
            # unmerged" does not prove the pop restored everything else.
            local vsix_rel="src/kiss/agents/vscode/kiss-sorcar.vsix"
            local unmerged
            unmerged=$(git -C "$PROJECT_DIR" diff --name-only --diff-filter=U 2>/dev/null || true)
            if [ "$unmerged" = "$vsix_rel" ]; then
                guard_vsix_tracking "$PROJECT_DIR" || true
            fi
            echo "   WARNING: 'git stash pop' did not apply cleanly; your local"
            echo "   changes are preserved in 'git stash list'."
        fi
        STASHED_CHANGES=0
    fi
}

update_repo() {
    # Pull the latest kiss_ai sources before building.  If the working tree is
    # dirty, stash the changes first (so ``git pull`` applies cleanly), then
    # pop them back via the EXIT trap once the install finishes.
    #
    # KISS_SKIP_UPDATE exists for callers that deliberately install a checkout
    # they already control, commit for commit — ``sorcar-cloud`` has just put
    # the remote checkout on the branch it deploys (the laptop's uncommitted
    # edits committed and synced through origin first), and a pull would drag
    # it to whatever is on origin/main instead.
    if [ -n "${KISS_SKIP_UPDATE:-}" ]; then
        echo "   KISS_SKIP_UPDATE set — installing this checkout as-is, no pull."
        return 0
    fi
    if ! git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
        echo "   Not a git checkout — skipping pull."
        return 0
    fi
    # A stale rebuilt VSIX (or the unmerged stages a conflicted pop left in
    # an older install) must never reach the dirty check below: stashing it
    # makes the EXIT-trap ``git stash pop`` conflict with the release copy
    # guard_vsix_tracking restores in step [5/5], and an already-unmerged
    # entry makes ``git stash push`` fail with "needs merge" so the pull is
    # skipped forever.  Restoring HEAD's copy up front is lossless — the
    # rebuild is regenerated by step [4/5].  In a healthy development repo
    # the guard is a no-op; a development repo with a force-added VSIX must
    # fail HERE, loudly — the stash below would otherwise hide the staged
    # binary from the step [5/5] guard and the EXIT-trap pop would restore
    # it after that guard passed, silently bypassing the hard error.
    # One run only: the output is captured and re-emitted on failure
    # (success stays quiet here — step [5/5]'s own guard call reports the
    # restoration), instead of a silenced probe plus a loud re-run that
    # repeated every git operation of the guard.
    local _kiss_guard_out
    if ! _kiss_guard_out=$(guard_vsix_tracking "$PROJECT_DIR" 2>&1); then
        printf '%s\n' "$_kiss_guard_out" >&2
        exit 1
    fi
    if [ -n "$(git -C "$PROJECT_DIR" status --porcelain)" ]; then
        echo "   Repository is dirty — stashing local changes..."
        if git -C "$PROJECT_DIR" stash push --include-untracked -m "install.sh auto-stash"; then
            STASHED_CHANGES=1
            # The update lock (see the kiss-update-lock block) is a
            # kernel lock the process's death releases, so this may be
            # the script's only EXIT trap.
            trap restore_stashed_changes EXIT
        else
            echo "   WARNING: git stash failed; continuing without pulling."
            return 0
        fi
    fi
    echo "   Pulling latest changes..."
    # Non-fatal: offline machines must still be able to rebuild/reinstall
    # from the current checkout instead of crashing under `set -e`.
    #
    # Strategy:
    #   1. ``git fetch`` so we know the remote state even if the working tree
    #      ends up untouched.
    #   2. Try a fast-forward pull (the common, safe case).
    #   3. If fast-forward fails, the local branch has diverged from upstream
    #      — typically because the remote was force-pushed (e.g. release
    #      retag).  Reset hard to the upstream tip so the "Update" action in
    #      the settings panel actually updates.  Any local edits were already
    #      stashed above, so this is non-destructive.
    if ! git -C "$PROJECT_DIR" fetch --tags --prune origin 2>/dev/null; then
        echo "   WARNING: git fetch failed (offline?); continuing with the current checkout."
        return 0
    fi
    if git -C "$PROJECT_DIR" pull --ff-only; then
        return 0
    fi
    if git -C "$PROJECT_DIR" rev-parse --abbrev-ref '@{upstream}' &>/dev/null; then
        echo "   Branches diverged (upstream likely force-pushed) — resetting to upstream..."
        git -C "$PROJECT_DIR" reset --hard '@{upstream}' \
            || echo "   WARNING: git reset to upstream failed; continuing with the current checkout."
    else
        echo "   WARNING: no upstream tracking branch; continuing with the current checkout."
    fi
}

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
# the pipe.  tee inherits the update-lock fd 9 as well, harmlessly: its
# lifetime ends with the pipe, i.e. with this shell.
exec > >(trap '' INT TERM; exec tee -a "$LOG_FILE") 2>&1

{
    echo "=== KISS Sorcar Source Install ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Directory: $PROJECT_DIR"
    echo "OS: $OS ($ARCH)"
    if [ "$_KISS_INTERACTIVE" = 1 ]; then
        echo "Mode: interactive (asks before installing Homebrew or upgrading tools; pass --non-interactive to skip the questions)"
    else
        echo "Mode: non-interactive (outdated tools are upgraded without asking)"
    fi
    echo ""

    if [ "$OS" = "Darwin" ]; then
        echo ">>> Checking Xcode Command Line Tools..."
        ensure_xcode_clt
        echo ""

        echo ">>> Checking Homebrew..."
        ensure_homebrew
        echo ""
    fi

    echo ">>> [1/5] Checking git..."
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
    if [ -n "$REQUIRED_GIT_VERSION" ] && [ -n "$INSTALLED_GIT" ] && ! version_gte "$INSTALLED_GIT" "$REQUIRED_GIT_VERSION"; then
        echo "   git $INSTALLED_GIT is older than the required version $REQUIRED_GIT_VERSION."
        if confirm "Upgrade git to $REQUIRED_GIT_VERSION or later?"; then
            upgrade_git
            INSTALLED_GIT=$(git --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
            if [ -n "$INSTALLED_GIT" ] && ! version_gte "$INSTALLED_GIT" "$REQUIRED_GIT_VERSION"; then
                echo "   WARNING: git is still $INSTALLED_GIT (< $REQUIRED_GIT_VERSION); some features may not work."
            fi
        else
            echo "   Skipping the git upgrade; some features may not work with git $INSTALLED_GIT."
        fi
    fi
    echo "   git $INSTALLED_GIT ready"
    echo ""

    echo ">>> Updating kiss_ai repository..."
    update_repo
    echo ""

    echo ">>> Checking uv..."
    if command -v uv &>/dev/null; then
        INSTALLED_UV=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
        if [ -n "$REQUIRED_UV_VERSION" ] && [ -n "$INSTALLED_UV" ] && ! version_gte "$INSTALLED_UV" "$REQUIRED_UV_VERSION"; then
            echo "   uv $INSTALLED_UV is older than the required version $REQUIRED_UV_VERSION."
            if confirm "Upgrade uv to $REQUIRED_UV_VERSION?"; then
                upgrade_uv
                INSTALLED_UV=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
            else
                echo "   Skipping the uv upgrade; continuing with uv $INSTALLED_UV."
            fi
        fi
        echo "   uv $INSTALLED_UV ready"
    else
        echo "   uv not found — will be installed by the VS Code extension"
    fi
    echo ""

    echo ">>> [2/5] Checking Node.js..."
    if ! command -v node &>/dev/null || ! command -v npm &>/dev/null || ! command -v npx &>/dev/null; then
        install_node || true
    fi
    if command -v node &>/dev/null && command -v npm &>/dev/null && command -v npx &>/dev/null; then
        INSTALLED_NODE=$(node --version 2>/dev/null | sed 's/^v//' || true)
        if [ -n "$REQUIRED_NODE_VERSION" ] && [ -n "$INSTALLED_NODE" ] && ! version_gte "$INSTALLED_NODE" "$REQUIRED_NODE_VERSION"; then
            echo "   Node.js $INSTALLED_NODE is older than the required version $REQUIRED_NODE_VERSION."
            if confirm "Upgrade Node.js to $REQUIRED_NODE_VERSION?"; then
                upgrade_node
                INSTALLED_NODE=$(node --version 2>/dev/null | sed 's/^v//' || true)
            else
                echo "   Skipping the Node.js upgrade; the extension build may fail with node v$INSTALLED_NODE."
            fi
        fi
        echo "   node v$INSTALLED_NODE ready"
        echo "   npm $(npm --version) ready"
    else
        echo "   ERROR: Node.js, npm, and npx are required to build the extension."
        echo "   Install Node.js from https://nodejs.org and re-run this script."
        exit 1
    fi
    echo ""

    echo ">>> [3/5] Checking VS Code CLI..."
    if ! find_code_cli; then
        install_code_cli || true
        find_code_cli || true
    fi
    if [ -n "$CODE_CLI" ]; then
        INSTALLED_VSCODE=$("$CODE_CLI" --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1 || true)
        if [ -n "$REQUIRED_VSCODE_VERSION" ] && [ -n "$INSTALLED_VSCODE" ] && ! version_gte "$INSTALLED_VSCODE" "$REQUIRED_VSCODE_VERSION"; then
            echo "   VS Code $INSTALLED_VSCODE is older than the required version $REQUIRED_VSCODE_VERSION."
            if confirm "Upgrade VS Code?"; then
                upgrade_vscode
                INSTALLED_VSCODE=$("$CODE_CLI" --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1 || true)
                if [ -n "$INSTALLED_VSCODE" ] && ! version_gte "$INSTALLED_VSCODE" "$REQUIRED_VSCODE_VERSION"; then
                    echo "   WARNING: VS Code is still $INSTALLED_VSCODE (< $REQUIRED_VSCODE_VERSION); the extension may refuse to install."
                fi
            else
                echo "   Skipping the VS Code upgrade; the extension may refuse to install into VS Code $INSTALLED_VSCODE."
            fi
        fi
        echo "   code CLI ready: $CODE_CLI (v$INSTALLED_VSCODE)"
    else
        echo "   ERROR: VS Code CLI not found — cannot install the extension."
        echo "   Install VS Code from https://code.visualstudio.com and re-run this script."
        exit 1
    fi
    echo ""

    echo ">>> [4/5] Building VS Code extension..."
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
    echo "   Copying bundled KISS runtime..."
    run_with_heartbeat "copy-kiss" npm run copy-kiss
    echo "   Packaging VSIX..."
    run_with_heartbeat "vsce package" npm run package
    cd "$PROJECT_DIR"
    if [ ! -f "$VSIX" ]; then
        echo "   ERROR: Failed to build VSIX"
        exit 1
    fi
    echo "   Built $VSIX"
    echo ""

    # The ``sorcar`` CLI itself is installed into ~/.local/bin by the VS Code
    # extension (``installCliScript`` in DependencyInstaller.ts).  Install the
    # companion repo-root scripts the same way so they too can be run from
    # anywhere.
    echo ">>> Installing rsorcar and sorcar-docker launchers..."
    install_repo_script_launcher rsorcar
    install_repo_script_launcher sorcar-docker
    echo ""

    # BEGIN: kiss-step-5-5-terminal-freeze  (tests extract this block verbatim)
    echo ">>> [5/5] Installing VS Code extension..."
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
    # Keep the freshly built VSIX from dirtying git (see guard_vsix_tracking
    # for the full rationale).  Public kiss_ai clones ship the VSIX as a
    # tracked file in every release commit, so "tracked" is a healthy state
    # there and the guard restores HEAD's copy instead of failing; in the
    # development repo a tracked VSIX means an accidental ``git add -f``
    # and remains a hard error.
    if ! guard_vsix_tracking "$PROJECT_DIR"; then
        exit 1
    fi

    # The kiss-web daemon is deliberately NOT touched by this script — no
    # kill, no socket cleanup, no launchctl/systemctl restart.  Restarting
    # kiss-web is owned entirely by the VS Code extension: after the
    # ``.extension-updated`` marker written below triggers the window
    # reload, the extension's DependencyInstaller rebuilds the bundled
    # Python environment (``uv sync``) and restarts the daemon itself
    # (``restartKissWebDaemon`` — the fingerprint of the freshly installed
    # extension never matches ``~/.kiss/.kiss-web.fingerprint``), deferring
    # while tasks are in flight (``daemonHasActiveTasks``).  Until that
    # restart the old daemon keeps serving, so running this script never
    # clobbers in-flight agent runs.

    # MODEL_INFO.json is intentionally NOT copied into the user's kiss
    # home directory.  The bundled
    # ``src/kiss/core/models/MODEL_INFO.json`` is read directly from
    # the installed package at runtime by ``kiss.core.models.model_info``,
    # so every extension upgrade automatically delivers the latest model
    # pricing/context table without leaving a stale user-side copy
    # shadowing the freshly installed bundled file.
    #
    # User-curated model overrides / extensions live in
    # ``~/.kiss/MY_MODELS.json`` — auto-seeded on first import with a
    # short documentation block and one commented-out example entry —
    # matching the ``MY_INJECTION.md`` / ``MY_TASK_TEMPLATES.md`` pattern.
    #
    # Re-introducing the copy here would mean a stale user-side
    # ``~/.kiss/MODEL_INFO.json`` shadowing the freshly installed
    # bundled file forever after the first install.

    # INJECTIONS.md is intentionally NOT copied into the user's kiss
    # home directory.  The bundled ``src/kiss/INJECTIONS.md`` is read
    # directly from the installed package at runtime by
    # ``kiss.server.tricks.read_tricks`` and ``getTricks`` in
    # ``SorcarTab.ts``, so every extension upgrade automatically
    # delivers the latest bundled tricks without clobbering user
    # edits.  User-curated tricks live in ``~/.kiss/MY_INJECTION.md``
    # — auto-seeded on first read with a single ``## Trick`` starter
    # ("Write end-to-end 100% coverage tests for the feature first.
    # Then implement the feature.") — matching the
    # ``MY_TASK_TEMPLATES.md`` / ``SAMPLE_TASKS.md`` pattern.
    #
    # Re-introducing the copy here would mean a stale user-side
    # ``~/.kiss/INJECTIONS.md`` shadowing the freshly installed
    # bundled file forever after the first install.
    KISS_HOME_DIR="${KISS_HOME:-$HOME/.kiss}"
    mkdir -p "$KISS_HOME_DIR"

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

    echo "=== Source bootstrap complete ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Project: $PROJECT_DIR"
    echo ""
    echo "KISS Sorcar runtime setup will finish inside VS Code."
    echo "The extension will install/check uv, Python dependencies, Playwright,"
    echo "cloudflared, shell PATH entries, API keys, remote access auth, and kiss-web."
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
