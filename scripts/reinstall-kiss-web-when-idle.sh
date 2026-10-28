#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Re-run ./install.sh once the kiss-web daemon has no in-flight task, so the
# daemon restarts on the freshly installed code without killing anything.
#
#   scripts/reinstall-kiss-web-when-idle.sh           # launch the detached job and return
#   scripts/reinstall-kiss-web-when-idle.sh --worker  # the job body (started by the above)
#
# Why a detached job: install.sh refuses to restart kiss-web while a task is in
# flight, and a Sorcar task that runs install.sh is itself in flight, so the
# restart must happen from a process that outlives the task and does not belong
# to the daemon's process group / service.
#
# Why ONE-SHOT matters: on 2026-10-28 an ad-hoc copy of this loop was started
# with `launchctl submit`, which tells launchd to keep the program alive.  Every
# time the worker finished, launchd restarted it; it waited for an idle daemon,
# reinstalled, restarted kiss-web, exited, and was restarted again -- nine
# installs in four minutes.  The macOS branch below therefore loads a plist with
# KeepAlive=false (and the worker boots its own label out when it exits); the
# Linux branch uses `systemd-run --collect`, a transient unit that runs once.
#
# Environment (all optional):
#   KISS_REINSTALL_POLL_SECONDS      seconds between idle probes (default 5)
#   KISS_REINSTALL_DEADLINE_SECONDS  give up after this long without an idle window (default 14400)
#   KISS_REINSTALL_LABEL             job label / unit name (default kiss-reinstall-<epoch>-<pid>)
#
# The worker logs to <repo>/tmp/reinstall-kiss-web/<label>.log and keeps each
# install.sh transcript next to it.  It exits 0 once install.sh reports
# "kiss-web is up" or "already serves this runtime", retries after "restart
# deferred", and gives up (non-zero) on any other outcome or at the deadline.
set -u

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
POLL_SECONDS="${KISS_REINSTALL_POLL_SECONDS:-5}"
DEADLINE_SECONDS="${KISS_REINSTALL_DEADLINE_SECONDS:-14400}"
LABEL="${KISS_REINSTALL_LABEL:-kiss-reinstall-$(date +%s)-$$}"
LOGDIR="$REPO/tmp/reinstall-kiss-web"
LOG="$LOGDIR/$LABEL.log"

ts() { date -u +%FT%TZ; }

# Everything the detached job needs, as NAME=VALUE entries.
JOB_ENV=(
    "HOME=$HOME"
    "PATH=$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    "KISS_REINSTALL_LABEL=$LABEL"
    "KISS_REINSTALL_POLL_SECONDS=$POLL_SECONDS"
    "KISS_REINSTALL_DEADLINE_SECONDS=$DEADLINE_SECONDS"
)

xml_escape() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

# macOS: a one-shot launchd job in the user's GUI domain.  RunAtLoad starts it
# as soon as it is bootstrapped; KeepAlive=false means launchd never restarts
# it.  The job gets ppid 1 and its own process group, so restarting the
# com.kiss.web-server agent does not kill it.
launch_with_launchd() {
    local plist="$LOGDIR/$LABEL.plist" domain="gui/$(id -u)" entry
    {
        echo '<?xml version="1.0" encoding="UTF-8"?>'
        echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
        echo '<plist version="1.0"><dict>'
        echo "  <key>Label</key><string>$(printf '%s' "$LABEL" | xml_escape)</string>"
        echo '  <key>ProgramArguments</key><array>'
        echo '    <string>/bin/bash</string>'
        echo "    <string>$(printf '%s' "$SCRIPT" | xml_escape)</string>"
        echo '    <string>--worker</string>'
        echo '  </array>'
        echo '  <key>RunAtLoad</key><true/>'
        echo '  <key>KeepAlive</key><false/>'
        echo '  <key>EnvironmentVariables</key><dict>'
        echo '    <key>KISS_REINSTALL_LAUNCHD</key><string>1</string>'
        for entry in "${JOB_ENV[@]}"; do
            echo "    <key>${entry%%=*}</key><string>$(printf '%s' "${entry#*=}" | xml_escape)</string>"
        done
        echo '  </dict>'
        echo '</dict></plist>'
    } >"$plist"
    launchctl bootout "$domain/$LABEL" 2>/dev/null
    launchctl bootstrap "$domain" "$plist"
}

# Linux: a transient systemd user unit; --collect removes it once it has run.
launch_with_systemd() {
    systemd-run --user --unit="$LABEL" --collect "${JOB_ENV[@]/#/--setenv=}" /bin/bash "$SCRIPT" --worker
}

# Neither launchd nor systemd: a new session via setsid, which the daemon's
# process-group kill cannot reach.  stderr stays attached so a failing setsid
# is heard; the worker moves its own output to its log as its first act.
launch_with_setsid() {
    setsid -f env "${JOB_ENV[@]}" /bin/bash "$SCRIPT" --worker </dev/null >/dev/null
}

launch() {
    mkdir -p "$LOGDIR"
    if [ "$(uname)" = "Darwin" ]; then
        launch_with_launchd || return 1
    elif command -v systemd-run >/dev/null 2>&1; then
        launch_with_systemd || return 1
    elif command -v setsid >/dev/null 2>&1; then
        launch_with_setsid || return 1
    else
        echo "reinstall-kiss-web-when-idle: need launchctl (macOS), systemd-run or setsid to detach the job" >&2
        return 1
    fi
    echo "Detached job $LABEL will re-run install.sh once kiss-web is idle; log: $LOG"
}

# Unload our own launchd label so `launchctl list` does not accumulate
# finished one-shot jobs.  launchd answers with SIGTERM, which is fine: this
# runs as the very last thing on exit.
unload_launchd_label() {
    if [ "${KISS_REINSTALL_LAUNCHD:-}" = "1" ]; then
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
    fi
}

kiss_web_is_idle() {
    local python="$HOME/.kiss/kiss/.venv/bin/python"
    [ -x "$python" ] || python=python3
    KISS_ACTIVE_TASKS_STRICT=1 "$python" "$REPO/scripts/check-kiss-web-active-tasks.py" >/dev/null 2>&1
}

worker() {
    local deadline=$(( $(date +%s) + DEADLINE_SECONDS )) idle_streak=0 attempt=0 runlog rc
    export KISS_NONINTERACTIVE=1
    unset KISS_FORCE_RESTART
    mkdir -p "$LOGDIR"
    exec >>"$LOG" 2>&1
    trap unload_launchd_label EXIT
    echo "=== $(ts) worker started (pid $$, pgid $(ps -o pgid= -p $$ | tr -d ' '), label $LABEL)"
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if kiss_web_is_idle; then
            idle_streak=$((idle_streak + 1))
        else
            idle_streak=0
        fi
        if [ "$idle_streak" -lt 3 ]; then
            sleep "$POLL_SECONDS"
            continue
        fi
        attempt=$((attempt + 1))
        runlog="$LOGDIR/$LABEL.install-$attempt.log"
        echo "=== $(ts) kiss-web idle for 3 polls; running install.sh (attempt $attempt)"
        ( cd "$REPO" && /bin/bash ./install.sh --non-interactive </dev/null ) >"$runlog" 2>&1
        rc=$?
        echo "=== $(ts) install.sh exit=$rc, transcript: $runlog"
        if grep -aq "kiss-web is up" "$runlog" || grep -aq "already serves this runtime" "$runlog"; then
            echo "=== $(ts) kiss-web restarted on the new code; done."
            return 0
        fi
        if grep -aq "restart deferred" "$runlog"; then
            echo "=== $(ts) tasks became active again; waiting for an idle window and retrying."
            idle_streak=0
            sleep "$POLL_SECONDS"
            continue
        fi
        echo "=== $(ts) install.sh ended without a restart status (exit=$rc); giving up."
        [ "$rc" -ne 0 ] && return "$rc"
        return 1
    done
    echo "=== $(ts) deadline reached without an idle window; giving up."
    return 2
}

if [ "${1:-}" = "--worker" ]; then
    worker
else
    launch
fi
