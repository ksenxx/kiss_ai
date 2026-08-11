#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Sync the task database (~/.kiss/sorcar.db) between this machine and a remote
# host in *both* directions, so the History panel on either machine lists every
# task either machine ever ran.
#
# Usage:  scripts/sync-task-db.sh user@ip-address [local-project-dir remote-project-dir]
#
# Called by ./sorcar-cloud on every deploy; also useful on its own to refresh a
# running deployment's history — and this machine's — without redeploying.
# Pass the two project directories to keep the History panel's "Workspace" chip
# working (see below); without them the tasks arrive but that chip hides them.
#
# Two passes, in this order:
#
#   1. remote -> here.  The remote takes a consistent snapshot of its database
#      (``VACUUM INTO``, so a running web app there is not disturbed), the
#      recorded work directories in that snapshot are re-pointed at *this*
#      checkout, and src/kiss/scripts/sync_db.py merges the rows this machine
#      is missing into ~/.kiss/sorcar.db in place.  Only the delta crosses the
#      network, and nothing here is deleted or overwritten wholesale: a local
#      web app can keep running throughout.  When this machine has no database
#      at all, the snapshot is downloaded in full instead, verified, and put in
#      place.
#   2. here -> remote.  The mirror image: a snapshot of this machine's
#      database, its work directories re-pointed at the deployment, then a
#      delta merge into the remote's database in place.
#
# Nothing is lost, by construction:
#
#   * sync_db.py only inserts rows and updates a task row that the incoming
#     copy has carried further along (more steps, tokens, cost, or a later end
#     time; a favourite marked on either machine survives).  A row that merely
#     differs is left alone, no row is ever deleted, and no table other than
#     ``task_history`` and ``events`` is touched.
#   * Neither live database is ever rewritten by the relocation step: work
#     directories are translated in the throw-away snapshot that travels, and
#     a relocation that fails stops its direction rather than let one
#     machine's paths overwrite the other's.
#   * Pass 1 runs first so that the tasks which ran only on the remote are
#     already safe here before pass 2 touches anything there.
#   * The one destructive operation is pass 2's fallback: replacing the
#     remote's database wholesale.  It only ever runs when nothing can be lost
#     by it — the remote has no database, or none that any reader could get a
#     row out of, or pass 1 has already brought its rows here *and the remote
#     still holds exactly what it held then* -- the same tasks, the same
#     progress on them, the same events -- compared once before the decision and
#     again once the web app there is stopped and nothing can write any more.
#     Anything that happened there in between, down to a single event, is
#     something this machine has never seen, and stops the replacement.
#     A probe that did not run authorises nothing.  Whatever file
#     it replaces is kept as ~/.kiss/sorcar.db.replaced-<time>, complete: the
#     pages that were still only in its -wal are folded into it first, and if
#     they cannot be, that -wal is kept alongside instead.  An older backup is
#     never written over, and the three tables no sync moves — that machine's
#     own usage counters — are carried from that file into the database which
#     replaced it.
#   * The fallback stops the remote web app, and stopping it kills the task it
#     is running, so it refuses to run while one is: SORCAR_FORCE_RESTART=1
#     says go ahead anyway, and SORCAR_LIVE_TASK_WINDOW sets how recent a
#     task's newest event has to be for the task to count as running (300
#     seconds by default; 0 turns the check off).
#   * A pass that could not finish leaves both databases as they were, says so,
#     and makes the script exit non-zero.
#
# The full upload of that fallback has four traps in it, and all four are
# handled here:
#
#   * The live database is a trio — sorcar.db plus its -wal (pages committed
#     but not yet folded into the main file) and -shm (an ephemeral,
#     machine-local index that must never be copied at all).  While an agent
#     runs, all three keep changing, so copying them one after another over a
#     slow link produces a set that never existed at any single instant.
#     ``VACUUM INTO`` instead yields one self-contained file holding one
#     consistent point in time.
#   * A kiss-web left running on the remote from an earlier deploy holds the
#     old database open.  Deleting the file underneath it makes it create a
#     fresh empty one; the upload then lands on that same inode, and the first
#     checkpoint afterwards writes that connection's stale, empty view back
#     over it — which is what left the remote History panel nearly blank.  So
#     the remote web app is stopped for the swap and started again afterwards,
#     whether the swap succeeded or not.
#   * A multi-gigabyte upload can be cut short.  The file therefore arrives
#     under a temporary name and is promoted only once it proves to hold
#     exactly the tasks that were sent, so a failed transfer keeps the previous
#     database instead of replacing it with a truncated one.
#   * Every task remembers the directory it ran in, and the History panel hides
#     tasks from other workspaces by default.  The deployment is this project
#     at a different path, so the recorded directories are re-pointed at it —
#     otherwise the imported history arrives complete and invisible.  The same
#     translation, in reverse, is what pass 1 applies to what it brings back.
#
# Needs python3 here and on the remote, and a SQLite new enough for
# ``VACUUM INTO`` (3.27, released 2019) on both.
set -euo pipefail

info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
step() { printf '\033[0;34m[STEP]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()  { printf '\033[0;31m[ERR]\033[0m  %s\n' "$*" >&2; exit 1; }

# Quote a value for the remote shell: a project path may hold a space or an
# apostrophe, which would otherwise break — or extend — the command line.
shquote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }

TARGET="${1:-}"
LOCAL_DIR="${2:-}"
REMOTE_DIR="${3:-}"
[[ -n "$TARGET" ]] \
    || die "Usage: $0 user@ip-address [local-project-dir remote-project-dir]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SYNC_DB="$PROJECT_ROOT/src/kiss/scripts/sync_db.py"
RELOCATE="$PROJECT_ROOT/src/kiss/scripts/relocate_work_dir.py"
CARRY_OVER="$PROJECT_ROOT/src/kiss/scripts/carry_over_tables.py"
RUNNING_TASKS="$PROJECT_ROOT/src/kiss/scripts/running_tasks.py"
FINGERPRINT="$PROJECT_ROOT/src/kiss/scripts/db_fingerprint.py"
# How recent a task's newest event has to be for the task to count as running,
# which is what stops the fallback from stopping the web app under one.
LIVE_TASK_WINDOW="${SORCAR_LIVE_TASK_WINDOW:-300}"

KISS_DIR="${KISS_HOME:-$HOME/.kiss}"
DB="$KISS_DIR/sorcar.db"
# Where the remote leaves the snapshot pass 1 reads.  Expanded by the remote
# shell, so it is written with a literal $HOME everywhere below.
REMOTE_SNAPSHOT='$HOME/.kiss/sorcar.db.outgoing'

TMP_DIR="$(mktemp -d)"
SNAPSHOT="$TMP_DIR/sorcar.db"

# One exit path for everything that has to be undone, armed before any of it
# happens: an ssh connection can drop after the remote has already acted.
REMOTE_SNAPSHOT_MADE=0
REMOTE_WEB_APP_STOPPED=0

# Set when the remote's tasks are known to be here now.  Replacing the
# remote's database is only safe once that has happened, so a pass 1 that
# could not finish has to be able to stop pass 2 from doing it.
PULL_COMPLETE=0
# What the remote held when pass 1 read it (see
# src/kiss/scripts/db_fingerprint.py).  A wholesale replacement is only as safe
# as that is current: anything that happened there since -- a task started, a
# task carried further, an event written -- is something this machine has never
# seen and the replacement would throw away.
REMOTE_FINGERPRINT_AT_PULL=""
# Set by anything that did not do what it set out to do.  Such a run has not
# synced the two machines, and must not report that it has.
INCOMPLETE=0
# IDs that were live on this machine before any remote rows were merged here.
# After the pull, row contents alone cannot say which machine a live task came
# from, and a genuinely remote task must survive a forced replacement.
LOCAL_LIVE_TASK_IDS=""

# Report a step that failed without giving up on the rest of the sync: the
# other direction, or the other machine, may still have work that can travel.
incomplete() {
    warn "$@"
    INCOMPLETE=1
}

cleanup() {
    if (( REMOTE_SNAPSHOT_MADE )); then
        ssh "$TARGET" "rm -f \"$REMOTE_SNAPSHOT\"" >/dev/null 2>&1 \
            || warn "Could not remove $TARGET:$REMOTE_SNAPSHOT — delete it by hand."
    fi
    if (( REMOTE_WEB_APP_STOPPED )); then
        ssh "$TARGET" 'systemctl --user start kiss-web.service >/dev/null 2>&1 || true' \
            || warn "Could not start the web app on $TARGET — start it by hand."
    fi
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# What the two sides have
#
# A delta merge needs a database with the two synced tables on the receiving
# end.  A missing or unreadable one (first deploy, torn earlier upload) or a
# database without those tables is reported here and decides both passes:
# pass 1 has nothing to bring back, and pass 2 falls back to a full upload.
#
# ``unknown`` -- the probe itself did not run, so this says nothing about what
# is on the remote -- is deliberately a state of its own: a dropped connection
# must never be read as "there is nothing there to lose".
# ---------------------------------------------------------------------------
step "Looking at the task database on $TARGET ..."
REMOTE_DB_STATE="$(ssh "$TARGET" 'python3 -' <<'PY' 2>/dev/null || echo unknown
import os
import sqlite3

path = os.path.expanduser("~/.kiss/sorcar.db")
if not os.path.isfile(path):
    print("missing")
    raise SystemExit
try:
    con = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
    tables = {row[0] for row in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
except sqlite3.Error:
    print("unreadable")
    raise SystemExit
print("ok" if {"task_history", "events"} <= tables else "incompatible")
PY
)"
# A connection that dies mid-command can leave a line of output behind before
# the "unknown" appended above, so only a whole, recognised answer counts as
# one: anything else is an answer that never arrived.
REMOTE_DB_STATE="$(printf '%s' "$REMOTE_DB_STATE" | tr -d '[:space:]')"
case "$REMOTE_DB_STATE" in
    ok|missing|unreadable|incompatible) ;;
    *) REMOTE_DB_STATE=unknown ;;
esac

# ---------------------------------------------------------------------------
# A consistent, self-contained copy of a live database
#
# timeout: an agent may be writing to the same database, and the snapshot only
# needs to wait out its current commit.
# ---------------------------------------------------------------------------
snapshot_local_db() {
    python3 -c 'import sqlite3, sys
con = sqlite3.connect(sys.argv[1], timeout=60)
con.execute("VACUUM INTO ?", (sys.argv[2],))
con.close()' "$DB" "$SNAPSHOT"
}

# A deployment can itself be a task in this machine's database.  Sending that
# unfinished row to the server makes the final restart guard mistake it for a
# task that began *there* during the deploy, so the deploy refuses to restart
# itself.  Capture local identity before the pull: afterwards a genuinely
# remote live row has been merged here too and must not be filtered back out.
capture_live_local_tasks() {
    local answer count
    [[ -f "$DB" && -f "$RUNNING_TASKS" ]] || return 0
    answer="$(python3 "$RUNNING_TASKS" "$DB" "$LIVE_TASK_WINDOW" 2>/dev/null || true)"
    count="$(printf '%s\n' "$answer" | head -1 | tr -d '[:space:]')"
    [[ "$count" =~ ^[0-9]+$ ]] || {
        warn "Could not identify live local tasks before syncing; continuing."
        return 0
    }
    ((count > 0)) || return 0
    LOCAL_LIVE_TASK_IDS="$(printf '%s\n' "$answer" | tail -n +2 | sed 's/ [^ ]*$//')"
}

# Leave the captured local rows out of the outgoing snapshot.  Completed tasks,
# stale interrupted tasks, and live tasks pulled from the remote still travel;
# deferred local work travels on the next sync after it finishes.  The live
# database itself is never changed.
omit_live_tasks_from_snapshot() {
    local removed
    [[ -n "$LOCAL_LIVE_TASK_IDS" ]] || return 0
    removed="$(printf '%s\n' "$LOCAL_LIVE_TASK_IDS" \
        | python3 -c 'import sqlite3, sys
snapshot = sys.argv[1]
task_ids = [line.rstrip("\n") for line in sys.stdin if line.rstrip("\n")]
con = sqlite3.connect(snapshot, timeout=60)
try:
    con.execute("CREATE TEMP TABLE live_local_tasks(id TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO live_local_tasks(id) VALUES (?)",
                    ((task_id,) for task_id in task_ids))
    con.execute("DELETE FROM events WHERE task_id IN"
                " (SELECT id FROM live_local_tasks)")
    removed = con.execute("DELETE FROM task_history WHERE id IN"
                          " (SELECT id FROM live_local_tasks)").rowcount
    con.commit()
    print(removed)
finally:
    con.close()' "$SNAPSHOT")"
    if ((removed > 0)); then
        info "Deferred $removed live local task(s); they will sync after they finish."
    fi
}

# What the remote's database holds, in one line, for comparing two moments in
# time (see src/kiss/scripts/db_fingerprint.py).  Empty means the question was
# not answered -- which no caller may read as "nothing has changed".
remote_fingerprint() {
    local path="$1" answer
    [[ -f "$FINGERPRINT" ]] || return 0
    answer="$(ssh "$TARGET" "python3 - \"$path\"" < "$FINGERPRINT" 2>/dev/null \
        | head -1 || true)"
    # Six counts and the digest of the task rows.  Anything else -- a python
    # traceback, a login banner, a shell's "command not found" -- is not an
    # answer, and must not be compared with one.
    [[ "$answer" =~ ^[0-9]+([[:space:]]+[0-9.eE+-]+){5}[[:space:]]+[0-9a-f]+$ ]] \
        || return 0
    printf '%s' "$answer"
}

count_tasks() {
    python3 -c 'import sqlite3, sys
con = sqlite3.connect("file:" + sys.argv[1] + "?mode=ro", uri=True)
print(con.execute("SELECT count(*) FROM task_history").fetchone()[0])' "$1"
}

# Rewrite the work directories of a snapshot so the receiving machine's
# History panel shows the tasks instead of filtering them out.  Only the paths
# inside the project move; a database that records none is left alone.
#
# A failure here stops the direction it belongs to: rows carrying the sending
# machine's paths would be written over rows that carry the receiving
# machine's, hiding tasks that were visible before.  Nothing has been sent at
# this point, so refusing costs only this run.
relocate_snapshot() {
    local snapshot="$1" from_dir="$2" to_dir="$3" moved
    [[ -n "$LOCAL_DIR" && -n "$REMOTE_DIR" ]] || return 0
    if [[ ! -f "$RELOCATE" ]]; then
        incomplete "$RELOCATE is missing — refusing to sync work directories" \
                   "this machine does not recognise."
        return 1
    fi
    if ! moved="$(python3 "$RELOCATE" "$snapshot" "$from_dir" "$to_dir")"; then
        incomplete "Could not re-point the recorded work directories" \
                   "from $from_dir to $to_dir — nothing was sent."
        return 1
    fi
    info "Re-pointed $moved task(s) from $from_dir to $to_dir."
}

# ---------------------------------------------------------------------------
# Pass 1: bring back what ran on the remote
#
# The snapshot is taken and rewritten on the remote, so that the rows arriving
# here already carry this machine's paths: the digests sync_db.py compares then
# match the rows already here, and a repeated sync moves nothing.
# ---------------------------------------------------------------------------
pull_from_remote() {
    local remote_tasks
    step "Snapshotting the task database on $TARGET ..."
    REMOTE_SNAPSHOT_MADE=1
    if ! remote_tasks="$(ssh "$TARGET" 'python3 -' < "$TMP_DIR/remote-snapshot.py")" \
            || [[ ! "$remote_tasks" =~ ^[0-9]+$ ]]; then
        incomplete "Could not snapshot the task database on $TARGET" \
                   "(no space in ~/.kiss, or a stuck writer?) — nothing brought back."
        return 0
    fi
    info "$TARGET holds $remote_tasks task(s)."
    # What the snapshot holds, which is what this pass is about to bring here.
    # Taken from the snapshot rather than from the live database, so that it
    # describes exactly the rows that travel.
    REMOTE_FINGERPRINT_AT_PULL="$(remote_fingerprint "$REMOTE_SNAPSHOT")"

    if [[ -n "$LOCAL_DIR" && -n "$REMOTE_DIR" ]]; then
        if [[ ! -f "$RELOCATE" ]]; then
            incomplete "$RELOCATE is missing — nothing brought back from $TARGET."
            return 0
        fi
        if ! ssh "$TARGET" "python3 - \"$REMOTE_SNAPSHOT\" $(shquote "$REMOTE_DIR") \
                $(shquote "$LOCAL_DIR")" < "$RELOCATE" >/dev/null; then
            incomplete "Could not re-point $TARGET's work directories at $LOCAL_DIR" \
                       "— nothing brought back, so its paths cannot overwrite yours."
            return 0
        fi
    fi

    if [[ ! -f "$DB" ]]; then
        download_remote_db "$remote_tasks"
        return 0
    fi
    step "Merging $TARGET's tasks into $DB ..."
    if [[ ! -f "$SYNC_DB" ]]; then
        incomplete "$SYNC_DB is missing — $TARGET's tasks were left there."
    elif python3 "$SYNC_DB" "$TARGET:~/.kiss/sorcar.db.outgoing" "$DB"; then
        info "This machine's task database now holds $TARGET's tasks too."
        PULL_COMPLETE=1
    else
        incomplete "Could not merge $TARGET's tasks into $DB — they were left there." \
                   "This machine's own history is untouched."
    fi
}

# ---------------------------------------------------------------------------
# Pass 1, first time on this machine: no database to merge into, so take the
# remote's snapshot whole.  It is verified before it is put in place, and a
# database that appeared meanwhile (a web app starting up as this ran) is
# merged into rather than replaced.
# ---------------------------------------------------------------------------
download_remote_db() {
    local expected="$1" got incoming="$TMP_DIR/incoming.db"
    step "Downloading $expected task(s) from $TARGET ..."
    ssh "$TARGET" "gzip -1 -c \"$REMOTE_SNAPSHOT\"" | gzip -dc > "$incoming" \
        || { incomplete "Downloading the task database from $TARGET failed."; return 0; }
    got="$(python3 -c 'import sqlite3, sys
con = sqlite3.connect("file:" + sys.argv[1] + "?mode=ro&immutable=1", uri=True)
if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
    raise SystemExit("the downloaded database is corrupt")
print(con.execute("SELECT count(*) FROM task_history").fetchone()[0])' "$incoming")" \
        || { incomplete "The database downloaded from $TARGET is not readable."; return 0; }
    if [[ "$got" != "$expected" ]]; then
        incomplete "Only $got of $expected task(s) arrived — the download was discarded."
        return 0
    fi
    mkdir -p "$KISS_DIR" && chmod 700 "$KISS_DIR"
    if [[ ! -f "$DB" ]]; then
        mv "$incoming" "$DB"
        info "Task database created at $DB from $TARGET ($got task(s))."
        PULL_COMPLETE=1
        return 0
    fi
    # A web app started up while the download ran and created a database:
    # merge into it rather than throw its rows away.
    if [[ -f "$SYNC_DB" ]] && python3 "$SYNC_DB" "$incoming" "$DB"; then
        PULL_COMPLETE=1
    else
        incomplete "A task database appeared at $DB while downloading" \
                   "— $TARGET's tasks were left there."
    fi
}

# ---------------------------------------------------------------------------
# Pass 2: send what ran here
#
# The delta merge is the rule: sync_db.py asks the remote what it already has,
# ships only the missing task and event rows, and merges them in one
# transaction — so a repeated deploy moves kilobytes instead of gigabytes, the
# remote web app never has to stop, and the tasks that ran only there survive.
# ---------------------------------------------------------------------------
push_to_remote() {
    local tasks
    step "Snapshotting $DB ..."
    snapshot_local_db \
        || die "Could not snapshot $DB (no space in $TMP_DIR, or a stuck writer?)."
    omit_live_tasks_from_snapshot
    tasks="$(count_tasks "$SNAPSHOT" 2>/dev/null || true)"
    [[ "$tasks" =~ ^[0-9]+$ ]] || die "The snapshot of $DB is not a readable database."
    info "Snapshot taken: $tasks tasks, $(du -h "$SNAPSHOT" | cut -f1)."

    relocate_snapshot "$SNAPSHOT" "$LOCAL_DIR" "$REMOTE_DIR" || return 0

    if [[ ! -f "$SYNC_DB" ]]; then
        warn "$SYNC_DB is missing — uploading the database in full instead."
    elif [[ "$REMOTE_DB_STATE" == "ok" ]]; then
        step "Merging $tasks task(s) into $TARGET:~/.kiss/sorcar.db ..."
        if python3 "$SYNC_DB" "$SNAPSHOT" "$TARGET:~/.kiss/sorcar.db"; then
            info "Task database synced on $TARGET."
            return 0
        fi
        # Not "incomplete" yet: the fallback below can still put the two
        # machines in sync, and it says so itself when it cannot.
        warn "The merge into $TARGET failed — falling back to replacing its database."
    else
        info "No usable task database on $TARGET yet ($REMOTE_DB_STATE)."
    fi
    may_replace_remote_db || return 0
    upload_whole_db "$tasks"
}

# ---------------------------------------------------------------------------
# May the remote's database be thrown away and replaced by this machine's?
#
# Only when nothing it holds can be lost by doing so:
#
#   * ``missing``    -- there is no database there at all.
#   * ``unreadable`` -- not a database: no row can be read out of it by this
#                       script or by anything else, and the file itself is
#                       kept as a backup.
#   * ``incompatible`` -- a database without the two tables a merge needs.  It
#                       is kept as a backup, whole, and named in the warning.
#   * anything else  -- the remote has rows that a merge could not take, so
#                       replacing it would strand them.  The only case where
#                       that is acceptable is when pass 1 already brought them
#                       here, which is what PULL_COMPLETE records.
#
# A probe that did not run (``unknown``) says nothing about the remote, so it
# never authorises a replacement.
# ---------------------------------------------------------------------------
# Has the remote's database stood still since pass 1 read it?  Asked twice: once
# before the replacement is decided on, and again once the web app is stopped
# and nothing there can write any more -- the second answer is the one that
# makes the replacement safe rather than merely likely to be safe.
remote_unchanged_since_pull() {
    local now
    if [[ -z "$REMOTE_FINGERPRINT_AT_PULL" ]]; then
        incomplete "It is not recorded what $TARGET held when its tasks were" \
                   "brought here, so its database was left as it is."
        return 1
    fi
    now="$(remote_fingerprint '$HOME/.kiss/sorcar.db')"
    if [[ -z "$now" ]]; then
        incomplete "Cannot re-read what $TARGET holds, so it is not certain that" \
                   "the tasks brought back are still all of it: its database was" \
                   "left as it is."
        return 1
    fi
    if [[ "$now" != "$REMOTE_FINGERPRINT_AT_PULL" ]]; then
        incomplete "$TARGET has moved on since its tasks were brought here" \
                   "($REMOTE_FINGERPRINT_AT_PULL then, $now now — tasks, steps," \
                   "tokens, events, last event, what the tasks say), so replacing" \
                   "its database would" \
                   "throw that away: it was left as it is. Sync again to pick it up."
        return 1
    fi
}

may_replace_remote_db() {
    case "$REMOTE_DB_STATE" in
        missing) return 0 ;;
        unreadable|incompatible)
            warn "The database on $TARGET is $REMOTE_DB_STATE — replacing it," \
                 "and keeping it as ~/.kiss/sorcar.db.replaced-<time>."
            return 0
            ;;
    esac
    if (( PULL_COMPLETE )); then
        # ... as long as that is still true.  Pass 1 read the remote minutes
        # ago; anything that happened there since -- a task started, a task
        # carried further, an event written -- is something this machine has
        # never seen, and the replacement would throw it away.
        remote_unchanged_since_pull || return 1
        warn "Replacing the database on $TARGET wholesale; its own tasks are" \
             "already here, and it is kept as ~/.kiss/sorcar.db.replaced-<time>."
        return 0
    fi
    incomplete "Refusing to replace the task database on $TARGET: its tasks" \
               "could not be brought here first, and nothing on either machine" \
               "was changed. Start kiss-web on $TARGET once (it migrates its" \
               "own database on open) and deploy again."
    return 1
}

# ---------------------------------------------------------------------------
# Pass 2's fallback: swap the database in while nothing on the remote has it
# open.  The web app is brought back by the exit trap — including when the
# upload fails — so a deployment is never left without one.  ``pkill`` is
# anchored on the remote's home directory: only a web app running out of that
# home is a candidate, never some unrelated process (or, when this script is
# driven by a test harness, the developer's own).
# ---------------------------------------------------------------------------
upload_whole_db() {
    local tasks="$1" running backup
    # Stopping the web app kills the task it is running: the agent stops
    # mid-step and the steps it had left are gone.  A task counts as running
    # when its newest event is younger than the window (see
    # src/kiss/scripts/running_tasks.py); the rows an old database holds with
    # no end time and no recent event are not running tasks.
    # Only a database a reader can get a row out of can say anything about a
    # running task; the other states got here because nobody can.
    if [[ "${SORCAR_FORCE_RESTART:-}" != "1" && "$LIVE_TASK_WINDOW" != "0" \
          && "$REMOTE_DB_STATE" == "ok" ]]; then
        if [[ ! -f "$RUNNING_TASKS" ]]; then
            incomplete "$RUNNING_TASKS is missing, so it cannot be checked whether" \
                       "stopping the web app on $TARGET would kill a running task:" \
                       "its database was left as it is."
            return 0
        fi
        running="$(ssh "$TARGET" "python3 - \"\$HOME/.kiss/sorcar.db\" \
            $(shquote "$LIVE_TASK_WINDOW")" < "$RUNNING_TASKS" 2>/dev/null \
            | head -1 | tr -d '[:space:]' || true)"
        if [[ "$running" != "0" ]]; then
            incomplete "'${running:-no answer}' task(s) are running on $TARGET;" \
                       "replacing its database means stopping the web app, which" \
                       "would kill them. Nothing was changed on either machine —" \
                       "sync again when they are done, or set SORCAR_FORCE_RESTART=1."
            return 0
        fi
    fi
    step "Stopping the remote web app so nothing holds its database open ..."
    REMOTE_WEB_APP_STOPPED=1
    ssh "$TARGET" 'mkdir -p "$HOME/.kiss" && chmod 700 "$HOME/.kiss"
                   systemctl --user stop kiss-web.service >/dev/null 2>&1 || true
                   home_re=$(printf %s "$HOME" | sed "s/[][\\.^\$*+?(){}|\/]/\\\\&/g")
                   pkill -f "$home_re/.*kiss-web" >/dev/null 2>&1 || true
                   sleep 1
                   rm -f "$HOME"/.kiss/sorcar.db.incoming' \
        || die "Could not prepare $TARGET for the upload."

    # Nothing there can write any more, so this answer cannot go stale between
    # here and the swap.  It is the one that makes the replacement safe rather
    # than merely likely to be safe: whatever a last-second task wrote there
    # after pass 1 read the database is caught here, while the file it wrote it
    # to is still in place.
    if [[ "$REMOTE_DB_STATE" == "ok" ]] && ! remote_unchanged_since_pull; then
        return 0
    fi

    # gzip shrinks the database ~4x, so the upload takes a quarter as long.
    step "Uploading $tasks tasks to $TARGET ..."
    gzip -1 -c "$SNAPSHOT" \
        | ssh "$TARGET" 'gzip -dc > "$HOME/.kiss/sorcar.db.incoming"' \
        || die "Uploading the task database to $TARGET failed."

    if ! ssh "$TARGET" "EXPECTED='$tasks' bash -s" > "$TMP_DIR/swap.out" <<'SWAP'
set -e
cd "$HOME/.kiss"
got=$(python3 -c 'import sqlite3
con = sqlite3.connect("file:sorcar.db.incoming?mode=ro&immutable=1", uri=True)
if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
    raise SystemExit("the uploaded database is corrupt")
print(con.execute("SELECT count(*) FROM task_history").fetchone()[0])')
[ "$got" = "$EXPECTED" ] || { echo "ERROR: $got of $EXPECTED tasks arrived" >&2; exit 1; }
# The database being replaced is kept, not deleted.  The backup's name carries
# the time, so an earlier one -- possibly the only remaining copy of what it
# held -- is never written over.
#
# The backup is a second *link* to the same file rather than a rename of it, so
# the database stays where it is until the new one replaces it in one atomic
# step.  Renaming it away first leaves a window -- and, if what follows fails, a
# deployment with no database at all where a moment ago there was a full one.  A
# link costs no space; a filesystem that refuses one (some network mounts) gets
# a copy instead, and a backup that cannot be made at all stops the swap.
#
# The -wal comes first, before anything is kept: it holds pages the main file
# does not have yet, so a copy of the main file taken while they are still out
# there is a copy of an older database than the one being replaced.  Nothing
# holds the database open any more, so those pages can be folded in where they
# belong -- and whether they were is *read back* from the pragma rather than
# assumed: a checkpoint runs without error and still leaves frames behind when
# another reader holds an older view of the file (its first result is non-zero,
# "busy"), and deleting a -wal that was never folded in is losing exactly the
# rows that were only in it.
#
# A -wal that could not be folded in is kept beside the backup instead, where
# SQLite finds it again (it derives the -wal name from the file it opens), and
# it is *copied* there rather than moved, so that the database left behind is
# still complete if the last step below cannot be taken.  Either way the old
# -wal is gone before the new database takes its place: a -wal left beside a
# database it does not belong to is how a database gets corrupted.
if [ -f sorcar.db ]; then
    backup="sorcar.db.replaced-$(date -u +%Y%m%dT%H%M%SZ)"
    while [ -e "$backup" ]; do backup="$backup+"; done
    folded=1
    if [ -f sorcar.db-wal ]; then
        python3 -c 'import sqlite3, sys
con = sqlite3.connect("sorcar.db", timeout=30)
try:
    busy, frames, _ = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
finally:
    con.close()
# TRUNCATE leaves an empty log behind when it folded everything in, so
# anything else -- a busy reader, frames still in the log -- means it did not.
sys.exit(0 if busy == 0 and frames <= 0 else 1)' 2>/dev/null || folded=0
    fi
    ln sorcar.db "$backup" 2>/dev/null || cp -p sorcar.db "$backup" \
        || { echo "ERROR: cannot keep a copy of the database being replaced" >&2; exit 1; }
    if [ "$folded" = 0 ]; then
        cp -p sorcar.db-wal "$backup-wal" \
            || { echo "ERROR: cannot keep the -wal of the database being replaced" >&2; exit 1; }
    fi
    echo "SORCAR_DB_BACKUP=$backup"
fi
# The -shm is an index into the -wal, and the socket describes a process that is
# gone; neither says anything about the database arriving.
rm -f sorcar.db-wal sorcar.db-shm sorcar.sock
# The one step that takes the old database away.  If it cannot be taken, the old
# database is still here under its own name -- and gets its -wal back, so that
# what is here is the database that was here, whole.
if ! mv -f sorcar.db.incoming sorcar.db; then
    # No backup means there was no database here to put back.
    if [ -n "${backup:-}" ] && [ -f "$backup-wal" ]; then
        cp -p "$backup-wal" sorcar.db-wal || true
    fi
    echo "ERROR: cannot put the new database in place; the previous one is still here" >&2
    exit 1
fi
SWAP
    then
        die "The upload did not verify on $TARGET — its previous database was kept."
    fi
    backup="$(sed -n 's/^SORCAR_DB_BACKUP=//p' "$TMP_DIR/swap.out" | tail -1)"
    info "Task database replaced on $TARGET ($tasks tasks)."
    if [[ -n "$backup" ]]; then
        info "The database it replaced is kept as ~/.kiss/$backup."
        carry_over_counters "$backup"
    fi
}

# ---------------------------------------------------------------------------
# What the replacement does not bring with it
#
# A sync moves the two tables that hold a machine's history.  A sorcar.db has
# three more that no sync moves -- how often each model was chosen, each file
# opened, each task text run -- and a wholesale replacement would take those
# down with the file, in the one step that is supposed to lose nothing.  They
# are put back from the copy the replacement kept: rows only it had are
# inserted and every counter is raised to the larger of the two values, so
# running this twice changes nothing the second time.
#
# A failure here is worth saying out loud but not worth failing the sync over:
# the history -- what the two passes exist for -- is already in place.
# ---------------------------------------------------------------------------
carry_over_counters() {
    local backup="$1"
    # The name comes back from the remote, and goes into a remote command line.
    [[ "$backup" =~ ^sorcar\.db\.replaced-[0-9A-Za-z]+\+*$ ]] \
        || { warn "'$backup' is not a name this script gave a backup — leaving it alone."
             return 0; }
    [[ -f "$CARRY_OVER" ]] || { warn "$CARRY_OVER is missing — the usage counters" \
        "of the replaced database were left in ~/.kiss/$backup."; return 0; }
    if ! ssh "$TARGET" "python3 - \"\$HOME/.kiss/$backup\" \"\$HOME/.kiss/sorcar.db\"" \
            < "$CARRY_OVER"; then
        warn "Could not carry the usage counters of the replaced database over;" \
             "they are in ~/.kiss/$backup on $TARGET."
    fi
}

# ---------------------------------------------------------------------------
# The script the remote runs for pass 1's snapshot.  It is a file rather than
# a heredoc inside a command substitution so that a failing ssh can be caught
# instead of aborting the whole sync.
# ---------------------------------------------------------------------------
cat > "$TMP_DIR/remote-snapshot.py" <<'PY'
"""Copy this machine's live task database to a self-contained snapshot."""
import os
import sqlite3

database = os.path.expanduser("~/.kiss/sorcar.db")
snapshot = database + ".outgoing"
if os.path.exists(snapshot):
    os.unlink(snapshot)
con = sqlite3.connect(database, timeout=60)
con.execute("VACUUM INTO ?", (snapshot,))
con.close()
con = sqlite3.connect("file:" + snapshot + "?mode=ro", uri=True)
print(con.execute("SELECT count(*) FROM task_history").fetchone()[0])
con.close()
PY

# Preserve which live rows belonged to this machine before pull_from_remote
# makes the two histories indistinguishable.
capture_live_local_tasks

case "$REMOTE_DB_STATE" in
    ok) pull_from_remote ;;
    unknown)
        # The probe never ran, so this says nothing about the remote: there
        # may well be tasks there that this run should have brought back.
        incomplete "Could not read the task database on $TARGET at all" \
                   "— nothing was brought back, and nothing there was replaced."
        ;;
    *)
        info "No usable task database on $TARGET yet ($REMOTE_DB_STATE)" \
             "— nothing to bring back."
        ;;
esac

# The remote's snapshot has served its purpose; do not leave a copy of the
# database lying around while the upload below needs the same disk.
if (( REMOTE_SNAPSHOT_MADE )); then
    ssh "$TARGET" "rm -f \"$REMOTE_SNAPSHOT\"" >/dev/null 2>&1 \
        && REMOTE_SNAPSHOT_MADE=0 \
        || warn "Could not remove $TARGET:$REMOTE_SNAPSHOT yet."
fi

if [[ -f "$DB" ]]; then
    push_to_remote
else
    warn "No $DB on this machine — nothing to send to $TARGET."
fi

# A run that could not finish one of its two passes has not synced the two
# machines, and says so with its exit status: the caller decides whether that
# is worth stopping for.  Nothing has been lost either way — every step above
# either did what it said or left both databases as they were.
if (( INCOMPLETE )); then
    die "The task databases here and on $TARGET are not in sync yet" \
        "(see the warnings above)."
fi
info "The task databases here and on $TARGET hold the same tasks."
