#!/bin/bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# Ship this machine's task database (~/.kiss/sorcar.db) to a remote host, so
# the web app running there lists every task this machine ever ran.
#
# Usage:  scripts/ship-task-db.sh user@ip-address [local-project-dir remote-project-dir]
#
# Called by ./sorcar-cloud on every deploy; also useful on its own to refresh
# a running deployment's history without redeploying.  Pass the two project
# directories to keep the History panel's "Workspace" chip working (see
# below); without them the tasks arrive but that chip hides them.
#
# When the remote already has a task database, only the *difference* travels:
# src/kiss/scripts/sync_db.py merges the missing task and event rows into the
# remote database in one transaction, while the remote web app keeps running
# and tasks that ran only on the remote are preserved.  The full upload below
# happens only when there is no usable remote database yet (first deploy) or
# when the delta sync cannot proceed (say, the two schemas have drifted
# apart) -- replacing the database wholesale is the fallback, not the rule.
#
# For the full-upload fallback, the whole list appears remotely only if the
# whole database arrives and the panel is willing to show it.  Four things
# get in the way, and all four are handled here:
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
#     fresh empty one; the upload then lands on that same inode, and the
#     first checkpoint afterwards writes that connection's stale, empty view
#     back over it — which is what left the remote History panel nearly
#     blank.  So the remote web app is stopped for the swap and started again
#     afterwards, whether the swap succeeded or not.
#   * A multi-gigabyte upload can be cut short.  The file therefore arrives
#     under a temporary name and is promoted only once it proves to hold
#     exactly the tasks that were sent, so a failed transfer keeps the
#     previous database instead of replacing it with a truncated one.
#   * Every task remembers the directory it ran in, and the History panel
#     hides tasks from other workspaces by default.  The deployment is this
#     project at a different path, so the recorded directories are re-pointed
#     at it — otherwise the imported history arrives complete and invisible.
#
# Needs python3 here and on the remote, and a SQLite new enough for
# ``VACUUM INTO`` (3.27, released 2019) on this machine.
set -euo pipefail

info() { printf '\033[0;32m[INFO]\033[0m  %s\n' "$*"; }
step() { printf '\033[0;34m[STEP]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()  { printf '\033[0;31m[ERR]\033[0m  %s\n' "$*" >&2; exit 1; }

TARGET="${1:-}"
LOCAL_DIR="${2:-}"
REMOTE_DIR="${3:-}"
[[ -n "$TARGET" ]] \
    || die "Usage: $0 user@ip-address [local-project-dir remote-project-dir]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_DB="$(dirname "$SCRIPT_DIR")/src/kiss/scripts/sync_db.py"

DB="${KISS_HOME:-$HOME/.kiss}/sorcar.db"
if [[ ! -f "$DB" ]]; then
    warn "No $DB on this machine — nothing to ship."
    exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
SNAPSHOT="$TMP_DIR/sorcar.db"

# ---------------------------------------------------------------------------
# A consistent, self-contained copy of the database
#
# timeout: the agent running on this machine writes to the same database, and
# the snapshot only needs to wait out its current commit.
# ---------------------------------------------------------------------------
step "Snapshotting $DB ..."
python3 -c 'import sqlite3, sys
con = sqlite3.connect(sys.argv[1], timeout=60)
con.execute("VACUUM INTO ?", (sys.argv[2],))
con.close()' "$DB" "$SNAPSHOT" \
    || die "Could not snapshot $DB (no space in $TMP_DIR, or a stuck writer?)."

TASKS="$(python3 -c 'import sqlite3, sys
print(sqlite3.connect(sys.argv[1]).execute(
    "SELECT count(*) FROM task_history").fetchone()[0])' "$SNAPSHOT" 2>/dev/null || true)"
[[ "$TASKS" =~ ^[0-9]+$ ]] || die "The snapshot of $DB is not a readable database."
info "Snapshot taken: $TASKS tasks, $(du -h "$SNAPSHOT" | cut -f1)."

# ---------------------------------------------------------------------------
# Re-point the recorded work directories at the remote checkout
#
# Only the paths inside the project move, and only on the copy being shipped.
# Databases written before the flat ``work_dir`` column existed keep the path
# inside their ``extra`` JSON, and the migration that runs on the remote's
# first open copies that value across verbatim — so those are rewritten here
# too, otherwise they would arrive already filtered out of view.
# ---------------------------------------------------------------------------
if [[ -n "$LOCAL_DIR" && -n "$REMOTE_DIR" ]]; then
    moved="$(python3 -c 'import json, sqlite3, sys
snapshot, local, remote = sys.argv[1:4]
local, remote = local.rstrip("/"), remote.rstrip("/")
if not local or not remote:
    print(0)                         # "/" is not a project directory
    raise SystemExit


def relocate(path):
    """Return *path* under the remote checkout, or None if it is elsewhere."""
    if path == local:
        return remote
    if path.startswith(local + "/"):
        return remote + path[len(local):]
    return None


con = sqlite3.connect(snapshot)
columns = {row[1] for row in con.execute("PRAGMA table_info(task_history)")}
moved = 0
if "work_dir" in columns:
    cur = con.execute(
        "UPDATE task_history SET work_dir = ? || substr(work_dir, ?) "
        "WHERE substr(work_dir, 1, ?) = ? "
        "AND (length(work_dir) = ? OR substr(work_dir, ?, 1) = ?)",
        (remote, len(local) + 1, len(local), local, len(local),
         len(local) + 1, "/"),
    )
    moved = cur.rowcount
elif "extra" in columns:             # the schema kiss-web migrates on open
    for row_id, extra in con.execute(
            "SELECT id, extra FROM task_history WHERE extra IS NOT NULL"):
        try:
            payload = json.loads(extra)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        moved_to = relocate(str(payload.get("work_dir", "")))
        if moved_to is None:
            continue
        payload["work_dir"] = moved_to
        con.execute("UPDATE task_history SET extra = ? WHERE id = ?",
                    (json.dumps(payload), row_id))
        moved += 1
con.commit()
print(moved)
con.close()' "$SNAPSHOT" "$LOCAL_DIR" "$REMOTE_DIR")" \
        || die "Could not relocate the recorded work directories."
    info "Relocated $moved task(s) from $LOCAL_DIR to $REMOTE_DIR."
fi

# ---------------------------------------------------------------------------
# Preferred path: delta-sync into the remote database, in place
#
# sync_db.py asks the remote what it already has, ships only the missing
# task and event rows, and merges them in one transaction -- so a repeated
# deploy moves kilobytes instead of gigabytes, the remote web app never has
# to stop, and tasks that ran only on the remote survive.  It needs an
# existing remote database with the two synced tables, so probe first; a
# missing or unreadable database (first deploy, torn earlier upload) or a
# failed sync (schemas drifted apart) falls through to the full upload.
# ---------------------------------------------------------------------------
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

if [[ ! -f "$SYNC_DB" ]]; then
    warn "$SYNC_DB is missing — uploading the database in full instead."
elif [[ "$REMOTE_DB_STATE" == "ok" ]]; then
    step "Delta-syncing $TASKS tasks into $TARGET:~/.kiss/sorcar.db ..."
    if python3 "$SYNC_DB" "$SNAPSHOT" "$TARGET:~/.kiss/sorcar.db"; then
        info "Task database synced on $TARGET."
        exit 0
    fi
    warn "Delta sync failed — replacing the remote database wholesale instead" \
         "(tasks that ran only on $TARGET will be lost)."
else
    info "No usable task database on $TARGET yet ($REMOTE_DB_STATE) — uploading in full."
fi

# ---------------------------------------------------------------------------
# Fallback: swap the database in while nothing on the remote has it open
#
# The web app is brought back on the way out — including when the upload
# fails — so a deployment is never left without one.  ``pkill`` is anchored
# on the remote's home directory: only a web app running out of that home is
# a candidate, never some unrelated process (or, when this script is driven
# by a test harness, the developer's own).
# ---------------------------------------------------------------------------
start_remote_web_app() {
    ssh "$TARGET" 'systemctl --user start kiss-web.service >/dev/null 2>&1 || true' \
        || warn "Could not start the web app on $TARGET — start it by hand."
}

step "Stopping the remote web app so nothing holds its database open ..."
# Armed before the stop, not after: a connection that drops midway through
# the command below may already have stopped the service.
trap 'start_remote_web_app; rm -rf "$TMP_DIR"' EXIT
ssh "$TARGET" 'mkdir -p "$HOME/.kiss" && chmod 700 "$HOME/.kiss"
               systemctl --user stop kiss-web.service >/dev/null 2>&1 || true
               home_re=$(printf %s "$HOME" | sed "s/[][\\.^\$*+?(){}|\/]/\\\\&/g")
               pkill -f "$home_re/.*kiss-web" >/dev/null 2>&1 || true
               sleep 1
               rm -f "$HOME"/.kiss/sorcar.db.incoming' \
    || die "Could not prepare $TARGET for the upload."

# gzip shrinks the database ~4x, so the upload takes a quarter as long.
step "Uploading $TASKS tasks to $TARGET ..."
gzip -1 -c "$SNAPSHOT" \
    | ssh "$TARGET" 'gzip -dc > "$HOME/.kiss/sorcar.db.incoming"' \
    || die "Uploading the task database to $TARGET failed."

ssh "$TARGET" "EXPECTED='$TASKS' bash -s" <<'SWAP' \
    || die "The upload did not verify on $TARGET — its previous database was kept."
set -e
cd "$HOME/.kiss"
got=$(python3 -c 'import sqlite3
con = sqlite3.connect("file:sorcar.db.incoming?mode=ro&immutable=1", uri=True)
if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
    raise SystemExit("the uploaded database is corrupt")
print(con.execute("SELECT count(*) FROM task_history").fetchone()[0])')
[ "$got" = "$EXPECTED" ] || { echo "ERROR: $got of $EXPECTED tasks arrived" >&2; exit 1; }
# These belong to the database being replaced: a leftover -wal would be
# replayed into the new file, and -shm/.sock describe a process that is gone.
rm -f sorcar.db-wal sorcar.db-shm sorcar.sock
mv -f sorcar.db.incoming sorcar.db
SWAP
info "Task database replaced on $TARGET ($TASKS tasks)."
