#!/bin/bash
# End-to-end test of the remote half of ./rsorcar: the restart decision.
# Run: bash scripts/test_rsorcar_remote_restart.sh
#
# Runs the script rsorcar feeds the remote's ``bash -s`` (steps 5-7, the
# heredoc between ``<<'REMOTE'`` and ``REMOTE``) for real, in a fake remote:
# HOME is a scratch directory with the files step 4 ships there, the
# checkout is a scratch repository with a recorder install.sh, and uv,
# cloudflared, systemctl, loginctl, pkill and the public-URL wait are PATH
# stubs (systemctl records every call it gets).  The real
# src/kiss/scripts/running_tasks.py answers the running-task probe out of a
# real history.db.  What is checked:
#   * a task running on the remote (a row without end time, writing events
#     now) leaves kiss-web untouched: no stop, no restart, the published URL
#     kept, the deploy going on to its end and succeeding;
#   * no running task: kiss-web is restarted and the old URL dropped;
#   * FORCE_RESTART=1, a web app that is not running, or LIVE_TASK_WINDOW=0
#     restart it even with a task running;
#   * a database the probe cannot read is not read as "nothing is running":
#     kiss-web is left as it is, with a warning.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

# The remote script, as rsorcar ships it.
awk '/bash -s" <<'"'"'REMOTE'"'"'$/ {inside=1; next} /^REMOTE$/ {inside=0} inside' \
    "$REPO_ROOT/rsorcar" > "$WORK/remote.sh"
grep -q "7d. Wait for the local endpoint" "$WORK/remote.sh" \
    || fail "could not extract the remote script from rsorcar"

# --- One fake remote: $1 is the fixture dir, $2 the age in seconds of the
# newest event of the one task in its history.db ("" for a finished task,
# "corrupt" for a database that cannot be read).
make_remote() {
    local fix="$1" age="$2"
    mkdir -p "$fix/home/.kiss" "$fix/home/.local/bin" "$fix/bin" \
             "$fix/kiss/.venv/bin" "$fix/kiss/src/kiss/core"
    git -C "$fix/kiss" init -q
    printf '#!/bin/bash\necho "install.sh $*" >> "%s/install.log"\n' "$fix" > "$fix/kiss/install.sh"
    echo "def load_api_keys(): pass" > "$fix/kiss/src/kiss/core/user_config.py"
    printf '#!/bin/bash\nexit 0\n' > "$fix/kiss/.venv/bin/kiss-web"
    printf '#!/bin/bash\nexit 0\n' > "$fix/kiss/.venv/bin/python"
    chmod +x "$fix/kiss/install.sh" "$fix/kiss/.venv/bin/"*
    # What step 4 ships into ~/.kiss.
    cp "$REPO_ROOT/src/kiss/scripts/running_tasks.py" "$fix/home/.kiss/"
    printf 'import sys\nopen(sys.argv[1], "w").write("{}")\n' > "$fix/home/.kiss/remote_config.py"
    printf '#!/bin/bash\necho "SORCAR_PUBLIC_URL=https://fake.example"\n' \
        > "$fix/home/.kiss/wait-for-public-url.sh"
    # The URL the running web app published; a restart drops it.
    echo '{"url": "https://old.example"}' > "$fix/home/.kiss/remote-url.json"
    if [[ "$age" == corrupt ]]; then
        printf 'SQLite format 3\0' > "$fix/home/.kiss/history.db"
        head -c 512 /dev/zero >> "$fix/home/.kiss/history.db"
    else
        python3 - "$fix/home/.kiss/history.db" "$age" <<'PY'
import sqlite3, sys, time
con = sqlite3.connect(sys.argv[1])
con.executescript(
    "CREATE TABLE task_history(id TEXT PRIMARY KEY, timestamp REAL, task TEXT,"
    " end_ts INTEGER, result TEXT);"
    "CREATE TABLE events(task_id TEXT, seq INTEGER, event_json TEXT, timestamp REAL);")
now = time.time()
if sys.argv[2]:
    con.execute("INSERT INTO task_history VALUES ('T1', ?, 'a task', 0, 'Agent Failed Abruptly')", (now,))
    con.execute("INSERT INTO events VALUES ('T1', 0, '{}', ?)", (now - float(sys.argv[2]),))
else:
    con.execute("INSERT INTO task_history VALUES ('T1', ?, 'a task', ?, 'done')", (now, int(now)))
    con.execute("INSERT INTO events VALUES ('T1', 0, '{}', ?)", (now,))
con.commit()
PY
    fi
    # PATH stubs.  systemctl records what it is asked and answers is-active
    # with FAKE_SERVICE_RC (0: kiss-web is running).
    cat > "$fix/bin/systemctl" <<EOF
#!/bin/bash
echo "\$*" >> "$fix/systemctl.log"
case "\$*" in *is-active*) exit "\${FAKE_SERVICE_RC:-0}" ;; esac
exit 0
EOF
    for tool in uv cloudflared loginctl pkill; do
        printf '#!/bin/bash\nexit 0\n' > "$fix/bin/$tool"
    done
    chmod +x "$fix/bin/"*
}

# $1: fixture dir; the rest: extra VAR=value settings for the remote script.
run_remote() {
    local fix="$1"; shift
    env -i HOME="$fix/home" PATH="$fix/bin:$PATH" USER="${USER:-tester}" \
        REMOTE_DIR="$fix/kiss" WEB_PORT=8000 PASSWORD=pw LOCAL_NTFY_TOPIC='' \
        LIVE_TASK_WINDOW=300 FORCE_RESTART='' "$@" bash "$WORK/remote.sh" 2>&1
}

restarted() { grep -q "restart kiss-web.service" "$1/systemctl.log" 2>/dev/null; }
stopped() { grep -q "stop kiss-web.service" "$1/systemctl.log" 2>/dev/null; }

# --- Test 1: a running task leaves kiss-web alone, and the deploy succeeds ---
make_remote "$WORK/busy" 5
OUT=$(run_remote "$WORK/busy") || fail "the remote script failed with a task running:
$OUT"
restarted "$WORK/busy" && fail "kiss-web was restarted with a task running:
$OUT"
stopped "$WORK/busy" && fail "kiss-web was stopped with a task running:
$OUT"
echo "$OUT" | grep -q "'1' task(s) are running on this host; kiss-web is left running" \
    || fail "the deploy did not say why kiss-web was left running:
$OUT"
echo "$OUT" | grep -q "task T1 (last event [0-9]*s ago)" \
    || fail "the running task was not listed:
$OUT"
[[ -f "$WORK/busy/home/.kiss/remote-url.json" ]] \
    || fail "the running web app's published URL was dropped"
grep -q "install.sh --non-interactive" "$WORK/busy/install.log" \
    || fail "install.sh did not run on the remote: $(cat "$WORK/busy/install.log" 2>/dev/null)"
echo "$OUT" | grep -q "SORCAR_PUBLIC_URL=https://fake.example" \
    || fail "the deploy did not go on to the public URL after leaving kiss-web running:
$OUT"
pass "a task running on the remote keeps kiss-web as it is; the deploy still installs and succeeds"

# --- Test 2: nothing running: kiss-web is restarted ---------------------------
make_remote "$WORK/idle" ""
OUT=$(run_remote "$WORK/idle") || fail "the remote script failed with nothing running:
$OUT"
restarted "$WORK/idle" || fail "kiss-web was not restarted although nothing is running:
$OUT
$(cat "$WORK/idle/systemctl.log")"
[[ ! -e "$WORK/idle/home/.kiss/remote-url.json" ]] \
    || fail "the old published URL survived the restart"
echo "$OUT" | grep -q "is left running" && fail "a restart was reported as skipped:
$OUT"
pass "with no task running, kiss-web is restarted and the old URL dropped"

# --- Test 3: FORCE_RESTART=1 restarts over a running task --------------------
make_remote "$WORK/force" 5
OUT=$(run_remote "$WORK/force" FORCE_RESTART=1) || fail "the remote script failed with FORCE_RESTART=1:
$OUT"
restarted "$WORK/force" || fail "FORCE_RESTART=1 did not restart kiss-web:
$OUT"
pass "SORCAR_FORCE_RESTART=1 restarts kiss-web even with a task running"

# --- Test 4: a web app that is not running has no task to protect -------------
make_remote "$WORK/down" 5
OUT=$(run_remote "$WORK/down" FAKE_SERVICE_RC=3) || fail "the remote script failed with kiss-web down:
$OUT"
restarted "$WORK/down" || fail "a kiss-web that was not running was not started:
$OUT"
pass "a kiss-web that is not running is started whatever the database says"

# --- Test 5: LIVE_TASK_WINDOW=0 turns the check off -----------------------------
make_remote "$WORK/nowindow" 5
OUT=$(run_remote "$WORK/nowindow" LIVE_TASK_WINDOW=0) || fail "the remote script failed with LIVE_TASK_WINDOW=0:
$OUT"
restarted "$WORK/nowindow" || fail "LIVE_TASK_WINDOW=0 did not restart kiss-web:
$OUT"
pass "SORCAR_LIVE_TASK_WINDOW=0 restarts without looking"

# --- Test 6: a database the probe cannot read is not "nothing running" ---------
make_remote "$WORK/unreadable" corrupt
OUT=$(run_remote "$WORK/unreadable") || fail "the remote script failed on an unreadable database:
$OUT"
restarted "$WORK/unreadable" && fail "kiss-web was restarted although the probe had no answer:
$OUT"
echo "$OUT" | grep -q "WARNING: could not tell whether a task is running on this host" \
    || fail "no warning about the unreadable database:
$OUT"
pass "an unreadable database leaves kiss-web as it is, with a warning"

echo
echo "ALL TESTS PASSED"
