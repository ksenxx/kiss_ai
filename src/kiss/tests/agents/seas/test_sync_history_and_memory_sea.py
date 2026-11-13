# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``/sync_history_and_memory`` SEA and the engine behind it.

The engine is run for real against sandbox "machines": a fake ``ssh`` on
PATH runs the remote half locally with ``HOME`` pointing at a directory
per host, so every phase -- the shipped scripts, the sync_db phases, the
pack and merge streams -- is the real code end to end; only the network
is missing.  The databases have the web app's real schema
(``persistence._init_tables``) and the pages are written by ``MemoryDir``.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import sqlite3
import stat
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as persistence
from kiss.agents.seas.sync_history_and_memory import (
    memory_delta,
    peer_sync,
)
from kiss.agents.seas.sync_history_and_memory import (
    sync_history_and_memory_sea as sea_module,
)
from kiss.agents.seas.sync_history_and_memory.peer_sync import PeerSync
from kiss.agents.seas.sync_history_and_memory.ssh_peer import PeerError, SshPeer, control_directory
from kiss.agents.seas.sync_history_and_memory.sync_history_and_memory_sea import (
    SyncHistoryAndMemorySea,
    format_reports,
    sync_history_and_memory,
)
from kiss.agents.sorcar import history_columns, sea_commands
from kiss.agents.sorcar import machines as machine_registry
from kiss.agents.sorcar.machines import Machine, save_machines
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.memoryfield.pages import MemoryDir
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_worker_contract,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body
from kiss.tests.conftest import posix_only, wait_for

pytestmark = posix_only("drives the engine through a bash ssh stand-in")

# The stand-in ssh: options are dropped (``-o X``, ``-p N``, ``-O exit`` take
# a value; ``-N``, ``-f`` do not), the destination picks the fake machine's
# HOME under $REMOTE_ROOT, the remote command runs in bash.  The control
# socket is a plain file: created by the master (``-N``), removed by
# ``-O exit``.  A destination named ``dead`` refuses the connection as a
# real client would (exit 255); ``flaky<N>`` dies after the number of calls
# in $FLAKY_AFTER_flaky<N>.  A machine with a ``pysite`` directory in its
# home runs python with it on PYTHONPATH, so a ``sitecustomize.py`` there
# can skew the machine's clock.  A ``<host>.gate`` file under $REMOTE_ROOT
# holds the machine's commands until the test removes it; the first one
# to wait there creates ``<host>.gate.reached``.
FAKE_SSH = """#!/bin/bash
socket=""; master=no; control=""
while [ $# -gt 0 ]; do
    case "$1" in
        -o) case "$2" in ControlPath=*) : "${socket:=${2#ControlPath=}}" ;; esac; shift 2 ;;
        -O) control="$2"; shift 2 ;;
        -p|-i|-l|-F|-c) shift 2 ;;
        -N) master=yes; shift ;;
        -*) shift ;;
        *) break ;;
    esac
done
host="${1#*@}"
shift
case "$host" in
    dead) echo "ssh: connect to host dead: Connection refused" >&2; exit 255 ;;
    flaky*)
        # One byte appended per call, never a rewrite: calls made at once are each counted.
        # A call is refused once the count, its own byte included, is past the limit.
        printf x >> "$REMOTE_ROOT/$host.count"
        count=$(( $(wc -c < "$REMOTE_ROOT/$host.count") - 1 ))
        limit_var="FLAKY_AFTER_$host"
        if [ "$count" -ge "${!limit_var:-99}" ]; then
            echo "ssh: connection to $host closed" >&2; exit 255
        fi ;;
esac
if [ "$master" = yes ]; then touch "$socket"; exit 0; fi
if [ "$control" = exit ]; then rm -f "$socket"; exit 0; fi
gate="$REMOTE_ROOT/$host.gate"
if [ -e "$gate" ]; then touch "$gate.reached"; while [ -e "$gate" ]; do sleep 0.05; done; fi
export HOME="$REMOTE_ROOT/$host"
[ -d "$HOME/pysite" ] && export PYTHONPATH="$HOME/pysite"
exec bash -c "$*"
"""

PLACEHOLDER = peer_sync.UNFINISHED_RESULT


class Sandbox:
    """A local KISS home and one fake remote machine per host, reached through the fake ssh."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch, *hosts: str) -> None:
        """Lay out the homes and put the fake ssh on PATH."""
        self.root = root
        self.home = root / "local" / ".kiss"
        self.home.mkdir(parents=True)
        bindir = root / "bin"
        bindir.mkdir()
        fake = bindir / "ssh"
        fake.write_text(FAKE_SSH)
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("REMOTE_ROOT", str(root / "remotes"))
        monkeypatch.setenv("KISS_HOME", str(self.home))
        # The host names a sync learns are also kept in this process-wide map: a name
        # an earlier test learnt would not be saved to this home, nor ours forgotten.
        monkeypatch.setattr(machine_registry, "_hosts", {})
        self.hosts = hosts
        for host in hosts:
            (root / "remotes" / host / ".kiss").mkdir(parents=True)

    def machines(self, *hosts: str) -> list[Machine]:
        """Registered ssh machines for *hosts* (all of them by default), named after the host."""
        return [Machine(host, ssh=f"me@{host}") for host in (hosts or self.hosts)]

    def kiss_dir(self, host: str = "") -> Path:
        """The KISS home of *host*, or the local one."""
        return self.root / "remotes" / host / ".kiss" if host else self.home

    def db(self, host: str = "") -> Path:
        """The history database of *host*, or the local one."""
        return self.kiss_dir(host) / "history.db"

    def memory(self, host: str = "") -> MemoryDir:
        """The memory of *host*, or the local one."""
        return MemoryDir(self.kiss_dir(host) / "memories")

    def run(self, dry_run: bool = False) -> list[peer_sync.PeerReport]:
        """Sync the local home with every host and return the reports."""
        return PeerSync(self.machines(), dry_run=dry_run, home=self.home).run()


def make_db(path: Path, tasks: dict[str, dict[str, Any]]) -> None:
    """Create a history database with the real schema holding *tasks*.

    Each task maps its id to options: ``events`` (count, default 3),
    ``result`` (default ``"done"``), ``end_ts`` (default 2.0),
    ``timestamp`` (start, default 1.0), ``event_ts`` (the events' time,
    default the start).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    persistence._init_tables(conn)
    for task_id, options in tasks.items():
        start = float(options.get("timestamp", 1.0))
        events = int(options.get("events", 3))
        event_ts = float(options.get("event_ts", start))
        conn.execute(
            "INSERT INTO task_history(id, timestamp, task, result, end_ts, has_events)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                task_id,
                start,
                f"task {task_id}",
                options.get("result", "done"),
                options.get("end_ts", 2.0),
                int(events > 0),
            ),
        )
        for seq in range(events):
            conn.execute(
                "INSERT INTO events(task_id, seq, event_json, timestamp) VALUES (?, ?, ?, ?)",
                (task_id, seq, json.dumps({"task": task_id, "seq": seq}), event_ts),
            )
    conn.commit()
    conn.close()


def tasks_in(path: Path) -> dict[str, int]:
    """Return every task id of a database mapped to its number of events."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT t.id, COUNT(e.seq) FROM task_history t"
                " LEFT JOIN events e ON e.task_id = t.id GROUP BY t.id"
            )
        }
    finally:
        conn.close()


def finish_task(path: Path, task_id: str) -> None:
    """End a running task the way the web app does: result and end time written, events old."""
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE task_history SET result = 'done', end_ts = ? WHERE id = ?", (time.time(), task_id)
    )
    conn.execute("UPDATE events SET timestamp = ? WHERE task_id = ?", (time.time() - 3600, task_id))
    conn.commit()
    conn.close()


def seqs_of(path: Path, task_id: str) -> list[int]:
    """Return the ``seq`` of every event of *task_id* in a database, in order."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT seq FROM events WHERE task_id = ? ORDER BY seq", (task_id,)
        ).fetchall()
    finally:
        conn.close()
    return [int(row[0]) for row in rows]


def page_text(memory: MemoryDir, name: str) -> str:
    """The body of a page, or ``""`` when the page is not there."""
    path = memory.root / f"{name}.md"
    return path.read_text().split("---\n", 2)[-1] if path.is_file() else ""


def stamp_page(memory: MemoryDir, name: str, when: str) -> None:
    """Rewrite the ``updated`` stamp of a page (``YYYY-mm-ddTHH:MM:SSZ``)."""
    path = memory.root / f"{name}.md"
    text = path.read_text()
    head, _, rest = text.partition("updated:")
    _, _, tail = rest.partition("\n")
    path.write_text(f"{head}updated: {when}\n{tail}")


# ---------------------------------------------------------------- contract


def test_sea_methods_follow_the_user_contract() -> None:
    """The SEA pins the run: no built-in tools, its own prompt and system prompt, one tool."""
    sea = SyncHistoryAndMemorySea()
    assert_worker_contract(
        sea_module,
        sea,
        {
            "tool_profile": "none",
            "max_budget": sea_module.MAX_BUDGET_USD,
            "locked": ["tool_profile"],
        },
    )
    assert "history" in sea.description() and "/sync_history_and_memory" in sea.description()
    assert sea.prompt("") == sea_module.ADD_TO_PROMPT
    assert sea.prompt("  dry run ") == f"dry run\n\n{sea_module.ADD_TO_PROMPT}"
    assert sea.system_prompt("ignored") == sea_module.SYSTEM_PROMPT

    def other() -> None: ...

    assert sea.tools([other]) == [other, sync_history_and_memory]
    assert sea_commands.bundled_commands()["sync_history_and_memory"] == Path(sea_module.__file__)


# ---------------------------------------------------------------- the engine


def test_every_machine_ends_up_with_the_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disjoint histories and memories on three machines converge in one run, nothing lost."""
    box = Sandbox(tmp_path, monkeypatch, "alpha", "beta")
    make_db(box.db(), {"L1": {}, "L2": {"events": 5}})
    make_db(box.db("alpha"), {"A1": {"events": 7}})
    make_db(box.db("beta"), {"B1": {}, "B2": {"events": 0}})
    box.memory().write("local-page", "mine\n")
    box.memory("alpha").write("alpha-page", "alpha\n")
    MemoryDir(box.memory("beta").root / "kiss").write("repo-note", "beta's note\n")

    reports = box.run()

    expected = {"L1": 3, "L2": 5, "A1": 7, "B1": 3, "B2": 0}
    for host in ("", "alpha", "beta"):
        assert tasks_in(box.db(host)) == expected, host
    prints = [memory_delta.fingerprints(box.memory(host).root) for host in ("", "alpha", "beta")]
    assert prints[0] == prints[1] == prints[2]
    assert set(prints[0]) == {"local-page.md", "alpha-page.md", "kiss/repo-note.md"}
    assert page_text(box.memory("alpha"), "local-page") == "mine\n"

    alpha, beta = reports
    assert alpha.error == "" and beta.error == ""
    # Each machine's host name is saved, so the CHATS panel can name the
    # rows pulled from it before its daemon ever answers (the fake ssh
    # runs the peers here, on this host).
    saved = machine_registry._saved_hosts()
    assert saved["alpha"] == saved["beta"] == history_columns.host_name()
    assert alpha.pulled["tasks_inserted"] == 1 and alpha.pulled["events_inserted"] == 7
    assert beta.pulled["tasks_inserted"] == 2
    # Pushed after both pulls: each machine receives what the other brought.
    assert alpha.pushed["tasks_inserted"] == 4 and beta.pushed["tasks_inserted"] == 3
    assert alpha.memory_pulled["entries"] == 1 and alpha.memory_pulled["counts"]["added"] == 1
    assert alpha.memory_pushed["entries"] == 2 and alpha.memory_pushed["counts"]["added"] == 2
    assert beta.memory_pushed["entries"] == 2
    assert alpha.conflicts == [] and alpha.seconds > 0

    # A second run has nothing to carry.
    again = box.run()
    assert [report.name for report in again] == ["alpha", "beta"]
    for report in again:
        assert report.pulled["events_inserted"] == 0 and report.pushed["events_inserted"] == 0
        assert report.memory_pulled == {"entries": 0} and report.memory_pushed == {"entries": 0}


def test_branded_and_stock_installs_sync_with_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer's KISS home is found: its daemon's, the brand's, the stock ``~/.kiss``, a new one."""
    box = Sandbox(tmp_path, monkeypatch, "stock", "branded", "fresh", "multi", "odd")
    remotes = tmp_path / "remotes"
    make_db(box.db(), {"L1": {}})
    box.memory().write("local-page", "mine\n")
    # stock: a stock install, plus a stray memory an older sync left under the brand's name.
    make_db(box.db("stock"), {"S1": {}})
    MemoryDir(remotes / "stock" / ".brand" / "memories").write("stray", "old\n")
    # branded: the same brand as this machine, and a leftover empty stock directory.
    make_db(remotes / "branded" / ".brand" / "history.db", {"B1": {}})
    # fresh: has never run anything.
    (remotes / "fresh" / ".kiss").rmdir()
    # multi: two daemons of another brand; the registered one (port 8787) owns .other-b,
    # which beats a stock .kiss holding a history.
    make_db(box.db("multi"), {"K1": {}})
    for name, port in ((".other-a", 9999), (".other-b", 8787)):
        make_db(remotes / "multi" / name / "history.db", {f"M{port}": {}})
        (remotes / "multi" / name / "sorcar-local.json").write_text(
            json.dumps({"url": f"ws://127.0.0.1:{port}/ws"})
        )
    machines = box.machines("stock", "branded", "fresh")
    machines.append(Machine("multi", url="https://multi:8787", ssh="me@multi"))

    stock, branded, fresh, multi = PeerSync(
        machines, home=box.home, home_candidates=(".brand", ".kiss")
    ).run()

    homes = (stock.home, branded.home, fresh.home, multi.home)
    assert homes == ("~/.kiss", "~/.brand", "~/.brand", "~/.other-b")
    assert stock.error == branded.error == fresh.error == multi.error == ""
    assert tasks_in(remotes / "multi" / ".other-a" / "history.db") == {"M9999": 3}
    assert tasks_in(box.db("multi")) == {"K1": 3}
    union = {"L1": 3, "S1": 3, "B1": 3, "M8787": 3}
    assert tasks_in(remotes / "multi" / ".other-b" / "history.db") == union
    assert tasks_in(box.db()) == union and tasks_in(box.db("stock")) == union
    assert tasks_in(remotes / "branded" / ".brand" / "history.db") == union
    # A machine that never ran a task has no database: its history waits, its memory syncs.
    assert "database not found" in fresh.pushed["skipped"]
    assert page_text(MemoryDir(remotes / "fresh" / ".brand" / "memories"), "local-page")
    assert page_text(box.memory("stock"), "local-page") == "mine\n"
    assert page_text(box.memory(), "stray") == ""
    assert page_text(MemoryDir(remotes / "branded" / ".brand" / "memories"), "local-page")
    assert not (remotes / "branded" / ".kiss" / "memories").exists()

    # A home that is not one directory under $HOME is refused, not guessed at.
    (odd,) = PeerSync(box.machines("odd"), home=box.home, home_candidates=("a/b",)).run()
    assert odd.error == "odd (me@odd): unexpected KISS home 'a/b'" and odd.home == ""


def test_running_tasks_wait_for_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task in flight on either side stays where it is; it travels once it has ended."""
    box = Sandbox(tmp_path, monkeypatch, "alpha")
    now = time.time()
    make_db(
        box.db(),
        {
            "done-here": {},
            # Running here: the start placeholder and a fresh event.
            "running-here": {
                "result": PLACEHOLDER,
                "end_ts": 0,
                "timestamp": now - 60,
                "event_ts": now - 5,
            },
        },
    )
    make_db(
        box.db("alpha"),
        {
            "done-there": {},
            # Running there without the placeholder (an older daemon): a fresh event is enough.
            "busy-there": {
                "result": "",
                "end_ts": 0,
                "timestamp": now - 900,
                "event_ts": now - 100,
            },
            # The placeholder a day and more after the last event marks a dead agent: history.
            "died-there": {
                "result": PLACEHOLDER,
                "end_ts": 0,
                "timestamp": now - 3 * 86400,
                "event_ts": now - 2 * 86400,
            },
        },
    )

    [report] = box.run()

    assert report.error == ""
    assert report.held_back_there == ["busy-there"]
    assert report.held_back_here == ["running-here"]
    assert set(tasks_in(box.db())) == {"done-here", "running-here", "done-there", "died-there"}
    assert set(tasks_in(box.db("alpha"))) == {"done-there", "busy-there", "died-there", "done-here"}

    finish_task(box.db(), "running-here")
    finish_task(box.db("alpha"), "busy-there")
    [report] = box.run()
    assert report.held_back_there == [] and report.held_back_here == []
    assert (
        set(tasks_in(box.db()))
        == set(tasks_in(box.db("alpha")))
        == {"done-here", "running-here", "done-there", "busy-there", "died-there"}
    )


def test_unfinished_filter_reads_the_sources_clock_and_odd_schemas(tmp_path: Path) -> None:
    """The hold-back judges by the clock given, and copes with deltas missing tables or columns."""
    delta = tmp_path / "delta.db"
    make_db(delta, {"t": {"result": "", "end_ts": 0, "timestamp": 1000.0, "event_ts": 1000.0}})
    assert peer_sync.unfinished_tasks(str(delta), now=1100.0) == ["t"]
    assert peer_sync.unfinished_tasks(str(delta), now=1000.0 + peer_sync.RUNNING_WINDOW + 1) == []
    # Events alone (the target has the task's row already) are never held back: the row is
    # in place and events only ever append, so fresh ones are no sign of an unfinished row.
    conn = sqlite3.connect(delta)
    conn.execute("DELETE FROM task_history")
    conn.commit()
    conn.close()
    assert peer_sync.unfinished_tasks(str(delta), now=1100.0) == []
    conn = sqlite3.connect(delta)
    conn.execute("DROP TABLE task_history")
    conn.commit()
    conn.close()
    assert peer_sync.unfinished_tasks(str(delta), now=1100.0) == []
    # A delta of an ancient schema: no events table, task rows without end_ts or result.
    bare = tmp_path / "bare.db"
    conn = sqlite3.connect(bare)
    conn.execute("CREATE TABLE task_history (id TEXT PRIMARY KEY, timestamp REAL)")
    conn.execute("INSERT INTO task_history VALUES ('fresh', 5000.0), ('old', 1.0)")
    conn.commit()
    conn.close()
    assert peer_sync.unfinished_tasks(str(bare), now=5010.0) == ["fresh"]
    held = peer_sync.UnfinishedFilter(now=5010.0)
    held(str(bare))
    assert held.held == ["fresh"]
    conn = sqlite3.connect(bare)
    assert [r[0] for r in conn.execute("SELECT id FROM task_history")] == ["old"]
    conn.close()
    empty = peer_sync.UnfinishedFilter(now=1e9)
    empty(str(bare))
    assert empty.held == []


def test_memory_rules_newest_wins_deletions_travel_conflicts_stay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """merge_memory_pages' rules: newer stamp wins, tombstones delete, the same moment conflicts."""
    box = Sandbox(tmp_path, monkeypatch, "alpha")
    mine, theirs = box.memory(), box.memory("alpha")
    for memory in (mine, theirs):
        memory.write("shared", "v1\n")
        memory.write("doomed", "to be deleted\n")
        stamp_page(memory, "doomed", "2020-01-01T00:00:00Z")  # written well before its deletion
        memory.write("tie", "same stamp\n")
        stamp_page(memory, "tie", "2020-01-01T00:00:00Z")  # two writes may straddle a second
    # Newer on their side; the same stamp on both sides, different text; deleted on ours.
    theirs.write("shared", "v2 from alpha\n")
    stamp_page(theirs, "shared", "2030-01-01T00:00:00Z")
    stamp_page(mine, "shared", "2020-01-01T00:00:00Z")
    (mine.root / "tie.md").write_text(
        (mine.root / "tie.md").read_text().replace("same stamp", "mine")
    )
    (theirs.root / "tie.md").write_text(
        (theirs.root / "tie.md").read_text().replace("same stamp", "theirs")
    )
    when = time.time() - 60
    os.utime(mine.root / "tie.md", (when, when))
    os.utime(theirs.root / "tie.md", (when, when))
    mine.delete("doomed")
    # Debris and a stray file never travel.
    (theirs.root / "notes.txt").write_text("not a page\n")
    (theirs.root / ".DS_Store").write_bytes(b"\0")

    [report] = box.run()

    # Neither side has run a task yet: no history.db, the history is skipped, the memory syncs.
    assert report.error == ""
    assert report.pulled["skipped"].startswith("database not found") and "skipped" in report.pushed
    assert "history: pulled nothing (database not found" in format_reports([report], dry_run=False)
    assert page_text(mine, "shared") == page_text(theirs, "shared") == "v2 from alpha\n"
    assert not (theirs.root / "doomed.md").exists() and not (mine.root / "doomed.md").exists()
    assert (theirs.root / ".tombstones" / "doomed").is_file()
    assert "mine" in page_text(mine, "tie") and "theirs" in page_text(theirs, "tie")
    assert report.conflicts == ["tie"]
    # "kept": their copy of the deleted page came over and was kept out by our newer tombstone.
    assert report.memory_pulled["counts"] == {
        "added": 0,
        "updated": 1,
        "kept": 1,
        "conflicts": 1,
        "removed": 0,
    }
    assert report.memory_pushed["counts"]["removed"] == 1
    assert not (mine.root / "notes.txt").exists()
    prints = memory_delta.fingerprints(mine.root)
    assert set(prints) == {"shared.md", "tie.md", ".tombstones/doomed"}


def test_new_events_of_a_finished_task_both_sides_hold_travel_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh events of a task whose row both sides have are not mistaken for a running task."""
    box = Sandbox(tmp_path, monkeypatch, "alpha")
    now = time.time()
    make_db(box.db(), {"done": {"events": 1, "end_ts": now - 10, "event_ts": now - 20}})
    make_db(box.db("alpha"), {"done": {"events": 3, "end_ts": now - 10, "event_ts": now - 20}})

    [report] = box.run()

    assert report.error == ""
    assert report.held_back_there == [] and report.held_back_here == []
    assert tasks_in(box.db()) == tasks_in(box.db("alpha")) == {"done": 3}


def test_a_stale_tombstone_beside_a_live_page_does_not_delete_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted delete leaves a newer tombstone beside its page; alone it would delete."""
    box = Sandbox(tmp_path, monkeypatch, "alpha")
    mine, theirs = box.memory(), box.memory("alpha")
    mine.write("note", "kept\n")
    theirs.root.mkdir()
    (theirs.root / "note.md").write_bytes((mine.root / "note.md").read_bytes())  # the same page
    (theirs.root / ".tombstones").mkdir()
    (theirs.root / ".tombstones" / "note").write_text(
        "2099-01-01T00:00:00Z\n"
    )  # newer than the page
    assert set(memory_delta.fingerprints(theirs.root)) == {"note.md"}

    [report] = box.run()

    assert report.error == ""
    assert report.memory_pulled == {"entries": 0} and report.memory_pushed == {"entries": 0}
    assert page_text(mine, "note") == page_text(theirs, "note") == "kept\n"


FAST_CLOCK = "import time\n_real = time.time\ntime.time = lambda: _real() + 120.0\n"


def test_pages_relayed_through_a_fast_clock_become_conflicts_not_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One tolerance, the spread of all clocks, serves every merge: a relayed stamp cannot win.

    alpha's clock is 120 s fast.  alpha wrote the page first (its stamp says
    00:02:00), beta wrote it a minute later (stamp 00:01:00, true clock).
    Whichever is pulled first, the other must come out a conflict, not be
    overwritten by the stamp from the fast clock.
    """
    box = Sandbox(tmp_path, monkeypatch, "alpha", "beta")
    site = box.kiss_dir("alpha").parent / "pysite"
    site.mkdir()
    (site / "sitecustomize.py").write_text(FAST_CLOCK)
    alpha, beta = box.memory("alpha"), box.memory("beta")
    alpha.write("note", "alpha's older words\n")
    stamp_page(alpha, "note", "2030-01-01T00:02:00Z")
    beta.write("note", "beta's newer words\n")
    stamp_page(beta, "note", "2030-01-01T00:01:00Z")

    alpha_report, beta_report = box.run()

    assert alpha_report.error == "" and beta_report.error == ""
    assert 110 < alpha_report.skew < 130 and abs(beta_report.skew) < 10
    assert page_text(alpha, "note") == "alpha's older words\n"
    assert page_text(beta, "note") == "beta's newer words\n"
    assert page_text(box.memory(), "note") in {"alpha's older words\n", "beta's newer words\n"}
    assert "note" in alpha_report.conflicts + beta_report.conflicts
    assert f"clock skew: {alpha_report.skew:+.1f}s" in format_reports(
        [alpha_report, beta_report], dry_run=False
    )


def test_a_merge_waits_for_an_agents_write_and_gives_up_after_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge holds the memory's and the domain memories' mutation locks, as MemoryDir does."""
    memory = MemoryDir(tmp_path / "memories")
    memory.write("note", "old\n")
    stamp_page(memory, "note", "2020-01-01T00:00:00Z")
    other = MemoryDir(tmp_path / "other")
    other.write("note", "incoming\n")
    stamp_page(other, "note", "2020-01-02T00:00:00Z")
    MemoryDir(other.root / "dom").write("deep", "deep\n")
    stream = io.BytesIO()
    memory_delta.pack(other.root, ["note.md", "dom/deep.md"], stream)
    results: list[dict[str, Any]] = []
    root_lock = Path(f"{memory.root}.lock")
    at_root_lock = threading.Event()
    real_lock = memory_delta._lock

    def noticed_lock(descriptor: int, path: Path) -> None:
        """The merge's own lock acquisition, flagging the test when it reaches the memory's lock."""
        if path == root_lock:
            at_root_lock.set()
        real_lock(descriptor, path)

    monkeypatch.setattr(memory_delta, "_lock", noticed_lock)

    def merge() -> None:
        results.append(
            memory_delta.unpack_and_merge(io.BytesIO(stream.getvalue()), memory.root, 0.0)
        )

    # An agent is in the middle of a write: it holds the lock MemoryDir.write takes.
    with exclusive_file_lock(root_lock, timeout=5.0):
        # daemon: a merge that never returns fails the joins below, not pytest's exit
        worker = threading.Thread(target=merge, daemon=True)
        worker.start()
        # The merge reaches the memory's lock while the agent holds it, and waits there:
        # the domain lock it takes next is not even created yet.
        assert at_root_lock.wait(10.0), "the merge must take the memory's lock"
        worker.join(0.5)
        assert worker.is_alive(), "the merge must wait for the agent's write"
        assert not (memory.root / "dom.lock").exists()
        memory.page_path("note").write_text(
            memory.page_path("note").read_text().replace("old", "agent's")
        )
        stamp_page(memory, "note", "2030-01-01T00:00:00Z")
    worker.join(10.0)
    assert not worker.is_alive()

    # The merge compared against the page as the agent left it: the write survived.
    assert page_text(memory, "note") == "agent's\n"
    assert page_text(MemoryDir(memory.root / "dom"), "deep") == "deep\n"
    [result] = results
    assert result["received"] == 2 and result["counts"]["kept"] == 1
    assert result["counts"]["added"] == 1 and (memory.root / "dom.lock").is_file()

    # A lock held past the timeout: the merge gives up rather than write under it.
    monkeypatch.setattr(memory_delta, "LOCK_TIMEOUT", 0.3)
    with (
        exclusive_file_lock(memory.root / "dom.lock", timeout=5.0),
        pytest.raises(TimeoutError, match="dom.lock still locked"),
    ):
        memory_delta.unpack_and_merge(io.BytesIO(stream.getvalue()), memory.root, 0.0)
    assert page_text(MemoryDir(memory.root / "dom"), "deep") == "deep\n"
    # The memory's own lock too (the domain locks are free): a merge that skipped
    # it would go through here, whatever the worker's schedule above was, and
    # overwrite the page once it is stamped older than the incoming one.
    stamp_page(memory, "note", "2010-01-01T00:00:00Z")
    with (
        exclusive_file_lock(root_lock, timeout=5.0),
        pytest.raises(TimeoutError, match="memories.lock still locked"),
    ):
        memory_delta.unpack_and_merge(io.BytesIO(stream.getvalue()), memory.root, 0.0)
    assert page_text(memory, "note") == "agent's\n"


def test_task_deletions_travel_but_spare_a_task_running_on_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task deleted here disappears there and does not come back; one running there is spared."""
    box = Sandbox(tmp_path, monkeypatch, "alpha")
    now = time.time()
    make_db(box.db(), {"keep": {}, "gone": {}, "busy": {"result": PLACEHOLDER, "end_ts": 0}})
    make_db(
        box.db("alpha"),
        {
            "keep": {},
            "gone": {},
            "busy": {
                "result": PLACEHOLDER,
                "end_ts": 0,
                "timestamp": now - 30,
                "event_ts": now - 10,
            },
        },
    )
    conn = sqlite3.connect(box.db())
    persistence._delete_task_rows(conn, ["gone", "busy"], "")
    conn.close()
    assert set(tasks_in(box.db())) == {"keep"}

    [report] = box.run()

    assert report.error == ""
    assert report.pushed["deletions_recorded"] == 2
    assert report.pushed["tasks_deleted"] == 1
    assert set(tasks_in(box.db("alpha"))) == {"keep", "busy"}
    assert set(tasks_in(box.db())) == {"keep"}
    # The extract already leaves out what the target deleted, so nothing was held back.
    assert report.held_back_there == [] and report.pulled["tasks_inserted"] == 0


def test_a_machine_that_cannot_be_reached_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dead, flaky and url-only machines are reported; the reachable one is synced all the same."""
    box = Sandbox(tmp_path, monkeypatch, "alpha", "dead", "flaky1", "flaky2")
    # flaky1: the master connection gets through, the home probe dies.  flaky2:
    # the three probes (home, host name, memory) get through too, the
    # history extract dies.
    monkeypatch.setenv("FLAKY_AFTER_flaky1", "1")
    monkeypatch.setenv("FLAKY_AFTER_flaky2", "4")
    make_db(box.db(), {"L1": {}})
    make_db(box.db("alpha"), {"A1": {}})
    make_db(box.db("flaky1"), {"F1": {}})
    make_db(box.db("flaky2"), {"F2": {}})
    box.memory().write("page", "x\n")
    machines = [*box.machines(), Machine("web", url="https://web:8787")]
    sync = PeerSync(machines, home=box.home)

    reports = sync.run()

    alpha, dead, flaky1, flaky2, web = reports
    assert alpha.error == "" and tasks_in(box.db("alpha")) == {"L1": 3, "A1": 3}
    assert "Connection refused" in dead.error and dead.pulled == {}
    assert flaky1.error == "flaky1 (me@flaky1): ssh: connection to flaky1 closed"
    assert "_extract failed on me@flaky2" in flaky2.error and "closed" in flaky2.error
    assert tasks_in(box.db("flaky1")) == {"F1": 3} and tasks_in(box.db("flaky2")) == {"F2": 3}
    assert web.error == "not an ssh machine (https://web:8787); only ssh machines are synced"
    assert web.seconds == 0
    assert tasks_in(box.db()) == {"L1": 3, "A1": 3}
    # Every master connection was closed and the socket directory removed.
    assert not sync.control_dir.exists()


def test_the_stand_in_ssh_refuses_every_call_past_a_flaky_machines_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calls made at once are each counted: none past the limit gets through.

    The pull wave calls a machine twice at once (the manifest beside the
    extract).  A count that was read, incremented and written back could be
    reset by two such calls, and flaky2 above, dead by then, was pulled from.
    """
    Sandbox(tmp_path, monkeypatch, "flaky3")
    monkeypatch.setenv("FLAKY_AFTER_flaky3", "2")
    call = ["ssh", "me@flaky3", "echo through"]
    for _ in range(2):
        assert subprocess.run(call, capture_output=True, timeout=60).stdout == b"through\n"
    burst = ["bash", "-c", "for i in {1..32}; do ssh me@flaky3 echo through & done; wait"]
    # Ten bursts: on a saturated machine one burst caught the old count only 6 times in 10.
    for _ in range(10):
        refused = subprocess.run(burst, capture_output=True, timeout=60)
        assert refused.stdout == b"" and refused.stderr.count(b"flaky3 closed") == 32


def test_dry_run_changes_nothing_and_counts_what_would_travel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run rolls the history merges back and only counts the pages that differ."""
    box = Sandbox(tmp_path, monkeypatch, "alpha")
    make_db(box.db(), {"L1": {}})
    make_db(box.db("alpha"), {"A1": {"events": 2}})
    box.memory().write("here", "x\n")
    box.memory("alpha").write("there", "y\n")

    [report] = box.run(dry_run=True)

    assert report.pulled["dry_run"] is True and report.pulled["events_inserted"] == 2
    assert report.pushed["tasks_inserted"] == 1
    assert report.memory_pulled == {"entries": 1} and report.memory_pushed == {"entries": 1}
    assert tasks_in(box.db()) == {"L1": 3} and tasks_in(box.db("alpha")) == {"A1": 2}
    assert not (box.memory().root / "there.md").exists()
    text = format_reports([report], dry_run=True)
    assert (
        text.startswith("DRY RUN: nothing was changed.") and "pulled 1 task(s), 2 event(s)" in text
    )
    # The counts are against this machine as it is now: with several machines a real run
    # pushes each one what the others bring too, and the header says so.
    assert "compare each machine with this one as it is now" in text
    assert "pulled 1 entry(ies); pushed 1 entry(ies)" in text


# ---------------------------------------------------------------- adversarial


def test_machines_busy_writing_during_the_sync_lose_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agents appending events and rewriting pages everywhere during the sync: no error, no loss.

    The overlap is proven, not assumed: the sync waits at the stand-in
    ssh's gate on alpha until every writer has completed a round.
    """
    box = Sandbox(tmp_path, monkeypatch, "alpha", "beta")
    now = time.time()
    for host in ("", "alpha", "beta"):
        make_db(box.db(host), {f"{host or 'local'}-{i}": {"events": 20} for i in range(5)})
        box.memory(host).write(f"{host or 'local'}-page", "first\n")
    before = {host: tasks_in(box.db(host)) for host in ("", "alpha", "beta")}
    # Every live task is committed before the sync starts, so every report must hold it
    # back; during the sync its writer only appends events to it.
    live = {"events": 0, "result": PLACEHOLDER, "end_ts": 0, "timestamp": now}
    for host in ("", "alpha", "beta"):
        make_db(box.db(host), {f"{host or 'local'}-live": live})
    stop = threading.Event()
    errors: list[BaseException] = []
    writes = {host: 0 for host in ("", "alpha", "beta")}  # rounds each writer completed

    def hammer(host: str) -> None:
        try:
            conn = sqlite3.connect(box.db(host), timeout=30)
            seq = 0
            while not stop.is_set():
                conn.execute(
                    "INSERT INTO events(task_id, seq, event_json, timestamp) "
                    "VALUES (?, ?, '{}', ?)",
                    (f"{host or 'local'}-live", seq, time.time()),
                )
                conn.commit()
                box.memory(host).write(f"{host or 'local'}-page", f"rewrite {seq}\n")
                seq += 1
                writes[host] = seq
            conn.close()
        except BaseException as exc:  # noqa: BLE001 — reported by the test, not lost on the thread
            errors.append(exc)

    # The sync's first command on alpha waits at the stand-in's gate until every writer has
    # completed a round while the sync is in flight; both waits share one 30 s deadline after
    # which the gate is lifted regardless, so a stalled writer fails the overlap assertion
    # below instead of parking the sync.
    gate = box.root / "remotes" / "alpha.gate"
    reached = box.root / "remotes" / "alpha.gate.reached"
    gate.touch()
    overlap: dict[str, int] = {}  # rounds each writer completed while the sync waited

    def lift_gate() -> None:
        deadline = time.monotonic() + 30
        try:
            if wait_for(reached.exists, timeout=30):
                seen = dict(writes)
                wait_for(
                    lambda: all(writes[host] > seen[host] for host in writes),
                    timeout=deadline - time.monotonic(),
                )
                overlap.update({host: writes[host] - seen[host] for host in writes})
        finally:
            gate.unlink()

    # daemon: a thread that never ends fails the assertions below, not pytest's exit
    threads = [threading.Thread(target=hammer, args=(host,), daemon=True) for host in writes]
    threads.append(threading.Thread(target=lift_gate, daemon=True))
    # Started inside the try, so that a start that fails (the runner out of threads) leaves
    # no writer running.  No test: failing a later start alone takes a test double.
    try:
        for thread in threads:
            thread.start()
        reports = box.run()
    finally:
        stop.set()
        deadline = time.monotonic() + 30
        for thread in threads:
            if thread.is_alive():  # one that never started cannot be joined
                thread.join(max(0.0, deadline - time.monotonic()))

    assert errors == []
    assert [thread.is_alive() for thread in threads] == [False] * 4
    assert overlap and min(overlap.values()) >= 1, overlap
    assert [report.error for report in reports] == ["", ""]
    union = {task: 20 for tasks in before.values() for task in tasks}
    for host in ("", "alpha", "beta"):
        have = tasks_in(box.db(host))
        assert {k: v for k, v in have.items() if not k.endswith("-live")} == union, host
        # The live task stayed on its own machine, with every event it wrote.
        assert [k for k in have if k.endswith("-live")] == [f"{host or 'local'}-live"]
        assert seqs_of(box.db(host), f"{host or 'local'}-live") == list(range(writes[host]))
        assert not list(box.memory(host).root.glob("*.incoming"))
    for report in reports:
        assert report.held_back_there == [f"{report.name}-live"] and report.held_back_here == [
            "local-live"
        ]
    # Every writer's page is on every machine, and each is a whole page: "first" or some
    # "rewrite N" its writer completed, never a torn one.
    for host in ("", "alpha", "beta"):
        pages = sorted(page.stem for page in box.memory(host).root.glob("*.md"))
        assert pages == ["alpha-page", "beta-page", "local-page"], host
        for stem in pages:
            body = page_text(box.memory(host), stem)
            writer = stem.removesuffix("-page").replace("local", "")
            whole = {"first\n", *(f"rewrite {i}\n" for i in range(writes[writer]))}
            assert body in whole, (host, stem, body)


def test_pages_deleted_and_written_again_during_the_sync_fail_no_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agents deleting a page and writing it again everywhere during the sync: no error.

    A deletion writes the tombstone and removes the page; writing the page
    again publishes it and removes the tombstone.  The sync lists and reads
    the memories unlocked, and an entry it had listed, gone before it was
    read, used to fail that machine's sync with "No such file or
    directory".  Here every machine's agent flips one page between deleted
    and live as fast as it can while the whole sync runs six times.
    """
    hosts = ("", "alpha", "beta")
    box = Sandbox(tmp_path, monkeypatch, "alpha", "beta")
    for host in hosts:
        make_db(box.db(host), {f"{host or 'local'}-0": {"events": 2}})
        box.memory(host).write(f"{host or 'local'}-steady", "kept\n")
        box.memory(host).write(f"{host or 'local'}-flip", "body\n")
    stop = threading.Event()
    errors: list[BaseException] = []
    flips = {host: 0 for host in hosts}  # rounds each agent completed

    def agent(host: str) -> None:
        memory, name = box.memory(host), f"{host or 'local'}-flip"
        try:
            while not stop.is_set():
                memory.delete(name)
                memory.write(name, "body\n")
                flips[host] += 1
        except BaseException as exc:  # noqa: BLE001 — reported by the test, not lost on the thread
            errors.append(exc)

    # daemon: a thread that never ends fails the assertions below, not pytest's exit
    threads = [threading.Thread(target=agent, args=(host,), daemon=True) for host in hosts]
    try:
        for thread in threads:
            thread.start()
        reports = [report for _ in range(6) for report in box.run()]
    finally:
        stop.set()
        deadline = time.monotonic() + 30
        for thread in threads:
            if thread.is_alive():  # one that never started cannot be joined
                thread.join(max(0.0, deadline - time.monotonic()))

    assert errors == []
    assert [thread.is_alive() for thread in threads] == [False] * 3
    # Every agent kept flipping its page while the twelve peer syncs ran.
    assert min(flips.values()) >= len(reports) == 12, flips
    assert [report.error for report in reports] == [""] * 12
    for host in hosts:
        memory = box.memory(host)
        # The page nobody touches reached every machine, and no page anywhere is torn.
        steady = {page.stem for page in memory.root.glob("*-steady.md")}
        assert steady == {"alpha-steady", "beta-steady", "local-steady"}, host
        for page in memory.root.glob("*.md"):
            assert page_text(memory, page.stem) in ("kept\n", "body\n"), (host, page)
        assert not list(memory.root.glob("*.incoming"))


def test_a_hostile_stream_cannot_write_outside_the_memory(tmp_path: Path) -> None:
    """unpack admits page and tombstone paths only; anything else in a tar is dropped."""
    memory = tmp_path / "memories"
    memory.mkdir()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w|gz") as archive:
        for name in (
            "../escape.md",
            "/etc/passwd.md",
            "ok.md",
            "dom/ok.md",
            ".tombstones/ok",
            "dom/.tombstones/ok",
            "Upper.md",
            "nested/too/deep.md",
            ".sqlite3",
            "x.md/",
        ):
            data = b"---\nupdated: 2026-01-01T00:00:00Z\n---\nbody\n"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("link.md")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)
    stream.seek(0)
    staging = tmp_path / "staging"
    staging.mkdir()
    assert memory_delta.unpack(io.BytesIO(stream.getvalue()), staging) == 4
    assert sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()) == [
        ".tombstones/ok",
        "dom/.tombstones/ok",
        "dom/ok.md",
        "ok.md",
    ]
    assert not (tmp_path / "escape.md").exists() and not list(tmp_path.glob("*passwd*"))
    result = memory_delta.unpack_and_merge(stream, memory, 0.0)
    assert result["received"] == 4 and result["counts"]["added"] == 2
    assert sorted(p.relative_to(memory).as_posix() for p in memory.rglob("*.md")) == [
        "dom/ok.md",
        "ok.md",
    ]


def test_pack_skips_what_is_not_a_page_or_vanished(tmp_path: Path) -> None:
    """pack leaves out invalid names, symlinks, directories and files that are not there.

    A file is not there when it is gone since the fingerprints were taken
    (``gone.md``), or when it lies under what is a file here and a domain
    memory on the other machine (``blocked/page.md``).
    """
    memory = tmp_path / "memories"
    memory.mkdir()
    (memory / "real.md").write_text("---\nupdated: 2026-01-01T00:00:00Z\n---\nx\n")
    (memory / "link.md").symlink_to(memory / "real.md")
    (memory / "folder.md").mkdir()
    (memory / "blocked").write_text("a file where the other machine has a domain memory\n")
    stream = io.BytesIO()
    written = memory_delta.pack(
        memory,
        ["real.md", "link.md", "gone.md", "folder.md", "blocked/page.md", "../up.md", "real.md"],
        stream,
    )
    assert written == 1
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        assert [m.name for m in archive] == ["real.md"]
    assert memory_delta.fingerprints(tmp_path / "missing") == {}
    assert memory_delta.differing({"a": "1", "b": "2"}, {"a": "1", "b": "3", "c": "4"}) == {
        "b",
        "c",
    }


def test_pack_sends_whole_entries_while_an_agent_rewrites_and_deletes_them(
    tmp_path: Path,
) -> None:
    """An entry replaced or deleted while pack reads it travels whole or not at all.

    An agent publishes a page by an atomic rename at any moment of a pack.
    pack used to take an entry's size from its path and to open the file
    afterwards (``TarFile.add``): a page replaced in between by a longer one
    arrived cut short, a shorter one aborted the stream ("unexpected end of
    data"), and an entry deleted in between raised.  Here an agent rewrites
    a page with bodies of two lengths, deleting it every fifth round, while
    pack and unpack run a thousand times, and until both bodies have
    travelled: the page did change under pack.
    """
    memory = MemoryDir(tmp_path / "memories")
    bodies = ("short\n", "a much longer body\n" * 40)
    memory.write("page", bodies[0])
    received = tmp_path / "received"
    page, tombstone = received / "page.md", received / ".tombstones" / "page"
    entries = ["page.md", ".tombstones/page"]

    # A quiet memory first: the page travels with its bytes and its modification time.
    stream = io.BytesIO()
    assert memory_delta.pack(memory.root, entries, stream) == 1
    stream.seek(0)
    assert memory_delta.unpack(stream, received) == 1
    source = memory.root / "page.md"
    assert page.read_bytes() == source.read_bytes()
    assert page.stat().st_mtime == pytest.approx(source.stat().st_mtime, abs=1e-3)
    page.unlink()

    stop = threading.Event()
    errors: list[BaseException] = []

    def agent() -> None:
        turn = 0
        try:
            while not stop.is_set():
                turn += 1
                if turn % 5 == 0:
                    memory.delete("page")
                memory.write("page", bodies[turn % 2])
        except BaseException as exc:  # noqa: BLE001 — reported by the test, not lost on the thread
            errors.append(exc)

    # daemon: a thread that never ends fails the assertions below, not pytest's exit
    thread = threading.Thread(target=agent, daemon=True)
    thread.start()
    travelled: set[str] = set()
    packs = 0
    deadline = time.monotonic() + 60
    try:
        while (packs < 1000 or len(travelled) < 2) and time.monotonic() < deadline:
            packs += 1
            stream = io.BytesIO()
            written = memory_delta.pack(memory.root, entries, stream)
            stream.seek(0)
            assert memory_delta.unpack(stream, received) == written
            if page.is_file():
                body = page.read_text().split("---\n", 2)[-1]
                assert body in bodies, body
                travelled.add(body)
                page.unlink()
            if tombstone.is_file():
                # The whole deletion stamp: strptime rejects a cut or an empty one.
                time.strptime(tombstone.read_text(), "%Y-%m-%dT%H:%M:%SZ\n")
                tombstone.unlink()
    finally:
        stop.set()
        thread.join(30)
    assert errors == [] and not thread.is_alive()
    assert packs >= 1000 and travelled == set(bodies)


def test_fingerprints_list_a_memory_whose_page_is_deleted_and_written_again_meanwhile(
    tmp_path: Path,
) -> None:
    """Listing a memory does not fail when an entry is removed between its listing and its reading.

    fingerprints reads the memory unlocked.  A page deleted, or a tombstone
    removed because its page was written again, after it was listed used to
    raise FileNotFoundError: out of the page's or the tombstone's digest,
    or out of ``tombstones`` reading the deletion stamp.  Here an agent
    deletes a page and writes it again as fast as it can while the memory
    is listed three hundred times, and until the page has been listed both
    live and deleted: the memory did change under the listing.
    """
    memory = MemoryDir(tmp_path / "memories")
    memory.write("steady", "kept\n")
    memory.write("page", "body\n")
    steady = memory_delta.fingerprints(memory.root)["steady.md"]
    stop = threading.Event()
    errors: list[BaseException] = []

    def agent() -> None:
        try:
            while not stop.is_set():
                memory.delete("page")
                memory.write("page", "body\n")
        except BaseException as exc:  # noqa: BLE001 — reported by the test, not lost on the thread
            errors.append(exc)

    # daemon: a thread that never ends fails the assertions below, not pytest's exit
    thread = threading.Thread(target=agent, daemon=True)
    thread.start()
    seen: set[str] = set()
    listings = 0
    deadline = time.monotonic() + 60
    try:
        while (listings < 300 or len(seen) < 2) and time.monotonic() < deadline:
            listings += 1
            found = memory_delta.fingerprints(memory.root)
            # The page nobody touches is always listed, whole.
            assert found.pop("steady.md") == steady
            # The other is listed live, deleted, or (caught between the two) not at all.
            assert set(found) in ({"page.md"}, {".tombstones/page"}, set()), found
            seen.update(found)
    finally:
        stop.set()
        thread.join(30)
    assert errors == [] and not thread.is_alive()
    assert listings >= 300 and seen == {"page.md", ".tombstones/page"}


def test_memory_delta_script_modes_run_with_a_bare_python(tmp_path: Path) -> None:
    """The shipped script's three modes work from the command line as the remote half runs them."""
    import sys

    memory = tmp_path / "memories"
    MemoryDir(memory).write("note", "hello\n")
    script = str(peer_sync.MEMORY_DELTA_PATH)
    probe = subprocess.run(
        [sys.executable, script, "probe", str(memory)], capture_output=True, check=True, timeout=60
    )
    report = json.loads(probe.stdout)
    assert set(report["entries"]) == {"note.md"} and abs(report["now"] - time.time()) < 60
    packed = subprocess.run(
        [sys.executable, script, "pack", str(memory)],
        input=b'["note.md"]',
        capture_output=True,
        check=True,
        timeout=60,
    )
    other = tmp_path / "other"
    merged = subprocess.run(
        [sys.executable, script, "merge", str(other), "0.5"],
        input=packed.stdout,
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert json.loads(merged.stdout)["counts"]["added"] == 1
    assert (other / "note.md").read_bytes() == (memory / "note.md").read_bytes()
    assert subprocess.run([sys.executable, script], capture_output=True, timeout=60).returncode == 2
    assert (
        subprocess.run(
            [sys.executable, script, "bogus", str(memory)], capture_output=True, timeout=60
        ).returncode
        == 2
    )


def test_memory_delta_runs_as_shipped_text_with_its_helper_inline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exec'd from text with SHIPPED_SOURCES bound, as the ssh bootstrap does, every mode works."""
    namespace: dict[str, object] = {
        "__name__": "shipped",
        "SHIPPED_SOURCES": {memory_delta.MERGE_MODULE: peer_sync.MERGE_PAGES_PATH.read_bytes()},
    }
    source = peer_sync.MEMORY_DELTA_PATH.read_bytes()
    exec(compile(source, str(peer_sync.MEMORY_DELTA_PATH), "exec"), namespace)
    shipped = namespace["merge_module"]()  # type: ignore[operator]
    assert shipped.TOMBSTONES_DIR == ".tombstones"
    memory = tmp_path / "memories"
    MemoryDir(memory).write("note", "hello\n")
    main = namespace["main"]

    def run(argv: list[str], stdin: bytes = b"") -> tuple[int, bytes]:
        out = io.TextIOWrapper(io.BytesIO())
        # Put back as the call returns: an undo left to the test's teardown runs after
        # ``capsys`` has closed its capture file and makes that closed file ``sys.stdout``.
        with pytest.MonkeyPatch.context() as streams:
            streams.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(stdin)))
            streams.setattr("sys.stdout", out)
            status = main(argv)  # type: ignore[operator]
        out.flush()
        return status, out.buffer.getvalue()  # type: ignore[attr-defined]

    status, probe = run(["probe", str(memory)])
    assert status == 0 and set(json.loads(probe)["entries"]) == {"note.md"}
    status, packed = run(["pack", str(memory)], b'["note.md"]')
    assert status == 0
    status, merged = run(["merge", str(tmp_path / "other"), "0.5"], packed)
    assert status == 0 and json.loads(merged)["counts"]["added"] == 1
    assert run(["merge", str(tmp_path / "third")], packed)[0] == 0  # tolerance defaults to 0
    assert run([])[0] == 2 and "usage:" in capsys.readouterr().err
    assert run(["bogus", str(memory)])[0] == 2 and "unknown mode" in capsys.readouterr().err


def test_ssh_peer_commands_ride_the_master_connection() -> None:
    """The peer's ssh command lines carry the control socket, the port, and ship scripts inline."""
    control = control_directory()
    try:
        assert len(str(control)) < 40 and oct(control.stat().st_mode & 0o777) == "0o700"
        peer = SshPeer("lab", "me@lab.example:2222", control)
        assert peer.login == "me@lab.example" and peer.port == "2222"
        command = peer.command("echo hi")
        assert command[:3] == ["ssh", "-o", f"ControlPath={peer.socket}"] and command[-2:] == [
            "me@lab.example",
            "echo hi",
        ]
        assert "-p" in command and command[command.index("-p") + 1] == "2222"
        assert "ControlMaster=no" in command  # the default stays; the socket alone joins the master
        runner = peer.runner()
        assert runner.port == "2222" and runner.ssh_options[0] == f"ControlPath={peer.socket}"
        remote = peer.python_command(
            b"print(1)", {"helper": b"x = 1"}, ["probe", "~/.kiss/memories"]
        )
        python, flag, bootstrap, *args = shlex.split(remote)
        assert (python, flag, args) == ("python3", "-c", ["probe", "~/.kiss/memories"])
        assert bootstrap.startswith(
            "import base64;SHIPPED_SOURCES={'helper':base64.b64decode('eCA9IDE=')};exec("
        )
        peer.close()  # never opened: nothing to do
    finally:
        for leftover in control.glob("*"):
            leftover.unlink()
        control.rmdir()


@pytest.mark.skipif(shutil.which("ssh") is None, reason="no ssh client on this host")
def test_ssh_peer_reports_a_refused_master_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``open()`` runs the real ``ssh``; a host that refuses is a ``PeerError``, not a hang."""
    # An empty configuration of the test's own: the developer's ``~/.ssh/config`` (a
    # proxy for every host, say) must not decide what this login dials.
    config = tmp_path / "ssh_config"
    config.write_text("")
    monkeypatch.setenv(machine_registry.SSH_CONFIG_ENV, str(config))
    control = control_directory()
    try:
        with pytest.raises(PeerError, match="cannot connect"):
            SshPeer("nowhere", "nobody@127.0.0.1:1", control).open()
    finally:
        for leftover in control.glob("*"):
            leftover.unlink()
        control.rmdir()


# ---------------------------------------------------------------- the tool and the agent


def test_tool_checks_the_registry_before_syncing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No machines, an unknown name, a chosen subset: the tool answers from the registry."""
    box = Sandbox(tmp_path, monkeypatch, "alpha", "beta")
    make_db(box.db(), {"L1": {}})
    make_db(box.db("alpha"), {"A1": {}})
    make_db(box.db("beta"), {"B1": {}})
    save_machines([])
    assert (
        sync_history_and_memory()
        == "No machine is registered (`/machine add user@host`); nothing to sync."
    )
    save_machines(box.machines())
    assert sync_history_and_memory("alpha, gamma") == "Not registered: gamma (see `/machine list`)."

    text = sync_history_and_memory("alpha")

    assert text.startswith("Synced 1 of 1 machine(s).")
    assert "alpha (me@alpha): ok in" in text and "beta" not in text
    assert (
        "history: pulled 1 task(s), 3 event(s), 0 deletion(s), 0 removed; pushed 1 task(s)" in text
    )
    assert tasks_in(box.db("alpha")) == {"L1": 3, "A1": 3} and tasks_in(box.db("beta")) == {"B1": 3}

    text = sync_history_and_memory()
    assert text.startswith("Synced 2 of 2 machine(s).") and tasks_in(box.db("beta")) == {
        "L1": 3,
        "A1": 3,
        "B1": 3,
    }


def test_format_reports_names_failures_held_back_tasks_conflicts_and_skew() -> None:
    """Every field of a report has a line, and a failed machine is marked as such."""
    good = peer_sync.PeerReport(
        "lab",
        "me@lab",
        home="~/.brand",
        skew=-2.5,
        seconds=3.21,
        pulled={"tasks_inserted": 1, "events_inserted": 2},
        pushed={
            "tasks_inserted": 0,
            "events_inserted": 0,
            "deletions_recorded": 1,
            "tasks_deleted": 1,
        },
        memory_pulled={
            "entries": 2,
            "counts": {"added": 1, "updated": 1, "kept": 0, "conflicts": 0, "removed": 0},
        },
        memory_pushed={"entries": 0},
        held_back_there=["t1"],
        held_back_here=["h1", "h2"],
        conflicts=["page"],
    )
    bad = peer_sync.PeerReport("box", "", error="not an ssh machine")
    text = format_reports([good, bad], dry_run=False)
    assert text.splitlines()[0] == "Synced 1 of 2 machine(s)."
    assert "lab (me@lab): ok in 3.2s\n  KISS home there: ~/.brand\n" in text
    assert "pushed 0 task(s), 0 event(s), 1 deletion(s), 1 removed" in text
    assert "pulled 2 entry(ies): 1 added, 1 updated, 0 removed, 0 kept, 0 in conflict; " in text
    assert "in conflict; pushed 0 entry(ies)" in text
    assert "still running there, left for the next sync: t1" in text
    assert "still running here, left for the next sync: h1, h2" in text
    assert "conflicts (same page edited at the same moment on both sides, each side kept " in text
    assert "each side kept its own): page" in text
    assert "clock skew: -2.5s" in text
    assert "box (no ssh): FAILED: not an ssh machine" in text
    assert "history: pulled nothing (not reached); pushed nothing (not reached)" not in text
    unreached = peer_sync.PeerReport(
        "far",
        "me@far",
        error="timed out",
        pulled={"tasks_inserted": 1},
        memory_pulled={"entries": 1},
    )
    text = format_reports([unreached], dry_run=False)
    assert "0 removed; pushed nothing (not reached)" in text
    assert "memory: pulled 1 entry(ies); pushed nothing (not reached)" in text


def test_agent_run_calls_the_tool_once_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under the SEA's configuration the model sees only the sync tool and finish, and syncs."""
    box = Sandbox(tmp_path, monkeypatch, "alpha")
    make_db(box.db(), {"L1": {}})
    make_db(box.db("alpha"), {"A1": {}})
    save_machines(box.machines())
    script = [
        tool_call_body(
            "sync_history_and_memory", {"machines": "", "dry_run": False}, prompt_tokens=500
        ),
        finish_body("<p>alpha: pulled 1 task, pushed 1 task.</p>", prompt_tokens=600),
    ]
    run = sea_commands.evaluate_sea([SyncHistoryAndMemorySea()], "")
    assert run.prompt == sea_module.ADD_TO_PROMPT

    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)

    assert parsed["success"] is True and "alpha" in parsed["summary"]
    assert tasks_in(box.db()) == tasks_in(box.db("alpha")) == {"L1": 3, "A1": 3}
    assert len(agentic) == 2, [list(r) for r in agentic]
    for request in agentic:
        assert tool_names(request) == {"sync_history_and_memory", "finish"}, tool_names(request)
        assert system_message(request).startswith(sea_module.SYSTEM_PROMPT)
    [result] = tool_results(agentic[1])
    assert "Synced 1 of 1 machine(s)." in result and "alpha (me@alpha): ok in" in result
