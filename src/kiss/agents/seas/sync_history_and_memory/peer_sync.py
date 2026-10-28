# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bring the task history and the memory of this machine and its peers to the same state.

Two stores travel.  The task history (``history.db``: ``task_history``,
``events`` and the deletion records) is append-only with a global key
``(task_id, seq)``, so syncing it is a set union handled by
``kiss.scripts.sync_db`` -- only the rows the other side lacks travel,
compressed, and a deleted task stays deleted everywhere.  The memory
(``memories/``) is a directory of pages where the newest ``updated`` stamp
wins and deletions travel as tombstones (``merge_memory_pages``); only the
pages whose content hash differs travel (:mod:`memory_delta`).

Every machine is reached over one multiplexed ssh connection
(:class:`SshPeer`) and the machines are worked on at once, in three
waves: every peer is connected and asked for its clock and memory
fingerprints; then everything is pulled from every peer into this
machine; then this machine -- now holding the union -- is pushed to every
peer, so one run leaves all machines equal (a star; the barrier between
pulling and pushing is what makes it converge in one run).  Merges into
this machine's stores are serialised by a lock: two merges into the same
memory could otherwise each compare against the same old page and the
older copy win the race of renames.  Every memory merge, here and on the
peers, also holds the memory's own mutation lock so no agent's write is
lost under it (:func:`memory_delta.memory_locks`).

Page stamps come from the clock of whichever machine wrote them, and
after the pull wave this machine holds pages from every peer's clock, so
one tolerance serves every merge of a run: the spread of all the clocks
measured in the first wave.  Two writes closer together than that are
reported as a conflict and kept on both sides rather than ordered.

Tasks still running are held back on the side they run on, and travel
once they finish: a task whose newest event is younger than
:data:`RUNNING_WINDOW` seconds (by the clock of the machine it runs on),
or one still carrying the placeholder result the web app writes at a
task's start whose newest event is younger than :data:`ABANDONED_AFTER`.
A placeholder older than that marks a task whose agent died; its rows
are history like any other and travel.  Only a task whose row is about
to travel is judged: new events of a task the other side already has
always travel, since events are append-only and the row they belong to
is already in place.  Nothing else is held back and no row is ever
updated or dropped by the sync, so a sync run while tasks run on every
machine loses nothing: it only postpones the tasks in flight to the next
run.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiss.agents.seas.sync_history_and_memory import memory_delta
from kiss.agents.seas.sync_history_and_memory.ssh_peer import PeerError, SshPeer, control_directory
from kiss.agents.sorcar.machines import Machine
from kiss.core.brand import HOME_DIR
from kiss.core.config import kiss_home
from kiss.scripts import sync_db
from kiss.scripts.sync_db import PHASE_MANIFEST, PHASE_MERGE, Location, Runner, SyncError

RUNNING_WINDOW = 300.0
"""A task with an event younger than this many seconds is running and is held back."""

ABANDONED_AFTER = 24 * 3600.0
"""A task still showing the start placeholder this long after its last event is dead."""

UNFINISHED_RESULT = sync_db.UNFINISHED_RESULT
HISTORY_DB = "history.db"
MEMORIES_DIR = "memories"
MEMORY_DELTA_PATH = Path(memory_delta.__file__)
MERGE_PAGES_PATH = Path(sync_db.__file__).with_name("merge_memory_pages.py")


@dataclass
class PeerReport:
    """What one run did with one machine.

    Attributes:
        name: The machine's registered name.
        ssh: Its login.
        skew: The machine's clock minus this machine's, in seconds.
        pulled: ``sync_db`` statistics of the history pulled from it.
        pushed: ``sync_db`` statistics of the history pushed to it.
        held_back_there: Ids of its running tasks, left for the next run.
        held_back_here: Ids of this machine's running tasks, left for the next run.
        memory_pulled: Memory entries received from it and the merge counts.
        memory_pushed: Memory entries sent to it and its merge counts.
        conflicts: Pages edited at the same moment on both sides, kept as they were.
        seconds: Wall-clock time spent on the machine over both waves.
        error: Why the machine was skipped or left half-synced, or empty.
    """

    name: str
    ssh: str
    skew: float = 0.0
    pulled: dict[str, Any] = field(default_factory=dict)
    pushed: dict[str, Any] = field(default_factory=dict)
    held_back_there: list[str] = field(default_factory=list)
    held_back_here: list[str] = field(default_factory=list)
    memory_pulled: dict[str, Any] = field(default_factory=dict)
    memory_pushed: dict[str, Any] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""


class UnfinishedFilter:
    """A ``sync_db`` delta edit that drops the tasks still running on the source side.

    Attributes:
        now: The source machine's clock when the delta was taken.
        held: Ids of the tasks dropped, filled in when the edit runs.
    """

    def __init__(self, now: float) -> None:
        """Judge running tasks against *now*, the source machine's clock."""
        self.now = now
        self.held: list[str] = []

    def __call__(self, delta_path: str) -> None:
        """Delete the running tasks' rows and events from the uncompressed delta at *delta_path*."""
        self.held = unfinished_tasks(delta_path, self.now)
        if not self.held:
            return
        conn = sqlite3.connect(delta_path)
        try:
            marks = ",".join("?" * len(self.held))
            for table, key in (("events", "task_id"), ("task_history", "id")):
                if table in _tables(conn):
                    conn.execute(f'DELETE FROM "{table}" WHERE "{key}" IN ({marks})', self.held)
            conn.commit()
        finally:
            conn.close()


def unfinished_tasks(delta_path: str, now: float) -> list[str]:
    """Return the ids of the tasks whose row is in a ``sync_db`` delta and that are still running.

    The rule is the module docstring's: an event younger than
    :data:`RUNNING_WINDOW`, or the start placeholder with an event
    younger than :data:`ABANDONED_AFTER`.  A task whose events alone
    are in the delta (the other side has its row) is never held back.

    Args:
        delta_path: The uncompressed delta database (``task_history`` and
            ``events`` with the source's columns, only the rows about to travel).
        now: The source machine's clock, seconds since the epoch.
    """
    conn = sqlite3.connect(f"file:{delta_path}?mode=ro", uri=True)
    try:
        tables = _tables(conn)
        if "task_history" not in tables:
            return []
        newest: dict[str, float] = {}
        if "events" in tables and "timestamp" in _columns(conn, "events"):
            for task_id, stamp in conn.execute(
                'SELECT "task_id", MAX("timestamp") FROM "events" GROUP BY "task_id"'
            ):
                newest[str(task_id)] = float(stamp or 0.0)
        columns = _columns(conn, "task_history")
        selected = ", ".join(
            f'"{column}"' if column in columns else "NULL"
            for column in ("result", "timestamp", "end_ts")
        )
        rows = conn.execute(f'SELECT "id", {selected} FROM "task_history" ORDER BY "id"').fetchall()
    finally:
        conn.close()
    held: list[str] = []
    for task_id, result, started, ended in rows:
        last = max(newest.get(str(task_id), 0.0), float(started or 0.0))
        running = float(ended or 0.0) <= 0 and last > now - RUNNING_WINDOW
        placeholder = str(result or "") == UNFINISHED_RESULT and last > now - ABANDONED_AFTER
        if running or placeholder:
            held.append(str(task_id))
    return held


def _tables(conn: sqlite3.Connection) -> set[str]:
    """Return the table names of a database."""
    return {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the column names of *table*."""
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


class LockedRunner(Runner):
    """A ``sync_db`` runner that takes a lock around merges into this machine's database."""

    def __init__(self, inner: Runner, lock: threading.Lock) -> None:
        """Wrap *inner*; local merges run under *lock*."""
        super().__init__(inner.python, inner.port, inner.ssh_options)
        self.lock = lock

    def run(
        self, location: Location, phase: str, args: list[str], inp: str | None, out: str
    ) -> None:
        """Run a phase; a merge into a local database holds the lock, other phases do not."""
        if not location.is_remote and phase == PHASE_MERGE:
            with self.lock:
                super().run(location, phase, args, inp, out)
        else:
            super().run(location, phase, args, inp, out)


class PeerSync:
    """One run of the sync against a set of machines.

    Attributes:
        home: This machine's KISS home (``~/.kiss``).
        remote_home: The peers' KISS home as a remote path (``~/.kiss``).
        dry_run: Report what would travel without changing any machine.
        reports: One :class:`PeerReport` per machine, in the order given.
        tolerance: Seconds within which two page stamps count as the same
            moment in every merge of the run: the spread of the clocks of
            this machine and the peers reached, known after the first wave.
    """

    def __init__(
        self,
        machines: list[Machine],
        dry_run: bool = False,
        home: Path | None = None,
        remote_home: str = f"~/{HOME_DIR}",
    ) -> None:
        """Prepare a run against *machines*; ssh machines only, the rest are reported as skipped.

        Args:
            machines: Registered machines (``kiss.agents.sorcar.machines.load_machines``).
            dry_run: Roll the history merges back and only count the memory entries
                that would travel.
            home: This machine's KISS home; the configured one when None.
            remote_home: The peers' KISS home.
        """
        self.home = home or kiss_home()
        self.remote_home = remote_home
        self.dry_run = dry_run
        self.lock = threading.Lock()
        self.control_dir = control_directory()
        self.reports: list[PeerReport] = []
        self.peers: list[tuple[SshPeer, PeerReport]] = []
        self.remote_entries: dict[str, dict[str, str]] = {}
        self.manifests: dict[str, str] = {}
        self.tolerance = 0.0
        self.script = MEMORY_DELTA_PATH.read_bytes()
        self.shipped = {memory_delta.MERGE_MODULE: MERGE_PAGES_PATH.read_bytes()}
        for machine in machines:
            report = PeerReport(machine.name, machine.ssh)
            self.reports.append(report)
            if machine.ssh:
                self.peers.append((SshPeer(machine.name, machine.ssh, self.control_dir), report))
            else:
                report.error = f"not an ssh machine ({machine.url}); only ssh machines are synced"

    @property
    def local_db(self) -> Location:
        """This machine's history database."""
        return Location(str(self.home / HISTORY_DB))

    @property
    def local_memory(self) -> Path:
        """This machine's memory directory."""
        return self.home / MEMORIES_DIR

    def remote_db(self, peer: SshPeer) -> Location:
        """A peer's history database."""
        return Location(f"{peer.login}:{self.remote_home}/{HISTORY_DB}")

    def run(self) -> list[PeerReport]:
        """Sync every machine: connect to all, pull from all of them, then push the union to all.

        Returns:
            The per-machine reports; a machine that failed carries the error
            and whatever was done before it.
        """
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(self.peers))) as pool:
                self._wave(pool, self.connect)
                skews = [0.0] + [report.skew for _, report in self.peers if not report.error]
                self.tolerance = round(max(skews) - min(skews), 3)
                self._wave(pool, self.pull)
                self._wave(pool, self.push)
        finally:
            for peer, _ in self.peers:
                peer.close()
            for leftover in self.control_dir.glob("*"):
                leftover.unlink(missing_ok=True)
            self.control_dir.rmdir()
        return self.reports

    def _wave(self, pool: ThreadPoolExecutor, wave: Any) -> None:
        """Run *wave* for every peer not yet failed, all at once, and wait for the last."""
        futures = [
            pool.submit(self._guarded, wave, peer, report)
            for peer, report in self.peers
            if not report.error
        ]
        for future in futures:
            future.result()

    def _guarded(self, wave: Any, peer: SshPeer, report: PeerReport) -> None:
        """Run one wave for one peer, recording its time and any failure in the report."""
        started = time.monotonic()
        try:
            wave(peer, report)
        except (PeerError, SyncError, OSError, ValueError) as exc:
            report.error = str(exc)
        finally:
            report.seconds = round(report.seconds + time.monotonic() - started, 2)

    def connect(self, peer: SshPeer, report: PeerReport) -> None:
        """Wave one for one peer: connect, then read the peer's clock and memory fingerprints."""
        peer.open()
        before = time.time()
        probe = json.loads(peer.run(self.memory_command(peer, "probe")).decode("utf-8"))
        report.skew = round(float(probe["now"]) - (before + time.time()) / 2, 3)
        self.remote_entries[peer.name] = probe["entries"]

    def pull(self, peer: SshPeer, report: PeerReport) -> None:
        """Wave two for one peer: pull its history and pages into this machine.

        The peer's history manifest, which the push wave needs, is taken
        now, alongside the pull: both are reads on the peer and ride the
        same connection, so the wave costs the longer of the two instead
        of their sum.
        """
        theirs = self.remote_entries[peer.name]
        held = UnfinishedFilter(time.time() + report.skew)
        with ThreadPoolExecutor(max_workers=1) as side:
            listing = side.submit(self.remote_manifest, peer)
            report.pulled = self.history(
                self.remote_db(peer), self.local_db, LockedRunner(peer.runner(), self.lock), held
            )
        listing.result()
        report.held_back_there = held.held
        wanted = sorted(
            memory_delta.differing(memory_delta.fingerprints(self.local_memory), theirs)
            & theirs.keys()
        )
        report.memory_pulled = {"entries": len(wanted)}
        if not wanted or self.dry_run:
            return
        stream = peer.run(self.memory_command(peer, "pack"), json.dumps(wanted).encode("utf-8"))
        with self.lock:
            merged = memory_delta.unpack_and_merge(
                io.BytesIO(stream), self.local_memory, self.tolerance
            )
        report.memory_pulled.update(merged)
        report.conflicts += merged["conflicts"]

    def push(self, peer: SshPeer, report: PeerReport) -> None:
        """Wave three for one peer: push this machine's history and the pages the peer lacks."""
        held = UnfinishedFilter(time.time())
        report.pushed = self.history(
            self.local_db,
            self.remote_db(peer),
            peer.runner(),
            held,
            target_manifest=self.manifests.get(peer.name),
        )
        report.held_back_here = held.held
        theirs = self.remote_entries[peer.name]
        mine = memory_delta.fingerprints(self.local_memory)
        wanted = sorted(memory_delta.differing(mine, theirs) & mine.keys())
        report.memory_pushed = {"entries": len(wanted)}
        if not wanted or self.dry_run:
            return
        stream = io.BytesIO()
        memory_delta.pack(self.local_memory, wanted, stream)
        merged = json.loads(
            peer.run(
                self.memory_command(peer, "merge", str(self.tolerance)), stream.getvalue()
            ).decode("utf-8")
        )
        report.memory_pushed.update(merged)
        report.conflicts += [name for name in merged["conflicts"] if name not in report.conflicts]

    def remote_manifest(self, peer: SshPeer) -> None:
        """Take the peer's history manifest for the push wave; a peer without a database has none.

        Args:
            peer: The machine, connected.
        """
        path = self.control_dir / f"{peer.socket.name}.manifest.gz"
        try:
            peer.runner().run(self.remote_db(peer), PHASE_MANIFEST, [], None, str(path))
        except SyncError as exc:
            if "database not found" not in str(exc):
                raise
            return
        self.manifests[peer.name] = str(path)

    def history(
        self,
        source: Location,
        target: Location,
        runner: Runner,
        held: UnfinishedFilter,
        target_manifest: str | None = None,
    ) -> dict[str, Any]:
        """Copy the history rows *target* lacks from *source*, *held* dropping the running tasks.

        A machine that has never run a task has no database; the history
        is then skipped (the memory still syncs) and the statistics say so.

        Args:
            source: Database rows are read from.
            target: Database rows are written to.
            runner: Runs the phases, over the peer's connection.
            held: Drops the running tasks from the delta and records them.
            target_manifest: A manifest of *target* taken earlier, if any.

        Returns:
            The ``sync_db`` statistics, or ``{"skipped": reason}``.
        """
        try:
            return sync_db.synchronize(
                source,
                target,
                runner,
                dry_run=self.dry_run,
                delta_edits=[held],
                target_manifest=target_manifest,
            )
        except SyncError as exc:
            if "database not found" not in str(exc):
                raise
            return {"skipped": str(exc)}

    def memory_command(self, peer: SshPeer, mode: str, *args: str) -> str:
        """The remote command running :mod:`memory_delta` in *mode* on the peer's memory."""
        return peer.python_command(
            self.script, self.shipped, [mode, f"{self.remote_home}/{MEMORIES_DIR}", *args]
        )
