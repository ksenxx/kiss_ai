# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for wave-2 findings A4, A10, B1, C2.

A4  ``_get_db()`` read the global ``_db_generation`` twice — once for
    the staleness check and again (after connection creation) when
    stamping ``tl.gen``.  A concurrent ``_close_db()`` between the two
    reads made the thread stamp the *new* generation onto a connection
    created under the *old* one, so the connection survived an
    invalidation it should not have.

A10 ``_save_last_model`` did ``load_config() -> mutate -> save_config``
    with the full config dict.  ``save_config`` overlays every DEFAULTS
    key present in its argument onto a fresh re-read of the file, so
    the stale full-dict snapshot clobbered concurrent updates to
    *other* keys (e.g. ``work_dir``) made between the load and the
    save.

B1  The baseline commit SHA persisted to git config
    (``branch.<b>.kiss-baseline``) by ``save_baseline_commit`` must
    actually be persisted.  (The ``load_baseline_commit`` companion
    reader was later removed as production-dead API; the test reads
    the git config value directly.)

C2  ``_RunningAgentState.register()`` silently overwrote an existing
    live state registered under the same ``tab_id``, orphaning the old
    state's ``stop_event``/``task_thread``.  It must emit a WARNING on
    overwrite (semantics unchanged: last registration wins).

WAL ``_get_db()`` executed ``PRAGMA journal_mode=WAL`` on a fresh
    connection before arming ``busy_timeout`` and outside
    ``_init_tables_lock``.  The rollback→WAL transition needs an
    exclusive lock, and when a peer connection holds a RESERVED
    (write-transaction) lock SQLite reports ``SQLITE_BUSY``
    *immediately* — the busy handler is never consulted — so a peer
    mid-write on a brand-new ``sorcar.db`` (e.g. a parallel sub-agent
    running ``_init_tables``, or the kiss-web daemon initialising) made
    ``_get_db()`` die instantly with ``database is locked``.  The
    pragma now sits inside the init lock with a bounded busy/locked
    retry loop.

All tests drive real SQLite databases, real JSON config files, real git
repositories and real threads — no mocks, patches, or fakes.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
import kiss.agents.vscode.vscode_config as vc
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.running_agent_state import _RunningAgentState

_GIT_ID = (
    "-c",
    "user.email=test@example.com",
    "-c",
    "user.name=Test",
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git in *repo* with a fixed identity, optionally asserting success."""
    result = subprocess.run(
        ["git", *_GIT_ID, "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if check:
        assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result


def _init_repo(tmp_path: Path, name: str = "repo") -> Path:
    """Create a git repo with one commit on branch ``main``; return its root."""
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "f.txt").write_text("base\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", "c1")
    return repo


class _DBSandbox:
    """Redirect the persistence DB to a fresh temp dir per test."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._close_db()

    def teardown_method(self) -> None:
        th._close_db()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestA4GetDbGenerationRace(_DBSandbox):
    """A4: a connection created concurrently with ``_close_db`` must not survive it."""

    def test_connection_created_during_close_is_invalidated(self) -> None:
        """Deterministic interleaving via the real ``_init_tables_lock``.

        The worker's first ``_get_db()`` reads the generation for its
        staleness check, opens the SQLite connection, then blocks on
        ``_init_tables_lock`` (held by the test).  ``_close_db()`` runs
        while the worker is blocked, bumping the generation.  Once
        released, the worker finishes and stamps ``tl.gen``.  Its
        *second* ``_get_db()`` must detect the connection as stale and
        reconnect — the buggy code stamped the post-bump generation and
        returned the same (retired) connection object.
        """
        results: dict[str, object] = {}
        started = threading.Event()

        def worker() -> None:
            started.set()
            conn1 = th._get_db()
            conn2 = th._get_db()
            results["same"] = conn1 is conn2

        with th._init_tables_lock:
            t = threading.Thread(target=worker)
            t.start()
            assert started.wait(5)
            # Give the worker time to open its connection and block on
            # the DDL lock we are holding.
            time.sleep(0.3)
            th._close_db()  # bumps _db_generation while worker is mid-creation
        t.join(10)
        assert not t.is_alive()
        assert results["same"] is False, (
            "connection created concurrently with _close_db() survived the "
            "generation bump — _get_db stamped the post-bump generation"
        )


class TestA10SaveLastModelLostUpdate:
    """A10: ``_save_last_model`` must not clobber concurrent writes to other keys."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = Path(self.tmpdir) / ".kiss"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        vc.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    def teardown_method(self) -> None:
        vc.CONFIG_DIR, vc.CONFIG_PATH = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_work_dir_update_survives_save_last_model(self) -> None:
        """A ``work_dir`` write racing ``_save_last_model`` must never be lost.

        One thread hammers ``_save_last_model``; the main thread
        repeatedly persists a fresh ``work_dir`` marker and verifies it
        survives.  A large pass-through padding key widens the
        read-modify-write window so the buggy full-dict overlay loses
        the marker within a few seconds of attempts.
        """
        # Big non-DEFAULTS payload: passes through load/save untouched
        # but makes each JSON read/write measurably slower, widening
        # the race window without any artificial hooks.
        vc.CONFIG_PATH.write_text(json.dumps({"padding": "x" * 500_000}))

        stop = threading.Event()

        def hammer() -> None:
            i = 0
            while not stop.is_set():
                th._save_last_model(f"model-{i}")
                i += 1

        t = threading.Thread(target=hammer)
        t.start()
        lost: str | None = None
        try:
            deadline = time.time() + 8
            n = 0
            while time.time() < deadline:
                marker = f"/marker-{n}"
                vc.save_config({"work_dir": marker})
                time.sleep(random.random() * 0.01)
                if vc.load_config()["work_dir"] != marker:
                    lost = marker
                    break
                n += 1
        finally:
            stop.set()
            t.join(10)
        assert lost is None, (
            f"work_dir marker {lost!r} was clobbered by a concurrent "
            "_save_last_model full-config overlay"
        )
        # The hammered preference itself must have been persisted.
        assert str(vc.load_config()["last_model"]).startswith("model-")
        # And the unrelated pass-through key must still be intact.
        raw = json.loads(vc.CONFIG_PATH.read_text())
        assert len(raw["padding"]) == 500_000


class TestB1BaselineCommitRoundtrip:
    """B1: the baseline SHA must actually be persisted to git config."""

    def test_save_persists_to_git_config(self, tmp_path: Path) -> None:
        """``branch.<b>.kiss-baseline`` holds exactly what was saved."""
        repo = _init_repo(tmp_path)
        _git(repo, "branch", "kiss-wt-b1")
        sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
        assert GitWorktreeOps.save_baseline_commit(repo, "kiss-wt-b1", sha) is True
        stored = _git(repo, "config", "branch.kiss-wt-b1.kiss-baseline")
        assert stored.stdout.strip() == sha

    def test_config_unset_when_never_saved(self, tmp_path: Path) -> None:
        """Branches without a stored baseline have no config key."""
        repo = _init_repo(tmp_path)
        _git(repo, "branch", "kiss-wt-empty")
        result = _git(
            repo, "config", "branch.kiss-wt-empty.kiss-baseline", check=False
        )
        assert result.returncode != 0
        assert result.stdout.strip() == ""


class TestC2RegisterOverwriteWarning:
    """C2: silent overwrite of a live registry entry must emit a WARNING."""

    _TAB = "w2c2-tab"

    def teardown_method(self) -> None:
        _RunningAgentState.unregister(self._TAB)

    @staticmethod
    def _warnings(caplog: object) -> list[logging.LogRecord]:
        records: list[logging.LogRecord] = caplog.records  # type: ignore[attr-defined]
        return [
            r for r in records
            if r.levelno >= logging.WARNING
            and r.name == "kiss.agents.sorcar.running_agent_state"
        ]

    def test_overwrite_of_different_state_logs_warning(self, caplog) -> None:
        """Registering a second state under the same tab_id warns and wins."""
        s1 = _RunningAgentState(self._TAB, "test-model")
        s2 = _RunningAgentState(self._TAB, "test-model")
        with caplog.at_level(
            logging.WARNING, logger="kiss.agents.sorcar.running_agent_state",
        ):
            _RunningAgentState.register(self._TAB, s1)
            assert self._warnings(caplog) == []
            _RunningAgentState.register(self._TAB, s2)
        # Semantics unchanged: last registration wins.
        assert _RunningAgentState.running_agent_states[self._TAB] is s2
        warnings = self._warnings(caplog)
        assert warnings, "overwriting register() emitted no WARNING"
        assert any(self._TAB in r.getMessage() for r in warnings)

    def test_reregister_same_state_is_silent(self, caplog) -> None:
        """Re-registering the identical state object must not warn."""
        s1 = _RunningAgentState(self._TAB, "test-model")
        with caplog.at_level(
            logging.WARNING, logger="kiss.agents.sorcar.running_agent_state",
        ):
            _RunningAgentState.register(self._TAB, s1)
            _RunningAgentState.register(self._TAB, s1)
        assert self._warnings(caplog) == []
        assert _RunningAgentState.running_agent_states[self._TAB] is s1


class TestWalSwitchLockRace(_DBSandbox):
    """WAL: first-time ``_get_db()`` must survive a concurrent peer writer.

    Regression for the ``PRAGMA journal_mode=WAL`` race: two parallel
    sub-agents opening their first connections against a fresh
    ``sorcar.db`` made one die instantly with ``sqlite3.OperationalError:
    database is locked`` because the WAL switch ran outside
    ``_init_tables_lock`` with no retry — and when a peer connection
    holds a RESERVED (write-transaction) lock, the rollback→WAL
    transition reports ``SQLITE_BUSY`` *immediately*, without consulting
    the busy handler, so no ``busy_timeout``/``connect(timeout=...)``
    can save it.  This test recreates that collision deterministically
    with a real peer connection holding an uncommitted write transaction
    (``BEGIN IMMEDIATE`` + DDL — the same RESERVED lock a parallel
    sub-agent mid-``_init_tables`` holds) on the brand-new database file
    while a real thread performs its first ``_get_db()``.  The pre-fix
    code raises instantly here; the fixed code retries the WAL switch
    until the peer releases the lock.
    """

    def test_wal_pragma_survives_concurrent_reserved_writer(self) -> None:
        """``_get_db()`` retries the WAL switch instead of raising."""
        # A peer (standing in for a parallel sub-agent thread mid-DDL or
        # another process such as the kiss-web daemon) creates the fresh
        # database file and holds a RESERVED lock via an uncommitted
        # write transaction.  A RESERVED peer makes the WAL transition
        # fail with an IMMEDIATE "database is locked" (busy handler
        # bypassed), which is exactly the field failure.
        peer = sqlite3.connect(str(th._DB_PATH), check_same_thread=False)
        peer.execute("BEGIN IMMEDIATE")
        peer.execute("CREATE TABLE _peer_probe (x INTEGER)")

        errors: list[BaseException] = []
        results: dict[str, str] = {}
        started = threading.Event()

        def worker() -> None:
            started.set()
            try:
                conn = th._get_db()
                results["journal"] = str(
                    conn.execute("PRAGMA journal_mode").fetchone()[0]
                )
                # The schema must have been created despite the initial
                # lock collision.
                conn.execute("SELECT COUNT(*) FROM task_history").fetchone()
            except BaseException as exc:  # noqa: BLE001 — recorded for assert
                errors.append(exc)

        thread = threading.Thread(target=worker)
        try:
            thread.start()
            assert started.wait(5), "worker thread never started"
            # Keep the RESERVED lock held long enough that the worker's
            # WAL switch definitely collides with it (the buggy code
            # raised instantly here; the fixed code retries).
            time.sleep(0.5)
        finally:
            peer.execute("ROLLBACK")
            peer.close()
        thread.join(15)

        assert not thread.is_alive(), "worker thread hung in _get_db()"
        assert errors == [], f"_get_db() raised under lock contention: {errors!r}"
        assert results["journal"] == "wal"
