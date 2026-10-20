# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for Wave-2 findings (tmp/findings-2.md F5, F6, F8, F15).

Each test drives the real agent classes against a fresh git repo with an
isolated persistence DB.  No mocking libraries are used; where a code
path is only reachable through an LLM call, the *parent* ``run`` of
``SorcarAgent`` is temporarily replaced with a deterministic function
(the same convention as ``test_fixer3_agent_bugs.py``) and the commit
message generator runs with no provider configured
(:class:`OfflineFastModel`) so it takes its real fallback path.

Covered findings:

* F5 — ``_flush_warnings`` performed an unlocked check-then-clear on
  ``_merge_conflict_warning``: two concurrent
  flushes could broadcast the same warning twice, and a warning written
  while a flush was broadcasting was wiped by the flush's ``= None``.
* F6 — the commit toast id lived on ``self._commit_run_id``; a
  concurrent commit on the same instance overwrote it mid-flight,
  pairing one call's "committed" toast with the other call's
  "generating" toast (and the fallback minted a *fresh* id per stage,
  re-introducing the stacked-toast bug it existed to fix).  Now
  ``_commit_worktree`` mints the id per invocation and both stages of
  one commit carry it.
* F8 — ``ChatSorcarAgent.run`` minted ``chat_id`` with an inline
  ``uuid.uuid4().hex`` while ``ChatSorcarAgent.run`` (whose comment
  claimed the mintings were identical) routes through
  ``persistence._allocate_chat_id``.
* F15 — ``new_chat()`` dropped the warning set when it parked the
  pending worktree on its branch (which branch holds the work, and
  that it was not merged): it was only surfaced by a *later*
  ``run()``; a user who opened a new chat and never ran another task
  lost it silently.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.chat_sorcar_agent as chat_module
import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.json_printer import JsonPrinter
from kiss.tests.server.conftest import init_repo, run_git
from kiss.tests.server.parallel_agent_harness import OfflineFastModel

_PARENT_CLASS = cast(Any, SorcarAgent.__mro__[1])


def _ok_run(self: Any, *args: Any, **kwargs: Any) -> str:
    """Deterministic parent-run replacement: always succeeds fast."""
    return "success: true\nsummary: wave2 done\n"


class _RecordingPrinter(JsonPrinter):
    """JsonPrinter whose ``broadcast`` records events thread-safely.

    ``broadcast_delay`` (seconds) widens the race window for the F5
    concurrency tests: it makes the broadcast slow enough that a
    concurrent flush / writer deterministically lands inside it.
    """

    def __init__(self, broadcast_delay: float = 0.0) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.broadcast_delay = broadcast_delay
        self._events_lock = threading.Lock()
        self.in_broadcast = threading.Event()

    def broadcast(self, event: dict[str, Any]) -> None:  # type: ignore[override]
        self.in_broadcast.set()
        if self.broadcast_delay:
            time.sleep(self.broadcast_delay)
        with self._events_lock:
            self.events.append(dict(event))

    def by_type(self, event_type: str) -> list[dict[str, Any]]:
        with self._events_lock:
            return [e for e in self.events if e.get("type") == event_type]


class _Base(unittest.TestCase):
    """Fresh git repo + isolated persistence DB per test."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wave2-test-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        init_repo(self.repo)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "history.db"
        _persistence._db_conn = None

        self._original_parent_run = _PARENT_CLASS.run
        self.offline = OfflineFastModel()
        self.offline.__enter__()

    def tearDown(self) -> None:
        _PARENT_CLASS.run = self._original_parent_run
        self.offline.__exit__(None, None, None)

        from kiss.server import agent_state

        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

        if _persistence._db_conn is not None:
            try:
                _persistence._db_conn.close()
            except Exception:  # pragma: no cover — cleanup best-effort
                pass
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _setup_worktree_agent(self) -> ChatSorcarAgent:
        """Real worktree on a fresh branch with one uncommitted file."""
        agent = ChatSorcarAgent("wave2-wt")
        wt_work = agent._try_setup_worktree(Path(self.repo), self.repo)
        assert wt_work is not None, "worktree setup failed"
        assert agent._wt is not None
        Path(agent._wt.wt_dir, "uncommitted.txt").write_text("pending work\n")
        return agent


class TestF5ConcurrentFlushBroadcastsOnce(_Base):
    """F5: two concurrent flushes must not broadcast a warning twice."""

    def test_concurrent_flushes_emit_exactly_one_warning(self) -> None:
        agent = ChatSorcarAgent("wave2-f5-double")
        agent._merge_conflict_warning = "only-once warning"
        printer = _RecordingPrinter(broadcast_delay=0.05)
        barrier = threading.Barrier(2)

        def flush() -> None:
            barrier.wait()
            agent._flush_warnings(printer)

        threads = [threading.Thread(target=flush) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        warnings = printer.by_type("warning")
        self.assertEqual(
            len(warnings), 1,
            f"warning broadcast {len(warnings)} times, expected exactly "
            f"once: {warnings!r}",
        )
        self.assertIsNone(agent._merge_conflict_warning)

    def test_warning_set_during_flush_is_not_lost(self) -> None:
        """A warning written mid-broadcast must survive to the next flush.

        Mirrors the real interleaving: ``_release_worktree`` (or the
        daemon's ``task_runner``) sets a NEW warning while a starting
        ``run()`` is still broadcasting the previous one.  The old
        code cleared the attribute *after* the broadcast, wiping the
        new warning.
        """
        agent = ChatSorcarAgent("wave2-f5-lost")
        agent._merge_conflict_warning = "first warning"
        printer = _RecordingPrinter(broadcast_delay=0.1)

        flusher = threading.Thread(
            target=agent._flush_warnings, args=(printer,),
        )
        flusher.start()
        self.assertTrue(printer.in_broadcast.wait(timeout=5.0))
        agent.add_warning("second warning")
        flusher.join()

        agent._flush_warnings(printer)
        messages = [e.get("message") for e in printer.by_type("warning")]
        self.assertIn("first warning", messages)
        self.assertIn(
            "second warning", messages,
            "warning set during a concurrent flush was silently lost",
        )


class TestF6CommitToastIdIsPerInvocation(_Base):
    """F6: 'generating' and 'committed' toasts share one per-call id.

    ``_commit_worktree`` mints the toast id locally per invocation, so
    nothing on the instance can be re-stamped by a concurrent call; the
    test checks the observable contract that both stages of one commit
    carry the same id and the "committed" toast names the real commit.
    """

    def test_generating_and_committed_toasts_share_one_id(self) -> None:
        agent = self._setup_worktree_agent()
        assert agent._wt is not None
        wt_dir = agent._wt.wt_dir
        printer = _RecordingPrinter()
        agent.printer = printer

        committed = agent._commit_worktree()

        self.assertTrue(committed)
        subject = run_git(str(wt_dir), "log", "-1", "--format=%s").stdout
        self.assertTrue(subject.strip())
        tracked = run_git(str(wt_dir), "ls-files").stdout.split()
        self.assertIn("uncommitted.txt", tracked)

        notes = printer.by_type("notification")
        self.assertEqual(
            [n.get("message") for n in notes],
            ["Generating commit message", f"Committed {subject.strip()}"],
        )
        generating, done = notes
        self.assertTrue(generating.get("sticky"))
        self.assertNotIn("sticky", done)
        self.assertTrue(generating.get("id"))
        self.assertEqual(
            generating.get("id"), done.get("id"),
            "'generating' and 'committed' toasts must share one id so "
            "the chat page replaces the sticky toast in place — a "
            "concurrent auto-commit on the same instance must not be "
            "able to split them",
        )
        agent.discard()


def _fixed_chat_id() -> str:
    """Deterministic ``_allocate_chat_id`` replacement."""
    return "cafe" * 8


class TestF8ChatIdMintingIsShared(_Base):
    """F8: ChatSorcarAgent.run must mint via persistence._allocate_chat_id."""

    def test_chat_run_mints_through_allocate_chat_id(self) -> None:
        _PARENT_CLASS.run = _ok_run
        original = getattr(chat_module, "_allocate_chat_id", None)
        self.assertIsNotNone(
            original,
            "chat_sorcar_agent must route chat-id minting through "
            "persistence._allocate_chat_id (same helper as "
            "ChatSorcarAgent.run), not an inline uuid4().hex",
        )
        chat_module._allocate_chat_id = _fixed_chat_id
        try:
            agent = ChatSorcarAgent("wave2-f8")
            result = agent.run(
                prompt_template="trivial task", work_dir=self.repo,
            )
        finally:
            chat_module._allocate_chat_id = original  # type: ignore[assignment]

        self.assertIn("wave2 done", result)
        self.assertEqual(agent.chat_id, "cafe" * 8)
        conn = _persistence._get_db()
        rows = conn.execute("SELECT chat_id FROM task_history").fetchall()
        self.assertTrue(rows)
        self.assertEqual(rows[0][0], "cafe" * 8)


class TestF15NewChatSurfacesReleaseWarnings(_Base):
    """F15: new_chat must flush the parked-worktree warning to the printer."""

    def test_new_chat_broadcasts_parked_worktree_warning(self) -> None:
        agent = self._setup_worktree_agent()
        assert agent._wt is not None
        wt_dir = agent._wt.wt_dir
        branch = agent._wt.branch
        printer = _RecordingPrinter()
        agent.printer = printer
        agent._chat_id = "wave2-f15-chat"

        agent.new_chat()

        warnings = printer.by_type("warning")
        self.assertTrue(
            warnings,
            "new_chat() dropped the parked-worktree warning instead of "
            "broadcasting it to the attached printer",
        )
        message = warnings[0].get("message") or ""
        self.assertIn(f"committed to branch '{branch}'", message)
        self.assertIn("not merged into 'main'", message)
        self.assertIsNone(agent._merge_conflict_warning)
        self.assertEqual(agent.chat_id, "")
        self.assertIsNone(agent._wt)
        self.assertFalse(wt_dir.exists())
        files = run_git(
            self.repo, "show", "--name-only", "--pretty=format:", branch,
        ).stdout.split()
        self.assertIn("uncommitted.txt", files)
        self.assertTrue(
            GitWorktreeOps.load_preserve_marker(Path(self.repo), branch),
        )
        self.assertEqual(
            run_git(self.repo, "rev-parse", "--abbrev-ref", "HEAD")
            .stdout.strip(),
            "main",
        )
        self.assertFalse(Path(self.repo, "uncommitted.txt").exists())

    def test_new_chat_without_printer_retains_warning(self) -> None:
        """Without a printer the warning must survive for the next run."""
        agent = self._setup_worktree_agent()
        assert agent._wt is not None
        branch = agent._wt.branch

        agent.new_chat()

        warning = agent._merge_conflict_warning or ""
        self.assertIn(
            f"committed to branch '{branch}'", warning,
            "with no broadcast-capable printer attached, new_chat() "
            "must retain the warning for the next run()'s flush",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
