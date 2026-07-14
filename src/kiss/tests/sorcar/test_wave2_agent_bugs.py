# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for Wave-2 findings (tmp/findings-2.md F5, F6, F8, F15).

Each test drives the real agent classes against a fresh git repo with an
isolated persistence DB.  No mocking libraries are used; where a code
path is only reachable through an LLM call, the module-level
``_generate_commit_message`` / the *parent* ``run`` of ``SorcarAgent``
is temporarily replaced with a deterministic function (the same
convention as ``test_fixer3_agent_bugs.py``) so the real
``WorktreeSorcarAgent`` / ``ChatSorcarAgent`` code paths under test all
execute for real.

Covered findings:

* F5 — ``_flush_warnings`` performed an unlocked check-then-clear on
  ``_stash_pop_warning`` / ``_merge_conflict_warning``: two concurrent
  flushes could broadcast the same warning twice, and a warning written
  while a flush was broadcasting was wiped by the flush's ``= None``.
* F6 — the auto-commit toast id lived on ``self._commit_run_id``; a
  concurrent ``_auto_commit_worktree`` on the same instance overwrote
  it mid-flight, pairing one call's "committed" toast with the other
  call's "generating" toast (and the fallback minted a *fresh* id per
  stage, re-introducing the stacked-toast bug it existed to fix).
* F8 — ``ChatSorcarAgent.run`` minted ``chat_id`` with an inline
  ``uuid.uuid4().hex`` while ``WorktreeSorcarAgent.run`` (whose comment
  claimed the mintings were identical) routes through
  ``persistence._allocate_chat_id``.
* F15 — ``new_chat()`` ignored ``_release_worktree`` failures: the
  merge-conflict warning (with its manual recovery commands) was only
  surfaced by a *later* ``run()``; a user who opened a new chat and
  never ran another task lost it silently.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.chat_sorcar_agent as chat_module
import kiss.agents.sorcar.persistence as _persistence
import kiss.agents.sorcar.worktree_sorcar_agent as wt_module
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.json_printer import JsonPrinter

_PARENT_CLASS = cast(Any, SorcarAgent.__mro__[1])


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


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
        _init_repo(self.repo)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

        self._original_parent_run = _PARENT_CLASS.run
        self._original_commit_msg_fn = wt_module._generate_commit_message

    def tearDown(self) -> None:
        _PARENT_CLASS.run = self._original_parent_run
        wt_module._generate_commit_message = self._original_commit_msg_fn

        from kiss.agents.sorcar.running_agent_state import _RunningAgentState

        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
        with ChatSorcarAgent._running_agents_lock:
            ChatSorcarAgent.running_agents.clear()

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

    def _setup_worktree_agent(
        self, auto_commit: bool,
    ) -> WorktreeSorcarAgent:
        """Real worktree on a fresh branch with one uncommitted file."""
        agent = WorktreeSorcarAgent("wave2-wt")
        agent.auto_commit_enabled = auto_commit
        wt_work = agent._try_setup_worktree(Path(self.repo), self.repo)
        assert wt_work is not None, "worktree setup failed"
        assert agent._wt is not None
        Path(agent._wt.wt_dir, "uncommitted.txt").write_text("pending work\n")
        return agent


class TestF5ConcurrentFlushBroadcastsOnce(_Base):
    """F5: two concurrent flushes must not broadcast a warning twice."""

    def test_concurrent_flushes_emit_exactly_one_warning(self) -> None:
        agent = WorktreeSorcarAgent("wave2-f5-double")
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
        VS Code ``task_runner``) sets a NEW warning while a starting
        ``run()`` is still broadcasting the previous one.  The old
        code cleared the attribute *after* the broadcast, wiping the
        new warning.
        """
        agent = WorktreeSorcarAgent("wave2-f5-lost")
        agent._merge_conflict_warning = "first warning"
        printer = _RecordingPrinter(broadcast_delay=0.1)

        flusher = threading.Thread(
            target=agent._flush_warnings, args=(printer,),
        )
        flusher.start()
        # Wait until the flush is inside its (slow) broadcast, then
        # write the second warning — exactly the window in which the
        # old check-then-clear implementation wiped it.
        self.assertTrue(printer.in_broadcast.wait(timeout=5.0))
        agent._set_warnings(merge="second warning")
        flusher.join()

        # The second warning must still be pending and surface on the
        # next flush.
        agent._flush_warnings(printer)
        messages = [e.get("message") for e in printer.by_type("warning")]
        self.assertIn("first warning", messages)
        self.assertIn(
            "second warning", messages,
            "warning set during a concurrent flush was silently lost",
        )


class _IntruderCommitMessageFn:
    """Deterministic ``_generate_commit_message`` replacement.

    Simulates the F6 race: while THIS auto-commit is inside its (slow)
    LLM commit-message call, a concurrent ``_auto_commit_worktree`` on
    the same agent instance stamps a new run id.  With the old
    ``self._commit_run_id`` shared-state design, the subsequent
    "committed" toast then used the intruder's id and never replaced
    this call's sticky "Generating commit message" toast.
    """

    def __init__(self, agent: WorktreeSorcarAgent) -> None:
        self.agent = agent

    def __call__(
        self,
        commit_dir: Path,
        user_prompt: str | None,
        task_result: str | None = None,
    ) -> str:
        self.agent._commit_run_id = "intruder-run-id"  # type: ignore[attr-defined]
        return "wave2 test commit"


class TestF6CommitToastIdIsPerInvocation(_Base):
    """F6: 'generating' and 'committed' toasts share one per-call id."""

    def test_toast_id_survives_concurrent_run_id_overwrite(self) -> None:
        agent = self._setup_worktree_agent(auto_commit=True)
        assert agent._wt is not None
        wt_dir = agent._wt.wt_dir
        printer = _RecordingPrinter()
        agent.printer = printer
        wt_module._generate_commit_message = cast(
            Any, _IntruderCommitMessageFn(agent),
        )

        committed = agent._auto_commit_worktree()

        self.assertTrue(committed)
        # The commit really landed with the generated message.
        subject = _run_git(str(wt_dir), "log", "-1", "--format=%s").stdout
        self.assertEqual(subject.strip(), "wave2 test commit")

        notes = printer.by_type("notification")
        self.assertEqual(
            [n.get("message") for n in notes],
            ["Generating commit message", "Committed wave2 test commit"],
        )
        generating, done = notes
        self.assertTrue(generating.get("sticky"))
        self.assertNotIn("sticky", done)
        self.assertTrue(generating.get("id"))
        self.assertEqual(
            generating.get("id"), done.get("id"),
            "'generating' and 'committed' toasts must share one id so "
            "the webview replaces the sticky toast in place — a "
            "concurrent auto-commit on the same instance must not be "
            "able to split them",
        )
        self.assertNotEqual(done.get("id"), "intruder-run-id")
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
            "WorktreeSorcarAgent.run), not an inline uuid4().hex",
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
        # The persisted task row carries the id minted by the shared
        # helper.
        conn = _persistence._get_db()
        rows = conn.execute("SELECT chat_id FROM task_history").fetchall()
        self.assertTrue(rows)
        self.assertEqual(rows[0][0], "cafe" * 8)


class TestF15NewChatSurfacesReleaseWarnings(_Base):
    """F15: new_chat must flush release warnings to the live printer."""

    def test_new_chat_broadcasts_release_failure_warning(self) -> None:
        agent = self._setup_worktree_agent(auto_commit=False)
        assert agent._wt is not None
        wt_dir = agent._wt.wt_dir
        printer = _RecordingPrinter()
        agent.printer = printer
        agent._chat_id = "wave2-f15-chat"

        agent.new_chat()

        warnings = printer.by_type("warning")
        self.assertTrue(
            warnings,
            "new_chat() dropped the _release_worktree failure warning "
            "instead of broadcasting it to the attached printer",
        )
        message = warnings[0].get("message") or ""
        self.assertIn("Auto-commit is disabled", message)
        self.assertIn(str(wt_dir), message)
        # Flushed means cleared — and the chat state was reset.
        self.assertIsNone(agent._merge_conflict_warning)
        self.assertEqual(agent.chat_id, "")
        self.assertIsNone(agent._wt)

    def test_new_chat_without_printer_retains_warning(self) -> None:
        """Without a printer the warning must survive for the next run."""
        agent = self._setup_worktree_agent(auto_commit=False)

        agent.new_chat()

        warning = agent._merge_conflict_warning or ""
        self.assertIn(
            "Auto-commit is disabled", warning,
            "with no broadcast-capable printer attached, new_chat() "
            "must retain the warning for the next run()'s flush",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
