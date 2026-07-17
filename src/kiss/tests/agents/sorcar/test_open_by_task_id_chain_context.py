# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: opening a tab by task id seeds the first run's chat
context from the opened task's ``parent_task_id`` chain.

Spec
----
When a tab is opened by a task id (``resumeSession`` carrying
``taskId``) and NO task has been run in the tab after opening, the
next task the user issues in that tab must have its
:meth:`ChatSorcarAgent.build_chat_prompt` context built by traversing
the ``parent_task_id`` links starting at the opened task (walking
parents upward, then reversed to chronological order) and then taking
at most ``MAX_TASKS`` (10) entries exactly the way the current
chat-context trimming works (``del ctx[2:2 + len(ctx) - 10]`` — keep
the first 2 and the last 8).

After that first run consumes the opened-task seed, subsequent runs
in the same tab fall back to the normal chat-context path.  The seed
is also discarded when the tab is re-pointed at a chat without a
task id (``resumeSession`` without ``taskId``), when the user starts
a new chat in the tab (``newChat``), and when the tab is closed.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _load_task_chain_context
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.server import VSCodeServer


def _redirect_db(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _init_git_repo(tmpdir: str) -> None:
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmpdir,
                   capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir,
                   capture_output=True)
    Path(tmpdir, ".gitkeep").touch()
    # The redirected sorcar DB lives at <repo>/.kiss/; its -wal/-shm
    # sidecar files appearing mid-task would otherwise trip the
    # post-task dirty-files merge review and block follow-up runs.
    Path(tmpdir, ".gitignore").write_text(".kiss/\n")
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir,
                   capture_output=True)


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]], threading.Lock]:
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    printer = server.printer

    def capture(event: dict[str, Any]) -> None:
        ev = printer._inject_task_id(event)
        with printer._lock:
            printer._record_event(ev)
        printer._persist_event(ev)
        with lock:
            events.append(ev)

    printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


class _PromptCapture:
    """Captures the ``prompt_template`` the underlying agent receives
    and returns a canned YAML success result (no LLM calls)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.counter = 0

    def fake_run(self, agent_self: object, **kwargs: object) -> str:
        self.prompts.append(str(kwargs.get("prompt_template", "")))
        self.counter += 1
        summary = f"run-{self.counter} done"
        printer = kwargs.get("printer")
        if printer is not None and hasattr(printer, "broadcast"):
            cast(Any, printer).broadcast({
                "type": "result",
                "text": summary,
                "success": True,
            })
        return f"success: true\nsummary: {summary}\n"


def _patch_parent_run(capture: _PromptCapture) -> Any:
    parent = cast(Any, SorcarAgent.__mro__[1])
    original = parent.run

    def _run_proxy(self_agent: object, **kwargs: object) -> str:
        return capture.fake_run(self_agent, **kwargs)

    parent.run = _run_proxy
    return original


def _unpatch_parent_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


def _run_and_wait(server: VSCodeServer, tab_id: str, prompt: str,
                  work_dir: str) -> None:
    server._handle_command({
        "type": "run", "prompt": prompt,
        "model": "claude-fable-5", "workDir": work_dir,
        "tabId": tab_id,
    })
    # ``_cmd_run`` stamps ``task_thread`` before returning, and the
    # worker thread's outer ``finally`` (``_run_task``) resets it to
    # None once the task finishes.  A near-instant run (e.g. one
    # rejected by the ``is_merging`` merge-review guard) can complete
    # before this thread reads the attribute, so ``None`` here means
    # "already finished", not "never started" — only join a thread
    # that is still registered.
    t = server._get_tab(tab_id).task_thread
    if t is not None:
        t.join(timeout=10)
        assert not t.is_alive()


def _seed_chain(
    specs: list[tuple[str, str, str]],
) -> list[tuple[str, str]]:
    """Persist a parent-linked chain of finished tasks.

    Args:
        specs: List of ``(task_text, result_text, chat_id)`` tuples in
            chronological (root-first) order.  Each row's
            ``parent_task_id`` points at the previously inserted row.

    Returns:
        List of ``(task_id, chat_id)`` for the inserted rows.
    """
    out: list[tuple[str, str]] = []
    parent = ""
    for task_text, result_text, chat_id in specs:
        extra: dict[str, object] | None = (
            {"parent_task_id": parent} if parent else None
        )
        tid, cid = th._add_task(task_text, chat_id=chat_id, extra=extra)
        th._save_task_result(result_text, task_id=tid)
        out.append((tid, cid))
        parent = tid
    return out


def _open_by_task_id(server: VSCodeServer, tab_id: str, chat_id: str,
                     task_id: str) -> None:
    """Simulate the frontend opening a fresh tab for a history task."""
    server._handle_command({"type": "newChat", "tabId": tab_id})
    server._handle_command({
        "type": "resumeSession",
        "chatId": chat_id,
        "taskId": task_id,
        "tabId": tab_id,
    })


class _ChainContextBase(unittest.TestCase):
    """Shared fixture: temp DB + git repo + captured-prompt server."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server, self.events, self.lock = _make_server()
        self.capture = _PromptCapture()
        self.original_run = _patch_parent_run(self.capture)

    def tearDown(self) -> None:
        _unpatch_parent_run(self.original_run)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestChainContextFirstRun(_ChainContextBase):
    """First run after open-by-task-id uses the parent-chain context."""

    def test_chain_context_spans_chats_and_orders_root_first(self) -> None:
        # T1 <- T2 live in chat A; T3 (parent T2) lives in chat B, so
        # chain context and chat context are observably different.
        rows = _seed_chain([
            ("t1 magic word apple", "r1 apple stored", "chatA"),
            ("t2 magic word pear", "r2 pear stored", "chatA"),
            ("t3 magic word plum", "r3 plum stored", "chatB"),
        ])
        t3_id = rows[2][0]

        tab = "tab-chain-basic"
        _open_by_task_id(self.server, tab, "chatB", t3_id)
        _run_and_wait(self.server, tab, "recall all magic words",
                      self.tmpdir)

        assert self.capture.prompts, "expected a captured prompt"
        prompt = self.capture.prompts[0]
        assert "Previous tasks and results" in prompt, prompt
        assert "### Task 1\nt1 magic word apple" in prompt, prompt
        assert "### Result 1\nr1 apple stored" in prompt, prompt
        assert "### Task 2\nt2 magic word pear" in prompt, prompt
        assert "### Result 2\nr2 pear stored" in prompt, prompt
        assert "### Task 3\nt3 magic word plum" in prompt, prompt
        assert "### Result 3\nr3 plum stored" in prompt, prompt
        assert prompt.rstrip().endswith("recall all magic words")

    def test_second_run_falls_back_to_chat_context(self) -> None:
        rows = _seed_chain([
            ("t1 magic word apple", "r1 apple stored", "chatA"),
            ("t2 magic word pear", "r2 pear stored", "chatA"),
            ("t3 magic word plum", "r3 plum stored", "chatB"),
        ])
        t3_id = rows[2][0]

        tab = "tab-chain-oneshot"
        _open_by_task_id(self.server, tab, "chatB", t3_id)
        _run_and_wait(self.server, tab, "first follow-up quokka",
                      self.tmpdir)
        _run_and_wait(self.server, tab, "second follow-up wombat",
                      self.tmpdir)

        assert len(self.capture.prompts) == 2
        second = self.capture.prompts[1]
        # Normal chat-context path for chat B: T3 is excluded from
        # _load_chat_context (non-empty parent_task_id), but the first
        # follow-up run in this tab is included.  Chain-only members
        # from chat A must NOT leak in.
        assert "Previous tasks and results" in second, second
        assert "first follow-up quokka" in second, second
        assert "t1 magic word apple" not in second, second
        assert "t2 magic word pear" not in second, second

    def test_chain_trimmed_to_max_tasks(self) -> None:
        specs = [
            (f"chain-task-{i:02d} sentinel{i:02d}",
             f"chain-result-{i:02d}", "chatX")
            for i in range(1, 14)
        ]
        rows = _seed_chain(specs)
        last_id = rows[-1][0]

        tab = "tab-chain-trim"
        _open_by_task_id(self.server, tab, "chatX", last_id)
        _run_and_wait(self.server, tab, "trim check", self.tmpdir)

        prompt = self.capture.prompts[0]
        # 13 entries -> del [2:5] -> keep 1,2 then 6..13 (10 total).
        for kept in (1, 2, 6, 7, 8, 9, 10, 11, 12, 13):
            assert f"sentinel{kept:02d}" in prompt, (kept, prompt)
        for dropped in (3, 4, 5):
            assert f"sentinel{dropped:02d}" not in prompt, (dropped, prompt)
        # Numbering is contiguous after the trim.
        assert "### Task 1\nchain-task-01" in prompt, prompt
        assert "### Task 3\nchain-task-06" in prompt, prompt
        assert "### Task 10\nchain-task-13" in prompt, prompt

    def test_cycle_in_parent_links_terminates(self) -> None:
        rows = _seed_chain([
            ("cyc-a task", "cyc-a result", "chatC"),
            ("cyc-b task", "cyc-b result", "chatC"),
        ])
        a_id, b_id = rows[0][0], rows[1][0]
        # Close the loop: a's parent = b (b's parent is already a).
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET parent_task_id = ? WHERE id = ?",
                (b_id, a_id),
            )
            db.commit()

        tab = "tab-chain-cycle"
        _open_by_task_id(self.server, tab, "chatC", b_id)
        _run_and_wait(self.server, tab, "cycle check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert "cyc-a task" in prompt, prompt
        assert "cyc-b task" in prompt, prompt

    def test_rejected_run_preserves_seed(self) -> None:
        rows = _seed_chain([
            ("t1 magic word apple", "r1 apple stored", "chatA"),
            ("t2 magic word pear", "r2 pear stored", "chatB"),
        ])
        t2_id = rows[1][0]

        tab = "tab-chain-rejected"
        _open_by_task_id(self.server, tab, "chatB", t2_id)
        # A submit rejected by the merge-review guard runs no task, so
        # it must NOT consume the opened-by-task-id seed.
        self.server._get_tab(tab).is_merging = True
        _run_and_wait(self.server, tab, "rejected run", self.tmpdir)
        assert not self.capture.prompts, "rejected run must not reach agent"
        assert self.server._tab_opened_task_ids.get(tab) == t2_id

        self.server._get_tab(tab).is_merging = False
        _run_and_wait(self.server, tab, "accepted run", self.tmpdir)
        prompt = self.capture.prompts[0]
        assert "t1 magic word apple" in prompt, prompt
        assert "t2 magic word pear" in prompt, prompt

    def test_missing_parent_row_stops_traversal(self) -> None:
        tid, _ = th._add_task(
            "orphan task zebra", chat_id="chatD",
            extra={"parent_task_id": "no-such-task-id"},
        )
        th._save_task_result("orphan result", task_id=tid)

        tab = "tab-chain-orphan"
        _open_by_task_id(self.server, tab, "chatD", tid)
        _run_and_wait(self.server, tab, "orphan check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert "### Task 1\norphan task zebra" in prompt, prompt
        assert "### Task 2" not in prompt, prompt


class TestChainContextCleared(_ChainContextBase):
    """The opened-task seed is discarded on re-open / newChat / close."""

    def test_resume_without_task_id_clears_seed(self) -> None:
        rows = _seed_chain([
            ("t1 magic word apple", "r1 apple stored", "chatA"),
            ("t2 magic word pear", "r2 pear stored", "chatA"),
        ])
        t2_id = rows[1][0]

        tab = "tab-clear-resume"
        _open_by_task_id(self.server, tab, "chatA", t2_id)
        # Re-open the same tab pointed at the chat WITHOUT a task id.
        self.server._handle_command({
            "type": "resumeSession", "chatId": "chatA", "tabId": tab,
        })
        _run_and_wait(self.server, tab, "post-clear check", self.tmpdir)

        prompt = self.capture.prompts[0]
        # Chat-context path: T2 (non-empty parent_task_id) is filtered
        # out; T1 remains.
        assert "t1 magic word apple" in prompt, prompt
        assert "t2 magic word pear" not in prompt, prompt

    def test_unknown_task_id_falls_back_to_chat_context(self) -> None:
        _seed_chain([("plain chat task koala", "plain result", "chatE")])

        tab = "tab-fallback"
        _open_by_task_id(self.server, tab, "chatE", "missing-task-id")
        _run_and_wait(self.server, tab, "fallback check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert "plain chat task koala" in prompt, prompt

    def test_new_chat_clears_seed(self) -> None:
        rows = _seed_chain([
            ("t1 magic word apple", "r1 apple stored", "chatA"),
        ])
        t1_id = rows[0][0]

        tab = "tab-clear-newchat"
        _open_by_task_id(self.server, tab, "chatA", t1_id)
        self.server._handle_command({"type": "newChat", "tabId": tab})
        _run_and_wait(self.server, tab, "fresh chat check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert "Previous tasks and results" not in prompt, prompt
        assert prompt.startswith("# Task\n"), prompt

    def test_close_tab_clears_seed(self) -> None:
        rows = _seed_chain([
            ("t1 magic word apple", "r1 apple stored", "chatA"),
        ])
        t1_id = rows[0][0]

        tab = "tab-clear-close"
        _open_by_task_id(self.server, tab, "chatA", t1_id)
        assert self.server._tab_opened_task_ids.get(tab) == t1_id
        self.server._handle_command({"type": "closeTab", "tabId": tab})
        assert tab not in self.server._tab_opened_task_ids


class TestChainContextPrimitives(_ChainContextBase):
    """Direct behavior of the persistence helper and agent seed API."""

    def test_load_task_chain_context_empty_id(self) -> None:
        assert _load_task_chain_context("") == []

    def test_load_task_chain_context_unknown_id(self) -> None:
        assert _load_task_chain_context("nope") == []

    def test_load_task_chain_context_order(self) -> None:
        rows = _seed_chain([
            ("root", "root-res", "chatZ"),
            ("mid", "mid-res", "chatZ"),
            ("leaf", "leaf-res", "chatZ"),
        ])
        ctx = _load_task_chain_context(rows[-1][0])
        assert [e["task"] for e in ctx] == ["root", "mid", "leaf"]
        assert [e["result"] for e in ctx] == [
            "root-res", "mid-res", "leaf-res",
        ]

    def test_resume_from_task_id_empty_is_noop(self) -> None:
        agent = ChatSorcarAgent("chain-noop")
        agent.resume_from_task_id("")
        assert agent._context_task_id == ""

    def test_resume_from_task_id_seeds_one_prompt_only(self) -> None:
        rows = _seed_chain([
            ("root", "root-res", "chatZ"),
            ("leaf", "leaf-res", "chatZ"),
        ])
        agent = ChatSorcarAgent("chain-seed")
        agent.resume_from_task_id(rows[-1][0])
        first = agent.build_chat_prompt("go")
        assert "### Task 1\nroot" in first
        assert "### Task 2\nleaf" in first
        # Consumed: the next prompt uses the (empty) chat context.
        second = agent.build_chat_prompt("go again")
        assert second == "# Task\ngo again"


if __name__ == "__main__":
    unittest.main()
