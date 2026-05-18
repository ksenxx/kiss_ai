"""Test: loading a finished task from history and then running a new
task in the same chat tab must carry the prior chat context into the
augmented prompt the underlying agent receives.

Spec
----
1. A previous task (T1) with result R1 is persisted in ``sorcar.db``
   under chat id X.
2. The user clicks the history row for T1.  The frontend's
   ``createNewTab(s.id)`` allocates a tab with id == X and sends:

       ``newChat`` (tabId=X)  →  ``resumeSession`` (chatId=X, taskId=t1,
       tabId=X)

   so the ``tab_id == chat_id`` invariant holds.
3. The user types a new task (T2) and presses Run.  The backend's
   ``_run_task_inner`` builds a fresh agent that ultimately calls
   :meth:`SorcarAgent.run` with the augmented prompt produced by
   :meth:`ChatSorcarAgent.build_chat_prompt`.
4. The augmented prompt MUST contain both the prior task T1 and its
   result R1 — otherwise the new run starts with no memory of the
   conversation, defeating the purpose of "resume chat".

The bug being reproduced: when ``_replay_session`` finishes by
re-setting ``tab.chat_id = X``, but a later ``_cmd_run`` overwrites
``tab.chat_id`` with the tab id passed in the command and the chat
context lookup ends up pointing at a chat id that has no rows.

The test patches the LLM step so no real API calls are made; it
captures the ``prompt_template`` actually passed to the underlying
agent and asserts the prior task text appears verbatim.
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
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer


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
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir,
                   capture_output=True)


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]], threading.Lock]:
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    printer = server.printer

    def capture(event: dict[str, Any]) -> None:
        # Mirror the real ``BaseBrowserPrinter.broadcast`` side effects
        # so ``peek_recording`` (used by ``_extract_result_summary``)
        # and persistence see the event, just without the WSS transport.
        ev = printer._inject_tab_id(event)
        with printer._lock:
            printer._record_event(ev)
        printer._persist_event(ev)
        with lock:
            events.append(ev)

    printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


class _PromptCapture:
    """Captures the ``prompt_template`` argument the underlying agent
    receives, then returns a canned YAML success result so the task
    completes synchronously without any LLM calls."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.summaries = ["first done", "second done"]

    def fake_run(self, agent_self: object, **kwargs: object) -> str:
        self.prompts.append(str(kwargs.get("prompt_template", "")))
        printer = kwargs.get("printer")
        summary = (
            self.summaries.pop(0) if self.summaries else "done"
        )
        # Emit a ``result`` event so the task runner's
        # ``_extract_result_summary`` picks up a non-empty summary
        # and persists it into ``task_history.result`` — matching
        # the real LLM flow where the last result event carries the
        # task summary that subsequent ``build_chat_prompt`` calls
        # surface as "### Result N".
        if printer is not None and hasattr(printer, "broadcast"):
            printer.broadcast({
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
        "model": "claude-opus-4-6", "workDir": work_dir,
        "tabId": tab_id,
    })
    t = server._get_tab(tab_id).task_thread
    assert t is not None
    t.join(timeout=10)
    assert not t.is_alive()


class TestHistoryContinuationContext(unittest.TestCase):
    """After loading a finished task from history and starting a new
    task in the same tab, the augmented prompt must include the prior
    task and its result."""

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

    def test_new_task_after_history_load_includes_prior_context(
        self,
    ) -> None:
        # Step 1: run the first task in a fresh chat tab.
        first_tab = "tab-first"
        _run_and_wait(self.server, first_tab, "remember the magic word: banana",
                      self.tmpdir)
        first_prompt = self.capture.prompts[0]
        # New chat: prompt is just the task, no prior context.
        assert "Previous tasks and results" not in first_prompt
        assert "banana" in first_prompt

        # Look up the persisted task to discover its chat_id and id.
        rows = th._load_history(limit=10)
        assert rows, "expected first task to be persisted"
        first_row = rows[0]
        chat_id = str(first_row["chat_id"])
        task_id = cast(int, first_row["id"])
        assert chat_id, "expected non-empty chat_id"

        # Step 2: simulate the user clicking the history row for that
        # task.  The frontend's ``createNewTab(chat_id)`` sends
        # ``newChat`` first, then ``resumeSession``.  The tab id in
        # both messages is the chat id (the tab_id == chat_id
        # invariant the backend relies on).
        history_tab = chat_id
        self.server._handle_command(
            {"type": "newChat", "tabId": history_tab}
        )
        self.server._handle_command({
            "type": "resumeSession",
            "id": chat_id,
            "taskId": task_id,
            "tabId": history_tab,
        })

        # After resumeSession the tab's chat_id must point at the
        # resumed session.
        tab = self.server._get_tab(history_tab)
        assert tab.chat_id == chat_id, \
            f"tab.chat_id={tab.chat_id!r}, expected {chat_id!r}"

        # Step 3: the user types and runs a follow-up task in the
        # resumed tab.
        _run_and_wait(self.server, history_tab,
                      "what was the magic word?", self.tmpdir)

        # Step 4: the captured prompt for the second run must
        # contain the prior task and its result.
        assert len(self.capture.prompts) >= 2, \
            f"expected at least 2 runs, got {len(self.capture.prompts)}"
        second_prompt = self.capture.prompts[1]
        assert "Previous tasks and results" in second_prompt, (
            "second prompt should include the chat-history preamble "
            f"but was:\n{second_prompt}"
        )
        assert "banana" in second_prompt, (
            "second prompt should include the prior task text "
            f"'banana' but was:\n{second_prompt}"
        )
        assert "first done" in second_prompt, (
            "second prompt should include the prior result 'first done' "
            f"but was:\n{second_prompt}"
        )
        assert "what was the magic word?" in second_prompt


class TestHistoryContinuationWhenTabIdMismatch(unittest.TestCase):
    """When the frontend opens a fresh tab for a finished chat and the
    tab id is NOT equal to the chat id (e.g. because the chat id is
    already taken by another open tab, so ``createNewTab`` allocated a
    fresh uuid instead of the preset chat id), the new task must still
    carry the prior chat context.

    This reproduces the bug where ``_cmd_run`` overwrites the
    ``_RunningAgentState.chat_id`` set by ``_replay_session`` with the
    incoming ``tabId``, breaking the link to the persisted chat
    session.
    """

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

    def test_followup_with_distinct_tab_id_keeps_context(self) -> None:
        # First, run a task in a chat whose chat_id will not coincide
        # with the follow-up's tab id.
        first_tab = "tab-original"
        _run_and_wait(self.server, first_tab,
                      "remember the magic word: pineapple",
                      self.tmpdir)
        rows = th._load_history(limit=10)
        assert rows
        chat_id = str(rows[0]["chat_id"])
        task_id = cast(int, rows[0]["id"])

        # Simulate a viewer browser that allocates a DIFFERENT tab id
        # to view this chat (e.g. a second viewer opening the row, or
        # an existing chat tab whose id collides so the frontend mints
        # a fresh uuid).  ``_replay_session`` sets the tab's chat_id
        # but does NOT change the tab id; the follow-up run command
        # carries the viewer's tab id, which differs from chat_id.
        history_tab = "tab-viewer"
        self.server._handle_command(
            {"type": "newChat", "tabId": history_tab}
        )
        self.server._handle_command({
            "type": "resumeSession",
            "id": chat_id,
            "taskId": task_id,
            "tabId": history_tab,
        })
        tab = self.server._get_tab(history_tab)
        assert tab.chat_id == chat_id, \
            f"replay should have populated chat_id; got {tab.chat_id!r}"

        # Run a follow-up task in the viewer tab.
        _run_and_wait(self.server, history_tab,
                      "what was the magic word?", self.tmpdir)
        assert len(self.capture.prompts) >= 2
        second_prompt = self.capture.prompts[1]
        assert "Previous tasks and results" in second_prompt, (
            "follow-up prompt should carry chat context even when "
            "tab_id != chat_id, but was:\n" + second_prompt
        )
        assert "pineapple" in second_prompt, (
            "follow-up prompt should reference the prior task text "
            "'pineapple', but was:\n" + second_prompt
        )


if __name__ == "__main__":
    unittest.main()
