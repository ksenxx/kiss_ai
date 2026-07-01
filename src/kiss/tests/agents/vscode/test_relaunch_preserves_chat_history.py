# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: after VS Code is closed and relaunched, submitting a new
task to a restored tab must continue the tab's prior chat session — the
agent must receive the "Previous tasks and results" preamble built from
the tasks that ran in that chat before the restart.

Reproduces the following bug:

1. User runs a task in a VS Code tab.  The tab's frontend state records
   ``backendChatId = X`` and the daemon persists the task under
   ``chat_id = X``.
2. User closes VS Code (which stops the daemon).
3. User relaunches VS Code.  The webview restores its tabs from
   ``vscode.getState()`` and posts a ``ready`` message whose
   ``restoredTabs`` array names ``{tabId=T, chatId=X}``.  The
   extension replays this as a ``resumeSession`` command to the freshly
   started daemon.
4. The daemon's ``_replay_session`` handles ``resumeSession`` by
   recording ``_tab_chat_views[T] = X`` and, when a
   ``_RunningAgentState[T]`` already exists, by writing
   ``tab.chat_id = X``.  After a cold daemon restart no state exists
   yet, so *only* ``_tab_chat_views`` gets the association.
5. User submits a new prompt on the SAME tab.  ``_cmd_run`` opens a
   fresh ``_RunningAgentState`` for ``T`` with an empty ``chat_id``,
   mints a brand new UUID chat id, and runs the task.  The
   ``ChatSorcarAgent.build_chat_prompt`` call therefore queries
   ``_load_chat_context`` for the FRESH uuid, finds nothing, and sends
   the LLM an empty preamble — the previous conversation is lost.

The end-to-end test drives the real :class:`VSCodeServer` through
``_handle_command`` exactly as the extension layer would, from a cold
start.  Only :meth:`SorcarAgent.__mro__[1].run` (the grandparent's
plain ``run``) is stubbed so we can capture the augmented prompt that
:meth:`ChatSorcarAgent.build_chat_prompt` produced and inspect the
final ``tab.chat_id`` recorded on the tab state.
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

import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _flush_chat_events,
    _load_chat_context,
    _save_task_result,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer
from kiss.core.models.model_info import get_available_models


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
    subprocess.run(
        ["git", "config", "user.email", "t@t"],
        cwd=tmpdir, capture_output=True, check=False,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=tmpdir, capture_output=True, check=False,
    )
    Path(tmpdir, ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True,
    )


def _patch_grandparent_run_capture(
    captured_prompts: list[str], done: threading.Event,
) -> Any:
    """Replace the LLM ``run`` with a stub that records the prompt.

    The grandparent of :class:`SorcarAgent` (the base agent) is what
    :meth:`ChatSorcarAgent.run` ultimately delegates to via
    ``super().run(prompt_template=agent_prompt, ...)`` — where
    ``agent_prompt`` is exactly the chat-history-augmented prompt
    produced by :meth:`ChatSorcarAgent.build_chat_prompt`.  Capturing
    it here proves whether the previous conversation reached the LLM.
    """
    parent = cast(Any, SorcarAgent.__mro__[1])
    original = parent.run

    def _run_proxy(self_agent: Any, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt_template", "")
        if isinstance(prompt, str):
            captured_prompts.append(prompt)
        try:
            return str(yaml.dump({"success": True, "summary": "ok"}))
        finally:
            done.set()

    parent.run = _run_proxy
    return original


def _unpatch_grandparent_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


def _wait_until(predicate: Any, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestRelaunchPreservesChatHistory(unittest.TestCase):
    """End-to-end: after a daemon cold-start, a restored tab's follow-up
    task must inherit the prior chat_id (and the LLM prompt must include
    the previous conversation)."""

    def setUp(self) -> None:
        models = get_available_models()
        if not models:
            self.skipTest("no model API key configured")
        self.model = models[0]
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)

    def tearDown(self) -> None:
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_restored_tab_submit_preserves_prior_chat(self) -> None:
        # ------------------------------------------------------------------
        # 1) Simulate the state left behind by a prior VS Code session:
        #    one completed task persisted under a specific chat_id.
        #    In production this row would have been created by
        #    ``ChatSorcarAgent.run``'s _add_task/_save_task_result path.
        # ------------------------------------------------------------------
        prior_chat_id = "chat-relaunch-fixture-000000000001"
        task_id, chat_id = _add_task(
            "First user prompt from the prior session",
            chat_id=prior_chat_id,
            extra={
                "model": self.model,
                "work_dir": self.tmpdir,
                "version": "test",
            },
        )
        assert chat_id == prior_chat_id
        _save_task_result(
            task_id=task_id,
            result="Prior assistant answer that the LLM must see next time.",
        )
        _flush_chat_events()

        # Sanity — the chat context loads with exactly one entry.
        ctx = _load_chat_context(prior_chat_id)
        assert len(ctx) == 1, ctx
        assert "prior session" in str(ctx[0]["task"] or "")
        assert "Prior assistant answer" in str(ctx[0]["result"] or "")

        # ------------------------------------------------------------------
        # 2) Simulate VS Code relaunch by instantiating a FRESH
        #    :class:`VSCodeServer` (mirrors a daemon cold-start with an
        #    empty ``_RunningAgentState.running_agent_states`` /
        #    ``_tab_chat_views``).  The webview then restores its tabs
        #    and the extension replays ``resumeSession`` per restored
        #    tab, exactly as ``SorcarSidebarView.ts`` does on ``ready``.
        # ------------------------------------------------------------------
        # Guarantee that the fresh server does NOT inherit any per-tab
        # state from a previous test run of the same in-process module.
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState
        _RunningAgentState.running_agent_states.clear()

        server = VSCodeServer()

        captured_prompts: list[str] = []
        done = threading.Event()
        original_run = _patch_grandparent_run_capture(captured_prompts, done)
        try:
            tab_id = "tab-restored-after-relaunch"
            server._handle_command(
                {
                    "type": "resumeSession",
                    "chatId": prior_chat_id,
                    "tabId": tab_id,
                },
            )

            # ------------------------------------------------------------------
            # 3) User submits a follow-up prompt on the restored tab.
            #    Under the bug this call mints a fresh chat_id and the
            #    agent's ``build_chat_prompt`` returns just the new
            #    prompt with no "Previous tasks and results" preamble.
            # ------------------------------------------------------------------
            server._handle_command(
                {
                    "type": "run",
                    "prompt": "Second user prompt after the relaunch",
                    "model": self.model,
                    "workDir": self.tmpdir,
                    "tabId": tab_id,
                    "autoCommit": True,
                },
            )

            assert done.wait(timeout=30), (
                "grandparent run never fired for the submitted task"
            )
            # Wait for the worker thread's outer finally to complete so
            # ``tab.chat_id`` / persistence writes are settled.
            assert _wait_until(
                lambda: (
                    server._get_tab(tab_id).task_thread is None
                    or not server._get_tab(tab_id).task_thread.is_alive()  # type: ignore[union-attr]
                ),
                timeout=30,
            ), "task worker thread never completed"
        finally:
            _unpatch_grandparent_run(original_run)

        # ------------------------------------------------------------------
        # 4) The restored tab's chat_id MUST still be the prior chat_id.
        #    Under the bug it is a freshly-minted UUID unrelated to
        #    ``prior_chat_id``.
        # ------------------------------------------------------------------
        tab = server._get_tab(tab_id)
        self.assertEqual(
            tab.chat_id, prior_chat_id,
            f"Restored tab lost its prior chat_id after relaunch: "
            f"tab.chat_id={tab.chat_id!r} expected={prior_chat_id!r}",
        )

        # ------------------------------------------------------------------
        # 5) The prompt the LLM actually saw must contain the previous
        #    task's text and result — i.e.
        #    ``ChatSorcarAgent.build_chat_prompt`` must have found the
        #    prior chat context and prepended it.
        # ------------------------------------------------------------------
        self.assertTrue(
            captured_prompts,
            "grandparent run captured no prompt at all",
        )
        agent_prompt = captured_prompts[-1]
        self.assertIn(
            "Previous tasks and results", agent_prompt,
            "The LLM prompt is missing the chat-history preamble — "
            "the prior conversation was not surfaced to the model.",
        )
        self.assertIn(
            "First user prompt from the prior session", agent_prompt,
            "The prior task text is missing from the LLM prompt.",
        )
        self.assertIn(
            "Prior assistant answer", agent_prompt,
            "The prior task result is missing from the LLM prompt.",
        )
        self.assertIn(
            "Second user prompt after the relaunch", agent_prompt,
            "The newly submitted prompt is missing from the LLM prompt.",
        )

        # ------------------------------------------------------------------
        # 6) The new task must have been persisted under the SAME
        #    chat_id as the prior task — otherwise the chat is split
        #    into two orphan sessions in the history sidebar.
        # ------------------------------------------------------------------
        _flush_chat_events()
        ctx_after = _load_chat_context(prior_chat_id)
        titles = [str(e.get("task", "") or "") for e in ctx_after]
        self.assertTrue(
            any("Second user prompt after the relaunch" in t for t in titles),
            f"New task was not persisted under the prior chat_id; "
            f"history for {prior_chat_id!r} = {titles!r}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
