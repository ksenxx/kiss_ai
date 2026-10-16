# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: after a daemon restart and a page reload, submitting a new
task to a restored tab must continue the tab's prior chat session — the
agent must receive the "Previous tasks and results" preamble built from
the tasks that ran in that chat before the restart.

Reproduces the following bug:

1. User runs a task in a chat tab.  The tab's frontend state records
   ``backendChatId = X`` and the daemon persists the task under
   ``chat_id = X``.
2. The daemon is stopped.
3. The daemon is started again and the page is reloaded.  The page
   restores its tabs from persisted state and posts a ``ready`` message
   whose ``restoredTabs`` array names ``{tabId=T, chatId=X}``, which
   reaches the freshly started daemon as a ``resumeSession`` command.
4. The daemon's ``_replay_session`` handles ``resumeSession`` by
   recording ``_tab_chat_views[T] = X``.  After a cold daemon restart
   no agent state exists yet, so *only* ``_tab_chat_views`` gets the
   association.
5. User submits a new prompt on the SAME tab.  ``_cmd_run`` opens a
   fresh ``AgentState`` for ``T`` with an empty ``chat_id``,
   mints a brand new UUID chat id, and runs the task.  The
   ``ChatSorcarAgent.build_chat_prompt`` call therefore queries
   ``_load_chat_context`` for the FRESH uuid, finds nothing, and sends
   the LLM an empty preamble — the previous conversation is lost.

The end-to-end test drives the real :class:`AgentServer` through
``_handle_command`` exactly as the daemon's dispatcher would, from a cold
start.  Only :meth:`SorcarAgent.__mro__[1].run` (the grandparent's
plain ``run``) is stubbed so we can capture the augmented prompt that
:meth:`ChatSorcarAgent.build_chat_prompt` produced and inspect the
final ``chat_id`` recorded on the tab's agent state.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from typing import Any, cast

import yaml

from kiss.agents.sorcar.persistence import (
    _add_task,
    _flush_chat_events,
    _load_chat_context,
    _save_task_result,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import (
    init_repo,
    redirect_persistence,
    restore_persistence,
    restore_sorcar_parent_run,
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
        self.saved = redirect_persistence(self.tmpdir)
        init_repo(self.tmpdir)

    def tearDown(self) -> None:
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_restored_tab_submit_preserves_prior_chat(self) -> None:
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

        ctx = _load_chat_context(prior_chat_id)
        assert len(ctx) == 1, ctx
        assert "prior session" in str(ctx[0]["task"] or "")
        assert "Prior assistant answer" in str(ctx[0]["result"] or "")

        from kiss.server import agent_state
        agent_state.agent_states.clear()
        self.addCleanup(agent_state.agent_states.clear)

        server = AgentServer()

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

            def _worker_done() -> bool:
                state = agent_state.find_by_tab(tab_id)
                if state is None:
                    return True
                thread = state.task_thread
                return thread is None or not thread.is_alive()

            assert _wait_until(_worker_done, timeout=30), (
                "task worker thread never completed"
            )
        finally:
            restore_sorcar_parent_run(original_run)

        tab_state = agent_state.find_by_tab(tab_id)
        assert tab_state is not None, (
            "server-owned agent state for the tab disappeared after the run"
        )
        self.assertEqual(
            tab_state.chat_id, prior_chat_id,
            f"Restored tab lost its prior chat_id after relaunch: "
            f"chat_id={tab_state.chat_id!r} expected={prior_chat_id!r}",
        )

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
