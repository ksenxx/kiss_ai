# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A message typed into a running task is echoed as a ``steer`` prompt.

The chat webview shows the task's text in the accent "Task" panel and
must show every message the user types into the RUNNING task in a
panel like it (the "Message" panel): both are the user's words.  The
daemon tells the two apart by stamping the steering echo with
``steer: True`` — on the live echo (``_echo_injected_prompt``) and on
the durable ``recordOnly`` copy the drain hook emits for a message
queued before the task row existed — so the flag is persisted with the
event and a replay renders the same panel.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.tests.server.test_subagent_prompt_inject_stop import (
    _Harness,
    _persisted_prompts,
    isolated_db,  # noqa: F401 - pytest fixture
)


def _steer_prompts(rows: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    return [e for e in _persisted_prompts(rows, text) if e.get("steer") is True]


@pytest.mark.usefixtures("isolated_db")
class TestSteerPromptEchoFlag:
    def test_live_echo_and_persisted_event_carry_steer(self) -> None:
        h = _Harness()
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "FOCUS ON THE LOGIN BUG",
            "tabId": "tab-parent",
        })
        rec = [
            e for e in h.recording(h.parent_task_id)
            if e.get("type") == "prompt"
            and "FOCUS ON THE LOGIN BUG" in str(e.get("text", ""))
        ]
        assert rec, "the steering echo was not recorded"
        assert rec[0].get("steer") is True
        assert _steer_prompts(
            h.persisted_events(h.parent_task_id), "FOCUS ON THE LOGIN BUG",
        ), "the persisted echo lost its steer flag"

    def test_deferred_record_only_echo_carries_steer(self) -> None:
        h = _Harness()
        h.parent._last_task_id = None
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "QUEUED BEFORE THE ROW",
            "tabId": "tab-parent",
        })
        assert h.parent_state.unattributed_prompt_echoes == ["QUEUED BEFORE THE ROW"]
        h.parent._last_task_id = h.parent_task_id
        h.parent.printer = h.printer
        h.parent._tab_id = "tab-parent"  # type: ignore[attr-defined]
        model = AnthropicModel(
            "claude-haiku-4-5",
            os.environ.get("ANTHROPIC_API_KEY", "test-key"),
        )
        h.printer._thread_local.task_id = h.parent_task_id
        try:
            h.parent._drain_pending_user_messages(model)
        finally:
            h.printer._thread_local.task_id = ""
        persisted = _steer_prompts(
            h.persisted_events(h.parent_task_id), "QUEUED BEFORE THE ROW",
        )
        assert persisted, "the durable copy lost its steer flag"
        assert "recordOnly" not in persisted[0]

    def test_task_own_prompt_is_not_steer(self) -> None:
        """The task's own ``prompt`` event (the agent's print) has no flag."""
        h = _Harness()
        h.printer._thread_local.task_id = h.parent_task_id
        try:
            h.printer.print("the task text", "prompt")
        finally:
            h.printer._thread_local.task_id = ""
        rec = [
            e for e in h.recording(h.parent_task_id)
            if e.get("type") == "prompt" and e.get("text") == "the task text"
        ]
        assert rec and "steer" not in rec[0]
