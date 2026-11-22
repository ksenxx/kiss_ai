# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop that lands after a call's final usage arrived still bills that usage, once.

Cost audit of 2026-11-22, two adapters:

* Claude Code staged the CLI's terminal ``result`` (the turn's final
  usage) for ``take_partial_usage_response`` only when the turn then
  failed with a ``KISSError``.  A Stop raised by the wait for the CLI's
  exit that follows the ``result`` billed nothing, or the placeholder
  usage of an earlier ``message_start`` (1 output token for 100).
* Anthropic's ``_create_message`` dropped the staged final message before
  the SDK stream and its watchdog were closed, so a Stop raised during
  that cleanup recorded a completed, charged call as $0.

A normal return bills the call once and leaves nothing staged for a
later failure to bill again.

No mocks: a real ``KISSAgent`` runs against a stand-in ``claude``
executable on PATH, or through the real Anthropic SDK against a local
``/v1/messages`` SSE server.  The Stop is the task thread's real stop
event (and, for the stream cleanup, which reads no stop event, the
daemon's forced ``KeyboardInterrupt``), landed by holding the task thread
at the entry of the real function with :class:`Pause`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from types import CodeType
from typing import Any

import pytest

from kiss.core import stop_signal
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models import claude_code_model as cc_module
from kiss.core.models import stream_abort
from kiss.core.models.claude_code_model import _find_claude_cli
from kiss.core.models.model import _CLIProcess
from kiss.core.models.model_info import calculate_cost
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.conftest import install_cli_script
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer, sse
from kiss.tests.server.pause import Pause

_MODEL = "claude-opus-5-5"
_FINAL_USAGE = {"input_tokens": 1000, "output_tokens": 100}
_FINAL_COST = calculate_cost(_MODEL, 1000, 100)
_DEADLINE = 20.0

# A stand-in CLI: reads the prompt, prints its events (one per line), exits 0.
_FAKE_CLAUDE = """#!{python}
import json, sys
sys.stdin.read()
for event in json.loads({events!r}):
    print(json.dumps(event), flush=True)
"""
_CC_TURN: list[dict[str, Any]] = [
    {"type": "system", "subtype": "init", "model": _MODEL},
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "word"}]}},
    {"type": "result", "result": "word", "usage": _FINAL_USAGE},
]
# The usage a ``message_start`` reports: ``output_tokens`` is a placeholder.
_CC_MESSAGE_START = {
    "type": "message_start",
    "message": {"model": _MODEL, "usage": {**_FINAL_USAGE, "output_tokens": 1}},
}

_ANTHROPIC_TURN = [
    sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_terminal",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": _MODEL,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {**_FINAL_USAGE, "output_tokens": 1},
            },
        },
    ),
    sse(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
    ),
    sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
    sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": _FINAL_USAGE["output_tokens"]},
        },
    ),
    sse("message_stop", {"type": "message_stop"}),
]
_ANTHROPIC_RUN: dict[str, Any] = {
    "model_name": _MODEL,
    "prompt_template": "Say ok.",
    "is_agentic": False,
    "model_config": {"api_key": "sk-test"},
}


def _always(frame_locals: dict[str, Any]) -> bool:
    """A :class:`Pause` matcher: hold the first thread that enters the code."""
    return True


def _run_task(
    agent: KISSAgent, stop: threading.Event, run_kwargs: dict[str, Any], outcome: list[Any]
) -> None:
    """Run *agent* as a stoppable task thread; record what its run returned or raised."""
    stop_signal.set_thread_stop_event(stop)
    try:
        outcome.append(agent.run(verbose=False, **run_kwargs))
    except BaseException as error:  # noqa: BLE001 — reported to the test
        outcome.append(error)
    finally:
        stop_signal.set_thread_stop_event(None)


def _stopped_at(code: CodeType, forced: bool, run_kwargs: dict[str, Any]) -> KISSAgent:
    """Run a real agent and press Stop as its task thread enters *code*.

    Args:
        code: The function at whose entry the Stop lands.
        forced: Also inject the daemon's forced ``KeyboardInterrupt``,
            for a function that reads no stop event.
        run_kwargs: The arguments of ``KISSAgent.run``.

    Returns:
        The agent, whose run ended with ``KeyboardInterrupt``.
    """
    agent = KISSAgent("terminal usage kept on stop")
    stop = threading.Event()
    outcome: list[Any] = []
    worker = threading.Thread(
        target=_run_task, args=(agent, stop, run_kwargs, outcome), daemon=True
    )
    with Pause(code, _always) as held:
        worker.start()
        assert held.paused.wait(_DEADLINE), f"the call never reached {code.co_name}: {outcome}"
        stop.set()
        if forced:
            assert worker.ident is not None
            assert inject_async_exception(worker.ident, KeyboardInterrupt) == 1
    worker.join(_DEADLINE)
    assert not worker.is_alive(), "the stopped run never ended"
    assert isinstance(outcome[0], KeyboardInterrupt), repr(outcome[0])
    return agent


def _assert_billed_once(agent: KISSAgent) -> None:
    """The run was charged the call's final usage, and nothing is left to bill again."""
    assert agent.budget_used == pytest.approx(_FINAL_COST)
    assert agent.total_tokens_used == sum(_FINAL_USAGE.values())
    assert agent.model.take_partial_usage_response() is None


def _cc_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Put a stand-in ``claude`` printing *events* on PATH, paid by an API key.

    Returns:
        The ``KISSAgent.run`` arguments of one turn through it.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
    monkeypatch.setattr(DEFAULT_CONFIG, "CLAUDE_CODE_OAUTH_TOKEN", "")
    monkeypatch.setattr(DEFAULT_CONFIG, "claude_code_use_subscription", False)
    # The directory's conftest stubs the locator; the real one finds the stand-in.
    monkeypatch.setattr(cc_module, "_find_claude_cli", _find_claude_cli)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    install_cli_script(
        bin_dir / "claude", _FAKE_CLAUDE.format(python=sys.executable, events=json.dumps(events))
    )
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return {
        "model_name": "cc/opus",
        "prompt_template": "Say the word.",
        "is_agentic": False,
        "model_config": {"work_dir": str(tmp_path), "timeout": 30},
    }


@pytest.mark.parametrize("events", [_CC_TURN, [_CC_MESSAGE_START, *_CC_TURN]])
def test_cc_stop_after_the_result_bills_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[dict[str, Any]]
) -> None:
    """A Stop in the wait for the CLI's exit bills the ``result`` it had sent, not $0
    and not the placeholder usage of the turn's ``message_start``."""
    run_kwargs = _cc_run(tmp_path, monkeypatch, events)
    agent = _stopped_at(_CLIProcess.wait_for_exit.__code__, False, run_kwargs)
    _assert_billed_once(agent)


def test_cc_normal_return_bills_once_and_stages_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = KISSAgent("terminal usage kept on stop")
    agent.run(verbose=False, **_cc_run(tmp_path, monkeypatch, [_CC_MESSAGE_START, *_CC_TURN]))
    _assert_billed_once(agent)


def test_anthropic_stop_during_stream_cleanup_bills_the_finished_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced Stop landing as the finished stream is closed bills its final usage."""
    with ScriptedAnthropicServer(_ANTHROPIC_TURN) as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
        agent = _stopped_at(stream_abort._close_stream.__code__, True, _ANTHROPIC_RUN)
        assert len(server.requests) == 1
    _assert_billed_once(agent)


def test_anthropic_normal_return_bills_once_and_stages_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ScriptedAnthropicServer(_ANTHROPIC_TURN) as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
        agent = KISSAgent("terminal usage kept on stop")
        assert agent.run(verbose=False, **_ANTHROPIC_RUN) == "ok"
    _assert_billed_once(agent)
