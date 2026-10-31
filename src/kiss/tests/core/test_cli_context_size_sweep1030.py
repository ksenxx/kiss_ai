# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: a CLI turn of several API requests is not one huge context.

``codex exec`` reports one ``turn.completed`` usage summed over every API
request of the turn, and a turn that calls the KISS tools over the MCP
bridge makes one more request per call.  ``claude --print`` likewise sums
its messages.  The agent must size its context by one request, not by the
bill: five bridged calls over a 60K-token transcript bill 363K tokens,
past the 350K hand-off line of a 500K window, although no request came
near it.  Sized by the bill, the task failed on its first step with
``ContextWindowExceededError``.

The CLIs are scripted stand-ins on PATH, as in
``test_cli_models_kiss_driven.py``: the real ``codex`` would need a login
and hundreds of thousands of billed tokens per run to reach the line.
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.utils import finish as structured_finish
from kiss.tests.conftest import install_cli_script

HELPER_CALL = json.dumps({"tool_calls": [{"name": "_helper_tool", "arguments": {"text": "x"}}]})
FINISH_CALL = json.dumps(
    {
        "tool_calls": [
            {
                "name": "finish",
                "arguments": {"success": True, "is_continue": False, "summary_in_html": "Done."},
            }
        ]
    }
)

# Call N of the stand-in records its stdin prompt and prints the events
# of turns[N] (the last turn repeats), whatever its arguments.
_SCRIPT = """#!/usr/bin/env python3
import json, pathlib, sys
record_dir = pathlib.Path({record_dir!r})
n = len(list(record_dir.glob("prompt-*.txt")))
(record_dir / f"prompt-{{n}}.txt").write_text(sys.stdin.read())
turns = {turns!r}
for event in turns[min(n, len(turns) - 1)]:
    print(json.dumps(event), flush=True)
"""


def _install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, turns: list[list[dict]]
) -> Path:
    """Put a CLI *name* playing *turns* on PATH; return the dir of its recorded prompts."""
    bin_dir = tmp_path / "bin"
    record_dir = tmp_path / "records"
    bin_dir.mkdir()
    record_dir.mkdir()
    install_cli_script(bin_dir / name, _SCRIPT.format(record_dir=str(record_dir), turns=turns))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return record_dir


def _helper_tool(text: str) -> str:
    """Return *text*: a tool the first turn calls.

    Args:
        text: Any string.

    Returns:
        The same string.
    """
    return text


def _run(model_name: str) -> KISSAgent:
    """Run one agentic task on *model_name*; it must finish successfully."""
    agent = KISSAgent("cli context size")
    result = agent.run(
        model_name=model_name,
        prompt_template="Use the tool, then finish.",
        tools=[structured_finish, _helper_tool],
        is_agentic=True,
        verbose=False,
    )
    assert yaml.safe_load(result)["success"] is True
    return agent


def _codex_turn(text: str, tool_items: int, usage: dict[str, int]) -> list[dict]:
    """A ``codex exec --json`` turn of *tool_items* MCP calls answered by *text*."""
    events: list[dict] = [{"type": "thread.started", "thread_id": "t-1"}]
    events += [
        {"type": "item.completed", "item": {"id": f"item_{n}", "type": "mcp_tool_call"}}
        for n in range(tool_items)
    ]
    events += [
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": usage},
    ]
    return events


def test_codex_turn_of_bridged_calls_is_sized_per_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five MCP calls plus the answer are six requests: the context is a sixth of the bill."""
    usage = {"input_tokens": 360_000, "cached_input_tokens": 300_000, "output_tokens": 3_000}
    first = _codex_turn(HELPER_CALL, 5, usage)
    second = _codex_turn(FINISH_CALL, 0, {"input_tokens": 61_000, "output_tokens": 100})
    record_dir = _install(tmp_path, monkeypatch, "codex", [first, second])

    agent = _run("codex/gpt-5.5")

    assert agent.total_tokens_used == 363_000 + 61_100, "the bill keeps the summed usage"
    # What the model read about its context after the first turn.
    assert "Context: 60,500/500,000 tokens" in (record_dir / "prompt-1.txt").read_text()
    assert agent.context_tokens_used == 61_100


def _claude_message(input_tokens: int) -> list[dict]:
    """The usage events of one API request of *input_tokens* and 1,000 output tokens."""
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": input_tokens}}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1_000},
        },
    ]


def _claude_text(text: str) -> dict:
    """The final ``assistant`` event of a ``claude --print`` turn answering *text*."""
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def test_claude_turn_of_several_requests_is_sized_by_its_largest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two requests of 201K and 202K tokens: the context is 202K, not their 403K sum."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    first = [*_claude_message(200_000), *_claude_message(201_000), _claude_text(HELPER_CALL)]
    second = [*_claude_message(203_000), _claude_text(FINISH_CALL)]
    record_dir = _install(tmp_path, monkeypatch, "claude", [first, second])

    agent = _run("cc/sonnet")

    assert agent.total_tokens_used == 403_000 + 204_000, "the bill keeps the summed usage"
    assert "Context: 202,000/500,000 tokens" in (record_dir / "prompt-1.txt").read_text()
    assert agent.context_tokens_used == 204_000


def test_a_turn_without_per_request_usage_is_sized_by_its_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream that reports only the terminal ``result`` usage is one request."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    usage = {"input_tokens": 90, "output_tokens": 10}
    result = {"type": "result", "result": FINISH_CALL, "usage": usage}
    _install(tmp_path, monkeypatch, "claude", [[result]])

    agent = _run("cc/sonnet")

    assert agent.context_tokens_used == agent.total_tokens_used == 100
