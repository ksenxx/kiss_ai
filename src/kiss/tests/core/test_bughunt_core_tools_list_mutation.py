# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt regression: KISSAgent.run must not mutate the caller's tools list.

``KISSAgent._setup_tools`` historically appended the agent's own bound
``finish`` method directly to the list object the caller passed to
``run(tools=...)``.  The caller's list silently grew by one entry per
agent, and a second agent reusing the same list found "finish" already
present and registered the FIRST agent's bound ``finish`` method instead
of its own.

The test drives a full end-to-end ``run()`` through a real subprocess —
a fake ``claude`` CLI on PATH whose run-to-completion output is the final
text ``done`` — and asserts the caller's list is unchanged afterwards.
"""

import os
import sys
from pathlib import Path

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.tests.conftest import install_cli_script

_EVENTS = [
    {
        "type": "assistant",
        "message": {
            "id": "m1",
            "content": [{"type": "text", "text": "done"}],
        },
    },
    {
        "type": "result",
        "result": "done",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    },
]

_FAKE_CLAUDE = f"""#!{sys.executable}
import json, sys
sys.stdin.read()
for event in {_EVENTS!r}:
    print(json.dumps(event), flush=True)
"""


def echo_tool(text: str) -> str:
    """Echo the given text back.

    Args:
        text: The text to echo.
    """
    return text


def test_run_does_not_mutate_caller_tools_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After run(), the caller's tools list must be exactly as passed in."""
    install_cli_script(tmp_path / "claude", _FAKE_CLAUDE)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    shared_tools = [echo_tool]
    agent = KISSAgent("bughunt-tools-mutation")
    result = agent.run(
        model_name="cc/opus",
        prompt_template="Reply with exactly 'done'.",
        tools=shared_tools,
        max_steps=3,
        verbose=False,
    )
    assert result == "done"
    assert shared_tools == [echo_tool], (
        "run() mutated the caller's tools list: " f"{shared_tools!r}"
    )

    agent2 = KISSAgent("bughunt-tools-mutation-2")
    result2 = agent2.run(
        model_name="cc/opus",
        prompt_template="Reply with exactly 'done'.",
        tools=shared_tools,
        max_steps=3,
        verbose=False,
    )
    assert result2 == "done"
    assert shared_tools == [echo_tool]
    assert getattr(agent2.function_map["finish"], "__self__", None) is agent2
