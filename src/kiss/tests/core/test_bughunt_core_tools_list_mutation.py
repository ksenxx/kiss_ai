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
a fake ``claude`` CLI on PATH that returns a ``finish`` tool call — and
asserts the caller's list is unchanged afterwards.
"""

import json
from pathlib import Path

import pytest

from kiss.core.kiss_agent import KISSAgent

_EVENTS = [
    {
        "type": "assistant",
        "message": {
            "id": "m1",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "tool_calls": [
                                {"name": "finish", "arguments": {"result": "done"}}
                            ]
                        }
                    ),
                }
            ],
        },
    },
    {
        "type": "result",
        "result": "",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    },
]

_FAKE_CLAUDE = "#!/bin/bash\n/bin/cat > /dev/null\n" + "".join(
    f"echo '{json.dumps(event)}'\n" for event in _EVENTS
)


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
    cli = tmp_path / "claude"
    cli.write_text(_FAKE_CLAUDE)
    cli.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    shared_tools = [echo_tool]
    agent = KISSAgent("bughunt-tools-mutation")
    result = agent.run(
        model_name="cc/opus",
        prompt_template="Call finish with 'done'.",
        tools=shared_tools,
        max_steps=3,
        verbose=False,
    )
    assert result == "done"
    assert shared_tools == [echo_tool], (
        "run() mutated the caller's tools list: " f"{shared_tools!r}"
    )

    # A second agent reusing the same list must register ITS OWN finish
    # (the first agent's bound finish must not have leaked into the list).
    agent2 = KISSAgent("bughunt-tools-mutation-2")
    result2 = agent2.run(
        model_name="cc/opus",
        prompt_template="Call finish with 'done'.",
        tools=shared_tools,
        max_steps=3,
        verbose=False,
    )
    assert result2 == "done"
    assert shared_tools == [echo_tool]
    assert getattr(agent2.function_map["finish"], "__self__", None) is agent2
