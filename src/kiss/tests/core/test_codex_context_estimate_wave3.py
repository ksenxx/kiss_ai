# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: a Codex turn is sized by the transcript it leaves behind.

``codex exec --json`` reports one usage summed over every API request of
a turn and no per-request figures.  Spread evenly over the turn's
``requests`` (an upper bound on their number: one per tool item), the sum
understated the context whenever the requests differed.  A bridged tool
returning 400K tokens makes a first request of 1K tokens and a second of
401K; averaged to 201K, the agent never reached its 350K hand-off line,
although the next turn sends all 401K.  Five calls made by one request
look like six requests and were sized at a sixth of two.

The ``codex`` on PATH is a scripted stand-in that calls the KISS tools
through the real MCP bridge, as in ``test_codex_mcp_bridge.py``, and
reports the usage the real CLI would: the real one needs a login and
some 400K billed tokens per run.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import ContextWindowExceededError
from kiss.core.utils import finish as structured_finish
from kiss.tests.conftest import install_cli_script

FINISH_TEXT = json.dumps(
    {
        "tool_calls": [
            {
                "name": "finish",
                "arguments": {"success": True, "is_continue": False, "summary_in_html": "Done."},
            }
        ]
    }
)

# Call N of the stand-in records its stdin prompt and plays turns[N] (the
# last turn repeats): it makes the turn's MCP calls through the proxy,
# reporting each as a completed ``mcp_tool_call`` item like Codex, then
# answers with the turn's text and, if the turn has one, its usage.
_FAKE_CODEX = """#!{python}
import json, os, pathlib, sys
print(json.dumps({{"type": "thread.started", "thread_id": "t-1"}}), flush=True)
import asyncio, tomllib
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

record_dir = pathlib.Path({record_dir!r})
n = len(list(record_dir.glob("prompt-*.txt")))
(record_dir / f"prompt-{{n}}.txt").write_text(sys.stdin.read())
argv = sys.argv[1:]
overrides = tomllib.loads("\\n".join(
    argv[i + 1] for i in range(len(argv) - 1) if argv[i] == "-c"))
turns = {turns!r}
turn = turns[min(n, len(turns) - 1)]

async def call_tools():
    kiss = overrides["mcp_servers"]["kiss"]
    env = {{name: os.environ[name] for name in kiss.get("env_vars", [])}}
    params = StdioServerParameters(command=kiss["command"], args=kiss["args"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for tool, arguments in turn["mcp_calls"]:
                await session.call_tool(tool, arguments)
                print(json.dumps({{"type": "item.completed", "item": {{
                    "type": "mcp_tool_call", "server": "kiss", "tool": tool}}}}), flush=True)

if turn["mcp_calls"]:
    asyncio.run(call_tools())
print(json.dumps({{"type": "item.completed", "item": {{
    "type": "agent_message", "text": turn["text"]}}}}), flush=True)
if turn.get("usage"):
    print(json.dumps({{"type": "turn.completed", "usage": turn["usage"]}}), flush=True)
"""


def bulky(chars: int) -> str:
    """Return *chars* characters of filler text, as a big file read would.

    GPT-5's tokenizer (``o200k_base``) reads six characters of it per token.

    Args:
        chars: How many characters to return.

    Returns:
        The filler text.
    """
    return ("lorem ipsum " * (chars // 12 + 1))[:chars]


def _run_codex_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, turns: list[dict[str, Any]]
) -> tuple[KISSAgent, Path]:
    """Start a codex/* agent against the stand-in playing *turns*.

    Returns:
        The agent (its ``run`` is the caller's) and the directory of the
        stand-in's recorded prompts.
    """
    bin_dir = tmp_path / "bin"
    record_dir = tmp_path / "records"
    bin_dir.mkdir()
    record_dir.mkdir()
    script = _FAKE_CODEX.format(python=sys.executable, record_dir=str(record_dir), turns=turns)
    install_cli_script(bin_dir / "codex", script)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return KISSAgent("codex context estimate"), record_dir


def _run(agent: KISSAgent) -> dict[str, Any]:
    """Run the agentic task on ``codex/gpt-5.5``; return the finish payload."""
    result = agent.run(
        model_name="codex/gpt-5.5",
        prompt_template="Read the material with bulky, then finish.",
        tools=[structured_finish, bulky],
        is_agentic=True,
        verbose=False,
    )
    payload = yaml.safe_load(result)
    assert isinstance(payload, dict)
    return payload


def _reported_context(prompt: str) -> int:
    """The context size the agent's usage footer in *prompt* told the model."""
    sizes = re.findall(r"Context: ([\d,]+)/500,000 tokens", prompt)
    assert sizes, "the prompt carries the usage footer"
    return int(sizes[-1].replace(",", ""))


def test_one_huge_tool_result_reaches_the_hand_off_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requests of 1K and 401K tokens: the 401K transcript hands off, its 201K average did not."""
    turns: list[dict[str, Any]] = [
        {
            "mcp_calls": [["bulky", {"chars": 2_400_000}]],  # 400K tokens
            "text": "I read it.",
            "usage": {"input_tokens": 400_000, "output_tokens": 2_000},
        },
        {"mcp_calls": [], "text": FINISH_TEXT, "usage": {"input_tokens": 9, "output_tokens": 1}},
    ]
    agent, record_dir = _run_codex_agent(tmp_path, monkeypatch, turns)

    with pytest.raises(ContextWindowExceededError):
        _run(agent)

    assert 400_000 <= agent.context_tokens_used <= 402_000
    assert len(list(record_dir.glob("prompt-*.txt"))) == 1, "no turn was sent past the line"


def test_calls_made_by_one_request_are_not_averaged_over_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five calls of one request plus the answer are two requests, not six."""
    usage = {"input_tokens": 303_000, "cached_input_tokens": 1_000, "output_tokens": 2_000}
    turns: list[dict[str, Any]] = [
        {"mcp_calls": [["bulky", {"chars": 360_000}]] * 5, "text": "Read.", "usage": usage},
        {
            "mcp_calls": [],
            "text": FINISH_TEXT,
            "usage": {"input_tokens": 306_000, "output_tokens": 100},
        },
    ]
    agent, record_dir = _run_codex_agent(tmp_path, monkeypatch, turns)

    assert _run(agent)["success"] is True

    context = _reported_context((record_dir / "prompt-1.txt").read_text())
    assert 300_000 <= context <= 305_000, context
    assert agent.context_tokens_used == 306_100, "a one-request turn is its whole usage"


def test_a_turn_without_usage_leaves_the_context_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn that reports no usage keeps the size the previous turn measured."""
    turns: list[dict[str, Any]] = [
        {"mcp_calls": [], "text": "Thinking.", "usage": {"input_tokens": 7_000}},
        {"mcp_calls": [["bulky", {"chars": 10}]], "text": "Read."},
        {"mcp_calls": [], "text": FINISH_TEXT, "usage": {"input_tokens": 8_000}},
    ]
    agent, record_dir = _run_codex_agent(tmp_path, monkeypatch, turns)

    assert _run(agent)["success"] is True

    assert _reported_context((record_dir / "prompt-2.txt").read_text()) == 7_000
    assert agent.context_tokens_used == 8_000
