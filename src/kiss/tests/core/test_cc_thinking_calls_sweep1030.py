# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: a ``cc/*`` turn runs the calls its text makes, not its drafts.

A reasoning model may write a ``tool_calls`` block in its thinking and
then revise it before answering.  Parsing the thinking together with the
text ran both the draft and the revised call, and a draft cannot be
undone.  The thinking is read only when the text calls no tool (a model
that emits the block inside its thinking and then stops).

The ``claude`` CLI is a scripted stand-in on PATH: the real one cannot be
made to draft a call in its thinking on demand.
"""

import json
from pathlib import Path

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.utils import finish as structured_finish
from kiss.tests.core.test_cli_context_size_sweep1030 import FINISH_CALL, _install

RAN: list[str] = []


def record_command(command: str) -> str:
    """Record *command* instead of running it.

    Args:
        command: The shell command the model asked for.

    Returns:
        ``"ok"``.
    """
    RAN.append(command)
    return "ok"


def _call(command: str) -> str:
    """The ``tool_calls`` block asking ``record_command`` to run *command*."""
    call = {"name": "record_command", "arguments": {"command": command}}
    return json.dumps({"tool_calls": [call]})


def _turn(thinking: str, text: str) -> list[dict]:
    """A ``claude --print`` turn that thinks *thinking* and answers *text*."""
    blocks = [{"type": "thinking", "thinking": thinking}, {"type": "text", "text": text}]
    return [
        {"type": "assistant", "message": {"content": blocks}},
        {"type": "result", "result": text, "usage": {"input_tokens": 10, "output_tokens": 5}},
    ]


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: list[dict]) -> list[str]:
    """Run a ``cc/opus`` task whose first turn is *first*; return the commands that ran."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _install(tmp_path, monkeypatch, "claude", [first, _turn("", FINISH_CALL)])
    RAN.clear()
    result = KISSAgent("cc thinking calls").run(
        model_name="cc/opus",
        prompt_template="Clean the tmp dir.",
        tools=[structured_finish, record_command],
        is_agentic=True,
        verbose=False,
    )
    assert yaml.safe_load(result)["success"] is True
    return list(RAN)


def test_a_call_drafted_in_the_thinking_does_not_run_beside_the_text_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model drafted ``rm -rf build``, revised it and emitted ``rm -rf build/tmp``."""
    thinking = f"Draft: {_call('rm -rf build')} -- no, only the tmp dir."
    ran = _run(tmp_path, monkeypatch, _turn(thinking, _call("rm -rf build/tmp")))
    assert ran == ["rm -rf build/tmp"]


def test_a_call_only_in_the_thinking_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that writes the block in its thinking and stops still gets its call run."""
    ran = _run(tmp_path, monkeypatch, _turn(f"I will run {_call('echo hi')}", ""))
    assert ran == ["echo hi"]
