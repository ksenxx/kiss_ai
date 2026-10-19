# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Sorcar-level end-to-end test for the CLI-backed models.

Covers the wiring RelentlessAgent adds around
:class:`~kiss.core.kiss_agent.KISSAgent` for ``cc/*`` / ``codex/*`` models:
the task's ``work_dir`` reaches the CLI subprocess as its cwd (via the
framework-only ``model_config["work_dir"]`` key), the CLI gets Sorcar's
system prompt and tools as its system prompt, and its ``finish`` tool
call (the text-based protocol) ends the run.

Uses a real fake ``claude`` executable on PATH — no mocks.
"""

import json
import os
import stat
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.tests.conftest import posix_only

FINAL_TEXT = "sorcar cli done."
_FINISH_CALL = json.dumps(
    {
        "tool_calls": [
            {
                "name": "finish",
                "arguments": {
                    "success": True,
                    "is_continue": False,
                    "summary_in_html": FINAL_TEXT,
                },
            }
        ]
    }
)

_EVENTS = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 9, "output_tokens": 0}}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
    {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": _FINISH_CALL}]},
    },
    {
        "type": "result",
        "result": _FINISH_CALL,
        "usage": {"input_tokens": 9, "output_tokens": 4},
    },
]

_FAKE_CLI = """#!/usr/bin/env python3
import json, os, sys, pathlib
record_dir = pathlib.Path({record_dir!r})
n = len(list(record_dir.glob("call-*.json")))
prompt = sys.stdin.read()
(record_dir / f"call-{{n}}.json").write_text(
    json.dumps({{"argv": sys.argv[1:], "prompt": prompt, "cwd": os.getcwd()}}))
for event in {events!r}:
    print(json.dumps(event))
"""


def _install_fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install a fake ``claude`` on PATH; return its call-record directory.

    Args:
        tmp_path: The test's temporary directory.
        monkeypatch: Fixture used to prepend the fake's dir to PATH.

    Returns:
        The directory the fake writes its call records into.
    """
    bin_dir = tmp_path / "bin"
    record_dir = tmp_path / "records"
    bin_dir.mkdir()
    record_dir.mkdir()
    cli = bin_dir / "claude"
    cli.write_text(_FAKE_CLI.format(record_dir=str(record_dir), events=_EVENTS))
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return record_dir


@posix_only("shebang claude shim on PATH cannot shadow the real CLI on Windows")
def test_relentless_passes_work_dir_as_cli_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI child runs in the task's work_dir with Sorcar's prompt and
    tools as its system prompt, and its finish call ends the run."""
    record_dir = _install_fake_claude(tmp_path, monkeypatch)
    work_dir = tmp_path / "task-work"
    work_dir.mkdir()
    agent = RelentlessAgent("sorcar cli cwd")
    result = agent.run(
        model_name="cc/sonnet",
        prompt_template="Say done.",
        work_dir=str(work_dir),
        max_sub_sessions=1,
        verbose=False,
    )
    calls = [
        json.loads(p.read_text()) for p in sorted(record_dir.glob("call-*.json"))
    ]
    assert len(calls) == 1
    assert Path(calls[0]["cwd"]).resolve() == work_dir.resolve()
    argv = calls[0]["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    system_text = argv[argv.index("--system-prompt") + 1]
    assert "finish" in system_text and "tool_calls" in system_text
    assert "Say done." in calls[0]["prompt"]
    payload = yaml.safe_load(result)
    assert payload["success"] is True
    assert FINAL_TEXT in payload["summary"]
