# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Sorcar-level end-to-end tests for run-to-completion CLI models.

Covers the wiring RelentlessAgent adds around
:class:`~kiss.core.kiss_agent.KISSAgent` for ``cc/*`` / ``codex/*`` models:

1. The task's ``work_dir`` reaches the CLI subprocess as its cwd (via the
   framework-only ``model_config["work_dir"]`` key) — otherwise the CLI's
   native tools would act on the daemon's cwd instead of the task's
   (possibly worktree-redirected) work tree.
2. ``docker_image`` is refused for these models: their native tools run
   directly on the host, so the requested container isolation would be
   silently bypassed.

Uses a real fake ``claude`` executable on PATH — no mocks.
"""

import json
import os
import stat
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.kiss_error import KISSError

FINAL_TEXT = "sorcar-run-to-completion done."

_EVENTS = [
    {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": FINAL_TEXT}]},
    },
    {
        "type": "result",
        "result": FINAL_TEXT,
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


def test_relentless_passes_work_dir_as_cli_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI child runs in the task's work_dir, and the run succeeds."""
    record_dir = _install_fake_claude(tmp_path, monkeypatch)
    work_dir = tmp_path / "task-work"
    work_dir.mkdir()
    agent = RelentlessAgent("sorcar run-to-completion cwd")
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
    payload = yaml.safe_load(result)
    assert payload["success"] is True
    assert FINAL_TEXT in payload["summary"]


def test_docker_image_is_refused_for_cli_models(tmp_path: Path) -> None:
    """docker_image + cc/* raises instead of silently bypassing isolation."""
    agent = RelentlessAgent("sorcar docker guard")
    with pytest.raises(KISSError, match="cannot honor docker_image"):
        agent.run(
            model_name="cc/sonnet",
            prompt_template="Say done.",
            work_dir=str(tmp_path),
            docker_image="ubuntu:latest",
            max_sub_sessions=1,
            verbose=False,
        )
