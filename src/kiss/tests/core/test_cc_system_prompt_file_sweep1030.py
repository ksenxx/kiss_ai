# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: ``cc/*`` hands the system prompt to ``claude`` in a file.

The prompt used to travel on the command line (``--system-prompt``).
Windows caps a whole command line at 32,767 characters, which Sorcar's
system prompt alone (over 33,000) exceeds, so every non-trivial ``cc/*``
task failed to start there.  Linux caps one argument at 128 KiB, so these
tests use a larger prompt to hit the same wall on any POSIX host.

The ``claude`` CLI is a scripted stand-in on PATH that records what it
received, so no login or API spend is needed.
"""

import json
import os
from pathlib import Path

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.tests.conftest import install_cli_script

# Records argv and the system prompt file's content (read while the turn
# runs), then answers "ok" or, with ``fail``, exits with status 1.
_SCRIPT = """#!/usr/bin/env python3
import json, pathlib, sys
argv = sys.argv[1:]
sys.stdin.read()
system = None
if "--system-prompt-file" in argv:
    system = pathlib.Path(argv[argv.index("--system-prompt-file") + 1]).read_text()
pathlib.Path({record!r}).write_text(json.dumps({{"argv": argv, "system": system}}))
if {fail!r}:
    sys.exit(1)
for event in [
    {{"type": "assistant", "message": {{"content": [{{"type": "text", "text": "ok"}}]}}}},
    {{"type": "result", "result": "ok", "usage": {{"input_tokens": 1, "output_tokens": 1}}}},
]:
    print(json.dumps(event), flush=True)
"""

# Past Linux's 128 KiB limit on one argument (and far past Windows' limit
# on the whole command line).
BIG_PROMPT = "Follow the KISS rules.\n" * 9_000


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool = False) -> Path:
    """Put the recording ``claude`` on PATH; return the file it records into."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "record.json"
    install_cli_script(bin_dir / "claude", _SCRIPT.format(record=str(record), fail=fail))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return record


def _model(system_instruction: str | None) -> ClaudeCodeModel:
    """A ``cc/sonnet`` model with *system_instruction*, ready to generate."""
    config = {"system_instruction": system_instruction} if system_instruction else {}
    model = ClaudeCodeModel("cc/sonnet", model_config=config)
    model.initialize("Say ok.")
    return model


def test_a_system_prompt_too_long_for_a_command_line_reaches_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI starts, reads the whole prompt from the file, and the file is gone after."""
    assert len(BIG_PROMPT.encode()) > 128 * 1024
    record = _install(tmp_path, monkeypatch)

    content, _ = _model(BIG_PROMPT).generate()

    assert content == "ok"
    recorded = json.loads(record.read_text())
    assert recorded["system"] == BIG_PROMPT
    assert max(len(arg) for arg in recorded["argv"]) < 4096
    prompt_file = recorded["argv"][recorded["argv"].index("--system-prompt-file") + 1]
    assert not Path(prompt_file).exists(), "the system prompt file outlived the turn"


def test_the_system_prompt_file_is_removed_when_the_turn_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI that exits with an error still leaves no prompt file behind."""
    record = _install(tmp_path, monkeypatch, fail=True)

    with pytest.raises(KISSError):
        _model("Be brief.").generate()

    recorded = json.loads(record.read_text())
    assert recorded["system"] == "Be brief."
    prompt_file = recorded["argv"][recorded["argv"].index("--system-prompt-file") + 1]
    assert not Path(prompt_file).exists(), "the system prompt file outlived the failed turn"


def test_without_a_system_instruction_the_cli_keeps_its_own_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No instruction, no file: the CLI's own system prompt stays in place."""
    record = _install(tmp_path, monkeypatch)

    content, _ = _model(None).generate()

    assert content == "ok"
    recorded = json.loads(record.read_text())
    assert recorded["system"] is None
    assert "--system-prompt-file" not in recorded["argv"]
    assert "--system-prompt" not in recorded["argv"]
