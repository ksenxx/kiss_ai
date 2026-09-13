# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a CLI child emitting non-UTF-8 bytes must not wedge the turn.

``_CLIProcess`` reads the child's stdout and stderr through text-mode
pipes (``text=True, encoding="utf-8"``).  Without an ``errors=`` policy
the first undecodable byte raises ``UnicodeDecodeError`` inside the drain
thread, which the thread treats like a closed pipe:

* on **stderr** the drain thread dies, nobody reads the pipe any more,
  and a child that keeps logging blocks in ``write(2)`` once the 64 KiB
  buffer is full — its stdout goes silent and the turn is killed as a
  stall after the full ``timeout``;
* on **stdout** the EOF sentinel is posted early and everything the child
  said afterwards is silently dropped, so a truncated answer is reported
  as a completed turn.

A hook, plugin or shell command run by the CLI can print bytes in any
encoding, so both pipes must decode leniently.  The tests run a REAL
stand-in ``codex`` executable; nothing is mocked or patched.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from kiss.core.models.codex_model import CodexModel
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

_CODEX_LATIN1_STDERR_FLOOD = """
    import json
    import sys

    sys.stderr.buffer.write(b"warning: caf\\xe9 not found\\n")
    sys.stderr.buffer.write(b"x" * (200 * 1024))
    sys.stderr.buffer.flush()
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "OK"}}),
          flush=True)
    print(json.dumps({"type": "turn.completed", "usage": {}}), flush=True)
"""

_CODEX_LATIN1_STDOUT_LINE = """
    import json
    import sys

    sys.stdout.buffer.write(b"garbage \\xff\\xfe line\\n")
    sys.stdout.buffer.flush()
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": "AFTER"}}),
          flush=True)
    print(json.dumps({"type": "turn.completed", "usage": {}}), flush=True)
"""


def test_non_utf8_stderr_is_still_drained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An undecodable stderr byte must not stop the drain and stall the child."""
    install_cli(tmp_path, monkeypatch, "codex", _CODEX_LATIN1_STDERR_FLOOD)
    model = CodexModel("codex/default", model_config={"timeout": 8})
    model.initialize("hi")

    started = time.monotonic()
    content, _response = model.generate()
    elapsed = time.monotonic() - started

    assert content == "OK"
    assert elapsed < 6, f"the run blocked for {elapsed:.1f}s on an undrained stderr pipe"


def test_non_utf8_stdout_line_does_not_truncate_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output after an undecodable stdout byte must still be parsed."""
    install_cli(tmp_path, monkeypatch, "codex", _CODEX_LATIN1_STDOUT_LINE)
    model = CodexModel("codex/default", model_config={"timeout": 8})
    model.initialize("hi")

    content, _response = model.generate()

    assert content == "AFTER"
