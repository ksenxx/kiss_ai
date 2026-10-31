# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of two hand-off digest failures.

* A one-line ``name(k=v)`` snippet in the model's prose parses as a call;
  the executor appends the real call reprs after the prose, so the
  phantom came first, its ``[name]: `` paragraph was missing, and the
  result matcher gave up: no call of that turn got its result (and the
  turn's errors vanished from the open-errors section).
* :func:`describe_written_files` checked ``is_file()`` and then called
  ``stat()``: a file deleted in between (a background job, a sibling
  sub-agent) raised ``FileNotFoundError`` out of the whole digest.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.handoff_digest import (
    describe_written_files,
    handoff_digest,
    parse_tool_calls,
)


def _call(name: str, **args: Any) -> str:
    """Render a tool call the way the executor records it in a model message."""
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"```python\n{name}({rendered})\n```"


def _turn_with_prose_snippets() -> list[dict[str, Any]]:
    """A turn whose prose holds call-shaped snippets before the two real calls."""
    prose = (
        "Configure it with:\n```python\nsetup(debug=True)\n```\n"
        "and finish the line:\n```python\nresult = compute(\n```\n"
        "The cleanup would be:\n" + _call("Bash", command="rm -rf build") + "\nRunning:\n"
    )
    return [
        {"role": "user", "content": "Fix the build."},
        {
            "role": "model",
            "content": prose + _call("Bash", command="ls") + "\n"
            + _call("Read", file_path="a.py") + "\n```text\nusage\n```\n",
        },
        {"role": "user", "content": "[Bash]: a.py\nb.py\n\n[Read]: Error: no such file"},
    ]


def test_prose_snippets_do_not_take_the_turns_results() -> None:
    calls = parse_tool_calls(_turn_with_prose_snippets())
    assert [(c.name, c.args, c.result) for c in calls] == [
        ("Bash", {"command": "ls"}, "a.py\nb.py"),
        ("Read", {"file_path": "a.py"}, "Error: no such file"),
    ]
    digest = handoff_digest(_turn_with_prose_snippets(), [])
    assert "## Last failing tool calls\n- Read(file_path=a.py)\n  Error: no such file" in digest
    assert "## Last 2 of 2 tool calls (oldest first)\n1. Bash(command=ls) -> a.py" in digest
    assert "setup(" not in digest and "rm -rf" not in digest


def test_calls_without_a_results_message_keep_empty_results() -> None:
    """A turn the session never got results for keeps every parsed call."""
    turn = _turn_with_prose_snippets()[:2]
    turn.append({"role": "user", "content": "a steering message, not tool results"})
    assert [c.name for c in parse_tool_calls(turn)] == [
        "setup", "result = compute", "Bash", "Bash", "Read",
    ]
    assert all(c.result == "" for c in parse_tool_calls(turn))


def _churn(path: Path, stop: threading.Event) -> None:
    """Create and delete *path* until *stop* is set."""
    while not stop.is_set():
        path.write_bytes(b"x\n")
        path.unlink()


def test_a_file_deleted_while_it_is_described_is_reported_gone(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    stop = threading.Event()
    churner = threading.Thread(target=_churn, args=(path, stop), daemon=True)
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    churner.start()
    seen: set[str] = set()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            (line,) = describe_written_files([path])
            seen.add(line)
    finally:
        stop.set()
        churner.join(timeout=10)
        sys.setswitchinterval(interval)
    assert seen <= {f"- {path} (2 bytes)", f"- {path} (0 bytes)", f"- {path} (no longer exists)"}
    assert f"- {path} (no longer exists)" in seen
    # A path that is now a directory is no longer the file the task wrote.
    path.mkdir()
    assert describe_written_files([path]) == [f"- {path} (no longer exists)"]
