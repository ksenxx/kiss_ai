# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of three hand-off digest failures.

* A one-line ``name(k=v)`` snippet in the model's prose parses as a call;
  the executor appends the real call reprs after the prose, so the
  phantom came first, its ``[name]: `` paragraph was missing, and the
  result matcher gave up: no call of that turn got its result (and the
  turn's errors vanished from the open-errors section).
* :func:`describe_written_files` checked ``is_file()`` and then called
  ``stat()``: a file deleted in between (a background job, a sibling
  sub-agent) raised ``FileNotFoundError`` out of the whole digest.
* Python the parser gives up on (``MemoryError``, ``RecursionError``) in
  a written file or a one-line snippet, and a literal too long to
  print, raised out of the whole digest as well (sweep 2026-11-11).
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.handoff_digest import (
    describe_written_files,
    handoff_digest,
    parse_tool_calls,
)

PARSER_STACK = 8 << 20
"""Bytes of stack of the thread the deep sources are digested on.

The length of the sum ``ast.parse`` gives up on is the interpreter's:
Python 3.13 counts nested nodes (10,000, whatever the stack: a thread
with 512 KiB crashes at 5,000 terms), 3.14 measures the stack its
thread has left (34,770 terms fit in 8 MiB, 200,000 under ``ulimit -s
unlimited``).  A 20,000-term sum parsed on the main thread was thus
refused by 3.13 and taken by 3.14.  On a thread whose stack size is
set here, :data:`SUM_TERMS` terms are too many for both.
"""
SUM_TERMS = 200_000
"""Terms of the sum the parser gives up on: over five times what :data:`PARSER_STACK` holds."""


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
    churner.start()
    interval = sys.getswitchinterval()
    seen: set[str] = set()
    try:
        sys.setswitchinterval(1e-6)
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


def _on_parser_stack[T](function: Callable[..., T], *args: Any) -> T:
    """Return ``function(*args)`` run on a thread with :data:`PARSER_STACK` bytes of stack."""
    previous = threading.stack_size(PARSER_STACK)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(function, *args).result()
    finally:
        threading.stack_size(previous)


def test_source_the_parser_gives_up_on_costs_only_its_own_detail(tmp_path: Path) -> None:
    """Python too deep to parse, or a literal too long to print, leaves the digest standing.

    ``ast.parse`` raises ``MemoryError`` (its stack overflowed) on an
    ``elif`` chain of some 6,000 arms or as many unary signs, and
    ``RecursionError`` on a long sum (how long: :data:`PARSER_STACK`);
    an integer literal past the interpreter's 4300-digit limit evaluates
    (a hexadecimal one) but ``str`` of it raises ``ValueError``.  Each
    escaped ``handoff_digest``, and the next session got the plain
    ``Agent failed`` fallback instead of the summary.
    """
    arms = "".join(f"    elif x == {i}:\n        return {i}\n" for i in range(1, 8000))
    chain = tmp_path / "dispatch.py"
    chain.write_text(f"def dispatch(x):\n    if x == 0:\n        return 0\n{arms}")
    total = tmp_path / "total.py"
    total.write_text("def total():\n    return " + "+".join(["1"] * SUM_TERMS) + "\n")
    plain = tmp_path / "plain.py"
    plain.write_text("def plain():\n    return 1\n")
    assert _on_parser_stack(describe_written_files, [chain, total, plain]) == [
        f"- {chain} ({chain.stat().st_size:,} bytes)",
        f"- {plain} ({plain.stat().st_size} bytes): plain",
        f"- {total} ({total.stat().st_size:,} bytes)",
    ]
    snippets = (
        "```python\nverify(n=0x" + "F" * 5000 + ", e=65537)\n```\n"
        "```python\nnegate(x=" + "-" * 200000 + "1)\n```\n"
        "```python\nadd(x=" + "+".join(["1"] * SUM_TERMS) + ")\n```\n"
    )
    messages = [
        {"role": "user", "content": "Generate the tables."},
        {"role": "model", "content": "Roughly:\n" + snippets + _call("Bash", command="ls")},
    ]
    assert [(c.name, c.args) for c in _on_parser_stack(parse_tool_calls, messages)] == [
        ("verify", {}), ("negate", {}), ("add", {}), ("Bash", {"command": "ls"}),
    ]
    digest = _on_parser_stack(handoff_digest, messages, [chain, total, plain])
    assert f"- {plain} ({plain.stat().st_size} bytes): plain" in digest
    assert "1. verify() -> \n2. negate() -> \n3. add() -> \n4. Bash(command=ls) -> " in digest
