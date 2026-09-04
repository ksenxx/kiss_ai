# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the 2026-09-03 audit: prompt loss above fd 1023.

``_CLIProcess._write_prompt`` guarded each non-blocking ``os.write`` to
the child's stdin with ``select.select``, which raises ``ValueError:
filedescriptor out of range in select()`` for ANY descriptor >= 1024
(``FD_SETSIZE``).  The method's blanket ``except (OSError, ValueError)``
— there for the legitimate "child exited before reading its prompt"
case — swallowed that error and closed stdin after writing ZERO bytes.
In a long-running daemon with more than ~1024 descriptors open (many
concurrent tasks, exactly the environment ``_CLIProcess``'s own
docstring defends against with its Errno-24 argument), every CLI-backed
turn therefore silently sent the child an EMPTY prompt: no error, no
retry — the CLI agent just answered nothing, or the wrong thing.

The 2026-09-03 follow-up review found that the first fix traded one
platform limit for another: it replaced ``select.select`` with
``select.poll``, an API that does not exist on native Windows (which the
repo supports — ``.github/workflows/windows-test.yml``).  There the
writer thread died with ``AttributeError`` *outside* the writer's
``except (OSError, ValueError)``, stdin closed after ZERO bytes, and
every ``cc/*``/``codex/*`` turn silently got an empty prompt — the very
bug the patch fixed for Unix.  The final fix drops the readiness
primitive entirely: the descriptor is already non-blocking, so the
writer attempts ``os.write`` directly and, on ``BlockingIOError`` (or a
zero-byte write), waits up to ``_STOP_POLL_SECONDS`` on the cancel event
before retrying.  No ``select`` API of any flavor is involved.

Every test here drives a REAL stand-in CLI executable (a Python script)
through real pipes and real threads — nothing is mocked.  The high-fd
condition is created for real by opening descriptors until new ones land
above 1023.  The fd-pressure machinery needs the Unix-only ``resource``
module, so it is imported with ``pytest.importorskip`` *inside* the
context manager: on Windows the module still collects, the
platform-neutral tests run, and only the fd-pressure cases skip.

Branch coverage of the fixed writer loop:

* the ``BlockingIOError`` "pipe full, wait and retry" branch is
  exercised by both slow-reader tests (the child sleeps before draining,
  so the 64 KiB pipe fills and ``os.write`` raises EAGAIN many times);
* the cancel/child-exit/deadline break branch is exercised by the
  existing grandchild tests in
  ``test_audit0902_core_models_cli_process.py``, which run unchanged on
  the fixed code;
* the zero-byte-write retry branch is unreachable on POSIX without test
  doubles (a non-blocking POSIX pipe write either transfers at least one
  byte or raises ``BlockingIOError``; the branch exists for Windows
  pipes) and stays ``pragma: no cover`` in the source.

``TestNoSelectApiInvolved`` is the regression gate for the Windows bug
itself: it deletes ``select.poll`` AND ``select.select`` from the live
``select`` module — exactly what a native Windows interpreter looks like
to this code path, minus the OS — and proves a real child still receives
the whole prompt.  Nothing in the code under test is patched; only the
stdlib module object is temporarily stripped, with a ``try/finally``
restore.  This gate is unavoidable on Linux CI: fd pressure catches a
reintroduced ``select.select`` (it raises above FD_SETSIZE) but can
never catch a reintroduced ``select.poll``, which works fine on any
Linux descriptor.
"""

import contextlib
import json
import os
import select
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.core.models import claude_code_model as cc_module
from kiss.core.models.claude_code_model import ClaudeCodeModel, _find_claude_cli
from kiss.core.models.model import _CLIProcess

# How high the occupied descriptors reach; comfortably past FD_SETSIZE
# (1024) so every pipe subprocess.Popen creates lands above it.
_FD_CEILING = 1300
# Well past the 64 KiB pipe buffer, so the writer needs many chunks.
_BIG_PROMPT = "y" * (300 * 1024)
_TURN_TIMEOUT = 30


@contextlib.contextmanager
def _descriptors_above_fd_setsize() -> Iterator[None]:
    """Occupy descriptors so newly created fds land above 1023.

    Raises the soft ``RLIMIT_NOFILE`` when it is below the ceiling (the
    daemon does the same at startup), and releases everything afterwards.
    Skips (never errors) on platforms without the Unix-only ``resource``
    module, so this test file stays collectable and the platform-neutral
    tests below still run on Windows.
    """
    resource = pytest.importorskip(
        "resource", reason="fd-pressure setup needs the Unix-only resource module"
    )
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < _FD_CEILING + 64:
        if hard != resource.RLIM_INFINITY and hard < _FD_CEILING + 64:
            pytest.skip(f"RLIMIT_NOFILE hard limit {hard} is too low")
        resource.setrlimit(
            resource.RLIMIT_NOFILE, (_FD_CEILING + 64, hard)
        )
    held: list[int] = []
    try:
        while True:
            fd = os.open(os.devnull, os.O_RDONLY)
            held.append(fd)
            if fd >= _FD_CEILING:
                break
        yield
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def _install_script(tmp_path: Path, name: str, body: str) -> str:
    """Write an executable Python stand-in CLI and return its path."""
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    script.chmod(0o755)
    return str(script)


def _echo_length_cli(tmp_path: Path, sleep_first: float = 0.0) -> str:
    """A CLI that reads ALL of stdin and prints the byte count it received."""
    return _install_script(
        tmp_path,
        "echo-length",
        f"""
        import sys, time
        time.sleep({sleep_first})
        data = sys.stdin.buffer.read()
        print(len(data), flush=True)
        """,
    )


class TestPromptSurvivesAboveFdSetsize:
    """The stdin writer must not depend on the descriptor being < 1024."""

    def test_small_prompt_arrives_intact_above_fd_1024(
        self, tmp_path: Path
    ) -> None:
        prompt = "hello from a busy daemon"
        cli = _echo_length_cli(tmp_path)
        with _descriptors_above_fd_setsize():
            with _CLIProcess([cli], "Fake CLI", _TURN_TIMEOUT) as proc:
                assert proc._proc.stdin is not None
                assert proc._proc.stdin.fileno() >= 1024
                proc.send_prompt(prompt)
                received = [line.strip() for line in proc.lines()]
        assert received == [str(len(prompt.encode("utf-8")))]

    def test_slow_reader_gets_the_whole_big_prompt_above_fd_1024(
        self, tmp_path: Path
    ) -> None:
        # The child sleeps before draining stdin, so the pipe buffer
        # fills and the writer's readiness poller must time out and
        # retry (the "not yet writable" branch) many times.
        cli = _echo_length_cli(tmp_path, sleep_first=0.5)
        with _descriptors_above_fd_setsize():
            with _CLIProcess([cli], "Fake CLI", _TURN_TIMEOUT) as proc:
                proc.send_prompt(_BIG_PROMPT)
                received = [line.strip() for line in proc.lines()]
        assert received == [str(len(_BIG_PROMPT))]

    def test_child_that_closes_stdin_early_still_replies(
        self, tmp_path: Path
    ) -> None:
        # The child closes its stdin without reading it, so the writer's
        # os.write hits EPIPE mid-prompt (the pipe buffer is far smaller
        # than the prompt).  That is the legitimate "child exited before
        # reading its prompt" case: the writer must swallow it, close the
        # pipe, and the reply the child wrote must still come through.
        cli = _install_script(
            tmp_path,
            "close-stdin",
            """
            import os, time
            os.close(0)
            time.sleep(0.3)
            print("done", flush=True)
            """,
        )
        with _descriptors_above_fd_setsize():
            with _CLIProcess([cli], "Fake CLI", _TURN_TIMEOUT) as proc:
                proc.send_prompt(_BIG_PROMPT)
                received = [line.strip() for line in proc.lines()]
        assert received == ["done"]

    def test_claude_code_turn_sees_its_prompt_above_fd_1024(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Full adapter path: a stand-in ``claude`` echoes the prompt it
        # received back as the terminal ``result`` event.  On the broken
        # writer the model returns "" because the CLI got an empty stdin.
        _install_script(
            tmp_path,
            "claude",
            """
            import json, sys
            prompt = sys.stdin.read()
            print(json.dumps({"type": "result", "result": prompt}), flush=True)
            """,
        )
        monkeypatch.setenv("PATH", str(tmp_path))
        # Undo the directory's autouse locator stub (it points at a fake
        # /usr/bin/claude); this test spawns the real stand-in above.
        monkeypatch.setattr(cc_module, "_find_claude_cli", _find_claude_cli)
        prompt = "audit0903: the prompt must reach the CLI"
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": _TURN_TIMEOUT})
        model.initialize(prompt)
        with _descriptors_above_fd_setsize():
            content, result_json = model.generate()
        assert content == prompt
        assert json.loads(json.dumps(result_json))["type"] == "result"


class TestPromptDeliveryIsPlatformNeutral:
    """Prompt delivery under ordinary descriptor numbers, no Unix-only setup.

    These tests use no ``resource`` machinery and no fd pressure, so they
    run identically on native Windows: a real child, real pipes, real
    threads.  On the ``select.poll`` regression they fail there with the
    child reporting 0 bytes received.
    """

    def test_small_prompt_arrives_intact(self, tmp_path: Path) -> None:
        prompt = "hello under normal fd conditions"
        cli = _echo_length_cli(tmp_path)
        with _CLIProcess([cli], "Fake CLI", _TURN_TIMEOUT) as proc:
            proc.send_prompt(prompt)
            received = [line.strip() for line in proc.lines()]
        assert received == [str(len(prompt.encode("utf-8")))]

    def test_slow_reader_gets_the_whole_big_prompt(self, tmp_path: Path) -> None:
        # The child sleeps before draining stdin, so the 64 KiB pipe
        # buffer fills and the writer's os.write raises BlockingIOError
        # (the "pipe full, wait and retry" branch) many times before the
        # whole 300 KiB prompt is through.
        cli = _echo_length_cli(tmp_path, sleep_first=0.5)
        with _CLIProcess([cli], "Fake CLI", _TURN_TIMEOUT) as proc:
            proc.send_prompt(_BIG_PROMPT)
            received = [line.strip() for line in proc.lines()]
        assert received == [str(len(_BIG_PROMPT))]


class TestNoSelectApiInvolved:
    """The writer must deliver prompts with NO ``select`` readiness API.

    Native Windows has no ``select.poll``, and ``select.select`` rejects
    descriptors >= FD_SETSIZE on Unix — the writer may depend on neither.
    Fd pressure alone cannot gate the ``select.poll`` half on Linux CI
    (``poll`` accepts any descriptor number), so this test reproduces the
    Windows condition directly: it temporarily deletes ``select.poll``
    and ``select.select`` from the live stdlib module — nothing in the
    code under test is patched — and drives a real slow-reading child
    with a prompt far larger than the pipe buffer.  On the ``select.poll``
    regression the writer thread died with ``AttributeError``, stdin
    closed after zero bytes, and the child reported ``0``.
    """

    def test_big_prompt_delivered_without_select_poll_or_select(
        self, tmp_path: Path
    ) -> None:
        cli = _echo_length_cli(tmp_path, sleep_first=0.5)
        saved = {
            name: getattr(select, name)
            for name in ("poll", "select")
            if hasattr(select, name)
        }
        try:
            for name in saved:
                delattr(select, name)
            with _CLIProcess([cli], "Fake CLI", _TURN_TIMEOUT) as proc:
                proc.send_prompt(_BIG_PROMPT)
                received = [line.strip() for line in proc.lines()]
        finally:
            for name, value in saved.items():
                setattr(select, name, value)
        assert received == [str(len(_BIG_PROMPT))]
