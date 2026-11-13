# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 fix round (core-models): three ``_CLIProcess`` /
``_cli_turn`` defects found by the read-only review.

Every test drives a REAL stand-in ``claude`` executable (a Python script
installed on ``PATH``) through the adapter's own locator, real pipes and
real threads — nothing is mocked.

* **#4 — abandoned writer thread + stdin fd.**  The stdin writer used a
  blocking buffered ``write()``.  A grandchild that inherited the prompt
  pipe kept the writer parked inside ``write(2)`` holding the buffer
  lock, so ``close()`` skipped the pipe and both the thread and the fd
  lived as long as the grandchild.  The writer is now a cancellable
  non-blocking ``select`` + ``os.write`` loop that ``close()`` cancels,
  joins, and then closes stdin deterministically.
* **#7 — Stop ignored after stdout EOF.**  ``wait_for_exit()`` blocked in
  one ``proc.wait()`` for up to 10 s without polling the stop signal, so
  a Stop pressed while a lingering CLI wound down was ignored and the
  turn returned success.
* **#8 — thinking left open on an ordinary exception.**  ``_cli_turn``
  closed an open thinking bracket only for the deadline and Stop; a
  parser error (here a valid JSON *array* line, which the event iterator
  ``.get()``s) left ``_thinking_open=True`` and the retried turn's text
  was rendered as reasoning.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from kiss.core import stop_signal
from kiss.core.kiss_error import KISSError
from kiss.core.models import claude_code_model as cc_module
from kiss.core.models.claude_code_model import ClaudeCodeModel, _find_claude_cli
from kiss.core.models.model import _CLIProcess
from kiss.core.processes import pid_alive
from kiss.tests.conftest import install_cli_script

# Well past the 64 KiB pipe buffer, so an unread prompt blocks a naive writer.
_BIG_PROMPT = "x" * (300 * 1024)
# Far longer than any bound in the supervisor: only a deterministic cancel
# can bring the writer back before the grandchild dies.
_GRANDCHILD_LIFETIME = 120
_TURN_TIMEOUT = 30


def _install_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    """Install an executable stand-in ``claude`` and let the adapter discover it."""
    script = tmp_path / "claude"
    install_cli_script(script, f"#!{sys.executable}\n" + textwrap.dedent(body))
    # Prepended, not replaced: the stand-in still wins the lookup, and the
    # supervisor's Windows tree kill (``taskkill``) stays reachable.
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(cc_module, "_find_claude_cli", _find_claude_cli)
    return script


def _kill(pid_file: Path) -> None:
    """Kill the grandchild whose pid the stand-in recorded in *pid_file*."""
    if pid_file.exists():
        pid = int(pid_file.read_text())
        # Windows has no SIGKILL (os.kill(pid, SIGTERM) is TerminateProcess
        # there) and refuses, with PermissionError, to open a pid that has
        # already exited -- hence the liveness probe first.
        if pid_alive(pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    """Return whether *pid* disappears within *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not pid_alive(pid)


def _grandchild_holds_stdin(pid_file: Path) -> str:
    """Stand-in body: spawn a grandchild that inherits stdin, reply, exit.

    The grandchild starts its own session so that ``close()``'s group
    kill does not reach it (on POSIX; Windows ignores
    ``start_new_session`` and the grandchild dies with the Job Object):
    it keeps the prompt pipe open for its whole lifetime, and only the
    writer's cancellation can bring the supervisor back.
    """
    return f"""
    import json
    import subprocess
    import sys

    # The grandchild inherits OUR stdin -- the adapter's prompt pipe -- and
    # outlives us, but not stdout/stderr, so the adapter sees EOF on both.
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep({_GRANDCHILD_LIFETIME})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    with open({str(pid_file)!r}, "w") as f:
        f.write(str(grandchild.pid))
    print(json.dumps({{"type": "assistant",
                       "message": {{"id": "m1",
                                    "content": [{{"type": "text", "text": "DONE"}}]}}}}),
          flush=True)
    print(json.dumps({{"type": "result", "result": "DONE",
                       "usage": {{"input_tokens": 10, "output_tokens": 5}}}}),
          flush=True)
    """


def _lingers_after_closing_stdout(eof_marker: Path) -> str:
    """Stand-in body: full reply, close fd 1, announce it, then linger."""
    return f"""
    import json
    import os
    import sys
    import time

    sys.stdin.read()
    print(json.dumps({{"type": "assistant",
                       "message": {{"id": "m1",
                                    "content": [{{"type": "text", "text": "DONE"}}]}}}}),
          flush=True)
    print(json.dumps({{"type": "result", "result": "DONE",
                       "usage": {{"input_tokens": 10, "output_tokens": 5}}}}),
          flush=True)
    sys.stdout.flush()
    os.close(1)
    with open({str(eof_marker)!r}, "w") as f:
        f.write("eof")
    time.sleep(60)
    """


def _thinking_then_bad_event_then_ok(counter: Path) -> str:
    """Stand-in body: first run opens thinking then emits a malformed event; later runs reply."""
    return f"""
    import json
    import os
    import sys

    sys.stdin.read()
    counter = {str(counter)!r}
    runs = int(open(counter).read()) if os.path.exists(counter) else 0
    with open(counter, "w") as f:
        f.write(str(runs + 1))
    if runs == 0:
        print(json.dumps({{"type": "content_block_start",
                           "content_block": {{"type": "thinking"}}}}), flush=True)
        print(json.dumps({{"type": "content_block_delta",
                           "delta": {{"type": "thinking_delta",
                                      "thinking": "let me think"}}}}), flush=True)
        # An event whose delta is not an object: the parser calls .get() on it
        # (a line that is no event object at all is skipped as noise).
        print(json.dumps({{"type": "content_block_delta", "delta": "oops"}}), flush=True)
        sys.exit(0)
    print(json.dumps({{"type": "assistant",
                       "message": {{"id": "m2",
                                    "content": [{{"type": "text", "text": "hello"}}]}}}}),
          flush=True)
    print(json.dumps({{"type": "result", "result": "hello",
                       "usage": {{"input_tokens": 10, "output_tokens": 5}}}}),
          flush=True)
    """


def _writer_threads() -> list[threading.Thread]:
    """Every live ``cli-stdin-write`` thread in this process."""
    return [t for t in threading.enumerate() if t.name == "cli-stdin-write"]


class TestWriterIsCancelledAndStdinReleased:
    """Review #4: ``close()`` joins the writer and closes stdin, grandchild or not."""

    def test_supervisor_close_reclaims_writer_and_stdin(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "grandchild.pid"
        script = tmp_path / "child.py"
        script.write_text(textwrap.dedent(_grandchild_holds_stdin(pid_file)))
        assert not _writer_threads()
        try:
            started = time.monotonic()
            with _CLIProcess([sys.executable, str(script)], "probe", _TURN_TIMEOUT) as proc:
                proc.send_prompt(_BIG_PROMPT)
                lines = list(proc.lines())
                proc.raise_for_exit()
                assert any('"DONE"' in line for line in lines)
            elapsed = time.monotonic() - started

            writer = proc._writer
            assert writer is not None
            assert not writer.is_alive(), "close() left the stdin writer parked in write(2)"
            assert proc._proc.stdin is not None and proc._proc.stdin.closed, (
                "close() left the prompt pipe open for the grandchild's lifetime"
            )
            assert proc._proc.stdout is not None and proc._proc.stdout.closed
            assert proc._proc.stderr is not None and proc._proc.stderr.closed
            assert not _writer_threads()
            assert elapsed < _TURN_TIMEOUT / 2, f"turn took {elapsed:.1f}s"
            assert pid_file.exists(), "the stand-in never recorded its grandchild"
            grandchild = int(pid_file.read_text())
            if os.name == "nt":
                # No session to escape to: the grandchild is killed with the job.
                assert _wait_dead(grandchild), f"grandchild {grandchild} survived close()"
            else:
                # The grandchild is still alive: only cancellation could have freed us.
                assert pid_alive(grandchild)
        finally:
            _kill(pid_file)

    def test_adapter_generate_leaves_no_writer_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pid_file = tmp_path / "grandchild.pid"
        _install_claude(tmp_path, monkeypatch, _grandchild_holds_stdin(pid_file))
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": _TURN_TIMEOUT})
        model.initialize(_BIG_PROMPT)
        try:
            for _ in range(3):
                content, _response = model.generate()
                assert content == "DONE"
                assert not _writer_threads(), "generate() returned with a live stdin writer"
                _kill(pid_file)
        finally:
            _kill(pid_file)


class TestStopIsHonouredAfterStdoutEof:
    """Review #7: a Stop during the post-EOF exit wait raises promptly."""

    def test_raise_for_exit_raises_when_stop_is_already_set(self, tmp_path: Path) -> None:
        eof_marker = tmp_path / "eof"
        script = tmp_path / "child.py"
        script.write_text(textwrap.dedent(_lingers_after_closing_stdout(eof_marker)))
        stop = threading.Event()
        stop_signal.set_thread_stop_event(stop)
        try:
            with _CLIProcess([sys.executable, str(script)], "probe", _TURN_TIMEOUT) as proc:
                proc.send_prompt("hi")
                lines = list(proc.lines())  # returns at EOF; the child is still alive
                assert any('"DONE"' in line for line in lines)
                assert proc._proc.poll() is None
                stop.set()
                started = time.monotonic()
                with pytest.raises(KeyboardInterrupt):
                    proc.raise_for_exit()
                assert time.monotonic() - started < 3.0
        finally:
            stop_signal.set_thread_stop_event(None)

    def test_raise_for_exit_notices_stop_set_mid_wait(self, tmp_path: Path) -> None:
        eof_marker = tmp_path / "eof"
        script = tmp_path / "child.py"
        script.write_text(textwrap.dedent(_lingers_after_closing_stdout(eof_marker)))
        stop = threading.Event()
        stop_signal.set_thread_stop_event(stop)
        waiting = threading.Event()

        def press_stop_once_waiting() -> None:
            waiting.wait(10)
            stop.set()

        try:
            with _CLIProcess([sys.executable, str(script)], "probe", _TURN_TIMEOUT) as proc:
                proc.send_prompt("hi")
                list(proc.lines())
                threading.Thread(target=press_stop_once_waiting, daemon=True).start()
                started = time.monotonic()
                waiting.set()
                with pytest.raises(KeyboardInterrupt):
                    proc.raise_for_exit()
                # The unfixed wait blocks for the full 10 s exit grace.
                assert time.monotonic() - started < 5.0
        finally:
            stop_signal.set_thread_stop_event(None)

    def test_adapter_generate_does_not_return_success_after_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eof_marker = tmp_path / "eof"
        _install_claude(tmp_path, monkeypatch, _lingers_after_closing_stdout(eof_marker))
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": _TURN_TIMEOUT})
        model.initialize("hi")
        stop = threading.Event()
        stop_signal.set_thread_stop_event(stop)

        def press_stop_after_eof() -> None:
            deadline = time.monotonic() + 20
            while not eof_marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            stop.set()

        try:
            threading.Thread(target=press_stop_after_eof, daemon=True).start()
            with pytest.raises(KeyboardInterrupt):
                model.generate()
            assert all(m["role"] != "assistant" for m in model.conversation)
        finally:
            stop_signal.set_thread_stop_event(None)


class TestThinkingIsClosedOnAnyTurnException:
    """Review #8: an ordinary exception inside the turn closes the bracket."""

    def test_retry_after_parser_error_is_not_rendered_as_thinking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        counter = tmp_path / "runs"
        _install_claude(tmp_path, monkeypatch, _thinking_then_bad_event_then_ok(counter))
        thinking_events: list[bool] = []
        tokens: list[tuple[bool, str]] = []
        open_state = [False]

        def on_thinking(is_start: bool) -> None:
            open_state[0] = is_start
            thinking_events.append(is_start)

        def on_token(token: str) -> None:
            tokens.append((open_state[0], token))

        model = ClaudeCodeModel(
            "cc/opus",
            model_config={"timeout": _TURN_TIMEOUT},
            token_callback=on_token,
            thinking_callback=on_thinking,
        )
        model.initialize("hi")

        with pytest.raises(AttributeError):
            model.generate()
        assert thinking_events == [True, False], (
            f"thinking bracket not closed after the failed turn: {thinking_events}"
        )

        content, _response = model.generate()  # KISSAgent's retry
        assert content == "hello"
        assert counter.read_text() == "2"
        assert thinking_events == [True, False], thinking_events
        assert (False, "hello") in tokens, f"retry text rendered as reasoning: {tokens}"


_SLOW_READER_ECHOES_LENGTH = """
    import json
    import sys
    import time

    # Start reading only after the writer has filled the pipe and been
    # polled at least once, so the prompt arrives as many partial writes.
    time.sleep(0.5)
    data = sys.stdin.buffer.read()
    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1",
                                  "content": [{"type": "text",
                                               "text": str(len(data))}]}}),
          flush=True)
    print(json.dumps({"type": "result", "result": str(len(data)),
                      "usage": {"input_tokens": 10, "output_tokens": 5}}),
          flush=True)
"""

_CLOSES_STDIN_WITHOUT_READING = """
    import json
    import os
    import sys
    import time

    # Fill the pipe first (the writer blocks), then drop the read end: the
    # next write fails with EPIPE while we are still alive.
    time.sleep(0.3)
    os.close(0)
    time.sleep(0.3)
    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1",
                                  "content": [{"type": "text", "text": "DONE"}]}}),
          flush=True)
    print(json.dumps({"type": "result", "result": "DONE",
                      "usage": {"input_tokens": 10, "output_tokens": 5}}),
          flush=True)
"""


def _grandchild_holds_stderr(pid_file: Path, escapes_group: bool) -> str:
    """Stand-in body: spawn a grandchild that inherits only stderr, reply, exit.

    With *escapes_group* the grandchild starts its own session, so the
    process-group signals of ``close()`` do not reach it.
    """
    return f"""
    import json
    import subprocess
    import sys

    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep({_GRANDCHILD_LIFETIME})"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        start_new_session={escapes_group!r},
    )
    with open({str(pid_file)!r}, "w") as f:
        f.write(str(grandchild.pid))
    sys.stdin.read()
    print(json.dumps({{"type": "assistant",
                       "message": {{"id": "m1",
                                    "content": [{{"type": "text", "text": "DONE"}}]}}}}),
          flush=True)
    print(json.dumps({{"type": "result", "result": "DONE",
                       "usage": {{"input_tokens": 10, "output_tokens": 5}}}}),
          flush=True)
    """


def _reads_stdin_once_told(go: Path, received: Path) -> str:
    """Child source: wait for *go* to exist, read stdin to EOF, record its size in *received*."""
    part = f"{received}.part"
    return textwrap.dedent(f"""
        import os, sys, time
        while not os.path.exists({str(go)!r}):
            time.sleep(0.01)
        size = len(sys.stdin.buffer.read())
        with open({part!r}, "w") as f:
            f.write(str(size))
        os.replace({part!r}, {str(received)!r})
    """)


class _CallerHeldPastTheDeadline:
    """A ``caller_error`` poll that holds the caller once, past the deadline; it reports no error.

    It stands for a caller kept off the CPU between its deadline check
    and its look at the writer: it returns once the deadline is well
    past, the child was told to read and has read its stdin to EOF, and
    the writer thread is gone.
    """

    def __init__(self, go: Path, received: Path) -> None:
        self.go = go
        self.received = received
        self.deadline: float | None = None  # set once the supervisor is built

    def __call__(self) -> None:
        if self.deadline is None:
            return
        deadline, self.deadline = self.deadline, None
        # Past the deadline by more than the writer's poll: a writer with
        # a clock of its own has given up by then.
        time.sleep(max(deadline - time.monotonic(), 0.0) + 0.3)
        self.go.touch()
        limit = time.monotonic() + 30
        while (not self.received.exists() or _writer_threads()) and time.monotonic() < limit:
            time.sleep(0.01)


class TestNonBlockingWriterDeliversThePrompt:
    """The rewritten writer must still deliver every byte, in every ending."""

    def test_slow_reader_receives_the_whole_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial non-blocking writes are reassembled by the child into the
        complete prompt; ``send_prompt`` keeps polling while the writer is
        still busy and the child is alive."""
        _install_claude(tmp_path, monkeypatch, _SLOW_READER_ECHOES_LENGTH)
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": _TURN_TIMEOUT})
        model.initialize(_BIG_PROMPT)
        content, _response = model.generate()
        # The adapter wraps the prompt in its own framing, so the child sees
        # at least the prompt itself.
        assert int(content) >= len(_BIG_PROMPT.encode("utf-8"))
        assert not _writer_threads()

    def test_child_that_drops_stdin_while_alive_is_not_a_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``EPIPE`` on the prompt pipe (child closed fd 0 but is still
        running) ends the write quietly; the reply is still returned."""
        _install_claude(tmp_path, monkeypatch, _CLOSES_STDIN_WITHOUT_READING)
        model = ClaudeCodeModel("cc/opus", model_config={"timeout": _TURN_TIMEOUT})
        model.initialize(_BIG_PROMPT)
        content, _response = model.generate()
        assert content == "DONE"
        assert not _writer_threads()

    def test_a_prompt_counts_as_sent_only_when_the_child_got_all_of_it(
        self, tmp_path: Path
    ) -> None:
        """Only the caller ends a write at the deadline.  The writer used to give up at the
        deadline by itself and close stdin; a caller that looked a moment later found the
        writer finished, took the prompt for sent, and the child answered its first 64 KiB."""
        go, received = tmp_path / "go", tmp_path / "received"
        held = _CallerHeldPastTheDeadline(go, received)
        timeout = 2.0
        child = [sys.executable, "-c", _reads_stdin_once_told(go, received)]
        with _CLIProcess(child, "held-caller", timeout, caller_error=held) as proc:
            held.deadline = time.monotonic() + timeout
            proc.send_prompt(_BIG_PROMPT)
        assert int(received.read_text()) == len(_BIG_PROMPT)

    @pytest.mark.parametrize("escapes_group", [False, True], ids=["in-group", "own-session"])
    def test_reader_held_by_a_grandchild_is_reported_and_left_attached(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, escapes_group: bool
    ) -> None:
        """Only the prompt pipe is unconditionally reclaimed: a grandchild
        holding stderr keeps that reader parked, which ``close()`` logs and
        tolerates, while stdin and stdout are closed.

        That is the fate of a grandchild that left the child's process
        group (``own-session``); one still in the group is killed with it
        after the child exited (review 2026-10-12), so its reader drains
        to EOF and stderr is closed like the other pipes.
        """
        if escapes_group and os.name == "nt":
            pytest.skip("Windows ignores start_new_session: no grandchild leaves the Job Object")
        pid_file = tmp_path / "grandchild.pid"
        script = tmp_path / "child.py"
        script.write_text(textwrap.dedent(_grandchild_holds_stderr(pid_file, escapes_group)))
        try:
            with caplog.at_level("WARNING", logger="kiss.core.models.model"):
                with _CLIProcess([sys.executable, str(script)], "probe", _TURN_TIMEOUT) as proc:
                    proc.send_prompt("hi")
                    assert any('"DONE"' in line for line in proc.lines())
                    proc.raise_for_exit()
            assert proc._proc.stdin is not None and proc._proc.stdin.closed
            assert proc._proc.stdout is not None and proc._proc.stdout.closed
            assert proc._proc.stderr is not None
            if escapes_group:
                assert not proc._proc.stderr.closed
                assert proc._readers[1].is_alive()
                assert "left cli-stderr-drain attached to its pipe" in caplog.text
            else:
                assert proc._proc.stderr.closed
                assert not proc._readers[1].is_alive()
                assert "attached to its pipe" not in caplog.text
        finally:
            _kill(pid_file)


@pytest.mark.skipif(os.name == "nt", reason="a Popen handle keeps a Windows pid reserved")
@pytest.mark.parametrize(
    ("ending", "status"),
    [
        ("sys.exit(0)", 0),
        ("os.kill(os.getpid(), signal.SIGTERM)", -signal.SIGTERM),
    ],
    ids=["clean-exit", "killed-by-signal"],
)
def test_child_is_reaped_only_after_close_signalled_its_group(
    tmp_path: Path, ending: str, status: int
) -> None:
    """The exit status is read without reaping; ``close()``'s group kill comes first.

    ``close()`` signals the child's process group (SIGTERM, then SIGKILL
    for grandchildren that ignored it) by the child's pid, which the
    kernel keeps reserved as the group id only while the child is a
    zombie or a member lives: a child reaped by ``wait_for_exit`` (or by
    the prompt writer's poll) leaves a group id that any later session
    leader may inherit, and the SIGKILL would hit that stranger's group.
    So the status is observed with ``waitid(WNOWAIT)`` and the one reap
    happens in ``close()`` after the last signal.
    """
    script = tmp_path / "child.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, signal, sys
            print('{{"type": "result", "result": "DONE"}}', flush=True)
            {ending}
            """
        )
    )
    with _CLIProcess([sys.executable, str(script)], "probe", _TURN_TIMEOUT) as proc:
        proc.send_prompt("hi")
        assert any('"DONE"' in line for line in proc.lines())
        assert proc.wait_for_exit() == status
        pid = proc._proc.pid
        zombie = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        assert zombie is not None and zombie.si_pid == pid, "the child was reaped before close()"
        if status == 0:
            proc.raise_for_exit()
        else:
            with pytest.raises(KISSError, match=f"exit {status}"):
                proc.raise_for_exit()
    assert proc._proc.returncode == status
    with pytest.raises(ChildProcessError):  # reaped: no child of that pid is left
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
