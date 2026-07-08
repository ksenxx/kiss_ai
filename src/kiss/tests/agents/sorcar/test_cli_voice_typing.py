# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: typing during ``/voice`` + voice indicator in the steer header.

Two user-visible requirements of the sorcar CLI voice mode:

1. When voice is activated, TEXT INPUT MUST STILL WORK in the anchored
   input textbox — typing, editing, and Enter-to-submit behave exactly
   as without voice, while recognised speech is submitted concurrently
   as if it had been typed.
2. While a task is running with voice on, the STEER input textbox
   header (the panel's top border) MUST show the ``Listening ...`` /
   ``Transcribing ...`` indicator (steady red before the wake word,
   blinking after ``WAKE``).

Everything here is end-to-end in the established voice-test style: the
wake-word listener is a REAL child process substituted via the
``KISS_SORCAR_VOICE_CMD`` environment variable speaking the exact
``voice_wake`` stdout protocol, keystrokes are raw bytes written to a
real OS pipe standing in for stdin, and panel rendering is asserted on
the escape-code stream the anchored box writes.  No mocks.
"""

from __future__ import annotations

import functools
import io
import json
import os
import shlex
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import cli_voice
from kiss.agents.sorcar.cli_client import CliClient, _run_repl_loop
from kiss.agents.sorcar.cli_steering import AnchoredRepl
from kiss.core.print_to_console import ConsolePrinter

CYAN = "\x1b[36m"
RED = "\x1b[31m"
BLINK = "\x1b[5m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"
# Header prefixes: the top border opens "╭─" (cyan) and the indicator
# follows immediately — steady red before the wake word, blinking red
# after it (see ``voice_panel_top``).
HDR_PLAIN = f"{CYAN}╭─{RESET}{RED}"
HDR_BLINK = f"{CYAN}╭─{RESET}{BLINK}{RED}"


# ---------------------------------------------------------------------------
# Fake listener child processes (real subprocesses, no mocks)
# ---------------------------------------------------------------------------


def _speech(text: str) -> str:
    """Return one ``SPEECH <json>`` protocol line for *text*."""
    payload = {"text": text, "speaker": "tester", "language": "en-US"}
    return "SPEECH " + json.dumps(payload)


def _write_listener(
    tmp_path: Path,
    lines: list[str],
    *,
    delay: float = 0.0,
    tail_sleep: float = 60.0,
    exit_code: int | None = None,
    pid_file: Path | None = None,
) -> Path:
    """Write a tiny python script that emits *lines* on stdout.

    The script optionally records its own pid, prints ``READY``, waits
    *delay* seconds (so the test can type first), then prints each
    protocol line with a small gap, and finally either exits with
    *exit_code* or keeps running for *tail_sleep* seconds.
    """
    body = ["import sys, time"]
    if pid_file is not None:
        body += [
            "import os",
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))",
        ]
    body.append("print('READY', flush=True)")
    if delay:
        body.append(f"time.sleep({delay})")
    for line in lines:
        body += [
            f"print({line!r}, flush=True)",
            "time.sleep(0.03)",
        ]
    if exit_code is not None:
        body.append(f"sys.exit({exit_code})")
    else:
        body.append(f"time.sleep({tail_sleep})")
    script = tmp_path / "fake_listener.py"
    script.write_text("\n".join(body) + "\n")
    return script


def _set_voice_cmd(monkeypatch: pytest.MonkeyPatch, script: Path) -> None:
    """Point ``KISS_SORCAR_VOICE_CMD`` at *script*."""
    parts = [sys.executable, str(script)]
    monkeypatch.setenv(
        "KISS_SORCAR_VOICE_CMD", " ".join(shlex.quote(p) for p in parts)
    )


# ---------------------------------------------------------------------------
# stdin pipe + anchored REPL harness
# ---------------------------------------------------------------------------


@pytest.fixture
def stdin_pipe() -> Any:
    """Replace ``sys.stdin`` with the read end of a real OS pipe."""
    r, w = os.pipe()
    saved = sys.stdin
    sys.stdin = os.fdopen(r, "r", closefd=False)
    try:
        yield w
    finally:
        sys.stdin.close()
        sys.stdin = saved
        os.close(r)
        try:
            os.close(w)
        except OSError:
            pass


@pytest.fixture
def repl() -> AnchoredRepl:
    """An anchored REPL whose box renders into a captured StringIO."""
    r = AnchoredRepl()
    r.box._out = io.StringIO()
    r.box._active = True  # render without owning a real terminal
    return r


def _make_client(tmp_path: Path) -> CliClient:
    """A real, unstarted CliClient — construction touches no sockets."""
    return CliClient(
        tmp_path / "no.sock", str(tmp_path), "tab-test", ConsolePrinter(),
    )


def _wait_until(cond: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll *cond* until it holds or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def _pid_dead(pid: int) -> bool:
    """Return True once *pid* no longer exists."""
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    """Return True once process *pid* is really gone."""
    return _wait_until(lambda: _pid_dead(pid), timeout)


def _voice_start(repl: AnchoredRepl) -> Callable[[], Any]:
    """The anchored REPL's voice starter, as ``_run_anchored_client`` wires it."""
    return functools.partial(cli_voice.start_voice_anchored, repl.box)


# ---------------------------------------------------------------------------
# Requirement 1: typing MUST keep working while voice is active
# ---------------------------------------------------------------------------


class TestTypingDuringVoice:
    def test_typing_submits_while_voice_active(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """A line typed while voice mode is on submits exactly like normal."""
        pid_file = tmp_path / "listener.pid"
        script = _write_listener(tmp_path, [], pid_file=pid_file)
        _set_voice_cmd(monkeypatch, script)
        submitted: list[str] = []

        def writer() -> None:
            assert _wait_until(pid_file.exists)
            os.write(stdin_pipe, b"typed hello\r")
            assert _wait_until(lambda: bool(submitted))
            os.write(stdin_pipe, b"/exit\r")

        os.write(stdin_pipe, b"/voice\r")
        t = threading.Thread(target=writer)
        t.start()
        try:
            _run_repl_loop(
                _make_client(tmp_path),
                repl.read_idle_line,
                submitted.append,
                voice_start=_voice_start(repl),
            )
        finally:
            t.join()
        assert submitted == ["typed hello"]
        # The REPL exit closed the listener child.
        assert _wait_dead(int(pid_file.read_text()))

    def test_idle_header_shows_indicator_while_voice_on(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """The idle header opens with the steady red indicator + idle title."""
        pid_file = tmp_path / "listener.pid"
        script = _write_listener(tmp_path, [], pid_file=pid_file)
        _set_voice_cmd(monkeypatch, script)
        submitted: list[str] = []

        def writer() -> None:
            assert _wait_until(pid_file.exists)
            assert _wait_until(
                lambda: HDR_PLAIN + "Listening ..."
                in repl.box._out.getvalue()
            )
            os.write(stdin_pipe, b"/exit\r")

        os.write(stdin_pipe, b"/voice\r")
        t = threading.Thread(target=writer)
        t.start()
        try:
            _run_repl_loop(
                _make_client(tmp_path),
                repl.read_idle_line,
                submitted.append,
                voice_start=_voice_start(repl),
            )
        finally:
            t.join()
        rendered = repl.box._out.getvalue()
        assert HDR_PLAIN + "Listening ..." in rendered
        # The idle title follows the indicator on the same top border.
        assert f"Listening ...{RESET}{CYAN}KISS Sorcar" in rendered

    def test_typed_draft_preserved_when_speech_submits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """Speech submits its own line; a half-typed draft stays in the box."""
        script = _write_listener(
            tmp_path, ["WAKE", _speech("go")], delay=0.6,
        )
        _set_voice_cmd(monkeypatch, script)
        session = cli_voice.start_voice_anchored(repl.box)
        assert session is not None
        try:
            os.write(stdin_pipe, b"draft")
            line = repl.read_idle_line()
        finally:
            session.close()
        assert line == "go"
        assert repl.box.buf == "draft"

    def test_voice_toggle_off_via_typed_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Typing /voice while voice is on turns it off and kills the child."""
        pid_file = tmp_path / "listener.pid"
        script = _write_listener(tmp_path, [], pid_file=pid_file)
        _set_voice_cmd(monkeypatch, script)
        submitted: list[str] = []

        def writer() -> None:
            assert _wait_until(pid_file.exists)
            os.write(stdin_pipe, b"/voice\r")
            assert _wait_dead(int(pid_file.read_text()))
            os.write(stdin_pipe, b"still here\r")
            assert _wait_until(lambda: bool(submitted))
            os.write(stdin_pipe, b"/exit\r")

        os.write(stdin_pipe, b"/voice\r")
        t = threading.Thread(target=writer)
        t.start()
        try:
            _run_repl_loop(
                _make_client(tmp_path),
                repl.read_idle_line,
                submitted.append,
                voice_start=_voice_start(repl),
            )
        finally:
            t.join()
        assert submitted == ["still here"]
        assert repl.box.overlay == ""
        out = capsys.readouterr().out
        off = [ln for ln in out.splitlines() if "Voice mode off" in ln]
        assert off and off[0].startswith(YELLOW)

    def test_spoken_voice_command_toggles_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """Saying "/voice" while voice is on turns voice mode off."""
        pid_file = tmp_path / "listener.pid"
        script = _write_listener(
            tmp_path, [_speech("/voice")], pid_file=pid_file,
        )
        _set_voice_cmd(monkeypatch, script)
        submitted: list[str] = []

        def writer() -> None:
            assert _wait_until(pid_file.exists)
            assert _wait_dead(int(pid_file.read_text()))
            os.write(stdin_pipe, b"after\r")
            assert _wait_until(lambda: bool(submitted))
            os.write(stdin_pipe, b"/exit\r")

        os.write(stdin_pipe, b"/voice\r")
        t = threading.Thread(target=writer)
        t.start()
        try:
            _run_repl_loop(
                _make_client(tmp_path),
                repl.read_idle_line,
                submitted.append,
                voice_start=_voice_start(repl),
            )
        finally:
            t.join()
        assert submitted == ["after"]
        assert repl.box.overlay == ""

    def test_listener_death_clears_overlay_and_typing_still_works(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An unexpectedly dying listener leaves the keyboard fully usable."""
        pid_file = tmp_path / "listener.pid"
        script = _write_listener(
            tmp_path, [], pid_file=pid_file, exit_code=0,
        )
        _set_voice_cmd(monkeypatch, script)
        submitted: list[str] = []

        def writer() -> None:
            assert _wait_until(pid_file.exists)
            assert _wait_dead(int(pid_file.read_text()))
            assert _wait_until(lambda: repl.box.overlay == "")
            os.write(stdin_pipe, b"after death\r")
            assert _wait_until(lambda: bool(submitted))
            os.write(stdin_pipe, b"/exit\r")

        os.write(stdin_pipe, b"/voice\r")
        t = threading.Thread(target=writer)
        t.start()
        try:
            _run_repl_loop(
                _make_client(tmp_path),
                repl.read_idle_line,
                submitted.append,
                voice_start=_voice_start(repl),
            )
        finally:
            t.join()
        assert submitted == ["after death"]
        assert repl.box.overlay == ""
        out = capsys.readouterr().out
        err = [ln for ln in out.splitlines() if "unexpectedly" in ln]
        assert err and err[0].startswith(YELLOW)


# ---------------------------------------------------------------------------
# Requirement 2: the STEER header shows Listening ... / Transcribing ...
# ---------------------------------------------------------------------------


class TestSteerHeaderIndicator:
    def test_steer_header_shows_indicator_and_speech_queues(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """Listening/Transcribing render in the steer header; speech steers."""
        script = _write_listener(
            tmp_path,
            ["WAKE", "TRANSCRIBING", _speech("steer me")],
            delay=0.3,
        )
        _set_voice_cmd(monkeypatch, script)
        session = cli_voice.start_voice_anchored(repl.box)
        assert session is not None
        queued: list[str] = []
        done = threading.Event()

        def on_submit(line: str) -> None:
            queued.append(line)
            done.set()

        try:
            repl.run_steering_loop(on_submit, lambda: None, done.is_set)
        finally:
            session.close()
        assert queued == ["steer me"]
        rendered = repl.box._out.getvalue()
        # Steady red Listening ... before the wake word.
        assert HDR_PLAIN + "Listening ..." in rendered
        # Blinking after WAKE, with the STEER title following the
        # indicator on the same top border.
        assert (
            f"{HDR_BLINK}Listening ...{RESET}{CYAN} Dynamic steer"
            in rendered
        )
        assert HDR_BLINK + "Transcribing ..." in rendered
        # Closing voice mode clears the header indicator.
        assert repl.box.overlay == ""

    def test_close_discards_undrained_utterance(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """A spoken line still queued at voice-off never submits later."""
        script = _write_listener(tmp_path, [_speech("stale line")])
        _set_voice_cmd(monkeypatch, script)
        session = cli_voice.start_voice_anchored(repl.box)
        assert session is not None
        try:
            # Wait for the utterance to be injected but do NOT drain it
            # (no pump-stdin loop is running) — then turn voice off.
            assert _wait_until(lambda: bool(repl.box._injected))
        finally:
            session.close()
        # close() discarded the stale utterance and the indicator.
        assert repl.box.drain_injected() == []
        assert repl.box.overlay == ""

    def test_typing_steers_while_voice_on(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """Typed steer lines and spoken ones both queue while voice is on."""
        script = _write_listener(
            tmp_path, [_speech("spoken steer")], delay=0.5,
        )
        _set_voice_cmd(monkeypatch, script)
        session = cli_voice.start_voice_anchored(repl.box)
        assert session is not None
        queued: list[str] = []
        done = threading.Event()

        def on_submit(line: str) -> None:
            queued.append(line)
            if len(queued) >= 2:
                done.set()

        os.write(stdin_pipe, b"typed steer\r")
        try:
            repl.run_steering_loop(on_submit, lambda: None, done.is_set)
        finally:
            session.close()
        assert sorted(queued) == ["spoken steer", "typed steer"]
