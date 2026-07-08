# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: voice indicator in the input-box HEADER + yellow notifications.

The ``/voice`` indicator (``Listening ...`` / ``Transcribing ...``) must
be shown at the *beginning of the header* (the panel's top border) of
the sorcar CLI input textbox — not in the body — and it must start
*blinking only when the sorcar wake word is detected* (the listener's
``WAKE`` protocol line).  Before the wake word (and after ``NO_SPEECH``
or a malformed ``SPEECH`` payload) the indicator is steady red.

Additionally, all CLI notification toasts render in yellow.

Everything here is end-to-end, following the harness style of
``test_cli_voice.py``: the wake-word listener is a REAL child process
substituted via ``KISS_SORCAR_VOICE_CMD``, keystrokes travel over a
real OS pipe standing in for stdin, and assertions are made on the raw
escape-code stream the anchored box writes.  No mocks.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.cli_steering import _InputBox
from kiss.agents.sorcar.cli_voice import (
    VoiceListener,
    read_voice_line_plain,
    start_voice,
    start_voice_anchored,
)
from kiss.core.print_to_console import ConsolePrinter

CYAN = "\x1b[36m"
RED = "\x1b[31m"
BLINK = "\x1b[5m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"
# Header prefixes: the top border opens "╭─" (cyan) and the indicator
# follows immediately — steady red before the wake word, blinking red
# after it.
HDR_PLAIN = f"{CYAN}╭─{RESET}{RED}"
HDR_BLINK = f"{CYAN}╭─{RESET}{BLINK}{RED}"


def _speech(text: str) -> str:
    """Return one ``SPEECH <json>`` protocol line for *text*."""
    payload = {"text": text, "speaker": "tester", "language": "en-US"}
    return "SPEECH " + json.dumps(payload)


def _write_listener(
    tmp_path: Path,
    lines: list[str],
    *,
    tail_sleep: float = 60.0,
    exit_code: int | None = None,
) -> Path:
    """Write a tiny python script that emits *lines* on stdout."""
    body = ["import sys, time"]
    for line in lines:
        body += [f"print({line!r}, flush=True)", "time.sleep(0.03)"]
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


def _start_listener(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lines: list[str], **kw: Any
) -> VoiceListener:
    """Spawn a started :class:`VoiceListener` running the fake script."""
    _set_voice_cmd(monkeypatch, _write_listener(tmp_path, lines, **kw))
    listener = VoiceListener()
    listener.start()
    return listener


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
def box() -> _InputBox:
    """An anchored input box rendering into a captured StringIO."""
    b = _InputBox(threading.RLock(), io.StringIO())
    b._active = True  # render without owning a real terminal
    return b


def _capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    box: _InputBox,
    lines: list[str],
) -> tuple[str | None, str]:
    """Run one anchored background voice capture, return (speech, rendered).

    Starts the background pump session (the anchored REPL's voice
    mode), polls the box's injected-line queue the way ``_pump_stdin``
    does until the first utterance arrives (or the listener dies),
    then closes the session — so all header rendering happens exactly
    as in the live REPL.
    """
    _set_voice_cmd(monkeypatch, _write_listener(tmp_path, lines))
    session = start_voice_anchored(box)
    assert session is not None
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            injected = box.drain_injected()
            if injected:
                return injected[0], box._out.getvalue()
            if not session.active:
                break
            time.sleep(0.02)
        return None, box._out.getvalue()
    finally:
        session.close()


class TestHeaderIndicator:
    def test_indicator_in_header_not_in_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        """Pre-wake: steady red ``Listening ...`` at the header start."""
        text, rendered = _capture(
            monkeypatch, tmp_path, box, ["READY", _speech("hi")],
        )
        assert text == "hi"
        # Indicator opens the top border, steady red (no blink pre-wake).
        assert HDR_PLAIN + "Listening ..." in rendered
        assert BLINK + RED + "Listening ..." not in rendered
        # The box's normal title follows the indicator on the same top
        # border (voice no longer swaps in a modal title — the box
        # stays fully typeable).
        assert f"{RED}Listening ...{RESET}{CYAN} Dynamic steer" in rendered
        # The body never shows the indicator (chevron row is buffer/placeholder).
        assert "› Listening" not in rendered

    def test_blink_starts_on_wake(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        """Blinking MUST start when the sorcar wake word is detected."""
        text, rendered = _capture(
            monkeypatch, tmp_path, box, ["READY", "WAKE", _speech("x")],
        )
        assert text == "x"
        plain = rendered.find(HDR_PLAIN + "Listening ...")
        blink = rendered.find(HDR_BLINK + "Listening ...")
        assert plain != -1, "steady indicator missing before wake"
        assert blink != -1, "indicator did not blink after WAKE"
        assert plain < blink

    def test_transcribing_blinks_in_header(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        text, rendered = _capture(
            monkeypatch, tmp_path, box,
            ["READY", "WAKE", "TRANSCRIBING", _speech("x")],
        )
        assert text == "x"
        assert HDR_BLINK + "Transcribing ..." in rendered
        assert HDR_PLAIN + "Transcribing ..." not in rendered

    def test_no_speech_resets_to_steady_listening(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        """NO_SPEECH flips back to the steady (non-blinking) indicator."""
        text, rendered = _capture(
            monkeypatch, tmp_path, box,
            ["READY", "WAKE", "TRANSCRIBING", "NO_SPEECH", _speech("x")],
        )
        assert text == "x"
        last_plain = rendered.rfind(HDR_PLAIN + "Listening ...")
        last_blink = rendered.rfind(HDR_BLINK + "Transcribing ...")
        assert last_blink != -1
        assert last_plain > last_blink

    def test_malformed_speech_resets_to_steady_listening(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        text, rendered = _capture(
            monkeypatch, tmp_path, box,
            ["SPEECH {not json", _speech("good")],
        )
        assert text == "good"
        assert rendered.count(HDR_PLAIN + "Listening ...") >= 2
        assert HDR_BLINK not in rendered

    def test_overlay_state_cleared_after_capture(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        text, _rendered = _capture(
            monkeypatch, tmp_path, box, ["READY", "WAKE", _speech("done")],
        )
        assert text == "done"
        assert box.overlay == ""
        assert box.overlay_blink is False


class TestPlainReaderBlink:
    def test_plain_blink_only_after_wake(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        listener = _start_listener(
            monkeypatch, tmp_path, ["READY", "WAKE", _speech("x")],
        )
        try:
            text = read_voice_line_plain(listener)
        finally:
            listener.stop()
        assert text == "x"
        out = capsys.readouterr().out
        plain = out.find(f"{RED}Listening ...{RESET}")
        blink = out.find(f"{BLINK}{RED}Listening ...{RESET}")
        assert plain != -1, "steady inline indicator missing before wake"
        assert blink != -1, "inline indicator did not blink after WAKE"
        assert plain < blink


class TestYellowVoiceMessages:
    def test_voice_banner_is_yellow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _set_voice_cmd(monkeypatch, _write_listener(tmp_path, ["READY"]))
        session = start_voice(lambda _listener: None)
        try:
            assert session is not None
        finally:
            if session is not None:
                session.close()
        out = capsys.readouterr().out
        banner = [ln for ln in out.splitlines() if "Voice mode" in ln]
        assert banner and banner[0].startswith(YELLOW)

    def test_spawn_error_is_yellow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv(
            "KISS_SORCAR_VOICE_CMD", "/nonexistent-xyz/voice-listener"
        )
        assert start_voice(read_voice_line_plain) is None
        out = capsys.readouterr().out
        err = [ln for ln in out.splitlines() if "voice" in ln.lower()]
        assert err and err[0].startswith(YELLOW)

    def test_listener_exit_error_is_yellow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _set_voice_cmd(
            monkeypatch,
            _write_listener(tmp_path, ["READY"], exit_code=3),
        )
        session = start_voice_anchored(box)
        assert session is not None
        try:
            deadline = time.monotonic() + 10.0
            while session.active and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            session.close()
        assert not session.active
        out = capsys.readouterr().out
        err = [
            ln
            for ln in out.splitlines()
            if "listener" in ln.lower() and "exited" in ln.lower()
        ]
        assert err and err[0].startswith(YELLOW)


class TestYellowNotifications:
    @pytest.mark.parametrize("severity", ["info", "warning", "error", "bogus"])
    def test_notification_panel_is_yellow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        severity: str,
    ) -> None:
        """Every notification toast renders with a yellow border and text."""
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
        buf = io.StringIO()
        printer = ConsolePrinter(file=buf)
        printer.print(
            "something happened",
            type="notification",
            severity=severity,
            progress_message="details",
        )
        out = buf.getvalue()
        assert "something happened" in out
        assert "details" in out
        # Yellow border + yellow message text + dim-yellow progress
        # text; no cyan/red panel chrome.
        assert "\x1b[33" in out
        assert "\x1b[33msomething happened" in out
        assert "\x1b[2;33mdetails" in out
        assert "\x1b[36" not in out  # info no longer cyan
        assert "\x1b[31" not in out  # error no longer red
