# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``/voice`` wake-word voice-chat REPL feature.

The sorcar CLI REPL gains a ``/voice`` slash command that spawns the
wake-word listener process (``python -m kiss.server.voice_wake``)
and turns each recognised utterance into a submitted task, showing a
blinking red ``Listening ...`` indicator inside the anchored input
panel while waiting for speech.

Everything here is end-to-end: the listener is a REAL child process (a
tiny python script substituted via the ``KISS_SORCAR_VOICE_CMD``
environment variable) that speaks the exact stdout protocol of
``voice_wake`` (``READY`` / ``WAKE`` / ``TRANSCRIBING`` /
``SPEECH <json>`` / ``NO_SPEECH``), keystrokes are raw bytes written to
a real OS pipe standing in for stdin, and panel rendering is asserted
on the escape-code stream the box writes.  No mocks, no monkeypatched
functions.
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

from kiss.agents.sorcar.cli_client import CliClient, _run_repl_loop
from kiss.agents.sorcar.cli_panel import body_cursor_col, panel_cols
from kiss.agents.sorcar.cli_repl import SLASH_COMMANDS, CliCompleter
from kiss.agents.sorcar.cli_steering import _InputBox
from kiss.agents.sorcar.cli_voice import (
    VoiceListener,
    listener_command,
    read_voice_line_plain,
    start_voice,
    start_voice_anchored,
)
from kiss.core.print_to_console import ConsolePrinter

BLINK_RED = "\x1b[5m\x1b[31m"
# Steady (pre-wake) header indicator style: a reset then red, with no
# blink attribute (matches ``voice_panel_top``'s non-blinking render).
RED_ONLY = "\x1b[0m\x1b[31m"

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
    tail_sleep: float = 60.0,
    exit_code: int | None = None,
    pid_file: Path | None = None,
) -> Path:
    """Write a tiny python script that emits *lines* on stdout.

    The script optionally records its own pid (so tests can verify the
    child was really terminated), prints each protocol line with a
    small delay, then either exits with *exit_code* or keeps running
    for *tail_sleep* seconds (long enough that only an explicit
    terminate can end it within the test).
    """
    body = ["import sys, time"]
    if pid_file is not None:
        body += [
            "import os",
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))",
        ]
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


def _write_listener_with_grandchild(
    tmp_path: Path, pid_file: Path, child_pid_file: Path,
) -> Path:
    """Write a listener wrapper that owns a long-lived child process."""
    body = [
        "import os, subprocess, sys, time",
        "child = subprocess.Popen([",
        "    sys.executable, '-c', 'import time; time.sleep(60)'",
        "])",
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))",
        f"open({str(child_pid_file)!r}, 'w').write(str(child.pid))",
        "print('READY', flush=True)",
        "time.sleep(60)",
    ]
    script = tmp_path / "fake_listener_with_child.py"
    script.write_text("\n".join(body) + "\n")
    return script


def _write_exiting_wrapper_with_grandchild(
    tmp_path: Path, pid_file: Path, child_pid_file: Path,
) -> Path:
    """Write a wrapper that exits after spawning its long-lived child."""
    body = [
        "import os, subprocess, sys",
        "child = subprocess.Popen([",
        "    sys.executable, '-c', 'import time; time.sleep(60)'",
        "])",
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))",
        f"open({str(child_pid_file)!r}, 'w').write(str(child.pid))",
        "print('READY', flush=True)",
    ]
    script = tmp_path / "fake_exiting_wrapper_with_child.py"
    script.write_text("\n".join(body) + "\n")
    return script


def _write_nested_voice_listener(
    tmp_path: Path, counter_file: Path, pid_file: Path,
) -> Path:
    """Write a listener that reveals accidental nested /voice starts."""
    body = [
        "import json, os, time",
        f"counter = {str(counter_file)!r}",
        "try:",
        "    n = int(open(counter).read()) + 1",
        "except Exception:",
        "    n = 1",
        "open(counter, 'w').write(str(n))",
        f"open({str(pid_file)!r}, 'a').write(str(os.getpid()) + '\\n')",
        "def speech(text):",
        "    payload = {'text': text, 'speaker': None, 'language': 'en'}",
        "    print('SPEECH ' + json.dumps(payload), flush=True)",
        "print('READY', flush=True)",
        "if n == 1:",
        "    speech('/voice')",
        "    time.sleep(0.03)",
        "    speech('real task')",
        "else:",
        "    speech('nested task')",
    ]
    script = tmp_path / "fake_nested_voice_listener.py"
    script.write_text("\n".join(body) + "\n")
    return script


def _set_voice_cmd(
    monkeypatch: pytest.MonkeyPatch, script: Path, *extra: str
) -> None:
    """Point ``KISS_SORCAR_VOICE_CMD`` at *script* (env vars are fair game)."""
    parts = [sys.executable, str(script), *extra]
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


# ---------------------------------------------------------------------------
# stdin pipe + anchored box harness (same style as test_cli_steering.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def stdin_pipe() -> Any:
    """Replace ``sys.stdin`` with the read end of a real OS pipe.

    Yields the write fd; tests write raw keystroke bytes into it.
    """
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


def _capture_anchored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    box: _InputBox,
    lines: list[str],
    **kw: Any,
) -> tuple[str | None, str]:
    """Run one anchored background voice capture, return (speech, rendered).

    Starts the background pump session (the anchored REPL's voice
    mode), polls the box's injected-line queue the way ``_pump_stdin``
    does until the first utterance arrives (or the listener dies),
    then closes the session.
    """
    _set_voice_cmd(monkeypatch, _write_listener(tmp_path, lines, **kw))
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


def _wait_dead(proc_or_pid: Any, timeout: float = 5.0) -> bool:
    """Return True once the child process is really gone."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if isinstance(proc_or_pid, int):
            try:
                os.kill(proc_or_pid, 0)
            except OSError:
                return True
        else:
            if proc_or_pid.poll() is not None:
                return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# (k) command registration + autocomplete
# ---------------------------------------------------------------------------


class TestVoiceCommandRegistration:
    def test_voice_in_slash_commands(self) -> None:
        assert "/voice" in SLASH_COMMANDS
        help_text = SLASH_COMMANDS["/voice"].lower()
        assert "wake word" in help_text
        # Voice mode is a toggle and typing keeps working while it is on.
        assert "toggle" in help_text
        assert "typing" in help_text

    def test_voice_in_autocomplete_candidates(self, tmp_path: Path) -> None:
        completer = CliCompleter(str(tmp_path), "")
        menu = completer.build_menu("/voi")
        replacements = [r for r, _ in menu]
        assert any(r.startswith("/voice") for r in replacements)


# ---------------------------------------------------------------------------
# (h) KISS_SORCAR_VOICE_CMD parsing
# ---------------------------------------------------------------------------


class TestListenerCommand:
    def test_default_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KISS_SORCAR_VOICE_CMD", raising=False)
        assert listener_command() == [
            sys.executable, "-m", "kiss.server.voice_wake",
        ]

    def test_env_override_with_quoted_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "KISS_SORCAR_VOICE_CMD",
            "python3 '/tmp/my listener.py' --lang 'en US'",
        )
        assert listener_command() == [
            "python3", "/tmp/my listener.py", "--lang", "en US",
        ]


# ---------------------------------------------------------------------------
# (i) spawn failure
# ---------------------------------------------------------------------------


class TestSpawnFailure:
    def test_nonexistent_command_prints_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv(
            "KISS_SORCAR_VOICE_CMD", "/nonexistent-xyz/voice-listener"
        )
        session = start_voice(read_voice_line_plain)
        assert session is None
        out = capsys.readouterr().out
        assert "voice" in out.lower()
        assert "nonexistent-xyz" in out

    def test_malformed_env_override_prints_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("KISS_SORCAR_VOICE_CMD", "'unterminated")
        session = start_voice(read_voice_line_plain)
        assert session is None
        out = capsys.readouterr().out
        assert "could not start" in out
        assert "No closing quotation" in out

    def test_start_banner_lists_all_cancel_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        script = _write_listener(tmp_path, ["READY"])
        _set_voice_cmd(monkeypatch, script)
        session = start_voice(lambda _listener: None)
        try:
            assert session is not None
        finally:
            if session is not None:
                session.close()
        out = capsys.readouterr().out
        assert "Esc" in out
        assert "Ctrl+C" in out
        assert "Ctrl+D" in out
        assert "Enter" in out


# ---------------------------------------------------------------------------
# Anchored reader: display, speech, protocol handling
# ---------------------------------------------------------------------------


class TestAnchoredReader:
    def test_listening_display_and_speech_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """(a)(b)(c): blinking red indicator, Transcribing, SPEECH text."""
        del capsys
        line, rendered = _capture_anchored(
            monkeypatch, tmp_path, box,
            ["READY", "WAKE", "TRANSCRIBING", _speech("hello world")],
        )
        assert line == "hello world"
        assert BLINK_RED + "Listening ..." in rendered
        assert BLINK_RED + "Transcribing ..." in rendered
        # The overlay is cleared once voice mode is closed.
        assert box.overlay == ""

    def test_completion_menu_and_buffer_stay_editable_during_voice(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        """Voice never hijacks the box: menu, draft and caret keep working."""
        box.buf = "draft\nsecond"
        box.completer_fn = lambda _buf: ["/voice ", "/help "]
        box.feed(b"/\t", lambda _line: None, lambda: None)
        assert box._menu_open
        line, rendered = _capture_anchored(
            monkeypatch, tmp_path, box,
            ["READY", _speech("menu-free")],
        )
        assert line == "menu-free"
        # Typing stays fully usable while voice is on: the open
        # completion menu survives, still previewing the highlighted
        # candidate exactly as it did before voice started.
        assert box._menu_open
        assert box.buf == "/voice "
        # The indicator lives in the header (top border), steady red
        # pre-wake, and the caret parks on the still-visible buffer.
        assert "\x1b[36m╭─\x1b[0m\x1b[31mListening ..." in rendered
        cols = panel_cols()
        _row, col = body_cursor_col(box.buf, cols, box.cursor)
        assert f";{col}H" in rendered

    def test_no_speech_keeps_listening(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        """(f): NO_SPEECH goes back to listening until real speech."""
        line, _rendered = _capture_anchored(
            monkeypatch, tmp_path, box,
            ["READY", "WAKE", "TRANSCRIBING", "NO_SPEECH", _speech("after")],
        )
        assert line == "after"

    def test_malformed_speech_json_is_ignored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        """(l): a bad SPEECH payload keeps listening for the next one."""
        line, rendered = _capture_anchored(
            monkeypatch, tmp_path, box,
            ["SPEECH {not json", "SPEECH 42", 'SPEECH {"text": 5}',
             _speech("good one")],
        )
        assert line == "good one"
        # No wake word in this protocol stream: the header indicator is
        # re-shown steady red (never blinking) after each bad payload.
        assert rendered.count(RED_ONLY + "Listening ...") >= 2
        assert BLINK_RED + "Listening ..." not in rendered

    def test_legacy_json_string_speech_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        """The CLI accepts the legacy payload shape the wake service accepts."""
        line, _rendered = _capture_anchored(
            monkeypatch, tmp_path, box,
            ["SPEECH " + json.dumps("legacy text")],
        )
        assert line == "legacy text"

    def test_listener_unexpected_exit_prints_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """(j): a dying listener surfaces an error and leaves voice mode."""
        line, _rendered = _capture_anchored(
            monkeypatch, tmp_path, box, ["READY"], exit_code=3,
        )
        assert line is None
        assert "listener" in capsys.readouterr().out.lower()
        # The header indicator never outlives the dead listener.
        assert box.overlay == ""

    def test_listener_terminated_after_stop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """(g): stop() really terminates the child (terminate + kill)."""
        listener = _start_listener(monkeypatch, tmp_path, ["READY"])
        assert listener.alive()
        listener.stop()
        assert not listener.alive()
        assert listener.proc is not None
        assert listener.proc.poll() is not None
        # A second stop is a harmless no-op.
        listener.stop()

    @pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
    def test_stop_kills_listener_process_group_children(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """(g): stop() kills wrapper children, not only the wrapper."""
        pid_file = tmp_path / "listener.pid"
        child_pid_file = tmp_path / "listener-child.pid"
        _set_voice_cmd(
            monkeypatch,
            _write_listener_with_grandchild(tmp_path, pid_file, child_pid_file),
        )
        listener = VoiceListener()
        listener.start()
        deadline = time.monotonic() + 5.0
        while (
            (not pid_file.exists() or not child_pid_file.exists())
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert pid_file.exists()
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text())
        assert listener.alive()
        listener.stop()
        assert _wait_dead(listener.proc)
        assert _wait_dead(child_pid)

    @pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
    def test_stop_kills_group_child_after_wrapper_already_exited(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """(g): stop() also cleans up children of an exited wrapper."""
        pid_file = tmp_path / "wrapper.pid"
        child_pid_file = tmp_path / "wrapper-child.pid"
        _set_voice_cmd(
            monkeypatch,
            _write_exiting_wrapper_with_grandchild(
                tmp_path, pid_file, child_pid_file,
            ),
        )
        listener = VoiceListener()
        listener.start()
        deadline = time.monotonic() + 5.0
        while (
            (listener.proc is None or listener.proc.poll() is None)
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert listener.proc is not None
        assert listener.proc.poll() is not None
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text())
        assert not _wait_dead(child_pid, timeout=0.1)
        listener.stop()
        assert _wait_dead(child_pid)

    def test_stop_before_start_is_noop(self) -> None:
        """stop() on a never-started listener does nothing."""
        listener = VoiceListener(["true"])
        assert listener.proc is None
        assert not listener.alive()
        listener.stop()


# ---------------------------------------------------------------------------
# Fallback (non-anchored) plain reader
# ---------------------------------------------------------------------------


class TestPlainReader:
    def test_plain_listening_display_and_speech(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        listener = _start_listener(
            monkeypatch, tmp_path,
            ["READY", "WAKE", "TRANSCRIBING", _speech("fallback speech")],
        )
        try:
            line = read_voice_line_plain(listener)
        finally:
            listener.stop()
        assert line == "fallback speech"
        out = capsys.readouterr().out
        # Steady red before the wake word, blinking after WAKE.
        plain = out.find("\x1b[31mListening ...")
        blink = out.find(BLINK_RED + "Listening ...")
        assert plain != -1
        assert blink != -1
        assert plain < blink
        assert BLINK_RED + "Transcribing ..." in out

    def test_plain_reader_processes_speech_without_selectable_stdin(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Speech still works when fallback stdin has no fileno()."""
        listener = _start_listener(
            monkeypatch, tmp_path,
            ["READY", _speech("no fd speech")],
        )
        saved = sys.stdin
        sys.stdin = io.StringIO()
        try:
            line = read_voice_line_plain(listener)
        finally:
            sys.stdin = saved
            listener.stop()
        assert line == "no fd speech"

    def test_plain_cancel_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stdin_pipe: int,
    ) -> None:
        listener = _start_listener(monkeypatch, tmp_path, ["READY"])
        os.write(stdin_pipe, b"\x1b")
        try:
            line = read_voice_line_plain(listener)
        finally:
            listener.stop()
        assert line is None


# ---------------------------------------------------------------------------
# Full REPL-loop integration (real CliClient object, real child listener)
# ---------------------------------------------------------------------------


def _make_client(tmp_path: Path) -> CliClient:
    """A real, unstarted CliClient — construction touches no sockets."""
    return CliClient(
        tmp_path / "no.sock", str(tmp_path), "tab-test", ConsolePrinter(),
    )


class TestReplLoopIntegration:
    def test_voice_submits_speech_continuously(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """(c)(d)(j): speech lines submit as typed tasks, continuously."""
        del box
        script = _write_listener(
            tmp_path,
            ["READY", _speech("task one"), _speech("   "),
             _speech("task two")],
            exit_code=0,
        )
        _set_voice_cmd(monkeypatch, script)
        submitted: list[str] = []
        lines = iter(["/voice", "/exit"])

        def read_line() -> str | None:
            return next(lines, None)

        _run_repl_loop(
            _make_client(tmp_path),
            read_line,
            submitted.append,
        )
        # Blank speech is skipped; both real utterances are submitted;
        # after the listener exits the REPL falls back to typed input.
        assert submitted == ["task one", "task two"]
        assert "listener" in capsys.readouterr().out.lower()

    def test_spoken_voice_command_toggles_voice_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A spoken /voice while voice is active toggles voice OFF."""
        counter_file = tmp_path / "listener-count.txt"
        pid_file = tmp_path / "listener-pids.txt"
        _set_voice_cmd(
            monkeypatch,
            _write_nested_voice_listener(tmp_path, counter_file, pid_file),
        )
        submitted: list[str] = []
        lines = iter(["/voice", "/exit"])

        _run_repl_loop(
            _make_client(tmp_path),
            lambda: next(lines, None),
            submitted.append,
        )
        # The spoken "/voice" ends voice mode: nothing is submitted, no
        # nested listener is ever spawned, and the child is reaped.
        assert submitted == []
        assert counter_file.read_text() == "1"
        pids = pid_file.read_text().splitlines()
        assert len(pids) == 1
        assert _wait_dead(int(pids[0]))
        # The toggle-off notification is yellow too.
        out = capsys.readouterr().out
        off = [ln for ln in out.splitlines() if "Voice mode off" in ln]
        assert off and off[0].startswith("\x1b[33m")

    def test_voice_cancel_restores_prompt_and_kills_listener(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        """(e)(g): cancelling voice mode terminates the child listener."""
        pid_file = tmp_path / "listener.pid"
        script = _write_listener(
            tmp_path, ["READY"], pid_file=pid_file,
        )
        _set_voice_cmd(monkeypatch, script)
        submitted: list[str] = []
        lines = iter(["/voice", "/exit"])

        def read_line() -> str | None:
            return next(lines, None)

        def press_esc_once_listener_is_up() -> None:
            # Cancel only after the child proved it is running (wrote
            # its pid file) so the terminate-on-cancel is observable.
            deadline = time.monotonic() + 5.0
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            os.write(stdin_pipe, b"\x1b")

        presser = threading.Thread(target=press_esc_once_listener_is_up)
        presser.start()
        try:
            _run_repl_loop(
                _make_client(tmp_path),
                read_line,
                submitted.append,
            )
        finally:
            presser.join()
        assert submitted == []
        assert pid_file.exists()
        assert _wait_dead(int(pid_file.read_text()))

    def test_spoken_exit_word_exits_and_kills_listener(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        box: _InputBox,
        stdin_pipe: int,
    ) -> None:
        """Exit words spoken in voice mode end the REPL; child is reaped."""
        pid_file = tmp_path / "listener.pid"
        script = _write_listener(
            tmp_path, ["READY", _speech("exit")], pid_file=pid_file,
        )
        _set_voice_cmd(monkeypatch, script)
        submitted: list[str] = []

        _run_repl_loop(
            _make_client(tmp_path),
            lambda: "/voice",
            submitted.append,
        )
        assert submitted == []
        assert _wait_dead(int(pid_file.read_text()))

    def test_voice_uses_plain_reader_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without an explicit voice_reader the plain fallback is used."""
        script = _write_listener(
            tmp_path, ["READY", _speech("plain hello")], exit_code=0,
        )
        _set_voice_cmd(monkeypatch, script)
        submitted: list[str] = []
        lines = iter(["/voice", "/exit"])

        _run_repl_loop(
            _make_client(tmp_path),
            lambda: next(lines, None),
            submitted.append,
        )
        assert submitted == ["plain hello"]
        # No wake word spoken: the inline indicator stays steady red.
        out = capsys.readouterr().out
        assert "\x1b[31mListening ..." in out
        assert BLINK_RED + "Listening ..." not in out

    def test_voice_spawn_failure_keeps_repl_alive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """(i): /voice with an unspawnable listener does not crash."""
        monkeypatch.setenv(
            "KISS_SORCAR_VOICE_CMD", "/nonexistent-xyz/voice-listener"
        )
        submitted: list[str] = []
        lines = iter(["/voice", "still typing works", "/exit"])

        _run_repl_loop(
            _make_client(tmp_path),
            lambda: next(lines, None),
            submitted.append,
        )
        assert submitted == ["still typing works"]
        assert "nonexistent-xyz" in capsys.readouterr().out
