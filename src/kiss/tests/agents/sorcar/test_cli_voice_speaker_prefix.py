# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: CLI voice speech carries the "Speaker #N" prefix.

The chat webview (``src/kiss/agents/vscode/media/voice.js``,
``insertSpeech``) prefixes every recognised utterance with
``"Speaker #<n> says in the language <lang> that: <text>"`` (or
``"Speaker #<n> says that: <text>"`` without a language) whenever the
wake-word listener identified the speaker as an integer number >= 1,
and never submits blank speech.  The sorcar CLI voice mode MUST behave
identically, through the shared :func:`speaker_prefixed_text` helper in
:mod:`kiss.server.voice_wake` (the module that also produces the
``SPEECH`` payloads).

Everything here is end-to-end in the established voice-test style: the
wake-word listener is a REAL child process substituted via the
``KISS_SORCAR_VOICE_CMD`` environment variable speaking the exact
``voice_wake`` stdout protocol, and lines reach the REPL through the
real background pump (anchored) or the real modal capture (plain
fallback).  No mocks.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import cli_voice
from kiss.agents.sorcar.cli_steering import AnchoredRepl
from kiss.server.voice_wake import speaker_prefixed_text


def _speech_line(
    text: object, speaker: object = None, language: object = None,
) -> str:
    """Return one ``SPEECH <json dict>`` protocol line."""
    payload = {"text": text, "speaker": speaker, "language": language}
    return "SPEECH " + json.dumps(payload)


def _write_listener(tmp_path: Path, lines: list[str]) -> Path:
    """Write a tiny python listener script that emits *lines* on stdout."""
    body = ["import sys, time", "print('READY', flush=True)"]
    for line in lines:
        body += [f"print({line!r}, flush=True)", "time.sleep(0.03)"]
    body.append("time.sleep(60.0)")
    script = tmp_path / "fake_listener.py"
    script.write_text("\n".join(body) + "\n")
    return script


def _set_voice_cmd(monkeypatch: pytest.MonkeyPatch, script: Path) -> None:
    """Point ``KISS_SORCAR_VOICE_CMD`` at *script*."""
    parts = [sys.executable, str(script)]
    monkeypatch.setenv(
        "KISS_SORCAR_VOICE_CMD", " ".join(shlex.quote(p) for p in parts)
    )


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


def _capture_anchored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repl: AnchoredRepl,
    protocol_lines: list[str],
    n_lines: int = 1,
) -> list[str]:
    """Run a real background voice session and read *n_lines* utterances."""
    script = _write_listener(tmp_path, protocol_lines)
    _set_voice_cmd(monkeypatch, script)
    session = cli_voice.start_voice_anchored(repl.box)
    assert session is not None
    try:
        lines: list[str] = []
        for _ in range(n_lines):
            line = repl.read_idle_line()
            assert line is not None
            lines.append(line)
        return lines
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Anchored REPL (background pump): the submitted line must be prefixed
# ---------------------------------------------------------------------------


class TestAnchoredSpeakerPrefix:
    def test_speaker_and_language_prefixed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """speaker=1 + language → the full webview prefix on the CLI line."""
        lines = _capture_anchored(
            monkeypatch,
            tmp_path,
            repl,
            ["WAKE", "TRANSCRIBING", _speech_line("bonjour a tous", 1, "fr")],
        )
        assert lines == ["Speaker #1 says in the language fr that: bonjour a tous"]

    def test_speaker_without_language_prefixed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """speaker=2, language null → the short webview prefix."""
        lines = _capture_anchored(
            monkeypatch,
            tmp_path,
            repl,
            ["WAKE", _speech_line("run the tests", 2, None)],
        )
        assert lines == ["Speaker #2 says that: run the tests"]

    def test_blank_language_uses_short_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """language of only whitespace behaves like no language (JS trim)."""
        lines = _capture_anchored(
            monkeypatch,
            tmp_path,
            repl,
            [_speech_line("hello", 7, "   ")],
        )
        assert lines == ["Speaker #7 says that: hello"]

    def test_non_qualifying_speakers_get_no_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """null/0/-1/1.5/bool/string speakers submit the bare text."""
        cases = [None, 0, -1, 1.5, True, "tester"]
        protocol = [
            _speech_line(f"utterance {i}", speaker, "en")
            for i, speaker in enumerate(cases)
        ]
        lines = _capture_anchored(
            monkeypatch, tmp_path, repl, protocol, n_lines=len(cases),
        )
        assert lines == [f"utterance {i}" for i in range(len(cases))]

    def test_legacy_string_payload_gets_no_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """The legacy bare-JSON-string payload has no speaker → no prefix."""
        lines = _capture_anchored(
            monkeypatch,
            tmp_path,
            repl,
            ["SPEECH " + json.dumps("legacy text")],
        )
        assert lines == ["legacy text"]

    def test_speech_text_is_trimmed_like_webview(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """Surrounding whitespace is trimmed before the prefix is applied."""
        lines = _capture_anchored(
            monkeypatch,
            tmp_path,
            repl,
            [_speech_line("  fix the bug  ", 3, " de ")],
        )
        assert lines == ["Speaker #3 says in the language de that: fix the bug"]

    def test_blank_speech_never_submits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        repl: AnchoredRepl,
        stdin_pipe: int,
    ) -> None:
        """Whitespace-only speech is dropped (webview never submits it)."""
        lines = _capture_anchored(
            monkeypatch,
            tmp_path,
            repl,
            [
                _speech_line("   ", 1, "en"),
                _speech_line("real task", 1, None),
            ],
        )
        # The blank utterance was skipped; the first line read is the
        # real one.
        assert lines == ["Speaker #1 says that: real task"]


# ---------------------------------------------------------------------------
# Plain fallback REPL (modal capture): same prefix contract
# ---------------------------------------------------------------------------


class TestPlainSpeakerPrefix:
    def _capture_plain(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        protocol_lines: list[str],
    ) -> str | None:
        script = _write_listener(tmp_path, protocol_lines)
        _set_voice_cmd(monkeypatch, script)
        listener = cli_voice.VoiceListener()
        listener.start()
        try:
            return cli_voice.read_voice_line_plain(listener)
        finally:
            listener.stop()

    def test_plain_capture_prefixes_speaker_and_language(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stdin_pipe: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The fallback modal capture applies the same webview prefix."""
        text = self._capture_plain(
            monkeypatch,
            tmp_path,
            ["WAKE", "TRANSCRIBING", _speech_line("hola", 4, "es")],
        )
        assert text == "Speaker #4 says in the language es that: hola"
        # The 🎤 echo shows the exact submitted (prefixed) line.
        assert "🎤 Speaker #4 says in the language es that: hola" in (
            capsys.readouterr().out
        )

    def test_plain_capture_no_speaker_no_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stdin_pipe: int,
    ) -> None:
        """No identified speaker → the bare text, exactly as spoken."""
        text = self._capture_plain(
            monkeypatch, tmp_path, [_speech_line("plain text", None, None)],
        )
        assert text == "plain text"

    def test_plain_capture_skips_blank_speech(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stdin_pipe: int,
    ) -> None:
        """Blank speech keeps the modal capture listening for the next one."""
        text = self._capture_plain(
            monkeypatch,
            tmp_path,
            [_speech_line("", 1, "en"), _speech_line("next", 5, "en")],
        )
        assert text == "Speaker #5 says in the language en that: next"


# ---------------------------------------------------------------------------
# Shared helper: exact behavioural parity with voice.js insertSpeech
# ---------------------------------------------------------------------------


class TestSpeakerPrefixedTextParity:
    """Case-for-case parity with ``insertSpeech`` in media/voice.js."""

    @pytest.mark.parametrize(
        ("text", "speaker", "language", "expected"),
        [
            # Qualifying speakers (finite integer number >= 1).
            ("hi", 1, "fr", "Speaker #1 says in the language fr that: hi"),
            ("hi", 2, None, "Speaker #2 says that: hi"),
            ("hi", 2, "", "Speaker #2 says that: hi"),
            ("hi", 2, "  ", "Speaker #2 says that: hi"),
            ("hi", 12, " en-US ", "Speaker #12 says in the language en-US that: hi"),
            # JSON numbers have no int/float split in JS: an integral
            # float qualifies (Math.floor(2.0) === 2.0).
            ("hi", 2.0, None, "Speaker #2 says that: hi"),
            # Non-qualifying speakers → bare text.
            ("hi", None, "fr", "hi"),
            ("hi", 0, "fr", "hi"),
            ("hi", -3, "fr", "hi"),
            ("hi", 1.5, "fr", "hi"),
            ("hi", True, "fr", "hi"),  # JS typeof true !== 'number'
            ("hi", "tester", "fr", "hi"),
            ("hi", float("inf"), "fr", "hi"),
            ("hi", float("nan"), "fr", "hi"),
            # Text trimming; empty text yields "" (caller never submits).
            ("  hi  ", 1, None, "Speaker #1 says that: hi"),
            ("   ", 1, "fr", ""),
            ("", None, None, ""),
            # Non-string text behaves like JS String(undefined-guard): "".
            (None, 1, "fr", ""),
            (42, 1, "fr", ""),
            # Non-string language behaves like no language.
            ("hi", 1, 7, "Speaker #1 says that: hi"),
        ],
    )
    def test_parity(
        self, text: object, speaker: object, language: object, expected: str,
    ) -> None:
        assert speaker_prefixed_text(text, speaker, language) == expected


def test_prefix_survives_wait_between_lines() -> None:
    """Guard against accidental global state: two calls are independent."""
    a = speaker_prefixed_text("one", 1, "en")
    time.sleep(0.01)
    b = speaker_prefixed_text("two", None, None)
    assert a == "Speaker #1 says in the language en that: one"
    assert b == "two"
