# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the ``talk`` tool plays audio in the sorcar CLI.

The ``talk`` tool (sorcar_agent.py) broadcasts a
``{"type": "talk", "text", "language", "emotion", "talkId",
"audioB64", "audioMime"}`` event through the printer.  In the chat
webview the client plays the MP3 (``media/main.js`` ``playTalkAudio``)
or falls back to system TTS.  The sorcar CLI REPL uses
:class:`RecordingConsolePrinter`, so the CLI itself must play the
audio on the terminal machine's default speakers — historically it
did not, which is the bug reproduced here.

Everything is end-to-end in the established CLI-test style: the audio
player and the TTS fallback are REAL child processes substituted via
the ``KISS_SORCAR_PLAY_CMD`` / ``KISS_SORCAR_SAY_CMD`` environment
variables (same pattern as ``KISS_SORCAR_VOICE_CMD``), and assertions
are made on marker files those children write.  No mocks.
"""

from __future__ import annotations

import base64
import json
import shlex
import sys
import time
from pathlib import Path

import pytest

from kiss.ui.cli import cli_talk
from kiss.ui.cli.cli_printer import RecordingConsolePrinter

# A tiny-but-real MP3-looking byte string; the fake player only copies
# bytes, so any payload works — what matters is byte-exact delivery.
MP3_BYTES = b"ID3\x03\x00fake-mp3-frames-" + bytes(range(64))
MP3_B64 = base64.b64encode(MP3_BYTES).decode("ascii")


# ---------------------------------------------------------------------------
# Fake player / say child processes (real subprocesses, no mocks)
# ---------------------------------------------------------------------------


def _write_player(
    tmp_path: Path,
    marker_dir: Path,
    *,
    sleep: float = 0.0,
    exit_code: int = 0,
) -> Path:
    """Write a python script that stands in for the audio player.

    The script receives the audio file path as its LAST argument and
    writes one unique JSON marker file into *marker_dir* recording the
    start/end times, the audio bytes it read, and whether the audio
    file still existed.
    """
    marker_dir.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_player.py"
    script.write_text(
        "\n".join(
            [
                "import base64, json, os, sys, time",
                "path = sys.argv[-1]",
                "start = time.time()",
                "data = open(path, 'rb').read()",
                f"time.sleep({sleep})",
                "marker = {",
                "    'start': start,",
                "    'end': time.time(),",
                "    'data_b64': base64.b64encode(data).decode(),",
                "    'path': path,",
                "    'argv': sys.argv[1:],",
                "}",
                f"out = os.path.join({str(marker_dir)!r}, "
                "f'play-{time.time_ns()}-{os.getpid()}.json')",
                "tmp = out + '.tmp'",
                "open(tmp, 'w').write(json.dumps(marker))",
                "os.replace(tmp, out)",
                f"sys.exit({exit_code})",
            ]
        )
        + "\n"
    )
    return script


def _write_say(tmp_path: Path, marker_dir: Path) -> Path:
    """Write a python script standing in for the system TTS command.

    Records the spoken text (its LAST argument) in a unique marker
    file inside *marker_dir*.
    """
    marker_dir.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_say.py"
    script.write_text(
        "\n".join(
            [
                "import json, os, sys, time",
                "marker = {'text': sys.argv[-1], 'argv': sys.argv[1:]}",
                f"out = os.path.join({str(marker_dir)!r}, "
                "f'say-{time.time_ns()}-{os.getpid()}.json')",
                "tmp = out + '.tmp'",
                "open(tmp, 'w').write(json.dumps(marker))",
                "os.replace(tmp, out)",
            ]
        )
        + "\n"
    )
    return script


def _cmd(script: Path, *extra: str) -> str:
    """Build a quoted command string running *script* with *extra* args."""
    parts = [sys.executable, str(script), *extra]
    return " ".join(shlex.quote(p) for p in parts)


def _markers(marker_dir: Path, prefix: str) -> list[dict]:
    """Load all JSON marker files with *prefix* from *marker_dir*."""
    if not marker_dir.is_dir():
        return []
    out = []
    for p in sorted(marker_dir.glob(prefix + "-*.json")):
        out.append(json.loads(p.read_text()))
    return out


def _wait_markers(
    marker_dir: Path, prefix: str, count: int, timeout: float = 10.0
) -> list[dict]:
    """Poll until *count* markers with *prefix* exist (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = _markers(marker_dir, prefix)
        if len(found) >= count:
            return found
    return _markers(marker_dir, prefix)


def _talk_event(
    text: str = "hello there",
    *,
    audio: str | None = MP3_B64,
    talk_id: str | None = None,
) -> dict:
    """Build a ``talk`` broadcast event like the ``talk`` tool emits."""
    event: dict = {
        "type": "talk",
        "language": "en-US",
        "text": text,
        "emotion": "warm",
        "talkId": talk_id if talk_id is not None else f"t{time.time_ns():x}",
    }
    if audio is not None:
        event["audioB64"] = audio
        event["audioMime"] = "audio/mpeg"
    return event


@pytest.fixture(autouse=True)
def _fresh_player():
    """Give every test its own shared player (talkId dedupe is global)."""
    cli_talk.reset_shared_player_for_tests()
    yield
    cli_talk.reset_shared_player_for_tests()


# ---------------------------------------------------------------------------
# The reproduction + core playback behaviour through the CLI printer
# ---------------------------------------------------------------------------


class TestCliTalkPlaysAudio:
    """The CLI printer must actually play broadcast ``talk`` audio."""

    def test_talk_event_plays_audio_bytes_through_player(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE BUG: a talk broadcast on the CLI printer plays nothing.

        Post-fix the printer must hand the exact MP3 bytes from
        ``audioB64`` to the machine's audio player.
        """
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        printer = RecordingConsolePrinter()
        printer.broadcast(_talk_event("hi from the agent"))
        played = _wait_markers(marker_dir, "play", 1)
        assert len(played) == 1, "talk audio was never handed to the player"
        assert base64.b64decode(played[0]["data_b64"]) == MP3_BYTES

    def test_same_talk_id_plays_exactly_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Duplicate deliveries of one talkId speak once (webview parity)."""
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        printer = RecordingConsolePrinter()
        event = _talk_event("only once", talk_id="dup-1")
        printer.broadcast(event)
        printer.broadcast(dict(event))
        assert len(_wait_markers(marker_dir, "play", 1)) == 1
        time.sleep(0.5)
        assert len(_markers(marker_dir, "play")) == 1

    def test_back_to_back_talks_are_serialised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two talks never speak over each other (talk-queue parity)."""
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir, sleep=0.4)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        printer = RecordingConsolePrinter()
        printer.broadcast(_talk_event("first"))
        printer.broadcast(_talk_event("second"))
        played = _wait_markers(marker_dir, "play", 2)
        assert len(played) == 2
        played.sort(key=lambda m: m["start"])
        assert played[1]["start"] >= played[0]["end"]

    def test_event_without_talk_id_still_plays(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An id-less talk event is played (the webview plays those too)."""
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        printer = RecordingConsolePrinter()
        event = _talk_event("no id")
        del event["talkId"]
        printer.broadcast(event)
        assert len(_wait_markers(marker_dir, "play", 1)) == 1

    def test_temp_audio_file_removed_after_playback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decoded temp MP3 is deleted once the player finishes."""
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        printer = RecordingConsolePrinter()
        printer.broadcast(_talk_event("cleanup"))
        played = _wait_markers(marker_dir, "play", 1)
        assert len(played) == 1
        deadline = time.time() + 5.0
        audio_path = Path(played[0]["path"])
        while time.time() < deadline and audio_path.exists():
            time.sleep(0.02)
        assert not audio_path.exists()


# ---------------------------------------------------------------------------
# TTS fallback (no audio / broken audio / broken player)
# ---------------------------------------------------------------------------


class TestCliTalkSayFallback:
    """Without playable audio the CLI falls back to system TTS."""

    def test_no_audio_falls_back_to_say(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        say = _write_say(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        monkeypatch.setenv("KISS_SORCAR_SAY_CMD", _cmd(say))
        printer = RecordingConsolePrinter()
        printer.broadcast(_talk_event("speak me", audio=None))
        spoken = _wait_markers(marker_dir, "say", 1)
        assert len(spoken) == 1
        assert spoken[0]["text"] == "speak me"
        assert _markers(marker_dir, "play") == []

    def test_malformed_base64_falls_back_to_say(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "markers"
        say = _write_say(tmp_path, marker_dir)
        player = _write_player(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        monkeypatch.setenv("KISS_SORCAR_SAY_CMD", _cmd(say))
        printer = RecordingConsolePrinter()
        printer.broadcast(_talk_event("bad audio", audio="!!!not-base64!!!"))
        spoken = _wait_markers(marker_dir, "say", 1)
        assert len(spoken) == 1
        assert spoken[0]["text"] == "bad audio"

    def test_player_failure_falls_back_to_say(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A player exiting non-zero still speaks via the TTS fallback."""
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir, exit_code=3)
        say = _write_say(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        monkeypatch.setenv("KISS_SORCAR_SAY_CMD", _cmd(say))
        printer = RecordingConsolePrinter()
        printer.broadcast(_talk_event("resilient"))
        spoken = _wait_markers(marker_dir, "say", 1)
        assert len(spoken) == 1
        assert spoken[0]["text"] == "resilient"

    def test_unspawnable_player_falls_back_to_say(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "markers"
        say = _write_say(tmp_path, marker_dir)
        monkeypatch.setenv(
            "KISS_SORCAR_PLAY_CMD", str(tmp_path / "no-such-player")
        )
        monkeypatch.setenv("KISS_SORCAR_SAY_CMD", _cmd(say))
        printer = RecordingConsolePrinter()
        printer.broadcast(_talk_event("still audible"))
        spoken = _wait_markers(marker_dir, "say", 1)
        assert len(spoken) == 1
        assert spoken[0]["text"] == "still audible"

    def test_blank_text_and_no_audio_plays_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        say = _write_say(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        monkeypatch.setenv("KISS_SORCAR_SAY_CMD", _cmd(say))
        printer = RecordingConsolePrinter()
        printer.broadcast(_talk_event("   ", audio=None))
        time.sleep(0.8)
        assert _markers(marker_dir, "play") == []
        assert _markers(marker_dir, "say") == []

    def test_unspawnable_say_is_harmless(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No player AND no TTS: the CLI must not crash the agent loop."""
        monkeypatch.setenv(
            "KISS_SORCAR_PLAY_CMD", str(tmp_path / "no-such-player")
        )
        monkeypatch.setenv(
            "KISS_SORCAR_SAY_CMD", str(tmp_path / "no-such-say")
        )
        printer = RecordingConsolePrinter()
        printer.broadcast(_talk_event("into the void"))
        # Give the worker a moment; any crash surfaces as an exception
        # in later broadcasts or a dead worker thread.
        time.sleep(0.5)
        printer.broadcast(_talk_event("still alive"))
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Command resolution (env overrides, quoting)
# ---------------------------------------------------------------------------


class TestCommandResolution:
    """Env overrides are parsed with shlex so quoted args survive."""

    def test_player_command_env_quoting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "KISS_SORCAR_PLAY_CMD", "'/usr/local/my player' -q --rate 1.0"
        )
        assert cli_talk.player_command() == [
            "/usr/local/my player",
            "-q",
            "--rate",
            "1.0",
        ]

    def test_say_command_env_quoting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KISS_SORCAR_SAY_CMD", "'/opt/speak me' --slow")
        assert cli_talk.say_command() == ["/opt/speak me", "--slow"]

    def test_player_command_default_is_machine_player(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KISS_SORCAR_PLAY_CMD", raising=False)
        cmd = cli_talk.player_command()
        if sys.platform == "darwin":
            assert cmd == ["afplay"]
        else:
            assert cmd is None or cmd[0] in {"mpg123", "ffplay", "mpv"}

    def test_say_command_default_is_machine_tts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KISS_SORCAR_SAY_CMD", raising=False)
        cmd = cli_talk.say_command()
        if sys.platform == "darwin":
            assert cmd == ["say"]
        else:
            assert cmd is None or cmd[0] in {"espeak", "espeak-ng", "spd-say"}


# ---------------------------------------------------------------------------
# The interactive REPL path: daemon-fanned talk events via the dispatcher
# ---------------------------------------------------------------------------


class TestDispatcherTalkPlayback:
    """The interactive REPL (daemon client) must play fanned-out talks.

    The interactive ``sorcar`` REPL does not run the agent in-process:
    tasks execute in the ``sorcar web`` daemon and every event —
    including ``talk`` — arrives over the UDS connection at
    :class:`_EventDispatcher`.  Dropping ``talk`` there is exactly the
    reported "play tool plays nothing" bug for the interactive REPL.
    """

    def _dispatcher(self, tab_id: str = "tab-1"):
        import io

        from kiss.core.print_to_console import ConsolePrinter
        from kiss.ui.cli.cli_client import _EventDispatcher

        return _EventDispatcher(ConsolePrinter(file=io.StringIO()), tab_id=tab_id)

    def test_dispatched_talk_event_plays_audio_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE BUG (interactive REPL): dispatched talks play nothing."""
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        dispatcher = self._dispatcher()
        event = _talk_event("spoken via daemon")
        event["tabId"] = "tab-1"
        dispatcher.dispatch(event)
        played = _wait_markers(marker_dir, "play", 1)
        assert len(played) == 1, "dispatched talk never reached the player"
        assert base64.b64decode(played[0]["data_b64"]) == MP3_BYTES

    def test_dispatched_talk_without_tab_id_plays(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Untagged (global) talk copies play — talkId dedupe protects."""
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        dispatcher = self._dispatcher()
        dispatcher.dispatch(_talk_event("untagged"))
        assert len(_wait_markers(marker_dir, "play", 1)) == 1

    def test_foreign_tab_talk_event_is_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A copy stamped for another client's tab must stay silent."""
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        dispatcher = self._dispatcher(tab_id="mine")
        event = _talk_event("for another window")
        event["tabId"] = "someone-else"
        dispatcher.dispatch(event)
        time.sleep(0.6)
        assert _markers(marker_dir, "play") == []

    def test_dispatched_duplicate_talk_id_plays_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fan-out duplicates of one talkId speak once on this device."""
        marker_dir = tmp_path / "markers"
        player = _write_player(tmp_path, marker_dir)
        monkeypatch.setenv("KISS_SORCAR_PLAY_CMD", _cmd(player))
        dispatcher = self._dispatcher()
        event = _talk_event("once", talk_id="fan-1")
        event["tabId"] = "tab-1"
        dispatcher.dispatch(dict(event))
        dispatcher.dispatch(dict(event))
        assert len(_wait_markers(marker_dir, "play", 1)) == 1
        time.sleep(0.5)
        assert len(_markers(marker_dir, "play")) == 1
