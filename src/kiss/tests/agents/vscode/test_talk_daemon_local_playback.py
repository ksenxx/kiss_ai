# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the daemon plays talk clips for local VS Code webviews.

Regression tests for the "alien" talk voice: the ``talk`` tool ships a
natural GPT-synthesized MP3 inside the broadcast event, but a VS Code
chat webview cannot reliably play it — Chromium's autoplay policy
rejects ``Audio.play()`` in a webview unless the user interacted with
it seconds earlier (microsoft/vscode#197937 / #178642, closed as not
actionable), so the webview silently fell back to the robotic Web
Speech system voice.  The daemon now plays the clip natively on its
own machine's speakers whenever a local UDS webview tab is subscribed
and stamps every same-machine copy ``muted``.

These tests run a REAL ``RemoteAccessServer`` with a UDS listener and
REAL UDS client connections (no mocks), mirroring
``test_talk_double_playback.py``; the daemon's audio player is a REAL
child process substituted via ``KISS_SORCAR_PLAY_CMD`` whose marker
files carry the exact bytes it played.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import cli_talk
from kiss.agents.vscode.web_server import RemoteAccessServer

# A tiny-but-real MP3-looking byte string; the fake player only copies
# bytes, so any payload works — what matters is byte-exact delivery.
MP3_BYTES = b"ID3\x03\x00fake-mp3-frames-" + bytes(range(64))
MP3_B64 = base64.b64encode(MP3_BYTES).decode("ascii")


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _write_player(tmp_path: Path, marker_dir: Path) -> Path:
    """Write a python script standing in for the daemon's audio player.

    The script receives the audio file path as its LAST argument and
    writes one unique JSON marker file into *marker_dir* recording the
    audio bytes (base64) it read.
    """
    marker_dir.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_player.py"
    script.write_text(
        "import base64, json, os, sys, uuid\n"
        f"marker_dir = {str(marker_dir)!r}\n"
        "path = sys.argv[-1]\n"
        "with open(path, 'rb') as fh:\n"
        "    data = fh.read()\n"
        "marker = os.path.join(marker_dir, uuid.uuid4().hex + '.json')\n"
        "with open(marker, 'w') as fh:\n"
        "    json.dump({'audio_b64':"
        " base64.b64encode(data).decode('ascii')}, fh)\n"
    )
    return script


def _talk_event(
    task_id: str, talk_id: str, *, with_clip: bool
) -> dict[str, object]:
    """Build a ``talk`` broadcast event like the ``talk`` tool emits."""
    event: dict[str, object] = {
        "type": "talk",
        "taskId": task_id,
        "talkId": talk_id,
        "text": "hello from the agent",
        "language": "en-US",
        "emotion": "warm",
    }
    if with_clip:
        event["audioB64"] = MP3_B64
        event["audioMime"] = "audio/mpeg"
    return event


class TestTalkDaemonLocalPlayback(IsolatedAsyncioTestCase):
    """The daemon plays talk clips for local UDS webview tabs."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        self.marker_dir = Path(self.tmpdir) / "markers"
        player = _write_player(Path(self.tmpdir), self.marker_dir)
        self.saved_play_cmd = os.environ.get("KISS_SORCAR_PLAY_CMD")
        os.environ["KISS_SORCAR_PLAY_CMD"] = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(player))}"
        )
        cli_talk.reset_shared_player_for_tests()
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.agents.vscode.web_server import _generate_self_signed_cert

        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self.task_id = uuid.uuid4().hex

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self.saved_play_cmd is None:
            os.environ.pop("KISS_SORCAR_PLAY_CMD", None)
        else:
            os.environ["KISS_SORCAR_PLAY_CMD"] = self.saved_play_cmd
        cli_talk.reset_shared_player_for_tests()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self, tab_id: str, *, cli: bool = False
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open one UDS client, announce ``ready`` (and ``cliTabHello``)."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024
        )
        for cmd in (
            {"type": "setWorkDir", "workDir": self.tmpdir},
            {"type": "ready", "tabId": tab_id, "workDir": self.tmpdir},
        ):
            writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
        if cli:
            hello = {"type": "cliTabHello", "tabId": tab_id}
            writer.write((json.dumps(hello) + "\n").encode("utf-8"))
        await writer.drain()
        return reader, writer

    async def _collect_talks(
        self,
        reader: asyncio.StreamReader,
        count: int,
        timeout: float = 5.0,
    ) -> list[dict[str, object]]:
        """Read events until *count* ``talk`` copies arrive (or timeout)."""
        talks: list[dict[str, object]] = []
        deadline = asyncio.get_event_loop().time() + timeout
        while len(talks) < count:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                line = await asyncio.wait_for(
                    reader.readline(), timeout=remaining
                )
            except TimeoutError:
                break
            if not line:
                break
            msg = json.loads(line.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "talk":
                talks.append(msg)
        return talks

    def _wait_markers(self, count: int, timeout: float = 5.0) -> list[dict]:
        """Wait for *count* fake-player marker files; return their JSON."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            markers = sorted(self.marker_dir.glob("*.json"))
            if len(markers) >= count:
                return [json.loads(m.read_text()) for m in markers]
            time.sleep(0.05)
        markers = sorted(self.marker_dir.glob("*.json"))
        return [json.loads(m.read_text()) for m in markers]

    async def test_daemon_plays_clip_and_mutes_local_webview(self) -> None:
        """THE ALIEN-VOICE BUG: local webview must not speech-fallback.

        With a clip in the event and a local UDS webview subscribed,
        the daemon must play the clip natively (byte-exact) and mute
        the webview's copy — an unmuted copy makes the webview attempt
        ``Audio.play()``, get autoplay-rejected, and read the text with
        the robotic Web Speech voice instead.
        """
        web_tab = "webtab-" + uuid.uuid4().hex[:8]
        web_reader, web_writer = await self._connect(web_tab)
        self.server._printer.subscribe_tab(self.task_id, web_tab)
        await asyncio.sleep(0.05)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-daemon-1", with_clip=True)
        )

        talks = await self._collect_talks(web_reader, 1)
        self.assertEqual(len(talks), 1, "webview never got the talk copy")
        self.assertEqual(talks[0]["tabId"], web_tab)
        self.assertTrue(
            talks[0].get("muted"),
            "local webview copy must be muted when the daemon plays the "
            "clip — an unmuted copy falls back to the robotic Web Speech "
            "voice (the alien-voice bug)",
        )
        markers = await asyncio.to_thread(self._wait_markers, 1)
        self.assertEqual(
            len(markers), 1, "daemon never played the clip natively"
        )
        played = base64.b64decode(markers[0]["audio_b64"])
        self.assertEqual(played, MP3_BYTES, "clip bytes were corrupted")
        web_writer.close()

    async def test_daemon_playback_mutes_local_cli_tab_too(self) -> None:
        """The CLI tab on the daemon machine must not double-play."""
        web_tab = "webtab-" + uuid.uuid4().hex[:8]
        cli_tab = "clitab-" + uuid.uuid4().hex[:8]
        web_reader, web_writer = await self._connect(web_tab)
        cli_reader, cli_writer = await self._connect(cli_tab, cli=True)
        self.server._printer.subscribe_tab(self.task_id, web_tab)
        self.server._printer.subscribe_tab(self.task_id, cli_tab)
        await asyncio.sleep(0.05)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-daemon-2", with_clip=True)
        )

        talks = await self._collect_talks(web_reader, 2)
        by_tab = {t["tabId"]: t for t in talks}
        self.assertIn(web_tab, by_tab)
        self.assertIn(cli_tab, by_tab)
        self.assertTrue(by_tab[web_tab].get("muted"))
        self.assertTrue(
            by_tab[cli_tab].get("muted"),
            "the CLI tab's copy must be muted while the daemon plays — "
            "both are on the same machine",
        )
        markers = await asyncio.to_thread(self._wait_markers, 1)
        self.assertEqual(len(markers), 1, "expected exactly one playback")
        for w in (web_writer, cli_writer):
            w.close()

    async def test_remote_web_tab_keeps_playable_copy(self) -> None:
        """A remote WSS tab is another device; its copy stays playable."""
        web_tab = "webtab-" + uuid.uuid4().hex[:8]
        remote_tab = "remote-webtab-" + uuid.uuid4().hex[:8]
        web_reader, web_writer = await self._connect(web_tab)
        self.server._printer.subscribe_tab(self.task_id, web_tab)
        # Deliberately subscribe the remote tab WITHOUT a UDS
        # connection: this models a remote WSS browser tab.
        self.server._printer.subscribe_tab(self.task_id, remote_tab)
        await asyncio.sleep(0.05)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-daemon-3", with_clip=True)
        )

        talks = await self._collect_talks(web_reader, 2)
        by_tab = {t["tabId"]: t for t in talks}
        self.assertIn(web_tab, by_tab)
        self.assertIn(remote_tab, by_tab)
        self.assertTrue(by_tab[web_tab].get("muted"))
        self.assertFalse(
            by_tab[remote_tab].get("muted"),
            "the remote device's copy must stay playable there",
        )
        web_writer.close()

    async def test_clipless_talk_keeps_webview_copy_playable(self) -> None:
        """No clip → nothing to play natively; the webview still speaks."""
        web_tab = "webtab-" + uuid.uuid4().hex[:8]
        web_reader, web_writer = await self._connect(web_tab)
        self.server._printer.subscribe_tab(self.task_id, web_tab)
        await asyncio.sleep(0.05)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-daemon-4", with_clip=False)
        )

        talks = await self._collect_talks(web_reader, 1)
        self.assertEqual(len(talks), 1)
        self.assertFalse(
            talks[0].get("muted"),
            "a clipless talk must keep the webview copy playable so the "
            "Web Speech fallback can still read the text aloud",
        )
        await asyncio.sleep(0.3)
        self.assertEqual(
            len(list(self.marker_dir.glob("*.json"))),
            0,
            "the daemon must not spawn a player for a clipless talk",
        )
        web_writer.close()

    async def test_remote_only_subscribers_skip_daemon_playback(self) -> None:
        """Only remote tabs subscribed → daemon speakers stay silent."""
        remote_tab = "remote-webtab-" + uuid.uuid4().hex[:8]
        probe_tab = "probe-" + uuid.uuid4().hex[:8]
        # A UDS probe connection NOT subscribed to the task observes
        # the fanned-out copies without being a local player itself.
        probe_reader, probe_writer = await self._connect(probe_tab)
        self.server._printer.subscribe_tab(self.task_id, remote_tab)
        await asyncio.sleep(0.05)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-daemon-5", with_clip=True)
        )

        talks = await self._collect_talks(probe_reader, 1)
        self.assertEqual(len(talks), 1)
        self.assertEqual(talks[0]["tabId"], remote_tab)
        self.assertFalse(talks[0].get("muted"))
        await asyncio.sleep(0.3)
        self.assertEqual(
            len(list(self.marker_dir.glob("*.json"))),
            0,
            "no local webview tab is subscribed, so the daemon must not "
            "play the clip on its own speakers",
        )
        probe_writer.close()
