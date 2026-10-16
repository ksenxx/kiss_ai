# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the daemon plays talk clips for headless local clients.

The ``talk`` tool ships a GPT-synthesized MP3 inside the broadcast event.
A browser plays it itself, but a local client of the daemon's own
machine (a ``sorcar`` CLI or ``daemon_client`` run) has no player, so
the daemon plays the clip on its own speakers whenever such a client
addressed the talk's tab, and stays silent when only browsers did.

These tests run a REAL ``RemoteAccessServer`` and REAL token-bearing
local WSS client connections (no mocks); the daemon's audio player is a
REAL child process substituted via ``KISS_SORCAR_PLAY_CMD`` whose marker
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
from kiss.server import talk_player
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.server.conftest import redirect_persistence, restore_persistence, write_fake_player

MP3_BYTES = b"ID3\x03\x00fake-mp3-frames-" + bytes(range(64))
MP3_B64 = base64.b64encode(MP3_BYTES).decode("ascii")


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
    """The daemon plays talk clips for headless local clients."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        self.marker_dir = Path(self.tmpdir) / "markers"
        player = write_fake_player(Path(self.tmpdir), self.marker_dir)
        self.saved_play_cmd = os.environ.get("KISS_SORCAR_PLAY_CMD")
        os.environ["KISS_SORCAR_PLAY_CMD"] = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(player))}"
        )
        talk_player.reset_shared_player_for_tests()
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert

        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self.task_id = uuid.uuid4().hex

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self.saved_play_cmd is None:
            os.environ.pop("KISS_SORCAR_PLAY_CMD", None)
        else:
            os.environ["KISS_SORCAR_PLAY_CMD"] = self.saved_play_cmd
        talk_player.reset_shared_player_for_tests()
        if th._db_conn is not None:
            th._db_conn.close()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self, tab_id: str
    ) -> tuple[LocalReader, LocalWriter]:
        """Open one local client addressing the canonical tab *tab_id*.

        The tab is created in the shared tab registry first, as any run
        on it would do before emitting a talk; the ``ready`` this client
        announces next is what records its interest in the tab.
        """
        self.server._agent_server.tab_registry.update_tab(
            tab_id, title="cli run", create=True,
        )
        reader, writer = await open_local_connection(
            self.server, limit=16 * 1024 * 1024
        )
        for cmd in (
            {"type": "setWorkDir", "workDir": self.tmpdir},
            {"type": "ready", "tabId": tab_id, "workDir": self.tmpdir},
        ):
            writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
        await writer.drain()
        # The server handles ``ready`` on its own task; a talk broadcast
        # before that sees no interested local client.  Wait the way the
        # fan-out reads the interest.
        deadline = asyncio.get_event_loop().time() + 5.0
        while not self.server._printer._local_client_addressed({tab_id}):
            self.assertLess(
                asyncio.get_event_loop().time(), deadline,
                "daemon never recorded the local client's tab",
            )
            await asyncio.sleep(0.01)
        return reader, writer

    async def _collect_talks(
        self,
        reader: LocalReader,
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

    async def test_daemon_plays_clip_for_local_client(self) -> None:
        """A local client addressed the tab: the daemon plays the clip.

        The client still receives its copy of the event (byte-exact
        clip in the marker file proves the daemon played the same clip).
        """
        cli_tab = "cli-" + uuid.uuid4().hex[:8]
        reader, writer = await self._connect(cli_tab)
        self.server._printer.subscribe_tab(self.task_id, cli_tab)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-daemon-1", with_clip=True)
        )

        talks = await self._collect_talks(reader, 1)
        self.assertEqual(len(talks), 1, "local client never got the talk copy")
        self.assertEqual(talks[0]["tabId"], cli_tab)
        markers = await asyncio.to_thread(self._wait_markers, 1)
        self.assertEqual(len(markers), 1, "daemon never played the clip natively")
        played = base64.b64decode(markers[0]["audio_b64"])
        self.assertEqual(played, MP3_BYTES, "clip bytes were corrupted")
        writer.close()

    async def test_clipless_talk_spawns_no_player(self) -> None:
        """No clip: nothing to play natively; the event still fans out."""
        cli_tab = "cli-" + uuid.uuid4().hex[:8]
        reader, writer = await self._connect(cli_tab)
        self.server._printer.subscribe_tab(self.task_id, cli_tab)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-daemon-4", with_clip=False)
        )

        talks = await self._collect_talks(reader, 1)
        self.assertEqual(len(talks), 1)
        await asyncio.sleep(0.3)
        self.assertEqual(
            len(list(self.marker_dir.glob("*.json"))),
            0,
            "the daemon must not spawn a player for a clipless talk",
        )
        writer.close()

    async def test_remote_only_subscribers_skip_daemon_playback(self) -> None:
        """Only browser tabs subscribed: the daemon's speakers stay silent.

        The observer is a local client that addressed no tab (it sent a
        bare ``ready``), so it receives every event copy without
        counting as an interested client of the talk's tab.
        """
        remote_tab = "remote-webtab-" + uuid.uuid4().hex[:8]
        reader, writer = await open_local_connection(
            self.server, limit=16 * 1024 * 1024
        )
        writer.write((json.dumps({"type": "ready"}) + "\n").encode("utf-8"))
        await writer.drain()
        self.server._printer.subscribe_tab(self.task_id, remote_tab)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-daemon-5", with_clip=True)
        )

        talks = await self._collect_talks(reader, 1)
        self.assertEqual(len(talks), 1)
        self.assertEqual(talks[0]["tabId"], remote_tab)
        await asyncio.sleep(0.3)
        self.assertEqual(
            len(list(self.marker_dir.glob("*.json"))),
            0,
            "no local client addressed the tab, so the daemon must not "
            "play the clip on its own speakers",
        )
        writer.close()
