# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: talk muting is per-endpoint, not per-serialization.

The canonical tab registry mirrors the SAME tab ids to every client,
so a remote WSS browser and a local VS Code webview (UDS) both show
tab ``T``.  When the daemon plays a talk clip natively for the local
webview, only the SAME-MACHINE (UDS) copies must be muted — the
remote browser is a different device with its own speakers and must
keep a playable copy.

Two regressions covered, both against a REAL ``RemoteAccessServer``
with a real UDS listener, a real ``wss://`` client, and a real
audio-player child process (``KISS_SORCAR_PLAY_CMD``) — no mocks:

1. ``_fanout_talk`` used to build ONE serialization per tab (muted
   iff the tab id was in the UDS local-tab map) and send that same
   copy to BOTH transports, so the remote browser stayed silent
   whenever any local webview showed the tab.
2. After a webview reload, ``ready`` announces only the placeholder
   tab; canonical background tabs adopted from the ``tabs_state``
   snapshot never re-registered in ``_local_uds_tab_counts``, so a
   talk for a background tab skipped daemon-native playback entirely
   (the webview cannot autoplay → silence).  ``ready`` on a UDS
   connection now synchronizes the connection's local-tab membership
   from the canonical tab registry.
3. The local-tab bookkeeping used to be ADD-ONLY per connection:
   closing a canonical tab removed it from every client UI (via the
   ``tabs_state`` broadcast) but never pruned it from any live UDS
   connection's ``local_tabs`` set or from ``_local_uds_tab_counts``
   (decremented only on socket disconnect).  A still-running task's
   talk for the closed tab then triggered daemon-native playback even
   though NO local webview showed the tab, and a repeated ``ready``
   could not self-heal.  Registry removal (close / displacement) now
   prunes the id everywhere, and ``ready`` reconciles instead of only
   adding.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import shutil
import socket
import ssl
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
from kiss.server import talk_player
from kiss.server.web_server import RemoteAccessServer

MP3_BYTES = b"ID3\x03\x00fake-mp3-frames-" + bytes(range(64))
MP3_B64 = base64.b64encode(MP3_BYTES).decode("ascii")


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


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


def _talk_event(task_id: str, talk_id: str) -> dict[str, object]:
    """Build a clip-carrying ``talk`` event like the ``talk`` tool emits."""
    return {
        "type": "talk",
        "taskId": task_id,
        "talkId": talk_id,
        "text": "hello from the agent",
        "language": "en-US",
        "emotion": "warm",
        "audioB64": MP3_B64,
        "audioMime": "audio/mpeg",
    }


class TestTalkEndpointMuting(IsolatedAsyncioTestCase):
    """One canonical tab id, two transports, endpoint-specific muting."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        self.marker_dir = Path(self.tmpdir) / "markers"
        player = _write_player(Path(self.tmpdir), self.marker_dir)
        self.saved_play_cmd = os.environ.get("KISS_SORCAR_PLAY_CMD")
        os.environ["KISS_SORCAR_PLAY_CMD"] = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(player))}"
        )
        talk_player.reset_shared_player_for_tests()
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert

        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self.task_id = uuid.uuid4().hex
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await self.server.stop_async()
        if self.saved_play_cmd is None:
            os.environ.pop("KISS_SORCAR_PLAY_CMD", None)
        else:
            os.environ["KISS_SORCAR_PLAY_CMD"] = self.saved_play_cmd
        talk_player.reset_shared_player_for_tests()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_uds(
        self, tab_id: str
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open one UDS client (a VS Code webview) and announce ``ready``."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024
        )
        self._writers.append(writer)
        for cmd in (
            {"type": "setWorkDir", "workDir": self.tmpdir},
            {"type": "ready", "tabId": tab_id, "workDir": self.tmpdir},
        ):
            writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
        await writer.drain()
        return reader, writer

    async def _collect_uds_talks(
        self,
        reader: asyncio.StreamReader,
        count: int,
        timeout: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Read UDS events until *count* ``talk`` copies arrive."""
        talks: list[dict[str, Any]] = []
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

    async def _collect_wss_talks(
        self, ws: Any, count: int, timeout: float = 5.0
    ) -> list[dict[str, Any]]:
        """Read WSS frames until *count* ``talk`` copies arrive."""
        talks: list[dict[str, Any]] = []
        deadline = asyncio.get_event_loop().time() + timeout
        while len(talks) < count:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                break
            msg = json.loads(frame)
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

    async def test_wss_copy_unmuted_while_uds_copy_muted(self) -> None:
        """ITEM 1: the remote browser's copy of the SAME tab stays playable.

        One canonical tab id is observed by both a local UDS webview
        and a remote WSS browser (tab mirroring shows every registry
        tab on every client).  The daemon plays the clip natively for
        the local webview, so the UDS copy must be muted — but the
        WSS copy goes to a DIFFERENT device and must stay unmuted.
        """
        tab_id = "shared-tab-" + uuid.uuid4().hex[:8]
        uds_reader, _uds_writer = await self._connect_uds(tab_id)
        url = f"wss://127.0.0.1:{self.port}/ws"
        async with connect(url, ssl=_no_verify_ssl()) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")
            self.server._printer.subscribe_tab(self.task_id, tab_id)
            await asyncio.sleep(0.05)

            self.server._printer.broadcast(
                _talk_event(self.task_id, "talk-endpoint-1")
            )

            uds_talks = await self._collect_uds_talks(uds_reader, 1)
            wss_talks = await self._collect_wss_talks(ws, 1)
        self.assertEqual(
            len(uds_talks), 1, "UDS webview never got its talk copy"
        )
        self.assertEqual(
            len(wss_talks), 1, "WSS browser never got its talk copy"
        )
        self.assertEqual(uds_talks[0]["tabId"], tab_id)
        self.assertEqual(wss_talks[0]["tabId"], tab_id)
        self.assertTrue(
            uds_talks[0].get("muted"),
            "the same-machine UDS webview copy must be muted while the "
            "daemon plays the clip on this machine's speakers",
        )
        self.assertFalse(
            wss_talks[0].get("muted"),
            "the remote WSS browser is a DIFFERENT device: its copy of "
            "the same canonical tab must stay playable, not inherit the "
            "local webview's muted serialization",
        )
        markers = await asyncio.to_thread(self._wait_markers, 1)
        self.assertEqual(
            len(markers), 1, "daemon never played the clip natively"
        )
        self.assertEqual(base64.b64decode(markers[0]["audio_b64"]), MP3_BYTES)

    async def test_ready_registers_canonical_background_tabs(self) -> None:
        """ITEM 2: a fresh UDS reconnect covers snapshot-adopted tabs.

        After a webview reload, ``ready`` announces only the active
        placeholder tab; the canonical background tabs the client
        adopts from the ``tabs_state`` snapshot never arrive in
        tab-carrying commands.  A talk event for such a background tab
        must still trigger daemon-native playback (and mute the UDS
        copy) — the webview shows the tab and cannot autoplay.
        """
        active_tab = "active-tab-" + uuid.uuid4().hex[:8]
        bg_tab = "bg-tab-" + uuid.uuid4().hex[:8]
        registry = self.server._vscode_server.tab_registry
        registry.update_tab(active_tab, title="active chat", create=True)
        registry.update_tab(bg_tab, title="background chat", create=True)

        # Fresh reconnect: ready announces ONLY the active tab.
        uds_reader, _uds_writer = await self._connect_uds(active_tab)
        self.server._printer.subscribe_tab(self.task_id, bg_tab)
        await asyncio.sleep(0.1)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-endpoint-2")
        )

        talks = await self._collect_uds_talks(uds_reader, 1)
        self.assertEqual(
            len(talks), 1, "UDS webview never got the background-tab copy"
        )
        self.assertEqual(talks[0]["tabId"], bg_tab)
        markers = await asyncio.to_thread(self._wait_markers, 1)
        self.assertEqual(
            len(markers),
            1,
            "the daemon must play the clip natively for a canonical "
            "background tab shown by the reconnected local webview — "
            "ready must sync the connection's local-tab membership from "
            "the canonical tab registry",
        )
        self.assertEqual(base64.b64decode(markers[0]["audio_b64"]), MP3_BYTES)
        self.assertTrue(
            talks[0].get("muted"),
            "the local webview copy must be muted once the daemon owns "
            "the utterance",
        )

    async def _wait_uds_tabs_state_without(
        self,
        reader: asyncio.StreamReader,
        tab_id: str,
        timeout: float = 5.0,
    ) -> None:
        """Read UDS events until a ``tabs_state`` arrives sans *tab_id*."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            self.assertGreater(
                remaining, 0, "no tabs_state without the closed tab arrived"
            )
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
            self.assertTrue(line, "UDS connection closed unexpectedly")
            msg = json.loads(line.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "tabs_state":
                ids = [t.get("tabId") for t in msg.get("tabs", [])]
                if tab_id not in ids:
                    return

    async def test_closed_canonical_tab_stops_native_playback(self) -> None:
        """ITEM 3a: closing a busy canonical tab prunes muting bookkeeping.

        A closed busy tab keeps its task subscription until the task
        finishes (``_drop_tab_state`` defers teardown), so the still
        running agent keeps emitting events targeted at the closed
        tab id.  Once the ``tabs_state`` broadcast removed the tab
        from every client UI, NO local webview shows it anymore — a
        talk for that tab must therefore no longer trigger
        daemon-native playback, and the copies must stay playable.
        The bookkeeping used to be add-only (pruned only on socket
        disconnect), so the daemon kept playing such talks natively.
        """
        closed_tab = "closed-tab-" + uuid.uuid4().hex[:8]
        registry = self.server._vscode_server.tab_registry
        registry.update_tab(closed_tab, title="busy chat", create=True)

        # The UDS ready adopts the canonical tab into the local-tab
        # bookkeeping (regression 2 above).
        uds_reader, uds_writer = await self._connect_uds("placeholder-tab")
        await asyncio.sleep(0.1)

        # Real close path: registry removal + tabs_state broadcast.
        uds_writer.write(
            (json.dumps({"type": "closeTab", "tabId": closed_tab}) + "\n")
            .encode("utf-8")
        )
        await uds_writer.drain()
        await self._wait_uds_tabs_state_without(uds_reader, closed_tab)

        # The task outlives the close: its subscription to the closed
        # tab id is deliberately retained until the task finishes.
        self.server._printer.subscribe_tab(self.task_id, closed_tab)
        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-endpoint-4")
        )

        talks = await self._collect_uds_talks(uds_reader, 1)
        self.assertEqual(len(talks), 1, "talk copy for the tab never arrived")
        self.assertEqual(talks[0]["tabId"], closed_tab)
        self.assertFalse(
            talks[0].get("muted"),
            "no local webview shows the closed tab, so no copy may be "
            "muted in favor of daemon-native playback",
        )
        await asyncio.sleep(0.5)
        self.assertEqual(
            len(list(self.marker_dir.glob("*.json"))),
            0,
            "the canonical tab was closed on every client, so the daemon "
            "must NOT play the still-running task's talk clip natively — "
            "registry removal must prune the local-UDS tab bookkeeping",
        )

    async def test_repeated_ready_drops_stale_local_tabs(self) -> None:
        """ITEM 3b: a repeated ``ready`` reconciles, not merely adds.

        Pre-populate the connection's local-tab bookkeeping from a
        registry snapshot, shrink the registry behind the connection's
        back, then re-announce ``ready``: the stale id must be dropped
        (membership becomes the current snapshot), so a talk for the
        vanished tab no longer triggers daemon-native playback.
        """
        stale_tab = "stale-tab-" + uuid.uuid4().hex[:8]
        registry = self.server._vscode_server.tab_registry
        registry.update_tab(stale_tab, title="stale chat", create=True)

        # Pre-populate: ready adopts the registry snapshot.
        uds_reader, uds_writer = await self._connect_uds("placeholder-tab")
        await asyncio.sleep(0.1)

        # Shrink the registry WITHOUT the server close path, so only
        # the ready-time reconciliation can heal the bookkeeping.
        self.assertTrue(registry.close_tab(stale_tab))

        # Repeated ready on the SAME connection must drop the stale id.
        uds_writer.write(
            (
                json.dumps({
                    "type": "ready",
                    "tabId": "placeholder-tab",
                    "workDir": self.tmpdir,
                }) + "\n"
            ).encode("utf-8")
        )
        await uds_writer.drain()
        await asyncio.sleep(0.3)

        self.server._printer.subscribe_tab(self.task_id, stale_tab)
        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-endpoint-5")
        )

        talks = await self._collect_uds_talks(uds_reader, 1)
        self.assertEqual(len(talks), 1, "talk copy for the tab never arrived")
        self.assertEqual(talks[0]["tabId"], stale_tab)
        self.assertFalse(
            talks[0].get("muted"),
            "the tab is gone from the registry (and from every client), "
            "so its copy must stay playable",
        )
        await asyncio.sleep(0.5)
        self.assertEqual(
            len(list(self.marker_dir.glob("*.json"))),
            0,
            "a repeated ready must reconcile the connection's local-tab "
            "membership to the current registry snapshot (drop stale "
            "ids), so the daemon must not play the vanished tab's talk",
        )

    async def test_disconnect_unregisters_registry_synced_tabs(self) -> None:
        """Registry-synced local tabs get the SAME disconnect cleanup.

        After the only UDS client disconnects, a talk for a registry
        tab must no longer trigger daemon playback: the ready-time
        registry sync shares the connection's ``local_tabs`` cleanup.
        """
        bg_tab = "bg-tab-" + uuid.uuid4().hex[:8]
        registry = self.server._vscode_server.tab_registry
        registry.update_tab(bg_tab, title="background chat", create=True)
        _reader, writer = await self._connect_uds("placeholder-tab")
        await asyncio.sleep(0.1)
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.2)

        self.server._printer.subscribe_tab(self.task_id, bg_tab)
        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-endpoint-3")
        )
        await asyncio.sleep(0.5)
        self.assertEqual(
            len(list(self.marker_dir.glob("*.json"))),
            0,
            "no UDS webview is connected anymore, so the daemon must "
            "not play the clip on its own speakers",
        )
