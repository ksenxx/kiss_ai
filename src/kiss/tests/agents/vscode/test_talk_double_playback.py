# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: one ``talk`` playback per device (CLI + webview overlap).

Regression tests for the distorted / overlapping speech reintroduced
by the "cli: play talk tool audio on terminal speakers" feature: the
sorcar CLI plays every ``talk`` event on the terminal machine's
speakers, and the daemon then relays the SAME event to webview /
REPL clients on the SAME machine (all UDS peers are local), so one
utterance was played twice slightly offset — audibly distorted.

The daemon now arbitrates playback per device:

* CLI-originated talk events (``cliEvent`` envelope) were already
  played by the CLI process, so every UDS (same-machine) copy is
  stamped ``muted: true`` while WSS (remote-device) copies stay
  playable;
* daemon-hosted (REPL) talk events mute the copies stamped for CLI
  tabs (announced via ``cliTabHello``) whenever a webview tab is also
  subscribed, and let exactly one CLI tab play otherwise;
* ``cli_talk.TalkPlayer`` and ``media/main.js`` drop muted copies.

These tests run a REAL ``RemoteAccessServer`` with a UDS listener and
REAL UDS client connections (no mocks), mirroring
``test_web_server_uds.py``; the terminal player test uses a REAL
child process substituted via ``KISS_SORCAR_PLAY_CMD``.
"""

from __future__ import annotations

import asyncio
import json
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


def _talk_event(task_id: str, talk_id: str) -> dict[str, object]:
    """Build a ``talk`` broadcast event like the ``talk`` tool emits."""
    return {
        "type": "talk",
        "taskId": task_id,
        "talkId": talk_id,
        "text": "hello from the agent",
        "language": "en-US",
        "emotion": "warm",
    }


class TestTalkPlaybackArbitration(IsolatedAsyncioTestCase):
    """One playback per device across CLI terminal and webview clients."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
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
        """Read events until *count* ``talk`` copies arrive (or timeout).

        The daemon fans every tab-stamped copy out to every endpoint
        (clients filter by ``tabId``), so ONE connection observes the
        copies for every subscribed tab.
        """
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

    async def _no_more_talks(
        self, reader: asyncio.StreamReader, quiet: float = 0.3
    ) -> bool:
        """Return True when no further talk copy arrives within *quiet*."""
        extra = await self._collect_talks(reader, 1, timeout=quiet)
        return not extra

    async def test_cli_origin_talk_muted_for_all_uds_peers(self) -> None:
        """THE REGRESSION: a CLI-played talk must not replay locally.

        A CLI-launched task plays the clip on the terminal speakers
        and forwards the event via a ``cliEvent`` envelope; a local
        webview subscribed to the same task received a playable copy
        and played it again.  Post-fix every UDS copy is muted.
        """
        web_tab = "webtab-" + uuid.uuid4().hex[:8]
        web_reader, web_writer = await self._connect(web_tab)
        self.server._printer.subscribe_tab(self.task_id, web_tab)

        cli_reader, cli_writer = await self._connect(
            "clitab-" + uuid.uuid4().hex[:8], cli=True
        )
        envelope = {
            "type": "cliEvent",
            "event": _talk_event(self.task_id, "talk-cli-origin"),
        }
        cli_writer.write((json.dumps(envelope) + "\n").encode("utf-8"))
        await cli_writer.drain()

        talks = await self._collect_talks(web_reader, 1)
        self.assertEqual(len(talks), 1, "webview never got the talk copy")
        self.assertEqual(talks[0]["tabId"], web_tab)
        self.assertTrue(
            talks[0].get("muted"),
            "UDS webview copy of a CLI-played talk must be muted — an "
            "unmuted copy replays the clip on the same machine "
            "(distorted, overlapping speech)",
        )
        for w in (web_writer, cli_writer):
            w.close()

    async def test_repl_talk_muted_for_cli_tab_when_webview_subscribed(
        self,
    ) -> None:
        """Daemon-hosted talk: the webview plays, the CLI tab is muted."""
        web_tab = "webtab-" + uuid.uuid4().hex[:8]
        cli_tab = "clitab-" + uuid.uuid4().hex[:8]
        web_reader, web_writer = await self._connect(web_tab)
        cli_reader, cli_writer = await self._connect(cli_tab, cli=True)
        self.server._printer.subscribe_tab(self.task_id, web_tab)
        self.server._printer.subscribe_tab(self.task_id, cli_tab)
        await asyncio.sleep(0.05)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-repl-1")
        )

        talks = await self._collect_talks(web_reader, 2)
        by_tab = {t["tabId"]: t for t in talks}
        self.assertIn(web_tab, by_tab)
        self.assertIn(cli_tab, by_tab)
        self.assertFalse(
            by_tab[web_tab].get("muted"),
            "the webview tab's copy must stay playable",
        )
        self.assertTrue(
            by_tab[cli_tab].get("muted"),
            "the CLI REPL tab's copy must be muted while a webview tab "
            "is subscribed — both play on the same machine otherwise",
        )
        for w in (web_writer, cli_writer):
            w.close()

    async def test_remote_web_tab_does_not_mute_local_cli(self) -> None:
        """A WSS/remote web tab is another device; local CLI still plays."""
        remote_web_tab = "remote-webtab-" + uuid.uuid4().hex[:8]
        cli_tab = "clitab-" + uuid.uuid4().hex[:8]
        cli_reader, cli_writer = await self._connect(cli_tab, cli=True)
        # Deliberately subscribe the web tab WITHOUT a UDS connection:
        # this models a remote WSS browser tab.  It should receive a
        # playable copy on that remote device, but it must not suppress
        # the local terminal player's copy.
        self.server._printer.subscribe_tab(self.task_id, remote_web_tab)
        self.server._printer.subscribe_tab(self.task_id, cli_tab)
        await asyncio.sleep(0.05)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-remote-web-plus-cli")
        )

        talks = await self._collect_talks(cli_reader, 2)
        by_tab = {t["tabId"]: t for t in talks}
        self.assertIn(remote_web_tab, by_tab)
        self.assertIn(cli_tab, by_tab)
        self.assertFalse(
            by_tab[remote_web_tab].get("muted"),
            "remote web tab's copy must stay playable on the remote device",
        )
        self.assertFalse(
            by_tab[cli_tab].get("muted"),
            "a remote WSS tab must not suppress local CLI playback",
        )
        cli_writer.close()

    async def test_repl_talk_plays_on_sole_cli_tab(self) -> None:
        """A pure CLI REPL session (no webview) still hears the talk."""
        cli_tab = "clitab-" + uuid.uuid4().hex[:8]
        cli_reader, cli_writer = await self._connect(cli_tab, cli=True)
        self.server._printer.subscribe_tab(self.task_id, cli_tab)
        await asyncio.sleep(0.05)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-repl-solo")
        )

        talks = await self._collect_talks(cli_reader, 1)
        self.assertEqual(len(talks), 1)
        self.assertEqual(talks[0]["tabId"], cli_tab)
        self.assertFalse(
            talks[0].get("muted"),
            "with no webview subscribed the CLI tab must play the talk",
        )
        cli_writer.close()

    async def test_two_cli_tabs_exactly_one_plays(self) -> None:
        """Two REPLs on one task: exactly one terminal speaks."""
        tab_a = "clitab-a-" + uuid.uuid4().hex[:8]
        tab_b = "clitab-b-" + uuid.uuid4().hex[:8]
        reader_a, writer_a = await self._connect(tab_a, cli=True)
        reader_b, writer_b = await self._connect(tab_b, cli=True)
        self.server._printer.subscribe_tab(self.task_id, tab_a)
        self.server._printer.subscribe_tab(self.task_id, tab_b)
        await asyncio.sleep(0.05)

        self.server._printer.broadcast(
            _talk_event(self.task_id, "talk-two-repls")
        )

        talks = await self._collect_talks(reader_a, 2)
        self.assertEqual(len(talks), 2)
        unmuted = [t for t in talks if not t.get("muted")]
        self.assertEqual(
            len(unmuted),
            1,
            "exactly one CLI tab must play when only CLI tabs are "
            f"subscribed, got: {talks}",
        )
        for w in (writer_a, writer_b):
            w.close()

    async def test_non_talk_cli_events_relay_unchanged(self) -> None:
        """Ordinary CLI events keep flowing to webview tabs unmuted."""
        web_tab = "webtab-" + uuid.uuid4().hex[:8]
        web_reader, web_writer = await self._connect(web_tab)
        self.server._printer.subscribe_tab(self.task_id, web_tab)
        cli_reader, cli_writer = await self._connect(
            "clitab-" + uuid.uuid4().hex[:8], cli=True
        )
        envelope = {
            "type": "cliEvent",
            "event": {
                "type": "message",
                "taskId": self.task_id,
                "text": "plain event",
            },
        }
        cli_writer.write((json.dumps(envelope) + "\n").encode("utf-8"))
        await cli_writer.drain()
        deadline = asyncio.get_event_loop().time() + 5.0
        got = None
        while asyncio.get_event_loop().time() < deadline:
            line = await asyncio.wait_for(web_reader.readline(), timeout=5.0)
            if not line:
                break
            msg = json.loads(line.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "message":
                got = msg
                break
        assert got is not None, "relayed CLI event never reached the webview"
        self.assertEqual(got.get("text"), "plain event")
        self.assertNotIn("muted", got)
        for w in (web_writer, cli_writer):
            w.close()


class TestTalkPlayerHonoursMuted(IsolatedAsyncioTestCase):
    """The terminal player must drop daemon-muted talk copies."""

    async def test_muted_event_plays_nothing(self) -> None:
        """A ``muted: true`` copy triggers neither player nor TTS."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            marker_dir = tmpdir / "markers"
            marker_dir.mkdir()
            script = tmpdir / "fake_player.py"
            script.write_text(
                "import os, sys, time\n"
                f"open(os.path.join({str(marker_dir)!r}, "
                "f'play-{time.time_ns()}.marker'), 'w').write(sys.argv[-1])\n"
            )
            cmd = " ".join(
                shlex.quote(p) for p in (sys.executable, str(script))
            )
            import os

            os.environ["KISS_SORCAR_PLAY_CMD"] = cmd
            os.environ["KISS_SORCAR_SAY_CMD"] = cmd
            try:
                cli_talk.reset_shared_player_for_tests()
                player = cli_talk.shared_player()
                player.play(
                    {
                        "type": "talk",
                        "talkId": "muted-1",
                        "text": "should stay silent",
                        "muted": True,
                    }
                )
                time.sleep(0.5)
                self.assertEqual(
                    list(marker_dir.glob("play-*.marker")),
                    [],
                    "a muted talk copy must never reach the player/TTS",
                )
                # The muted copy must not poison the dedupe set: a
                # later playable copy of the same talkId still plays.
                player.play(
                    {
                        "type": "talk",
                        "talkId": "muted-1",
                        "text": "now audible",
                    }
                )
                deadline = time.time() + 5.0
                while (
                    not list(marker_dir.glob("play-*.marker"))
                    and time.time() < deadline
                ):
                    time.sleep(0.05)
                self.assertEqual(
                    len(list(marker_dir.glob("play-*.marker"))),
                    1,
                    "the unmuted copy of the talk must still play",
                )
            finally:
                os.environ.pop("KISS_SORCAR_PLAY_CMD", None)
                os.environ.pop("KISS_SORCAR_SAY_CMD", None)
                cli_talk.reset_shared_player_for_tests()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
