# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the settings sheet of a registered machine.

A real local daemon and a real remote daemon (:mod:`kiss.tests.remote_daemon`)
registered as machine ``box``, and a client on the local connection:

* ``getConfig`` / ``saveConfig`` stamped ``machine`` are answered by
  the machine's daemon -- its config, API keys and working directory,
  never this daemon's -- with the reply stamped ``machine`` and
  ``forwarded``, and a save lands in the machine's config;
* ``getMyModels`` / ``saveMyModel`` / ``deleteMyModel`` stamped
  ``machine`` read and write the machine's custom models, which this
  daemon's own list never shows;
* an unknown machine answers with an ``error`` reply of the right type.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.sync.client import connect as ws_connect

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon


def _as_another_client(url: str, commands: list[dict[str, Any]], reply_type: str) -> None:
    """Send *commands* to the daemon at *url* as a password-authenticated client, each answered."""
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with ws_connect(mach.ws_url(url), ssl=ctx, open_timeout=30) as ws:
        ws.send(json.dumps({"type": "auth", "password": "pw-remote"}))
        assert json.loads(ws.recv(timeout=30))["type"] == "auth_ok"
        for cmd in commands:
            ws.send(json.dumps({**cmd, "password": "pw-remote"}))
            for _ in range(50):
                if json.loads(ws.recv(timeout=30)).get("type") == reply_type:
                    break


def _save_models_as_another_client(url: str, names: list[str]) -> None:
    """Add custom models *names* on the daemon at *url* through another client."""
    _as_another_client(
        url, [{"type": "saveMyModel", "name": name} for name in names], "myModelsData",
    )


class RemoteMachineCase(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon registered as ``box``, a client on the local socket."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        root = Path(self.tmpdir)
        self.box_work = root / "box-work"
        self.box_work.mkdir()
        self.home_work = root / "home-work"
        self.home_work.mkdir()
        self.remote = await asyncio.to_thread(
            start_remote_daemon, root / "remote-home", password="pw-remote",
            config={"work_dir": str(self.box_work), "max_budget": 11.0},
        )
        certfile = root / "cert.pem"
        keyfile = root / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=root / "remote-url.json",
            local_endpoint_file=root / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)
        await self._ask({"type": "setWorkDir", "workDir": str(self.home_work)}, "workDirChanged")
        added = await self._ask({
            "type": "machineAction", "action": "add", "name": "box",
            "address": self.remote.url, "machinePassword": "pw-remote",
        }, "machines")
        self.assertEqual(added["error"], "")

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        self.remote.stop()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _ask(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* message."""
        await self._send(cmd)
        return await self._next(reply_type)

    async def _next(self, reply_type: str, forwarded: bool | None = None) -> dict[str, Any]:
        """The next *reply_type* message (with the given ``forwarded`` stamp, when asked)."""
        for _ in range(300):
            line = await asyncio.wait_for(self.reader.readline(), timeout=60.0)
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") != reply_type:
                continue
            if forwarded is None or bool(msg.get("forwarded")) == forwarded:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def _ask_box(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        """Send *cmd* for ``box`` and return its forwarded *reply_type* reply."""
        await self._send({"machine": "box", **cmd})
        return await self._next(reply_type, forwarded=True)


class TestRemoteMachineSettings(RemoteMachineCase):
    """The settings sheet's commands stamped ``machine`` are the machine's."""

    async def test_get_and_save_config_are_the_machines(self) -> None:
        reply = await self._ask_box({"type": "getConfig"}, "configData")
        self.assertEqual(reply.get("error", ""), "")
        self.assertEqual(reply["machine"], "box")
        self.assertIs(reply["forwarded"], True)
        self.assertEqual(reply["config"]["work_dir"], str(self.box_work))
        self.assertEqual(reply["config"]["max_budget"], 11.0)
        self.assertEqual(reply["config"]["remote_password"], "pw-remote")
        self.assertIn("apiKeys", reply)

        saved = await self._ask_box(
            {"type": "saveConfig", "config": {"max_budget": 77.0}}, "configData",
        )
        self.assertEqual(saved["machine"], "box")
        self.assertIs(saved["forwarded"], True)
        self.assertEqual(saved["config"]["max_budget"], 77.0)
        # The save landed on the machine, not here.
        again = await self._ask_box({"type": "getConfig"}, "configData")
        self.assertEqual(again["config"]["max_budget"], 77.0)
        mine = await self._ask({"type": "getConfig"}, "configData")
        self.assertNotIn("forwarded", mine)
        self.assertEqual(mine["config"]["work_dir"], str(self.home_work))
        self.assertNotEqual(mine["config"].get("max_budget"), 77.0)

    async def test_custom_models_are_the_machines(self) -> None:
        empty = await self._ask_box({"type": "getMyModels"}, "myModelsData")
        self.assertEqual(empty["machine"], "box")
        self.assertEqual(empty["models"], [])
        saved = await self._ask_box(
            {"type": "saveMyModel", "name": "box-model", "endpoint": "http://box:1/v1"},
            "myModelsData",
        )
        self.assertIs(saved["forwarded"], True)
        self.assertEqual([m["name"] for m in saved["models"]], ["box-model"])
        mine = await self._ask({"type": "getMyModels"}, "myModelsData")
        self.assertNotIn("forwarded", mine)
        self.assertNotIn("box-model", [m["name"] for m in mine["models"]])
        gone = await self._ask_box(
            {"type": "deleteMyModel", "name": "box-model"}, "myModelsData",
        )
        self.assertEqual(gone["models"], [])

    async def test_a_refused_model_save_is_an_error_reply_not_a_timeout(self) -> None:
        """The box answers a rejected name with an ``error`` frame: the
        link returns it as the reply (no 8 s wait, no back-off), and the
        box still answers the next request."""
        started = asyncio.get_running_loop().time()
        refused = await self._ask_box(
            {"type": "saveMyModel", "name": "_reserved"}, "myModelsData",
        )
        self.assertTrue(refused["error"], refused)
        self.assertNotIn("did not answer", refused["error"])
        self.assertLess(asyncio.get_running_loop().time() - started, 5.0)
        again = await self._ask_box({"type": "getMyModels"}, "myModelsData")
        self.assertEqual(again.get("error", ""), "")
        self.assertEqual(again["models"], [])

    async def test_broadcasts_queued_on_the_link_do_not_answer_a_later_request(self) -> None:
        """Edits by another client of the box push ``myModelsData`` to
        every connection, the link's included: a later forwarded request
        gets the box's current list, not a queued one."""
        # Open the link (and its socket) first, then edit on the box
        # through another client while the link sits idle.
        await self._ask_box({"type": "getMyModels"}, "myModelsData")
        await asyncio.to_thread(_save_models_as_another_client, self.remote.url, ["one", "two"])
        listed = await self._ask_box({"type": "getMyModels"}, "myModelsData")
        self.assertEqual(sorted(m["name"] for m in listed["models"]), ["one", "two"])
        saved = await self._ask_box({"type": "saveMyModel", "name": "three"}, "myModelsData")
        self.assertEqual(sorted(m["name"] for m in saved["models"]), ["one", "three", "two"])

    async def test_a_changed_remote_password_re_registers_the_machine(self) -> None:
        saved = await self._ask_box(
            {"type": "saveConfig", "config": {"remote_password": "pw-new"}}, "configData",
        )
        self.assertEqual(saved["config"]["remote_password"], "pw-new")
        (entry,) = [m for m in mach.load_machines() if m.name == "box"]
        self.assertEqual(entry.password, "pw-new")
        # The next request authenticates with the new password.
        again = await self._ask_box({"type": "getConfig"}, "configData")
        self.assertEqual(again.get("error", ""), "")
        self.assertEqual(again["config"]["remote_password"], "pw-new")

    async def test_a_password_changed_behind_the_link_is_a_rejection_not_a_reply(self) -> None:
        """The box's password changes through another client (not the
        sheet): the link's next frame is refused, and that refusal is an
        error ("rejected the password"), never an empty success."""
        await self._ask_box({"type": "getConfig"}, "configData")
        await asyncio.to_thread(
            _as_another_client, self.remote.url,
            [{"type": "saveConfig", "config": {"remote_password": "pw-other"}}], "configData",
        )
        refused = await self._ask_box({"type": "getConfig"}, "configData")
        self.assertIn("rejected the password", refused["error"])
        self.assertNotIn("config", refused)

    async def test_unknown_machine_answers_with_an_error_of_the_right_type(self) -> None:
        await self._send({"type": "getConfig", "machine": "nowhere"})
        reply = await self._next("configData", forwarded=True)
        self.assertEqual(reply["machine"], "nowhere")
        self.assertIn("no machine named 'nowhere'", reply["error"])
        await self._send({"type": "getMyModels", "machine": "nowhere"})
        models = await self._next("myModelsData", forwarded=True)
        self.assertIn("no machine named 'nowhere'", models["error"])
