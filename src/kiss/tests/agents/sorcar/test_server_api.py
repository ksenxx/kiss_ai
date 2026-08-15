# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Sorcar server API (``kiss.server.sorcar``).

The user interfaces (VS Code extension, remote webapp, CLI) may only
talk to the daemon through the command catalog defined in
``kiss.server.sorcar.API``.  These tests exercise the catalog's
validation logic directly and over a REAL Unix-domain-socket
connection to a live :class:`RemoteAccessServer` dispatcher — the
exact production ``_uds_handler`` code path the VS Code extension
uses — asserting that:

* commands outside the API are rejected with an ``error`` event
  delivered only to the sender (and stamped with the sender's tab id),
* commands missing a required field are rejected the same way, and
* commands inside the API flow through to their handlers (a valid
  ``activeTasksQuery`` gets its ``activeTasksResponse`` reply).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import tempfile
import threading
import unittest
from typing import Any

from kiss.server.sorcar import API, ApiCommand, validate_command
from kiss.server.web_server import RemoteAccessServer


class TestValidateCommand(unittest.TestCase):
    """Branch coverage for :func:`kiss.server.sorcar.validate_command`."""

    def test_non_dict_is_rejected(self) -> None:
        self.assertEqual(
            validate_command(["run"]),
            "Invalid command: expected a JSON object",
        )

    def test_missing_type_is_rejected(self) -> None:
        self.assertEqual(
            validate_command({}), "Invalid command: missing 'type'"
        )

    def test_non_string_type_is_rejected(self) -> None:
        self.assertEqual(
            validate_command({"type": 7}), "Invalid command: missing 'type'"
        )

    def test_unknown_command_is_rejected(self) -> None:
        self.assertEqual(
            validate_command({"type": "bogus"}), "Unknown command: bogus"
        )

    def test_missing_required_field_is_rejected(self) -> None:
        self.assertEqual(
            validate_command({"type": "userAnswer"}),
            "Invalid userAnswer command: missing answer",
        )

    def test_multiple_missing_fields_are_listed(self) -> None:
        self.assertEqual(
            validate_command({"type": "setFavorite"}),
            "Invalid setFavorite command: missing taskId, isFavorite",
        )

    def test_valid_command_passes(self) -> None:
        self.assertIsNone(validate_command({"type": "stop", "tabId": "t"}))
        self.assertIsNone(
            validate_command({"type": "run", "prompt": "hi", "tabId": "t"})
        )

    def test_empty_string_satisfies_required_field(self) -> None:
        self.assertIsNone(validate_command({"type": "auth", "password": ""}))

    def test_out_of_band_commands_are_in_the_catalog(self) -> None:
        self.assertIsNone(validate_command({"type": "getDefaultModel"}))
        self.assertIsNone(validate_command({"type": "readKissConfig"}))
        self.assertIsNone(
            validate_command({"type": "writeKissConfig", "config": {}})
        )
        self.assertIsNone(
            validate_command({"type": "voiceWakeStart", "sensitivity": 50})
        )
        self.assertIsNone(validate_command({"type": "voiceWakeStop"}))

    def test_write_kiss_config_requires_config(self) -> None:
        self.assertEqual(
            validate_command({"type": "writeKissConfig"}),
            "Invalid writeKissConfig command: missing config",
        )

    def test_catalog_covers_every_backend_handler(self) -> None:
        from kiss.server.server import VSCodeServer

        missing = set(VSCodeServer._HANDLERS) - set(API)
        self.assertEqual(missing, set())

    def test_api_command_is_frozen(self) -> None:
        cmd = API["run"]
        self.assertIsInstance(cmd, ApiCommand)
        self.assertEqual(cmd.required, ("prompt",))
        with self.assertRaises(Exception):
            cmd.name = "hacked"  # type: ignore[misc]


class TestServerApiOverUds(unittest.TestCase):
    """The live daemon dispatcher enforces the API over a real UDS."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.sock_path = os.path.join(self.tmp.name, "sorcar-test.sock")
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        self.server._printer._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path
            ),
            self.loop,
        ).result(timeout=5)

    def tearDown(self) -> None:
        async def _shutdown() -> None:
            self.uds_server.close()
            await self.uds_server.wait_closed()

        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)],
            timeout=5,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def _roundtrip(
        self, cmd: dict[str, Any], want_type: str
    ) -> dict[str, Any]:
        """Send *cmd* over a fresh UDS connection; return the first
        received event of type *want_type*."""

        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path
            )
            try:
                writer.write(json.dumps(cmd).encode() + b"\n")
                await writer.drain()
                while True:
                    line = await asyncio.wait_for(
                        reader.readline(), timeout=10
                    )
                    if not line:
                        raise AssertionError(
                            f"connection closed before a {want_type!r} event"
                        )
                    event: dict[str, Any] = json.loads(line)
                    if event.get("type") == want_type:
                        return event
            finally:
                writer.close()
                await writer.wait_closed()

        return asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )

    def test_unknown_command_gets_error_reply(self) -> None:
        event = self._roundtrip(
            {"type": "bogusCommand", "tabId": "tab-x"}, "error"
        )
        self.assertEqual(event["text"], "Unknown command: bogusCommand")
        self.assertEqual(event["tabId"], "tab-x")

    def test_missing_required_field_gets_error_reply(self) -> None:
        event = self._roundtrip({"type": "userAnswer"}, "error")
        self.assertEqual(
            event["text"], "Invalid userAnswer command: missing answer"
        )
        self.assertNotIn("tabId", event)

    def test_non_object_command_is_dropped_connection_survives(self) -> None:
        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path
            )
            try:
                writer.write(b'["not", "an", "object"]\n')
                writer.write(
                    json.dumps({"type": "bogusCommand"}).encode() + b"\n"
                )
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                event: dict[str, Any] = json.loads(line)
                return event
            finally:
                writer.close()
                await writer.wait_closed()

        event = asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )
        self.assertEqual(event.get("type"), "error")
        self.assertEqual(event.get("text"), "Unknown command: bogusCommand")

    def test_valid_api_command_reaches_its_handler(self) -> None:
        event = self._roundtrip(
            {"type": "activeTasksQuery"}, "activeTasksResponse"
        )
        self.assertIn("tabs", event)
        self.assertEqual(event["count"], 0)

    def test_vscode_only_command_is_dropped_silently(self) -> None:
        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path
            )
            try:
                writer.write(json.dumps({"type": "voiceAck"}).encode())
                writer.write(b"\n")
                writer.write(
                    json.dumps({"type": "activeTasksQuery"}).encode() + b"\n"
                )
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                event: dict[str, Any] = json.loads(line)
                return event
            finally:
                writer.close()
                await writer.wait_closed()

        event = asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )
        self.assertEqual(event.get("type"), "activeTasksResponse")

    def test_voice_dropped_frame_is_dropped_silently(self) -> None:
        """A ``voiceDropped`` frame must not draw an error banner.

        ``media/voice.js`` posts ``voiceDropped`` when the user switches
        chat tabs mid-utterance; only the VS Code extension host
        consumes it.  When the webview is served by the daemon the frame
        reaches this dispatcher, so the catalog must know it and drop
        it instead of replying ``Unknown command: voiceDropped``.
        """

        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path
            )
            try:
                writer.write(
                    json.dumps({
                        "type": "voiceDropped",
                        "tabId": "tab-x",
                        "text": "words spoken into a stale tab",
                    }).encode()
                    + b"\n"
                )
                writer.write(
                    json.dumps({"type": "activeTasksQuery"}).encode() + b"\n"
                )
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                event: dict[str, Any] = json.loads(line)
                return event
            finally:
                writer.close()
                await writer.wait_closed()

        event = asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )
        self.assertEqual(event.get("type"), "activeTasksResponse")

    def test_error_reply_goes_only_to_the_sender(self) -> None:
        async def _talk() -> tuple[dict[str, Any], dict[str, Any]]:
            reader_a, writer_a = await asyncio.open_unix_connection(
                self.sock_path
            )
            reader_b, writer_b = await asyncio.open_unix_connection(
                self.sock_path
            )
            try:
                writer_a.write(
                    json.dumps({"type": "bogusCommand"}).encode() + b"\n"
                )
                await writer_a.drain()
                line_a = await asyncio.wait_for(
                    reader_a.readline(), timeout=10
                )
                writer_b.write(
                    json.dumps({"type": "activeTasksQuery"}).encode() + b"\n"
                )
                await writer_b.drain()
                line_b = await asyncio.wait_for(
                    reader_b.readline(), timeout=10
                )
                return json.loads(line_a), json.loads(line_b)
            finally:
                writer_a.close()
                writer_b.close()
                await writer_a.wait_closed()
                await writer_b.wait_closed()

        event_a, event_b = asyncio.run_coroutine_threadsafe(
            _talk(), self.loop
        ).result(timeout=15)
        self.assertEqual(event_a.get("type"), "error")
        self.assertEqual(event_a.get("text"), "Unknown command: bogusCommand")
        self.assertEqual(event_b.get("type"), "activeTasksResponse")

    def test_get_default_model_round_trip(self) -> None:
        event = self._roundtrip({"type": "getDefaultModel"}, "defaultModel")
        self.assertIsInstance(event.get("model"), str)
        self.assertTrue(event["model"])

    def test_read_kiss_config_round_trip(self) -> None:
        from kiss.core.vscode_config import save_config

        save_config({"test_read_sentinel": "sentinel-value"})
        event = self._roundtrip({"type": "readKissConfig"}, "kissConfig")
        config = event.get("config")
        self.assertIsInstance(config, dict)
        assert isinstance(config, dict)
        self.assertEqual(config["test_read_sentinel"], "sentinel-value")
        # The reply carries the daemon's defaults-merged view.
        self.assertIn("max_budget", config)

    def test_write_kiss_config_round_trip(self) -> None:
        from kiss.core.vscode_config import load_config

        event = self._roundtrip(
            {
                "type": "writeKissConfig",
                "config": {"test_write_sentinel": "written-value"},
            },
            "kissConfigSaved",
        )
        self.assertTrue(event.get("ok"))
        self.assertNotIn("error", event)
        self.assertEqual(
            load_config().get("test_write_sentinel"), "written-value"
        )

    def test_write_kiss_config_rejects_non_object_payload(self) -> None:
        event = self._roundtrip(
            {"type": "writeKissConfig", "config": "junk"},
            "kissConfigSaved",
        )
        self.assertFalse(event.get("ok"))
        self.assertEqual(event.get("error"), "config must be a JSON object")

    def test_voice_wake_stop_without_listener_is_a_noop(self) -> None:
        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path
            )
            try:
                writer.write(
                    json.dumps({"type": "voiceWakeStop"}).encode() + b"\n"
                )
                writer.write(
                    json.dumps({"type": "activeTasksQuery"}).encode() + b"\n"
                )
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                event: dict[str, Any] = json.loads(line)
                return event
            finally:
                writer.close()
                await writer.wait_closed()

        event = asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )
        # No error and no voiceWakeState reply: the very next event is
        # the follow-up command's response.
        self.assertEqual(event.get("type"), "activeTasksResponse")

    def test_local_only_commands_are_dropped_for_remote_clients(self) -> None:
        """The UDS-gated handlers must ignore WSS-delivered commands.

        Each handler is invoked through the live server's API with a
        non-UDS context and NO endpoint: were the gate missing, the
        handler's direct reply would dereference the ``None`` endpoint
        and the future would raise.
        """
        from kiss.server.sorcar import ApiContext

        api = self.server._server_api
        ctx = ApiContext(
            endpoint=None,
            conn_state={"work_dir": "", "conn_id": "remote-conn"},
            is_uds=False,
        )
        calls = [
            api.read_kiss_config({"type": "readKissConfig"}, ctx),
            api.write_kiss_config(
                {"type": "writeKissConfig", "config": {"k": "v"}}, ctx
            ),
            api.voice_wake_start({"type": "voiceWakeStart"}, ctx),
            api.voice_wake_stop({"type": "voiceWakeStop"}, ctx),
        ]
        for coro in calls:
            asyncio.run_coroutine_threadsafe(coro, self.loop).result(
                timeout=10
            )
        leaked = self.server._voice_wake.running("remote-conn")
        # Reap a listener BEFORE asserting so a broken gate cannot
        # leak a live mic process past the failing test.
        asyncio.run_coroutine_threadsafe(
            self.server._voice_wake.stop("remote-conn"), self.loop
        ).result(timeout=10)
        self.assertFalse(leaked)


class TestCatalogSync(unittest.TestCase):
    """The handwritten client catalogs must not drift from the API."""

    def test_browser_catalog_is_a_subset_of_the_server_api(self) -> None:
        import re
        from pathlib import Path

        api_js = (
            Path(__file__).resolve().parents[3]
            / "agents"
            / "vscode"
            / "media"
            / "api.js"
        ).read_text()
        match = re.search(
            r"const SORCAR_API_COMMANDS = \[(.*?)\];", api_js, re.DOTALL
        )
        assert match is not None
        names = re.findall(r"'([A-Za-z]+)'", match.group(1))
        self.assertGreater(len(names), 30)
        missing = set(names) - set(API)
        self.assertEqual(
            missing,
            set(),
            f"media/api.js lists commands missing from the API: {missing}",
        )


class TestServerApiCodeBindings(unittest.TestCase):
    """The catalog's handler bindings define the server's code API."""

    def test_every_handler_is_a_server_api_coroutine(self) -> None:
        import inspect

        from kiss.server.sorcar import ServerApi

        for spec in API.values():
            if spec.handler == "drop":
                continue
            method = getattr(ServerApi, spec.handler, None)
            self.assertIsNotNone(
                method, f"{spec.name}: no ServerApi.{spec.handler}"
            )
            assert method is not None
            self.assertTrue(
                inspect.iscoroutinefunction(method),
                f"ServerApi.{spec.handler} must be async",
            )
            params = list(inspect.signature(method).parameters)
            self.assertEqual(
                params,
                ["self", "cmd", "ctx"],
                f"ServerApi.{spec.handler} has non-uniform signature",
            )

    def test_dropped_commands_match_drop_handlers(self) -> None:
        from kiss.server.sorcar import DROPPED_COMMANDS

        self.assertEqual(
            DROPPED_COMMANDS,
            frozenset(c.name for c in API.values() if c.handler == "drop"),
        )
        self.assertEqual(
            DROPPED_COMMANDS,
            frozenset({
                "focusEditor", "webviewFocusChanged", "activeTabChanged",
                "notificationAction", "sizeReport", "resolveDroppedPaths",
                "voiceToggle", "voiceSensitivity", "voiceAck",
                "voiceDropped", "auth",
            }),
        )

    def test_unknown_handler_name_fails_at_construction(self) -> None:
        from kiss.server import sorcar
        from kiss.server.sorcar import ApiCommand, ServerApi

        bogus = ApiCommand("bogusCmd", handler="no_such_method")
        sorcar.API["bogusCmd"] = bogus
        try:
            with self.assertRaises(TypeError):
                ServerApi(object())  # type: ignore[arg-type]
        finally:
            del sorcar.API["bogusCmd"]

    def test_translate_webview_command(self) -> None:
        from kiss.server.sorcar import translate_webview_command

        out = translate_webview_command(
            {"type": "resumeSession", "id": "c1", "tabId": "t"}
        )
        self.assertEqual(
            out, {"type": "resumeSession", "chatId": "c1", "tabId": "t"}
        )
        keep = {"type": "resumeSession", "id": "x", "chatId": "c2"}
        self.assertEqual(translate_webview_command(dict(keep)), keep)
        other = {"type": "stop", "tabId": "t"}
        self.assertEqual(translate_webview_command(dict(other)), other)


if __name__ == "__main__":
    unittest.main()
