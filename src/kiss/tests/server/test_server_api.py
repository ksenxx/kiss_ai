# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Sorcar server API (``kiss.server.sorcar``).

The user interfaces (the chat page, CLI) may only
talk to the daemon through the command catalog defined in
``kiss.server.sorcar.API``.  These tests exercise the catalog's
validation logic directly and over a REAL local WebSocket connection
to a live :class:`RemoteAccessServer` dispatcher — the exact production
``_ws_handler`` code path every client uses — asserting that:

* commands outside the API are rejected with an ``error`` event
  delivered only to the sender (and stamped with the sender's tab id),
* commands missing a required field are rejected the same way, and
* commands inside the API flow through to their handlers (a valid
  ``activeTasksQuery`` gets its ``activeTasksResponse`` reply).

Moved from ``kiss.tests.webapp.test_server_api`` because the
dependency closure of every test here touches only kiss.core,
kiss.agents.sorcar and kiss.server; the ``media/api.js`` catalog-drift
check stayed behind in tests/webapp.
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
from kiss.tests.local_ws import open_local_connection


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

    def test_catalog_covers_every_backend_handler(self) -> None:
        from kiss.server.server import AgentServer

        missing = set(AgentServer._HANDLERS) - set(API)
        self.assertEqual(missing, set())

    def test_api_command_is_frozen(self) -> None:
        cmd = API["run"]
        self.assertIsInstance(cmd, ApiCommand)
        self.assertEqual(cmd.required, ("prompt",))
        with self.assertRaises(Exception):
            cmd.name = "hacked"  # type: ignore[misc]


class TestServerApiOverLocalConnection(unittest.TestCase):
    """The live daemon dispatcher enforces the API over a real local connection."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            local_endpoint_file=os.path.join(self.tmp.name, "sorcar-local.json"),
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        # Loopback WSS on an ephemeral port: the production ``_ws_handler``
        # without the public daemon's tunnel, URL file and watchdogs.
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(), self.loop,
        ).result(timeout=30)

    def tearDown(self) -> None:
        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop)],
            timeout=30,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def _roundtrip(
        self, cmd: dict[str, Any], want_type: str
    ) -> dict[str, Any]:
        """Send *cmd* over a fresh local connection; return the first
        received event of type *want_type*."""

        async def _talk() -> dict[str, Any]:
            reader, writer = await open_local_connection(self.server)
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
            reader, writer = await open_local_connection(self.server)
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

    def test_repeated_auth_frame_is_discarded_silently(self) -> None:
        """An ``auth`` frame after the handshake draws no error banner.

        A client re-sending its queued batch after a reconnect can
        replay its ``auth`` frame; the catalog routes it to the no-op
        ``ServerApi.auth`` instead of replying ``Unknown command``.
        """

        async def _talk() -> dict[str, Any]:
            reader, writer = await open_local_connection(self.server)
            try:
                writer.write(
                    json.dumps({"type": "auth", "password": "x"}).encode() + b"\n"
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
        self.assertEqual(event.get("type"), "activeTasksResponse", event)

    def test_error_reply_goes_only_to_the_sender(self) -> None:
        async def _talk() -> tuple[dict[str, Any], dict[str, Any]]:
            reader_a, writer_a = await open_local_connection(self.server)
            reader_b, writer_b = await open_local_connection(self.server)
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


class TestServerApiCodeBindings(unittest.TestCase):
    """The catalog's handler bindings define the server's code API."""

    def test_every_handler_is_a_server_api_coroutine(self) -> None:
        import inspect

        from kiss.server.sorcar import ServerApi

        for spec in API.values():
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

    def test_translate_client_command(self) -> None:
        from kiss.server.sorcar import translate_client_command

        out = translate_client_command(
            {"type": "resumeSession", "id": "c1", "tabId": "t"}
        )
        self.assertEqual(
            out, {"type": "resumeSession", "chatId": "c1", "tabId": "t"}
        )
        keep = {"type": "resumeSession", "id": "x", "chatId": "c2"}
        self.assertEqual(translate_client_command(dict(keep)), keep)
        other = {"type": "stop", "tabId": "t"}
        self.assertEqual(translate_client_command(dict(other)), other)


if __name__ == "__main__":
    unittest.main()
