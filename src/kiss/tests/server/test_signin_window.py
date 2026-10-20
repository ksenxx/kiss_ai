# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the sign-in window's daemon side (:mod:`kiss.server.signin`).

A brand's executable ``$KISS_HOME/signin`` answers ``status`` and runs
``login <provider>``; the daemon turns that into the ``signin`` event of
a ``ready``, and the ``signinLogin`` / ``signinPaste`` / ``signinCancel``
commands into ``signinUrl``, ``signinDone`` and ``signinError`` events.

The daemon is real (local transport); the hook is a stub shell script
whose ``status`` answer is a file the tests write, and whose ``login``
prints a page URL, then waits for one pasted line on stdin.  The
``signin`` event is the tail of a ``ready``, sent by a background task
once the tabs are restored (see ``test_audit1020_c12_signin_ready``);
:class:`SigninHookCase` is the shared harness.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path

from kiss.server import signin
from kiss.server.signin import SIGNIN_HOOK
from kiss.tests.local_ws import LocalReader, LocalWriter
from kiss.tests.server.test_update_available_check import _UpdateCheckTestBase

HOOK = """#!/bin/bash
echo "$KISS_HOME" > "$KISS_HOME/hook-home"
case "$1" in
  status)
    echo run >> "$KISS_HOME/status-runs"
    [ -f "$KISS_HOME/status-sleep" ] && sleep "$(cat "$KISS_HOME/status-sleep")"
    cat "$KISS_HOME/status.json"
    exit "$(cat "$KISS_HOME/status-exit" 2>/dev/null || echo 0)"
    ;;
  login)
    case "$2" in
      nourl) echo '{"error": "unknown provider"}'; exit 1 ;;
      silent) exit 3 ;;
      crash) echo '{"url": "https://accounts.example/crash"}'; exit 4 ;;
      deaf)
        echo '{"url": "https://accounts.example/deaf"}'
        exec 0<&-; touch "$KISS_HOME/deaf-closed"; sleep 5; exit 5 ;;
      stubborn) echo '{"url": "https://accounts.example/stubborn"}'; trap "" TERM; exec sleep 30 ;;
      slow) sleep 0.5 ;;
    esac
    echo "{\\"url\\": \\"https://accounts.example/auth?p=$2\\"}"
    read -r line || exit 2
    echo "$line" > "$KISS_HOME/pasted"
    echo
    if [ "$line" = "bad" ]; then echo '{"error": "bad code"}'; exit 1; fi
    echo '{"logged_in": true, "email": "me@example.com"}'
    ;;
esac
"""

NOT_LOGGED_IN = {
    "logged_in": False,
    "title": "Sign in to the s10s hub",
    "providers": [
        {"id": "google", "label": "Login with Google"},
        {"id": "apple", "label": "Login with Apple"},
    ],
}


@unittest.skipIf(sys.platform == "win32", "the stub hook is a bash script")
class SigninHookCase(_UpdateCheckTestBase):
    """A real daemon whose ``$KISS_HOME`` holds the stub sign-in hook.

    The status cache is off (``STATUS_CACHE_TTL = 0``) so a test can
    change the hook's answer between two readies; the cache's own
    tests set the TTL they need.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.home = Path(self.tmpdir) / "kiss-home"
        self.home.mkdir()
        self._saved_home = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(self.home)
        self._saved_ttl = signin.STATUS_CACHE_TTL
        signin.STATUS_CACHE_TTL = 0.0
        signin.forget_status()
        self._writers: list[LocalWriter] = []
        self._readers_writers: dict[LocalReader, LocalWriter] = {}

    async def asyncTearDown(self) -> None:
        signin.STATUS_CACHE_TTL = self._saved_ttl
        signin.forget_status()
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        if self._saved_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_home
        await super().asyncTearDown()

    def _install_hook(self, status: dict[str, object] | str = NOT_LOGGED_IN) -> Path:
        hook = self.home / SIGNIN_HOOK
        hook.write_text(HOOK)
        hook.chmod(0o755)
        self._set_status(status)
        return hook

    def _set_status(self, status: dict[str, object] | str) -> None:
        text = status if isinstance(status, str) else json.dumps(status)
        (self.home / "status.json").write_text(text + "\n")

    async def _connect(self) -> LocalReader:
        """A new local connection's reader (its writer is kept for :meth:`_send`)."""
        reader, writer = await self._connect_local()
        self._writers.append(writer)
        self._readers_writers[reader] = writer
        return reader

    async def _ready(self, tab_id: str) -> tuple[LocalReader, list[dict[str, object]]]:
        """Connect, send ``ready``, return the reader and every event it produced.

        The events up to ``tabs_state``, then the sign-in events of the
        ``ready``'s tail (:meth:`_settle`), in arrival order.
        """
        reader = await self._connect()
        await self._send_ready(self._readers_writers[reader], tab_id)
        events = await self._read_until(reader, "tabs_state")
        return reader, events + await self._settle(reader)

    async def _read_until(self, reader: LocalReader, kind: str) -> list[dict[str, object]]:
        """The events on *reader* up to and including the first of type *kind*."""
        events: list[dict[str, object]] = []
        for _ in range(200):
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            self.assertTrue(line, f"local connection closed before {kind}")
            msg = json.loads(line.decode("utf-8"))
            events.append(msg)
            if msg.get("type") == kind:
                return events
        raise AssertionError(f"did not see {kind}")

    async def _settle(self, reader: LocalReader) -> list[dict[str, object]]:
        """The sign-in events the ``ready`` sent on *reader* produces, once its tail is done.

        A ping's answer proves the ``ready`` command itself has
        finished (commands run one at a time per connection), so its
        tail task exists; the daemon's background replies are then
        waited out, and a second ping flushes whatever the tail queued.
        """
        pending = await self._pending_signin_events(reader)
        replies = self.server._server_api._background_replies
        for _ in range(500):
            if not replies:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(replies, set(), "the ready's tail task never finished")
        return pending + await self._pending_signin_events(reader)

    async def _send(self, reader: LocalReader, cmd: dict[str, object]) -> None:
        writer = self._readers_writers[reader]
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _pending_signin_events(self, reader: LocalReader) -> list[dict[str, object]]:
        """Send a ping; the sign-in events that arrive before its answer."""
        await self._send(reader, {"type": "ping"})
        pending: list[dict[str, object]] = []
        for _ in range(50):
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            msg = json.loads(line.decode("utf-8"))
            kind = str(msg.get("type"))
            if kind == "pong":
                return pending
            if kind.startswith("signin"):
                pending.append(msg)
        raise AssertionError("no pong")

    @staticmethod
    def _signin_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
        return [ev for ev in events if ev.get("type") == "signin"]

    def _status_runs(self) -> int:
        """How many times the stub hook has answered ``status``."""
        runs = self.home / "status-runs"
        return len(runs.read_text().splitlines()) if runs.exists() else 0

    async def _wait_for_file(self, path: Path) -> None:
        for _ in range(500):
            if path.exists():
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"{path.name} never appeared")


class TestSigninWindow(SigninHookCase):
    """The ``signin`` event of a ``ready`` and the sign-in commands."""

    # ---- the ``signin`` event ----

    async def test_no_hook_sends_no_signin(self) -> None:
        _reader, events = await self._ready("tab-1")
        self.assertEqual(self._signin_events(events), [])
        self.assertIn("tipsData", [ev.get("type") for ev in events])

    async def test_hook_not_logged_in_sends_signin_with_the_providers(self) -> None:
        self._install_hook()
        _reader, events = await self._ready("tab-1")
        signin = self._signin_events(events)
        self.assertEqual(len(signin), 1)
        self.assertEqual(signin[0]["title"], "Sign in to the s10s hub")
        self.assertEqual(signin[0]["providers"], NOT_LOGGED_IN["providers"])
        # The hook ran with the daemon's KISS_HOME.
        self.assertEqual((self.home / "hook-home").read_text().strip(), str(self.home))
        # Every (re)connecting page gets it while the user is logged out...
        _reader, events = await self._ready("tab-2")
        self.assertEqual(len(self._signin_events(events)), 1)
        # ...and none once the hook says logged in.
        self._set_status({"logged_in": True, "email": "me@example.com"})
        _reader, events = await self._ready("tab-3")
        self.assertEqual(self._signin_events(events), [])

    async def test_signin_event_normalizes_the_providers(self) -> None:
        self._install_hook({
            "logged_in": False,
            "providers": [{"id": "google"}, {"label": "no id"}, "junk", {"id": "", "label": "x"}],
        })
        _reader, events = await self._ready("tab-1")
        signin = self._signin_events(events)[0]
        self.assertEqual(signin["title"], "Sign in")
        self.assertEqual(signin["providers"], [{"id": "google", "label": "google"}])
        # No provider to click: nothing to show.
        self._set_status({"logged_in": False, "providers": []})
        _reader, events = await self._ready("tab-2")
        self.assertEqual(self._signin_events(events), [])
        # Providers that are not a list: nothing to show, and the page still boots.
        self._set_status({"logged_in": False, "providers": 1})
        _reader, events = await self._ready("tab-3")
        self.assertEqual(self._signin_events(events), [])
        self.assertIn("tipsData", [ev.get("type") for ev in events])

    async def test_broken_hook_sends_no_signin(self) -> None:
        hook = self._install_hook("this is not json")
        _reader, events = await self._ready("tab-1")
        self.assertEqual(self._signin_events(events), [])
        self._set_status(NOT_LOGGED_IN)
        (self.home / "status-exit").write_text("1\n")
        _reader, events = await self._ready("tab-2")
        self.assertEqual(self._signin_events(events), [])
        (self.home / "status-exit").unlink()
        hook.chmod(0o644)
        _reader, events = await self._ready("tab-3")
        self.assertEqual(self._signin_events(events), [])
        hook.chmod(0o755)
        _reader, events = await self._ready("tab-4")
        self.assertEqual(len(self._signin_events(events)), 1)
        # JSON that is not an object, and a hook slower than STATUS_TIMEOUT.
        self._set_status("[1, 2]")
        _reader, events = await self._ready("tab-5")
        self.assertEqual(self._signin_events(events), [])
        self._set_status(NOT_LOGGED_IN)
        (self.home / "status-sleep").write_text("5\n")
        saved_timeout = signin.STATUS_TIMEOUT
        signin.STATUS_TIMEOUT = 0.5
        try:
            _reader, events = await self._ready("tab-6")
        finally:
            signin.STATUS_TIMEOUT = saved_timeout
        self.assertEqual(self._signin_events(events), [])
        (self.home / "status-sleep").unlink()
        # A hook the OS cannot run (a bogus interpreter) is no hook.
        hook.write_text("#!/nonexistent/interpreter\n")
        _reader, events = await self._ready("tab-7")
        self.assertEqual(self._signin_events(events), [])

    # ---- the sign-in commands ----

    async def test_login_url_to_the_clicker_and_done_to_every_window(self) -> None:
        self._install_hook()
        reader_a, _ = await self._ready("tab-a")
        reader_b, _ = await self._ready("tab-b")
        await self._send(reader_a, {"type": "signinLogin", "provider": "google"})
        url = await self._wait_for_event(reader_a, "signinUrl")
        self.assertEqual(url["provider"], "google")
        self.assertEqual(url["url"], "https://accounts.example/auth?p=google")
        # The paste reaches the hook's stdin and ends the sign-in.
        await self._send(reader_a, {"type": "signinPaste", "text": " http://127.0.0.1:1/?code=1 "})
        done_a = await self._wait_for_event(reader_a, "signinDone")
        self.assertEqual(done_a["email"], "me@example.com")
        events_b: list[dict[str, object]] = []
        while not events_b or events_b[-1].get("type") != "signinDone":
            line = await asyncio.wait_for(reader_b.readline(), timeout=5.0)
            events_b.append(json.loads(line.decode("utf-8")))
        self.assertEqual(events_b[-1]["email"], "me@example.com")
        self.assertNotIn("signinUrl", [ev.get("type") for ev in events_b])
        self.assertEqual((self.home / "pasted").read_text(), "http://127.0.0.1:1/?code=1\n")

    async def test_hook_failures_become_signin_errors(self) -> None:
        self._install_hook()
        reader, _ = await self._ready("tab-a")
        # A paste with no sign-in running.
        await self._send(reader, {"type": "signinPaste", "text": "x"})
        error = await self._wait_for_event(reader, "signinError")
        self.assertIn("No sign-in is in progress", str(error["error"]))
        # The hook rejects the pasted code.
        await self._send(reader, {"type": "signinLogin", "provider": "apple"})
        url = await self._wait_for_event(reader, "signinUrl")
        self.assertEqual(url["url"], "https://accounts.example/auth?p=apple")
        await self._send(reader, {"type": "signinPaste", "text": "bad"})
        error = await self._wait_for_event(reader, "signinError")
        self.assertEqual(error["provider"], "apple")
        self.assertEqual(error["error"], "bad code")
        # The hook fails before naming a page.
        await self._send(reader, {"type": "signinLogin", "provider": "nourl"})
        error = await self._wait_for_event(reader, "signinError")
        self.assertEqual(error["error"], "unknown provider")
        # The hook exits without a word.
        await self._send(reader, {"type": "signinLogin", "provider": "silent"})
        error = await self._wait_for_event(reader, "signinError")
        self.assertEqual(error["error"], "the sign-in hook printed no sign-in page")
        # The hook names the page, then dies without a verdict.
        await self._send(reader, {"type": "signinLogin", "provider": "crash"})
        url = await self._wait_for_event(reader, "signinUrl")
        self.assertEqual(url["url"], "https://accounts.example/crash")
        error = await self._wait_for_event(reader, "signinError")
        self.assertEqual(error["error"], "the sign-in hook exited with status 4")
        # The hook closed its stdin: the paste cannot be delivered.
        await self._send(reader, {"type": "signinLogin", "provider": "deaf"})
        await self._wait_for_event(reader, "signinUrl")
        await self._wait_for_file(self.home / "deaf-closed")
        await self._send(reader, {"type": "signinPaste", "text": "x"})
        error = await self._wait_for_event(reader, "signinError")
        self.assertIn("No sign-in is in progress", str(error["error"]))
        await self._send(reader, {"type": "signinCancel"})
        # A hook the OS cannot run.
        (self.home / SIGNIN_HOOK).write_text("#!/nonexistent/interpreter\n")
        await self._send(reader, {"type": "signinLogin", "provider": "google"})
        error = await self._wait_for_event(reader, "signinError")
        self.assertIn("No such file or directory", str(error["error"]))
        # No hook at all.
        (self.home / SIGNIN_HOOK).unlink()
        await self._send(reader, {"type": "signinLogin", "provider": "google"})
        error = await self._wait_for_event(reader, "signinError")
        self.assertEqual(error["error"], "no sign-in hook")

    async def test_cancel_and_restart_end_the_running_login_silently(self) -> None:
        self._install_hook()
        reader, _ = await self._ready("tab-a")
        await self._send(reader, {"type": "signinLogin", "provider": "google"})
        await self._wait_for_event(reader, "signinUrl")
        await self._send(reader, {"type": "signinCancel"})
        # The cancelled hook sends nothing; a paste now has nowhere to go.
        await self._send(reader, {"type": "signinPaste", "text": "x"})
        error = await self._wait_for_event(reader, "signinError")
        self.assertIn("No sign-in is in progress", str(error["error"]))
        self.assertFalse((self.home / "pasted").exists())
        # A second click ends the first hook; the paste goes to the second.
        await self._send(reader, {"type": "signinLogin", "provider": "google"})
        await self._wait_for_event(reader, "signinUrl")
        await self._send(reader, {"type": "signinLogin", "provider": "apple"})
        url = await self._wait_for_event(reader, "signinUrl")
        self.assertEqual(url["provider"], "apple")
        await self._send(reader, {"type": "signinPaste", "text": "ok"})
        done = await self._wait_for_event(reader, "signinDone")
        self.assertEqual(done["email"], "me@example.com")
        self.assertEqual((self.home / "pasted").read_text(), "ok\n")
        # Nothing else (no error from the ended first hook) is pending.
        self.assertEqual(await self._pending_signin_events(reader), [])
        # A hook still working out its URL when it is replaced never names it:
        # only the replacement's URL reaches the window.
        await self._send(reader, {"type": "signinLogin", "provider": "slow"})
        await self._send(reader, {"type": "signinLogin", "provider": "apple"})
        url = await self._wait_for_event(reader, "signinUrl")
        self.assertEqual(url["provider"], "apple")
        await asyncio.sleep(0.8)
        self.assertEqual(await self._pending_signin_events(reader), [])
        # A paste is cut to PASTE_LIMIT characters.
        await self._send(reader, {"type": "signinPaste", "text": "x" * (signin.PASTE_LIMIT + 50)})
        await self._wait_for_event(reader, "signinDone")
        self.assertEqual((self.home / "pasted").read_text(), "x" * signin.PASTE_LIMIT + "\n")

    async def test_shutdown_ends_the_running_login(self) -> None:
        self._install_hook()
        reader, _ = await self._ready("tab-a")
        await self._send(reader, {"type": "signinLogin", "provider": "google"})
        await self._wait_for_event(reader, "signinUrl")
        proc = self.server._server_api.signin._proc
        assert proc is not None
        await self.server._server_api.cancel_background_replies()
        self.assertIsNotNone(await asyncio.to_thread(proc.wait, 5))
        self.assertIsNone(self.server._server_api.signin._proc)


if __name__ == "__main__":
    unittest.main()
