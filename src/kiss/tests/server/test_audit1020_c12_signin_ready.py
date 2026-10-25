# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-20, slice C12: the sign-in hook and the ``ready`` sequence.

Findings C1-F1 / C2-03 / C4-F2 (the ``signin status`` subprocess used to
run inside every ``ready``, ahead of ``focusInput`` and the tab restore,
once per page load and reconnect), C4-F3 (the ``signin`` event came
before ``focusInput``, which took the window's focus) and C4-F6 (a
replaced or cancelled ``login`` hook was only sent SIGTERM).

Now the ``signin`` event is the tail of the ``ready``: a background
task sends it after the tab restore, and the hook's answer is cached
(:data:`kiss.server.signin.STATUS_CACHE_TTL`, dropped by a
``signinDone``).  The daemon is real; the hook is the stub of
:mod:`kiss.tests.server.test_signin_window`.
"""

from __future__ import annotations

import asyncio
import time
import unittest

from kiss.server import signin
from kiss.tests.server.test_signin_window import NOT_LOGGED_IN, SigninHookCase, forget_status


class TestSigninReadyTail(SigninHookCase):
    """The ``signin`` event follows the ready sequence and never holds it up."""

    async def test_signin_follows_focus_input_and_the_tab_restore(self) -> None:
        """C4-F3: the window must not be on screen when ``focusInput`` fires."""
        self._install_hook()
        _reader, events = await self._ready("tab-1")
        kinds = [str(ev.get("type")) for ev in events]
        self.assertIn("signin", kinds)
        self.assertLess(kinds.index("focusInput"), kinds.index("signin"))
        self.assertLess(kinds.index("tabs_state"), kinds.index("signin"))

    async def test_slow_status_hook_holds_up_neither_the_tab_restore_nor_the_next_command(
        self,
    ) -> None:
        """C1-F1 / C4-F2: the hook's process runs beside the ready, not inside it."""
        self._install_hook()
        (self.home / "status-sleep").write_text("2\n")
        reader = await self._connect()
        started = time.monotonic()
        await self._send_ready(self._readers_writers[reader], "tab-1")
        events = await self._read_until(reader, "tabs_state")
        self.assertLess(time.monotonic() - started, 1.5, "tabs_state waited for the hook")
        self.assertNotIn("signin", [ev.get("type") for ev in events])
        # The page's next command is answered while the hook still runs...
        await self._send(reader, {"type": "ping"})
        await self._read_until(reader, "pong")
        self.assertLess(time.monotonic() - started, 1.5, "the ping waited for the hook")
        # ...and the window still arrives, once the hook has answered.
        event = await self._wait_for_event(reader, "signin")
        self.assertEqual(event["providers"], NOT_LOGGED_IN["providers"])
        self.assertGreaterEqual(time.monotonic() - started, 2.0)


class TestStatusCache(SigninHookCase):
    """C2-03: one ``signin status`` run serves every ready of the TTL."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        signin.STATUS_CACHE_TTL = 60.0

    async def test_one_status_run_serves_every_ready_until_the_ttl_ends(self) -> None:
        self._install_hook()
        for tab in ("tab-1", "tab-2", "tab-3"):
            _reader, events = await self._ready(tab)
            self.assertEqual(len(self._signin_events(events)), 1, tab)
        self.assertEqual(self._status_runs(), 1)
        # After the TTL the hook is asked again, and a changed answer shows.
        signin.STATUS_CACHE_TTL = 0.2
        forget_status()
        _reader, events = await self._ready("tab-5")
        self.assertEqual(len(self._signin_events(events)), 1)
        self.assertEqual(self._status_runs(), 2)
        self._set_status({"logged_in": True, "email": "me@example.com"})
        await asyncio.sleep(0.25)
        _reader, events = await self._ready("tab-6")
        self.assertEqual(self._signin_events(events), [])
        self.assertEqual(self._status_runs(), 3)

    async def test_a_reconnect_storm_runs_the_hook_once(self) -> None:
        """Every window of a restarted daemon sends ``ready`` at once."""
        self._install_hook()
        (self.home / "status-sleep").write_text("0.5\n")
        results = await asyncio.gather(*(self._ready(f"tab-{i}") for i in range(4)))
        for _reader, events in results:
            self.assertEqual(len(self._signin_events(events)), 1)
        self.assertEqual(self._status_runs(), 1)

    async def test_a_sign_in_through_the_window_drops_the_cached_answer(self) -> None:
        self._install_hook()
        reader, events = await self._ready("tab-1")
        self.assertEqual(len(self._signin_events(events)), 1)
        # A login elsewhere is not seen before the TTL ends...
        self._set_status({"logged_in": True, "email": "me@example.com"})
        _reader, events = await self._ready("tab-2")
        self.assertEqual(len(self._signin_events(events)), 1)
        self.assertEqual(self._status_runs(), 1)
        # ...but one through the window is, at once.
        await self._send(reader, {"type": "signinLogin", "provider": "google"})
        await self._wait_for_event(reader, "signinUrl")
        await self._send(reader, {"type": "signinPaste", "text": "code"})
        await self._wait_for_event(reader, "signinDone")
        _reader, events = await self._ready("tab-3")
        self.assertEqual(self._signin_events(events), [])
        self.assertEqual(self._status_runs(), 2)


class TestHookEnd(SigninHookCase):
    """C4-F6: a replaced or cancelled ``login`` hook is killed when it ignores SIGTERM."""

    async def test_cancelled_hook_that_ignores_sigterm_is_killed(self) -> None:
        self._install_hook()
        saved = signin.HOOK_END_TIMEOUT
        signin.HOOK_END_TIMEOUT = 0.3
        self.addCleanup(setattr, signin, "HOOK_END_TIMEOUT", saved)
        reader, _ = await self._ready("tab-a")
        await self._send(reader, {"type": "signinLogin", "provider": "stubborn"})
        await self._wait_for_event(reader, "signinUrl")
        proc = self.server._server_api.signin._proc
        assert proc is not None
        await self._send(reader, {"type": "signinCancel"})
        self.assertEqual(await asyncio.to_thread(proc.wait, 5), -9)
        # The replacement hook of a second click gets the same end.
        await self._send(reader, {"type": "signinLogin", "provider": "stubborn"})
        await self._wait_for_event(reader, "signinUrl")
        proc = self.server._server_api.signin._proc
        assert proc is not None
        await self._send(reader, {"type": "signinLogin", "provider": "google"})
        url = await self._wait_for_event(reader, "signinUrl")
        self.assertEqual(url["provider"], "google")
        self.assertEqual(await asyncio.to_thread(proc.wait, 5), -9)
        await self._send(reader, {"type": "signinCancel"})
        self.assertEqual(await self._pending_signin_events(reader), [])


if __name__ == "__main__":
    unittest.main()
