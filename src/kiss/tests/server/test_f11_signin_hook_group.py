# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: cancelling a sign-in hook ends the hook's children too (F11).

``SigninLogin`` ran the hook with a plain ``Popen`` and ended it by
terminating (then killing) the hook alone.  A hook that started a
helper of its own (a browser opener, a poller) which inherited its
stdout kept the relay thread blocked on that pipe until the helper
went away on its own, and the helper lived on past the cancel.  Now
the hook runs in a process group of its own and the cancel ends the
whole group, as the talk player's playback does.

The daemon and the hook (a shell stub that backgrounds a ``sleep``
holding its stdout) are real.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

from kiss.server.signin import SIGNIN_HOOK
from kiss.tests.conftest import posix_only
from kiss.tests.server.test_signin_window import NOT_LOGGED_IN, SigninHookCase

pytestmark = posix_only("the hook stub is a shell script")

_HELPER_SECONDS = 20

# The ``login`` prints its URL, keeps a helper alive in the background
# that inherits its stdout (and records the helper's pid), then waits
# for the paste like a real hook.
_HOOK_WITH_HELPER = f"""#!/bin/bash
case "$1" in
  status) cat "$KISS_HOME/status.json" ;;
  login)
    echo '{{"url": "https://accounts.example/helper"}}'
    sleep {_HELPER_SECONDS} &
    echo $! > "$KISS_HOME/helper-pid"
    read -r line || exit 2
    echo '{{"logged_in": true, "email": "me@example.com"}}'
    ;;
esac
"""


def _alive(pid: int) -> bool:
    """Whether a process with *pid* exists (a zombie counts as alive).

    Signal 0 probes without delivering anything; it works on macOS too,
    which has no ``/proc``.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestCancelEndsTheHooksHelper(SigninHookCase):
    """``signinCancel`` ends the hook's helper and frees the relay thread at once."""

    async def test_cancel_ends_the_helper_and_the_relay_thread(self) -> None:
        hook = self.home / SIGNIN_HOOK
        hook.write_text(_HOOK_WITH_HELPER)
        hook.chmod(0o755)
        self._set_status(NOT_LOGGED_IN)
        reader, _ = await self._ready("tab-a")
        await self._send(reader, {"type": "signinLogin", "provider": "google"})
        url = await self._wait_for_event(reader, "signinUrl")
        self.assertEqual(url["url"], "https://accounts.example/helper")
        pid_file = self.home / "helper-pid"
        for _ in range(500):
            if pid_file.exists() and pid_file.read_text().strip():
                break
            await asyncio.sleep(0.01)
        helper_pid = int(pid_file.read_text().strip())
        self.assertTrue(_alive(helper_pid), "the hook's helper is running")
        relay = next(t for t in threading.enumerate() if t.name == "signin-login")

        started = time.monotonic()
        await self._send(reader, {"type": "signinCancel"})
        await asyncio.to_thread(relay.join, 5.0)
        self.assertFalse(relay.is_alive(), "the relay thread stayed blocked on the helper's pipe")
        self.assertLess(time.monotonic() - started, 5.0)
        for _ in range(500):
            if not _alive(helper_pid):
                break
            await asyncio.sleep(0.01)
        self.assertFalse(_alive(helper_pid), "the helper outlived the cancel")
        for _ in range(500):
            if not self.server._server_api.signin._ending:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.server._server_api.signin._ending, [])
        self.assertIsNone(self.server._server_api.signin._proc)
