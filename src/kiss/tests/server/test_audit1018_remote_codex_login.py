# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a machine's slow Codex device login is waited for (audit 2026-10-18).

A real local daemon and a real remote daemon registered as ``box``
whose ``codex`` (a fake on its PATH) takes longer than the link's usual
request budget to print its device code and then, like the real CLI,
stays alive waiting for the user to enter it: the forwarded
``startCodexLogin`` still brings the code back, the settings sheet's
other requests to the machine are answered meanwhile (the login travels
on a link lane of its own), and a second surface asking for the login
while the first is pending gets the same code, from the same CLI.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import time
from pathlib import Path

from kiss.agents.sorcar import machines as mach
from kiss.tests.conftest import posix_only
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.test_remote_machine_settings import RemoteMachineCase

pytestmark = posix_only("the fake codex is a POSIX script")

_SLOW_LOGIN = mach.REQUEST_TIMEOUT + 4
# Run ``n`` prints code ``CODE-RUNn``, then stays alive until the stop
# file next to the script exists (20 s at most).
_SLOW_CODEX = f"""#!/bin/sh
here=$(dirname "$0")
echo "$@" >> "$here/runs"
if [ "$1" = "login" ] && [ "$2" = "--device-auth" ]; then
  sleep {_SLOW_LOGIN:g}
  echo "   https://auth.openai.com/codex/device"
  echo "   CODE-RUN$(wc -l < "$here/runs" | tr -d ' ')"
  n=0
  while [ ! -e "$here/stop" ] && [ $n -lt 200 ]; do sleep 0.1; n=$((n+1)); done
  exit 0
fi
echo "Not logged in" >&2
exit 1
"""


class TestRemoteCodexLogin(RemoteMachineCase):
    """``startCodexLogin`` stamped ``machine`` waits for the machine's CLI."""

    async def asyncSetUp(self) -> None:
        self.previous_path = os.environ["PATH"]
        bin_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"slow-codex-{os.getpid()}"
        bin_dir.mkdir(exist_ok=True)
        script = bin_dir / "codex"
        script.write_text(_SLOW_CODEX)
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        self.script = script
        # The remote daemon inherits this PATH when the fixture starts it.
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{self.previous_path}"
        await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        os.environ["PATH"] = self.previous_path
        (self.script.parent / "stop").touch()
        await super().asyncTearDown()
        await asyncio.sleep(0.3)  # the CLI alive sees the stop file and exits
        shutil.rmtree(self.script.parent, ignore_errors=True)

    async def test_a_slow_device_login_answers_every_surface_with_its_code(self) -> None:
        reader, writer = await open_local_connection(self.server)
        self.addCleanup(writer.close)
        started = time.monotonic()
        await self._send({"type": "startCodexLogin", "machine": "box"})
        writer.write(b'{"type":"startCodexLogin","machine":"box"}\n')
        await writer.drain()
        # Meanwhile the sheet's other requests to the machine are answered.
        config = await asyncio.wait_for(self._ask_box({"type": "getConfig"}, "configData"), 5.0)
        self.assertEqual(config.get("error", ""), "")
        login = await self._next("codexLogin", forwarded=True)
        first_time = time.monotonic() - started
        self.assertGreater(first_time, _SLOW_LOGIN - 1)
        self.assertEqual(login.get("error", ""), "")
        self.assertEqual(login["machine"], "box")
        self.assertEqual(login["code"], "CODE-RUN1")
        self.assertEqual(login["url"], "https://auth.openai.com/codex/device")
        # The second surface's request waited its turn on the login
        # lane; the machine's CLI is still alive with CODE-RUN1, so it
        # is answered with that code, at once, from the same CLI.
        while True:
            line = await asyncio.wait_for(reader.readline(), 30.0)
            self.assertTrue(line, "second connection closed")
            second = json.loads(line)
            if second.get("type") == "codexLogin" and second.get("forwarded"):
                break
        self.assertLess(time.monotonic() - started - first_time, 3.0)
        self.assertEqual(second.get("error", ""), "")
        self.assertEqual(second["code"], "CODE-RUN1")
        self.assertEqual((self.script.parent / "runs").read_text().count("login"), 1)
