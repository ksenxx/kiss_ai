# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a machine's CLI status probe holds up none of the sender's commands.

The settings sheet asks the machine the chat header picked for
``getClaudeCodeStatus`` / ``getCodexStatus``, and the daemon waits for
the machine's answer up to the machine's own CLI bound (20 s / 30 s).
The forward ran inline in the sender's command loop, so every command
the sender sent after it (a submit, a stop) waited that long too, and
the Codex login's 5 s status poll piled up behind a slow probe.  The
probes now run in the background like the subpanel polls, one in
flight per connection, probe and machine.

A real local daemon and a "black hole" machine ``hole`` whose port
accepts the connection and never answers (a host that drops packets).
"""

from __future__ import annotations

import socket
import threading
import time

from kiss.agents.sorcar import machines as mach
from kiss.tests.server.test_remote_machine_settings import RemoteMachineCase


class TestCliStatusForwardInBackground(RemoteMachineCase):
    """A status probe to an unresponsive machine runs beside the sender's commands."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.listener.settimeout(0.2)
        self.accepted: list[socket.socket] = []
        self.stop_accepting = threading.Event()
        self.acceptor = threading.Thread(target=self._accept_forever, daemon=True)
        self.acceptor.start()
        hole = mach.Machine(
            name="hole", url=f"https://127.0.0.1:{self.listener.getsockname()[1]}",
            password="pw",
        )
        self.hole_cert = hole.cert_file()
        self.hole_cert.parent.mkdir(parents=True, exist_ok=True)
        (box,) = [m for m in mach.load_machines() if m.name == "box"]
        self.hole_cert.write_bytes(box.cert_file().read_bytes())
        mach.save_machines([box, hole])

    def _accept_forever(self) -> None:
        while not self.stop_accepting.is_set():
            try:
                self.accepted.append(self.listener.accept()[0])
            except TimeoutError:
                continue

    async def asyncTearDown(self) -> None:
        self.stop_accepting.set()
        self.acceptor.join(5)
        for conn in self.accepted:
            conn.close()
        self.listener.close()
        self.hole_cert.unlink(missing_ok=True)
        await super().asyncTearDown()

    async def test_status_probes_hold_up_no_later_command(self) -> None:
        for _ in range(2):
            await self._send({"type": "getCodexStatus", "machine": "hole"})
            await self._send({"type": "getClaudeCodeStatus", "machine": "hole"})
        started = time.monotonic()
        reply = await self._ask({"type": "getConfig"}, "configData")
        self.assertLess(time.monotonic() - started, mach.REQUEST_TIMEOUT / 2)
        self.assertNotIn("forwarded", reply)
        # One probe of each kind waits on the machine; the repeats sent
        # while it does were dropped.
        api = self.server._server_api
        pending = sorted(key[1] for key, task in api._panel_replies.items() if not task.done())
        self.assertEqual(pending, ["getClaudeCodeStatus", "getCodexStatus"])
        self.assertEqual(len(api._background_replies), 2)
