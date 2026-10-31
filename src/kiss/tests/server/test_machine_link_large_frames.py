# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A history click on a registered machine no longer drops the link to it.

Reproduces the report "34.55.131.190 is unreachable": a UI opened on
the remote machine clicked a chat whose transcript was 1.5 MB.  The
remote daemon broadcast that ``task_events`` replay to every connected
socket, including the local daemon's ``MachineLink`` request socket,
which was opened with the websockets default ``max_size`` of 1 MiB.
The link closed with 1009 ("message too big"), every ``ping`` failed
for the link's back-off period, and the MACHINES panel showed the
machine as unreachable.

The fix has two halves and this test exercises both against a real
second daemon (``kiss.tests.remote_daemon``) seeded with a task whose
text is larger than 1 MiB:

* ``MachineLink._connect`` opens its sockets with no frame-size limit,
  so a stream socket (:meth:`MachineLink.open_connection`, an ordinary
  peer that must receive its tab's events) takes the >1 MiB transcript
  without dying — the stream below proves it.
* The shared request socket declares itself a daemon link and the
  remote daemon leaves it out of the task-event fan-out; the ``ping``
  after the replay proves the link is still up.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from websockets.sync.client import ClientConnection

import kiss.agents.sorcar.machines as mach
from kiss.tests.remote_daemon import start_remote_daemon

ONE_MIB = 1024 * 1024
BIG_TASK = "Summarise this log: " + "x" * (int(1.5 * ONE_MIB))
CHAT_ID = "chat-big"


def _recv_frame(ws: ClientConnection, wanted: str, seconds: float) -> str:
    """Return the raw text of the first *wanted* frame *ws* receives within *seconds*."""
    deadline = time.monotonic() + seconds
    while True:
        raw = ws.recv(timeout=max(0.1, deadline - time.monotonic()))
        text = raw if isinstance(raw, str) else raw.decode("utf-8")
        frame: dict[str, Any] = json.loads(text)
        if frame.get("type") == wanted:
            return text


class TestMachineLinkLargeFrames(TestCase):
    """One real remote daemon with a seeded >1 MiB transcript, one ``MachineLink``."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-link-frames-")
        mach.save_machines([])
        self.remote = start_remote_daemon(
            Path(self.tmpdir) / "remote-home", password="pw-remote",
            seed=[{"chat_id": CHAT_ID, "task": BIG_TASK}],
        )
        self.machine = mach.add_machine("box", self.remote.url, "pw-remote")

    def tearDown(self) -> None:
        mach.close_links()
        self.remote.stop()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_transcript_replay_on_the_machine_keeps_the_link_alive(self) -> None:
        """A UI on the machine replays a 1.5 MB transcript; the link still answers."""
        link = mach.link_for(self.machine)
        self.assertEqual(link.request({"type": "ping"}, "pong")["type"], "pong")

        # A remote-tab stream (ordinary peer) and the "UI" on the machine
        # that clicks the big chat in its history.
        stream = link.open_connection()
        self.addCleanup(stream.close)
        ui = link.open_connection()
        self.addCleanup(ui.close)
        task_id = self.remote.task_ids[f"{CHAT_ID}/{BIG_TASK}"]
        # A raw frame on a password-authenticated socket carries the
        # machine's password, as RemoteTabStream.send does.
        ui.send(json.dumps({
            "type": "resumeSession", "chatId": CHAT_ID, "taskId": task_id,
            "tabId": "tab-ui", "password": "pw-remote",
        }))

        # The history click is broadcast: the stream receives the whole
        # transcript, which is larger than the old 1 MiB frame limit.
        replay = _recv_frame(stream, "task_events", 60.0)
        self.assertGreater(len(replay), ONE_MIB)
        self.assertIn(BIG_TASK[-64:], replay)

        # The request socket was not closed by the replay: the machine
        # is still reachable, with no back-off in the way.
        self.assertEqual(link.request({"type": "ping"}, "pong")["type"], "pong")
        self.assertEqual(link._health.last_error, "")
