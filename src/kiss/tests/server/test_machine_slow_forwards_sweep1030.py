# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: forwarded commands a machine may take longer than a request's budget on.

A real local daemon and a real remote daemon registered as ``box``,
whose working directory is a git repository with a ``post-checkout``
hook that runs past :data:`~kiss.agents.sorcar.machines.REQUEST_TIMEOUT`
(as a Git LFS or dependency-install hook does).  The Source Control
graph's "Checkout (Detached)" on the machine runs git, so the hook:

* the ``gitAction`` reply is the machine's own result, not a "did not
  answer: no gitActionResult within 8s" error while the checkout
  completes there anyway;
* while it is pending, another surface's Explorer request to the
  machine is answered at once (the action waits on a link lane of its
  own), and afterwards the machine is not backed off as down.

The settings sheet's ``getClaudeCodeStatus`` / ``getCodexStatus`` are
waited for as long as the machine's own bound on ``claude auth status``
/ ``codex login status`` (20 s / 30 s).  A CLI that slow cannot be had
end to end: it takes a stand-in ``claude`` / ``codex`` executable (a
test double), so for them the test checks only that the forwarded probe
is the machine's answer.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import machines as mach
from kiss.tests.conftest import posix_only
from kiss.tests.local_ws import LocalReader, open_local_connection
from kiss.tests.server.test_remote_machine_settings import RemoteMachineCase

pytestmark = posix_only("the slow git hook is a POSIX shell script")

_HOOK_SECONDS = mach.REQUEST_TIMEOUT + 2


def _git(cwd: Path, *args: str) -> str:
    """Run ``git *args`` in *cwd* and return its output, failing the test on an error."""
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout


async def _reply(reader: LocalReader, reply_type: str) -> dict[str, Any]:
    """Read *reader* up to its next *reply_type* message and return that message."""
    for _ in range(300):
        line = await asyncio.wait_for(reader.readline(), timeout=60.0)
        assert line, "connection closed"
        msg: dict[str, Any] = json.loads(line)
        if msg.get("type") == reply_type:
            return msg
    raise AssertionError(f"no {reply_type} reply")


class TestSlowMachineForwards(RemoteMachineCase):
    """A machine's slow Git action and CLI probes are waited for, on lanes of their own."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        _git(self.box_work, "init", "-q")
        _git(self.box_work, "config", "user.email", "t@t")
        _git(self.box_work, "config", "user.name", "t")
        for text in ("one", "two"):
            (self.box_work / "f.txt").write_text(text + "\n")
            _git(self.box_work, "add", "-A")
            _git(self.box_work, "commit", "-qm", text)
        self.first_sha = _git(self.box_work, "rev-parse", "HEAD~1").strip()
        self.hook_done = Path(self.tmpdir) / "hook-done"
        hook = self.box_work / ".git" / "hooks" / "post-checkout"
        hook.write_text(
            f"#!/bin/sh\nsleep {_HOOK_SECONDS:g}\ntouch {shlex.quote(str(self.hook_done))}\n",
        )
        hook.chmod(0o755)

    async def test_a_slow_git_action_is_answered_and_holds_up_nothing(self) -> None:
        started = time.monotonic()
        await self._send({
            "type": "gitAction", "action": "checkoutDetached", "sha": self.first_sha,
            "workDir": str(self.box_work), "tabId": "tab-box", "token": "act", "machine": "box",
        })
        # Meanwhile another surface browses the machine: answered at
        # once, not queued behind the action on the machine's link.
        reader, writer = await open_local_connection(self.server)
        self.addCleanup(writer.close)
        writer.write(json.dumps({
            "type": "listDir", "path": str(self.box_work), "token": "tree", "machine": "box",
        }).encode("utf-8") + b"\n")
        await writer.drain()
        listing = await _reply(reader, "dirListing")
        self.assertEqual(listing.get("error", ""), "")
        self.assertEqual(listing["machine"], "box")
        self.assertIn("f.txt", [entry["name"] for entry in listing["entries"]])
        self.assertFalse(self.hook_done.exists(), "the listing waited for the git action")

        acted = await self._next("gitActionResult")
        self.assertEqual(acted.get("error", ""), "")
        self.assertEqual(acted["machine"], "box")
        self.assertTrue(acted["ok"])
        self.assertEqual(acted["token"], "act")
        # The action really outlasted a request's usual budget.
        self.assertTrue(self.hook_done.exists())
        self.assertGreater(time.monotonic() - started, mach.REQUEST_TIMEOUT)
        self.assertEqual(_git(self.box_work, "rev-parse", "HEAD").strip(), self.first_sha)
        # The machine was not backed off as down: the next request goes through.
        status = await self._ask({
            "type": "gitStatus", "workDir": str(self.box_work), "token": "scm", "machine": "box",
        }, "gitStatus")
        self.assertEqual(status.get("error", ""), "")
        self.assertEqual(status["machine"], "box")

    async def test_cli_status_probes_are_the_machines_answers(self) -> None:
        for cmd_type, reply_type in (
            ("getClaudeCodeStatus", "claudeCodeStatus"), ("getCodexStatus", "codexStatus"),
        ):
            reply = await self._ask_box({"type": cmd_type}, reply_type)
            self.assertEqual(reply["machine"], "box")
            # ``installed`` is the machine's status report, never in a
            # link failure's reply.
            self.assertIn("installed", reply, reply)
