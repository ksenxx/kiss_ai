# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: ``SurfaceBridge`` reports why the node bridge could not serve a command.

Before this, a bridge that died at startup (for instance in an agent
worktree where ``out/wsClient.js`` was never compiled) or whose daemon
connection was refused left the driver waiting out its 30 s reply
timeout and failing with a bare ``queue.Empty``; node's stderr was piped
and never read, so the reason was lost.

The "gave no reply in 30 s" branch of :meth:`SurfaceBridge.call` is the
last-resort guard for a bridge that is alive but wedged; every known
command either answers or kills the process, so it is not exercised
here (and not faked).
"""

import json
import time
from pathlib import Path

from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness
from kiss.tests.server.test_subagent_tabs_all_surfaces import (
    SurfaceBridge,
    skip_unless_bridge_runs,
)


class SurfaceBridgeFailuresTest(DaemonLocalHarness):
    """A crashed bridge and a refused connection both fail fast, with the cause."""

    def setUp(self) -> None:
        skip_unless_bridge_runs(self)
        super().setUp()

    def test_bridge_crash_is_reported_at_once(self) -> None:
        """A bridge that exits before answering fails with its exit code and stderr."""
        bridge = SurfaceBridge(str(Path(self.tmpdir) / "no-such-endpoint.json"))
        self.addCleanup(bridge.quit)
        started = time.monotonic()
        with self.assertRaises(AssertionError) as ctx:
            bridge.open("sidebar")
        self.assertLess(time.monotonic() - started, 20)
        message = str(ctx.exception)
        self.assertIn("bridge exited on 'open' (exit code 1)", message)
        self.assertIn("no-such-endpoint.json", message)

    def test_refused_token_is_answered(self) -> None:
        """``open`` is answered with the connection error when the daemon refuses the token."""
        record = json.loads(self.endpoint_file.read_text())
        record["token"] = "0" * len(record["token"])
        bad_endpoint = Path(self.tmpdir) / "bad-token.json"
        bad_endpoint.write_text(json.dumps(record))
        bridge = SurfaceBridge(str(bad_endpoint))
        self.addCleanup(bridge.quit)
        started = time.monotonic()
        reply = bridge.call("open", name="sidebar", bodyAttrs="", state=None)
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(reply["op"], "error", reply)
        self.assertEqual(reply["name"], "sidebar")
        self.assertTrue(reply["error"].startswith("connect: "), reply)
        self.assertIn("auth_failed", reply["error"])
