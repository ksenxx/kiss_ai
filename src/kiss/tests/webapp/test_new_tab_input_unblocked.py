# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: "+" while a task runs in the current chat opens a chat
whose composer is ready for a new task.

``createNewTab()`` shows the new tab through ``restoreTab(tab)``, which
leaves the page's running state alone, so it must then call
``setRunningState(false)``.  Without that call the busy chat's running
state carried over to the new chat: its input and Send were blocked
(the input lock of the time) and Stop stayed on screen.

The test drives the real chat page (``media/chat.html`` with ``api.js``
and ``main.js`` in jsdom, built by
``src/kiss/webapp/test/simplify2_harness.js``): a ``status`` event marks
the first chat running, a click on the "+" button opens the new chat,
and the test reads the composer controls of both.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

_WEBAPP_DIR = Path(__file__).resolve().parents[2] / "webapp"
_HARNESS = _WEBAPP_DIR / "test" / "simplify2_harness.js"
_JSDOM_PKG = _WEBAPP_DIR / "node_modules" / "jsdom" / "package.json"

_SCRIPT = r"""
const {makePage, send, click, byId, stubRaf, readyTabId} = require(%s);
const {win, posted} = makePage({bodyClass: 'remote-chat', beforeScripts: stubRaf});
function controls() {
  return {
    tab: win._testApi.getActiveTabId(),
    runningTabs: win._testApi.openTabs().filter(t => t.isRunning).map(t => t.id),
    inputDisabled: byId(win, 'task-input').disabled,
    sendDisabled: byId(win, 'send-btn').disabled,
    sendDisplay: byId(win, 'send-btn').style.display,
    stopDisplay: byId(win, 'stop-btn').style.display,
  };
}
const boot = readyTabId(posted);
send(win, {type: 'status', running: true, tabId: boot});
const busy = controls();
click(win, byId(win, 'new-chat-btn'));
process.stdout.write(JSON.stringify({boot, busy, fresh: controls()}));
win.close();
"""


class TestCreateNewTabDoesNotBlockInput(unittest.TestCase):
    """The new chat opened from a busy chat is idle; the busy chat keeps running."""

    def test_new_tab_does_not_disable_input_when_prior_tab_running(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(f"jsdom is not installed under {_JSDOM_PKG.parents[1]}")
        result = subprocess.run(
            ["node", "-e", _SCRIPT % json.dumps(str(_HARNESS))],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(_WEBAPP_DIR),
        )
        assert result.returncode == 0, f"node error: {result.stderr}\nstdout: {result.stdout}"
        out = json.loads(result.stdout)
        boot, busy, fresh = out["boot"], out["busy"], out["fresh"]

        assert busy["tab"] == boot and busy["runningTabs"] == [boot], busy
        assert busy["stopDisplay"] == "flex", busy

        assert fresh["tab"] != boot, fresh
        assert fresh["runningTabs"] == [boot], "the busy chat keeps running"
        assert fresh["inputDisabled"] is False, fresh
        assert fresh["sendDisabled"] is False, fresh
        assert fresh["sendDisplay"] == "flex", fresh
        assert fresh["stopDisplay"] == "none", (
            f"the new chat shows the busy chat's Stop button: {fresh}"
        )


if __name__ == "__main__":
    unittest.main()
