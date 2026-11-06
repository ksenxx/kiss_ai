# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The extension's service worker (``chrome_extension/background.js``) in a Node VM.

The shipped script is loaded unmodified into a ``vm`` context whose
``chrome`` API is a small tab model scripted by the test (the one thing
Node does not have), the way the bridge tests script the extension's side
of the socket; ``handleCommand`` is driven with the frames KISS sends.
The model can hold one API call open, which lets a command's deadline
pass in the middle of it deterministically.

Focus-mutating commands serialize on one queue, and a command that
overran its deadline is answered and its queue slot released while its
own promise is still pending.  A timed-out screenshot therefore resumed
after the next command had opened its tab, activated the old tab and
captured it: the later, successful command no longer owned the focus.
Every focus-mutating step that follows an await now checks the deadline
first.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

BACKGROUND_JS = (
    Path(__file__).parents[3] / "agents" / "sorcar" / "chrome_extension" / "background.js"
)

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

#: Loads background.js, runs the command named on stdin with a 25 ms
#: deadline while the named chrome API call is held open, runs an
#: open_tab behind it, releases the held call and reports every tab or
#: focus mutation the model saw, in order, plus the replies.
HARNESS = r"""
const vm = require('vm');
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const calls = [];
const tabs = new Map([
  [1, { id: 1, windowId: 1, title: 'A', url: 'http://a/', active: true, status: 'complete' }],
]);
let active = 1;
let release = null;
function maybeHold(name, result) {
  if (name === input.hold && release === null) {
    return new Promise((resolve) => { release = () => resolve(result); });
  }
  return Promise.resolve(result);
}
const chrome = {
  runtime: { onStartup: { addListener() {} }, onInstalled: { addListener() {} }, getURL: (p) => p },
  action: { onClicked: { addListener() {} } },
  alarms: { onAlarm: { addListener() {} }, create() {}, clear() {} },
  tabs: {
    get: (id) => maybeHold('tabs.get', { ...tabs.get(id) }),
    update(id, props) {
      if (props.active) { active = id; calls.push('activate:' + id); }
      return maybeHold('tabs.update', { ...tabs.get(id) });
    },
    async create(props) {
      const id = tabs.size + 1;
      tabs.set(id, {
        id, windowId: 1, title: '', url: props.url, active: true, status: 'complete',
      });
      active = id;
      calls.push('open:' + id);
      return { ...tabs.get(id) };
    },
    async captureVisibleTab() {
      calls.push('capture:' + active);
      return 'data:image/png;base64,';
    },
  },
  windows: {
    update(windowId) { calls.push('focus:' + windowId); return maybeHold('windows.update', {}); },
  },
};
const replies = {};
const sock = {
  readyState: 1,
  send(text) { const reply = JSON.parse(text); replies[reply.id] = reply; },
};
const context = vm.createContext({
  chrome, WebSocket: { OPEN: 1 }, fetch: () => Promise.reject(new Error('no bridge')),
  setTimeout, clearTimeout, setInterval, clearInterval, console,
});
vm.runInContext(input.source, context, { filename: 'background.js' });
const handleCommand = vm.runInContext('handleCommand', context);
(async () => {
  await handleCommand(sock, { id: 1, cmd: input.cmd, args: { tabId: 1 }, timeout_ms: 25 });
  await handleCommand(sock, {
    id: 2, cmd: 'open_tab', args: { url: 'http://b/' }, timeout_ms: 1000,
  });
  release();
  // Past the screenshot's 250 ms paint wait.
  await new Promise((resolve) => setTimeout(resolve, 400));
  console.log(JSON.stringify({ calls, active, replies }));
  process.exit(0);
})();
"""


def _run_focus_race(cmd: str, hold: str) -> dict:
    """Run *cmd* with the chrome call *hold* held past its deadline, then an open_tab."""
    payload = {"source": BACKGROUND_JS.read_text(encoding="utf-8"), "cmd": cmd, "hold": hold}
    completed = subprocess.run(
        ["node", "-e", HARNESS], input=json.dumps(payload), capture_output=True, text=True,
        timeout=60, check=True,
    )
    return dict(json.loads(completed.stdout))


@pytest.mark.parametrize(
    ("cmd", "hold"),
    [("screenshot", "tabs.get"), ("screenshot", "windows.update"), ("activate_tab", "tabs.update")],
)
def test_a_timed_out_focus_command_does_not_resume_into_the_next_ones_focus(
    cmd: str, hold: str
) -> None:
    """Held past its deadline, the command is answered; once released it mutates nothing more."""
    out = _run_focus_race(cmd, hold)
    assert out["replies"]["1"]["ok"] is False
    assert "exceeded its 0.025s deadline" in out["replies"]["1"]["error"]
    assert out["replies"]["2"]["ok"] is True
    # Nothing after the next command opened its tab: no activation, focus
    # or capture of the old tab.
    assert out["calls"][out["calls"].index("open:2"):] == ["open:2"]
    assert out["active"] == 2
