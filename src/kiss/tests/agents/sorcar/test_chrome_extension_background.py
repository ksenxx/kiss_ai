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
  scripting: {
    // The page world: records the script a command runs and answers it.
    async executeScript({ args }) {
      calls.push('script:' + args[0]);
      return [{ result: { ok: true, value: 'ran' } }];
    },
  },
};
const replies = {};
const sock = {
  readyState: 1,
  send(text) { const reply = JSON.parse(text); replies[reply.id] = reply; },
};
const context = vm.createContext({
  chrome, WebSocket: { OPEN: 1 }, fetch: () => Promise.reject(new Error('no bridge')),
  setTimeout, clearTimeout, setInterval, clearInterval, console, URL,
});
vm.runInContext(input.source, context, { filename: 'background.js' });
const handleCommand = vm.runInContext('handleCommand', context);
(async () => {
  await handleCommand(sock, {
    id: 1, cmd: input.cmd, args: input.args || { tabId: 1 }, timeout_ms: input.timeout_ms || 25,
  });
  await handleCommand(sock, {
    id: 2, cmd: 'open_tab', args: { url: 'http://b/' }, timeout_ms: 1000,
  });
  if (release) release();
  // Past the screenshot's 250 ms paint wait.
  await new Promise((resolve) => setTimeout(resolve, 400));
  console.log(JSON.stringify({ calls, active, replies }));
  process.exit(0);
})();
"""


#: Loads background.js into a context whose clock and timers are the
#: test's: an open_tab is held open in tabs.create, a close_tab queued
#: behind it with a 10 ms deadline, its deadline timer fired while the
#: clock is still one tick short of the deadline, then the open_tab
#: released. Reports close_tab's reply at that point, every tab mutation
#: that followed, and the final replies.
EARLY_TIMER_HARNESS = r"""
const vm = require('vm');
const source = require('fs').readFileSync(0, 'utf8');
let now = 1000;
const timers = [];
const calls = [];
const replies = {};
let release = null;
const chrome = {
  runtime: { onStartup: { addListener() {} }, onInstalled: { addListener() {} }, getURL: (p) => p },
  action: { onClicked: { addListener() {} } },
  alarms: { onAlarm: { addListener() {} }, create() {}, clear() {} },
  tabs: {
    create: () => new Promise((resolve) => { release = () => resolve({ id: 2 }); }),
    async get(id) { return { id, windowId: 1, status: 'complete' }; },
    async remove(id) { calls.push('close:' + id); },
  },
};
const sock = { readyState: 1, send(text) { const r = JSON.parse(text); replies[r.id] = r; } };
const context = vm.createContext({
  chrome, WebSocket: { OPEN: 1 }, fetch: () => Promise.reject(new Error('no bridge')), console,
  URL, Date: { now: () => now },
  setTimeout(fn, delay) { const t = { fn, delay, live: true }; timers.push(t); return t; },
  clearTimeout(t) { if (t) t.live = false; },
  setInterval() {}, clearInterval() {},
});
vm.runInContext(source, context, { filename: 'background.js' });
const handleCommand = vm.runInContext('handleCommand', context);
const settle = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };
(async () => {
  const open = handleCommand(sock, { id: 1, cmd: 'open_tab', args: {}, timeout_ms: 100 });
  await settle();
  const close = handleCommand(sock, {
    id: 2, cmd: 'close_tab', args: { tabId: 5 }, timeout_ms: 10,
  });
  await settle();
  now = 1009; // the 10 ms timer fires a tick before Date.now() reaches 1010
  timers.find((t) => t.live && t.delay === 10).fn();
  await close;
  const answered = replies[2];
  release();
  await open;
  await settle();
  console.log(JSON.stringify({ answered, calls, replies }));
})();
"""


# Not covered: a closed socket's server proof completing after its replacement authenticated
# (the ``sock !== ws`` guard in connect()'s hello_ok handler). That handler exists only on a
# socket connect() opened, and the order only with a digest held open: both would be stand-ins.


def test_a_queued_command_answered_by_an_early_timer_never_runs() -> None:
    """Answered "exceeded" by its deadline timer, a queued command stays unrun once its slot frees.

    A timer may fire a fraction of a millisecond before ``Date.now()``
    reaches the deadline (the case ``pastDeadline`` is written for), so
    the context's clock and timers are the test's: the queued ``close_tab``
    was answered with the clock one tick short, and its slot, freed when
    the ``open_tab`` ahead of it completes, must not close the tab.
    """
    completed = subprocess.run(
        ["node", "-e", EARLY_TIMER_HARNESS],
        input=BACKGROUND_JS.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    out = json.loads(completed.stdout)
    assert out["answered"] == {"id": 2, "ok": False, "error": "command exceeded its 0.01s deadline"}
    assert out["replies"]["1"]["ok"] is True
    assert out["calls"] == []


def _run_focus_race(cmd: str, hold: str, **command: object) -> dict:
    """Run *cmd* with the chrome call *hold* held past its deadline, then an open_tab.

    *command* overrides the first command's ``args`` and ``timeout_ms``.
    """
    payload = {"source": BACKGROUND_JS.read_text(encoding="utf-8"), "cmd": cmd, "hold": hold,
               **command}
    completed = subprocess.run(
        ["node", "-e", HARNESS], input=json.dumps(payload), capture_output=True, text=True,
        timeout=60, check=True,
    )
    return dict(json.loads(completed.stdout))


@pytest.mark.parametrize(
    ("cmd", "hold"),
    [
        ("screenshot", "tabs.get"),
        ("screenshot", "windows.update"),
        ("screenshot", "tabs.update"),
        ("screenshot", ""),  # nothing held: the paint wait alone outlasts the deadline
        ("activate_tab", "tabs.update"),
    ],
)
def test_a_timed_out_focus_command_does_not_resume_into_the_next_ones_focus(
    cmd: str, hold: str
) -> None:
    """Held past its deadline, the command is answered; once released it mutates nothing more.

    A capture is a mutation too: a screenshot resumed after its deadline
    captured the next command's tab (and spent Chrome's capture quota).
    """
    out = _run_focus_race(cmd, hold)
    assert out["replies"]["1"]["ok"] is False
    assert "exceeded its 0.025s deadline" in out["replies"]["1"]["error"]
    assert out["replies"]["2"]["ok"] is True
    # Nothing after the next command opened its tab: no activation, focus
    # or capture of the old tab.
    assert out["calls"][out["calls"].index("open:2"):] == ["open:2"]
    assert out["active"] == 2


def test_type_text_with_an_empty_string_clears_the_field() -> None:
    """A15-8: ``type_text`` with ``''`` reaches the page (it replaces the value with nothing)."""
    out = _run_focus_race(
        "type_text", "", args={"tabId": 1, "uid": 4, "text": ""}, timeout_ms=1000,
    )
    assert out["replies"]["1"] == {"id": 1, "ok": True, "result": "ran"}
    assert [c for c in out["calls"] if c.startswith("script:")][0].endswith('(4, "", null)')
    # A missing text is still refused before the page is touched.
    out = _run_focus_race("type_text", "", args={"tabId": 1, "uid": 4}, timeout_ms=1000)
    assert out["replies"]["1"]["ok"] is False
    assert "text" in out["replies"]["1"]["error"]
    assert not any(c.startswith("script:") for c in out["calls"])
