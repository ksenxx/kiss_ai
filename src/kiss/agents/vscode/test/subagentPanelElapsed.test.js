// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Sub-agent tabs must show the per-panel time-spent label agent tabs
// show, in the bottom-right corner of every event panel.
//
// A sub-agent tab's transcript arrives in two ways:
//
//   1. live events streamed after the tab subscribed
//      (processOutputEventForBgTab), which stamp panels with the wall
//      clock exactly like the visible tab's stream, and
//   2. a `task_events` replay (the tab was opened or re-opened
//      mid-run, e.g. by expanding a collapsed run_parallel panel),
//      whose panels can NOT be stamped with the render-time clock --
//      that would measure the replay, not the tool.
//
// The replay path therefore derives every closed panel's duration from
// the persisted event timestamps (tool_result.ts - tool_call.ts), and
// hands the still-open panels of a still-RUNNING task back to the live
// ticker so they keep counting from their own event's wall-clock
// start. A finished task's unclosed panels (an interrupted tool call)
// keep no label: no duration exists for them. These tests pin all of
// that, on both clients (VS Code webview and remote webapp), plus the
// guard rails: closing events without a usable timestamp render
// nothing, and an adjacent-task replay never ticks even while its
// owner tab is running.

/* global require, console, process, __dirname */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const WEB_SERVER_PY = path.resolve(
  __dirname,
  '..',
  '..',
  '..',
  'server',
  'web_server.py',
);

function chatHtml() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  return html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
}

function newDom() {
  const dom = new JSDOM(chatHtml(), {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  return win;
}

function loadWebviewScripts(win) {
  for (const f of ['panelCopy.js', 'api.js', 'main.js']) {
    win.eval(fs.readFileSync(path.join(MEDIA, f), 'utf8'));
  }
}

function makeExtensionClient() {
  const win = newDom();
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  loadWebviewScripts(win);
  return {win, posted, name: 'extension webview'};
}

function readShimJs() {
  const src = fs.readFileSync(WEB_SERVER_PY, 'utf8');
  const m = src.match(/_WS_SHIM_JS\s*=\s*r"""([\s\S]*?)"""/);
  assert.ok(m, 'could not locate _WS_SHIM_JS literal in web_server.py');
  return m[1];
}

function installFakeWebSocket(win, sockets) {
  function FakeWebSocket(url) {
    this.url = url;
    this.readyState = FakeWebSocket.OPEN;
    this.sent = [];
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
    sockets.push(this);
  }
  FakeWebSocket.CONNECTING = 0;
  FakeWebSocket.OPEN = 1;
  FakeWebSocket.CLOSING = 2;
  FakeWebSocket.CLOSED = 3;
  FakeWebSocket.prototype.send = function (data) {
    this.sent.push(data);
  };
  FakeWebSocket.prototype.close = function () {
    this.readyState = FakeWebSocket.CLOSED;
    if (typeof this.onclose === 'function') this.onclose();
  };
  win.WebSocket = FakeWebSocket;
}

function makeWebappClient() {
  const win = newDom();
  const sockets = [];
  installFakeWebSocket(win, sockets);
  win.eval(readShimJs());
  assert.strictEqual(
    typeof win.acquireVsCodeApi,
    'function',
    "web_server.py's shim must define acquireVsCodeApi()",
  );
  loadWebviewScripts(win);
  const sock = sockets[0];
  assert.ok(sock, 'the webapp shim must open a WebSocket');
  sock.onopen();
  const authFrame = JSON.parse(sock.sent[0]);
  assert.strictEqual(
    authFrame.type,
    'auth',
    'the shim must authenticate before sending commands',
  );
  sock.onmessage({data: JSON.stringify({type: 'auth_ok'})});
  const posted = [];
  const drain = () => {
    while (posted.length < sock.sent.length) {
      posted.push(JSON.parse(sock.sent[posted.length]));
    }
    return posted;
  };
  return {win, posted, drain, sock, name: 'remote webapp'};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function activateTab(win, tabId) {
  const el = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"]`,
  );
  assert.ok(el, 'tab ' + tabId + ' must exist in the tab bar');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/**
 * The `.panel-elapsed` label of *panel*'s own footer bar, or null.
 *
 * @param {Element} panel The event panel.
 * @returns {?Element} The label element.
 */
function elapsedLabel(panel) {
  return panel.querySelector(':scope > .panel-time > .panel-elapsed');
}

/**
 * Parse a `.panel-elapsed` label ("450ms", "3.5s", "1m 2.0s") to ms.
 *
 * @param {string} text The label text.
 * @returns {number} Milliseconds.
 */
function parseElapsedMs(text) {
  let m = /^(\d+)ms$/.exec(text);
  if (m) return Number(m[1]);
  m = /^(\d+(?:\.\d+)?)s$/.exec(text);
  if (m) return Number(m[1]) * 1000;
  m = /^(\d+)m (\d+(?:\.\d+)?)s$/.exec(text);
  if (m) return Number(m[1]) * 60000 + Number(m[2]) * 1000;
  assert.fail('unparsable elapsed label: ' + JSON.stringify(text));
  return 0;
}

/**
 * Boot a client, run a one-task fan-out and capture the sub-agent tab
 * id the client minted when it resumed the sub-agent's session.
 *
 * @param {Function} makeClient Client factory (extension or webapp).
 * @returns {object} {win, all, parentId, taskId, subTabId}.
 */
function bootSubagent(makeClient) {
  const client = makeClient();
  const {win} = client;
  const all = () => (client.drain ? client.drain() : client.posted);
  const ready = all().find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'the client must post ready with a tabId');
  const parentId = ready.tabId;

  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
  });
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 1'])},
  });

  const taskId = 'sub-task-elapsed-1';
  send(win, {
    type: 'new_tab',
    task_id: taskId,
    parent_tab_id: parentId,
    taskId: '',
  });
  const resume = all()
    .filter(m => m.type === 'resumeSession' && m.taskId === taskId)
    .pop();
  assert.ok(resume, 'new_tab must make the client resume the sub-agent');
  const subTabId = resume.tabId;

  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: taskId,
    isSubagentTab: true,
    isDone: false,
  });
  send(win, {
    type: 'status',
    running: true,
    tabId: subTabId,
    startTs: Date.now(),
  });
  return {win, all, parentId, taskId, subTabId};
}

// A sub-agent transcript replayed mid-run (`task_events` for a RUNNING
// sub-agent tab): closed panels show the timestamp-derived duration,
// and the still-open trailing tool_call resumes the live tick from its
// own event's start -- then freezes when its live tool_result lands.
function testReplayedSubagentPanelsShowDurations(makeClient, label) {
  const {win, taskId, subTabId} = bootSubagent(makeClient);
  const t0 = Date.now() - 60000;
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    chat_id: 'chat-elapsed',
    extra: '',
    events: [
      {type: 'prompt', text: 'sub task body', ts: t0},
      {type: 'thinking_start', ts: t0 + 1000},
      {type: 'thinking_delta', text: 'pondering'},
      {type: 'thinking_end'},
      {type: 'tool_call', name: 'Bash', command: 'echo one', ts: t0 + 4500},
      {type: 'tool_result', content: 'one', is_error: false, ts: t0 + 7000},
      {type: 'tool_call', name: 'Bash', command: 'echo two', ts: t0 + 9000},
    ],
  });

  activateTab(win, subTabId);
  const out = win.document.getElementById('output');

  const thoughts = out.querySelector('.llm-panel');
  assert.ok(thoughts, label + ': replayed Thoughts panel exists');
  const thLabel = elapsedLabel(thoughts);
  assert.ok(thLabel, label + ': replayed Thoughts panel has a duration');
  assert.strictEqual(
    thLabel.textContent,
    '3.5s',
    label + ': Thoughts duration = tool_call.ts - thinking_start.ts',
  );

  const tcs = out.querySelectorAll('.ev.tc');
  assert.strictEqual(tcs.length, 2, label + ': both tool_call panels exist');
  const closedLabel = elapsedLabel(tcs[0]);
  assert.ok(closedLabel, label + ': closed tool_call has a duration');
  assert.strictEqual(
    closedLabel.textContent,
    '2.5s',
    label + ': closed tool_call duration = tool_result.ts - tool_call.ts',
  );

  // The trailing tool_call has not reported back: it must be TICKING
  // from its own event's wall-clock start (~51s ago), not from the
  // moment the replay rendered it.
  const openTc = tcs[1];
  const openLabel = elapsedLabel(openTc);
  assert.ok(openLabel, label + ': the open tool_call shows a live elapsed');
  const openMs = parseElapsedMs(openLabel.textContent);
  assert.ok(
    openMs >= 50000 && openMs < 70000,
    label +
      ': the open tool_call counts from its event start (~51s), got ' +
      openLabel.textContent,
  );
  assert.ok(
    openTc.dataset.startMs,
    label + ': the open tool_call joined the live tick',
  );
  assert.ok(
    !openTc.dataset.timeDone,
    label + ': the open tool_call is not frozen yet',
  );

  // Its live tool_result freezes the label at the true total duration:
  // the RESULT EVENT's own timestamp closes the panel, so a slow
  // delivery (here: the result stamped 2.5s after the call but
  // delivered ~51s later) must not inflate the duration.
  send(win, {
    type: 'tool_result',
    content: 'two',
    is_error: false,
    taskId: taskId,
    tabId: subTabId,
    ts: t0 + 9000 + 2500,
  });
  assert.strictEqual(
    openTc.dataset.timeDone,
    '1',
    label + ': the live tool_result freezes the replayed panel',
  );
  assert.strictEqual(
    elapsedLabel(openTc).textContent,
    '2.5s',
    label + ": the frozen duration is the events' own span, not delivery time",
  );

  const promptPanel = out.querySelector('.ev.prompt');
  assert.ok(promptPanel, label + ': replayed prompt panel exists');
  assert.strictEqual(
    elapsedLabel(promptPanel),
    null,
    label + ': a prompt panel never carries a duration',
  );
}

// Live-streamed sub-agent panels (the pre-existing behavior, pinned):
// panels stamped while the events stream into the background tab carry
// the same elapsed label the visible tab's panels do.
function testLiveSubagentPanelsShowElapsed(makeClient, label) {
  const {win, taskId, subTabId} = bootSubagent(makeClient);
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    chat_id: 'chat-elapsed-live',
    extra: '',
    events: [],
  });
  const stamp = ev => Object.assign({taskId: taskId, tabId: subTabId}, ev);
  send(win, stamp({type: 'prompt', text: 'sub task body', ts: Date.now()}));
  send(win, stamp({type: 'thinking_start', ts: Date.now()}));
  send(win, stamp({type: 'thinking_delta', text: 'pondering'}));
  send(win, stamp({type: 'thinking_end'}));
  send(win, stamp({type: 'tool_call', name: 'Bash', command: 'echo hi'}));
  send(win, stamp({type: 'tool_result', content: 'hi', is_error: false}));

  activateTab(win, subTabId);
  const out = win.document.getElementById('output');
  const tc = out.querySelector('.ev.tc');
  assert.ok(tc, label + ': live tool_call panel exists');
  const tcLabel = elapsedLabel(tc);
  assert.ok(tcLabel, label + ': live sub-agent tool_call shows elapsed');
  assert.ok(
    parseElapsedMs(tcLabel.textContent) < 60000,
    label + ': live elapsed is wall-clock small, got ' + tcLabel.textContent,
  );
  assert.strictEqual(tc.dataset.timeDone, '1', label + ': frozen on result');
  const thoughts = out.querySelector('.llm-panel');
  assert.ok(
    elapsedLabel(thoughts),
    label + ': live sub-agent Thoughts panel shows elapsed',
  );
}

// A FINISHED task's replay: closed panels get their historical
// durations, but the interrupted trailing tool_call must NOT tick --
// no event ever closed it, so no duration exists.
function testFinishedReplayOpenPanelDoesNotTick(makeClient, label) {
  const client = makeClient();
  const {win} = client;
  const all = () => (client.drain ? client.drain() : client.posted);
  const tabId = all().find(m => m.type === 'ready').tabId;
  const t0 = new Date(2021, 2, 5, 14, 7).getTime();
  send(win, {
    type: 'task_events',
    tabId: tabId,
    chat_id: 'chat-finished',
    task: 'old task',
    events: [
      {type: 'prompt', text: 'old prompt', ts: t0},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: t0 + 2000},
      {type: 'tool_result', content: 'ok', is_error: false, ts: t0 + 5500},
      {type: 'tool_call', name: 'Bash', command: 'oops', ts: t0 + 6000},
    ],
  });
  const out = win.document.getElementById('output');
  const tcs = out.querySelectorAll('.ev.tc');
  assert.strictEqual(tcs.length, 2, label + ': both tool_call panels exist');
  assert.strictEqual(
    elapsedLabel(tcs[0]).textContent,
    '3.5s',
    label + ': the finished tool_call keeps its historical duration',
  );
  assert.strictEqual(
    elapsedLabel(tcs[1]),
    null,
    label + ': an interrupted tool_call shows no duration',
  );
  assert.ok(
    !tcs[1].dataset.startMs,
    label + ': a finished replay puts nothing on the live tick',
  );
}

// A closing event with no usable timestamp renders nothing, and the
// panel must still be sealed: a later promotion pass may not put a
// CLOSED panel on the live tick.
function testCloseWithoutTimestampSealsWithoutLabel(makeClient, label) {
  const {win, taskId, subTabId} = bootSubagent(makeClient);
  const t0 = Date.now() - 30000;
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    chat_id: 'chat-nots',
    extra: '',
    events: [
      {type: 'prompt', text: 'sub task body', ts: t0},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: t0 + 1000},
      {type: 'tool_result', content: 'ok', is_error: false},
      {type: 'tool_call', name: 'Read', path: '/tmp/x'},
    ],
  });
  activateTab(win, subTabId);
  const out = win.document.getElementById('output');
  const tcs = out.querySelectorAll('.ev.tc');
  assert.strictEqual(tcs.length, 2, label + ': both tool_call panels exist');
  assert.strictEqual(
    elapsedLabel(tcs[0]),
    null,
    label + ': a tool_result without ts renders no duration',
  );
  assert.strictEqual(
    tcs[0].dataset.timeDone,
    '1',
    label + ': the closed panel is sealed all the same',
  );
  assert.ok(
    !tcs[0].dataset.startMs,
    label + ': the sealed panel must not join the live tick',
  );
  // The trailing tool_call carried no ts either: nothing to count from.
  assert.strictEqual(
    elapsedLabel(tcs[1]),
    null,
    label + ': an open tool_call without ts cannot tick',
  );
  assert.ok(
    !tcs[1].dataset.startMs && !tcs[1].dataset.startTs,
    label + ': no start was recorded for a ts-less replayed panel',
  );
}

// An adjacent-task replay is always a NEIGHBOURING task's finished
// transcript: its open panels never join the live tick, even though
// the owner tab is running its own live task.
function testAdjacentReplayNeverTicks(makeClient, label) {
  const client = makeClient();
  const {win} = client;
  const all = () => (client.drain ? client.drain() : client.posted);
  const tabId = all().find(m => m.type === 'ready').tabId;
  send(win, {type: 'status', running: true, tabId: tabId, startTs: Date.now()});
  send(win, {type: 'prompt', text: 'live task', tabId: tabId, taskId: '77'});
  const t0 = Date.now() - 40000;
  send(win, {
    type: 'adjacent_task_events',
    tabId: tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [
      {type: 'prompt', text: 'older prompt', ts: t0},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: t0 + 1000},
      {type: 'tool_result', content: 'ok', is_error: false, ts: t0 + 3000},
      {type: 'tool_call', name: 'Bash', command: 'oops', ts: t0 + 5000},
    ],
  });
  const out = win.document.getElementById('output');
  const adj = out.querySelector('.adjacent-task');
  assert.ok(adj, label + ': the adjacent task container rendered');
  const tcs = adj.querySelectorAll('.ev.tc');
  assert.strictEqual(tcs.length, 2, label + ': adjacent tool_calls exist');
  assert.strictEqual(
    elapsedLabel(tcs[0]).textContent,
    '2.0s',
    label + ': the adjacent closed tool_call keeps its duration',
  );
  assert.strictEqual(
    adj.querySelectorAll('[data-start-ms]').length,
    0,
    label + ': an adjacent replay must never join the live tick',
  );
  assert.strictEqual(
    elapsedLabel(tcs[1]),
    null,
    label + ': the adjacent open tool_call shows no duration',
  );
}

// A terminal event (task_done here; task_error / task_stopped /
// task_interrupted share the same seal) freezes every panel the
// transcript still has open — a replay-promoted tool_call included —
// at the task's own end timestamp, and the sealed panel can never
// rejoin the live tick.
function testTerminalEventSealsOpenPanels(makeClient, label) {
  const client = makeClient();
  const {win} = client;
  const all = () => (client.drain ? client.drain() : client.posted);
  const tabId = all().find(m => m.type === 'ready').tabId;
  const t0 = Date.now() - 60000;
  send(win, {type: 'status', running: true, tabId: tabId, startTs: t0});
  send(win, {
    type: 'task_events',
    tabId: tabId,
    chat_id: 'chat-terminal',
    task: 'running task',
    task_id: '91',
    events: [
      {type: 'prompt', text: 'go', ts: t0},
      {type: 'tool_call', name: 'Bash', command: 'sleep 999', ts: t0},
    ],
  });
  const out = win.document.getElementById('output');
  const tc = out.querySelector('.ev.tc');
  assert.ok(tc, label + ': the open tool_call panel exists');
  assert.ok(tc.dataset.startMs, label + ': the open panel was promoted');
  send(win, {
    type: 'task_done',
    success: true,
    tabId: tabId,
    startTs: t0,
    endTs: t0 + 12000,
  });
  assert.strictEqual(
    tc.dataset.timeDone,
    '1',
    label + ': task_done seals the open panel',
  );
  assert.strictEqual(
    elapsedLabel(tc).textContent,
    '12.0s',
    label + ": the seal uses the task's own end timestamp",
  );
}

// The daemon emits no tool_result for the `finish` tool — the `result`
// event is its close, so `result` must freeze the finish panel's label.
function testResultSealsFinishPanel(makeClient, label) {
  const client = makeClient();
  const {win} = client;
  const all = () => (client.drain ? client.drain() : client.posted);
  const tabId = all().find(m => m.type === 'ready').tabId;
  send(win, {type: 'status', running: true, tabId: tabId, startTs: Date.now()});
  send(win, {
    type: 'tool_call',
    name: 'finish',
    tabId: tabId,
    ts: Date.now(),
  });
  send(win, {
    type: 'result',
    summary: 'done',
    total_tokens: 1,
    cost: '$0',
    tabId: tabId,
    ts: Date.now(),
  });
  const out = win.document.getElementById('output');
  const tc = out.querySelector('.ev.tc');
  assert.ok(tc, label + ': the finish tool_call panel exists');
  assert.strictEqual(
    tc.dataset.timeDone,
    '1',
    label + ': the result event seals the finish panel',
  );
  assert.ok(
    elapsedLabel(tc),
    label + ': the sealed finish panel keeps its elapsed label',
  );
}

// The replay-only report suppression must not survive into the live
// stream the replay's tail state becomes: a report the task writes
// AFTER its transcript was replayed opens like any live report, while
// a report inside the replayed transcript still opens nothing.
function testReplayDoesNotSuppressLiveReports(makeClient, label) {
  const client = makeClient();
  const {win} = client;
  const all = () => (client.drain ? client.drain() : client.posted);
  const tabId = all().find(m => m.type === 'ready').tabId;
  const t0 = Date.now() - 30000;
  const oldDoc = '<!DOCTYPE html><html><body>old</body></html>';
  send(win, {type: 'status', running: true, tabId: tabId, startTs: t0});
  send(win, {
    type: 'task_events',
    tabId: tabId,
    chat_id: 'chat-report',
    task: 'reporting task',
    task_id: '92',
    events: [
      {type: 'prompt', text: 'write reports', ts: t0},
      {
        type: 'tool_call',
        name: 'Write',
        path: 'reports/old.html',
        content: oldDoc,
        ts: t0 + 1000,
      },
      {
        type: 'tool_result',
        content:
          'Successfully wrote ' +
          oldDoc.length +
          ' characters to reports/old.html',
        is_error: false,
        tool_name: 'Write',
        path: 'reports/old.html',
        ts: t0 + 2000,
      },
    ],
  });
  assert.strictEqual(
    win.document.querySelectorAll('#tab-list .chat-tab.content-tab').length,
    0,
    label + ': a replayed report opens no content tab',
  );
  const liveDoc = '<!DOCTYPE html><html><body>live</body></html>';
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/live.html',
    content: liveDoc,
    tabId: tabId,
    ts: Date.now(),
  });
  send(win, {
    type: 'tool_result',
    content:
      'Successfully wrote ' +
      liveDoc.length +
      ' characters to reports/live.html',
    is_error: false,
    tool_name: 'Write',
    path: 'reports/live.html',
    tabId: tabId,
    ts: Date.now(),
  });
  send(win, {type: 'task_done', success: true, tabId: tabId});
  assert.strictEqual(
    win.document.querySelectorAll('#tab-list .chat-tab.content-tab').length,
    1,
    label + ': the live report written after the replay opens its tab',
  );
}

const tests = [
  testReplayedSubagentPanelsShowDurations,
  testLiveSubagentPanelsShowElapsed,
  testFinishedReplayOpenPanelDoesNotTick,
  testCloseWithoutTimestampSealsWithoutLabel,
  testAdjacentReplayNeverTicks,
  testTerminalEventSealsOpenPanels,
  testResultSealsFinishPanel,
  testReplayDoesNotSuppressLiveReports,
];

let failures = 0;
for (const makeClient of [makeExtensionClient, makeWebappClient]) {
  const label =
    makeClient === makeExtensionClient ? 'extension webview' : 'remote webapp';
  for (const t of tests) {
    try {
      t(makeClient, label);
      console.log('ok - ' + t.name + ' (' + label + ')');
    } catch (e) {
      failures += 1;
      console.error('FAIL - ' + t.name + ' (' + label + ')');
      console.error(e && e.stack ? e.stack : String(e));
    }
  }
}
if (failures > 0) {
  console.error(failures + ' failure(s)');
  process.exit(1);
}
console.log('all subagentPanelElapsed tests passed');
// pretendToBeVisual keeps a rAF scheduler alive; exit explicitly.
process.exit(0);
