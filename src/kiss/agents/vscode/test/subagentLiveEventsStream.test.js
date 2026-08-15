// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A freshly spawned sub-agent's tab must show the sub-agent's events.
//
// Live sequence under test (the daemon side is pinned by
// test_replay_session_subscribes_running_subagent_no_events.py):
//
//   1. ChatSorcarAgent.run broadcasts `new_tab` with the sub-agent's
//      task id; the client opens a background sub-agent tab and posts
//      `resumeSession {taskId, tabId}`.
//   2. The daemon replays the (possibly empty) transcript to that tab
//      (`openSubagentTab` + `status running` + `task_events`) and
//      subscribes the tab to the sub-agent's live stream.
//   3. Every later sub-agent event is fanned out stamped with the
//      tab's id and the sub-agent's task id.
//
// These tests pin step 3's client half: the streamed events must land
// in the sub-agent tab's transcript -- both while the tab is in the
// background (rendered into its detached fragment) and once the user
// activates it -- and must NOT leak into the parent tab's transcript.
//
// The same assertions run twice: once with the VS Code webview host
// (stub acquireVsCodeApi) and once with the REAL remote-webapp host
// (the _WS_SHIM_JS literal lifted out of src/kiss/server/web_server.py
// driving a fake WebSocket), because the two clients must behave
// identically.

/* global require, __dirname, console */

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

// ---------------------------------------------------------------------------
// host 1: the VS Code webview
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// host 2: the remote webapp, running web_server.py's real shim
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// scenario helpers
// ---------------------------------------------------------------------------

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/**
 * Boot a client, run a one-task fan-out and answer the client's
 * `resumeSession` the way `_replay_session` does for a sub-agent that
 * has not persisted any events yet.
 *
 * @param {Function} makeClient Client factory (extension or webapp).
 * @returns {object} {win, all, parentId, taskId, subTabId}.
 */
function bootLiveSubagent(makeClient) {
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

  const taskId = 'sub-task-live-1';
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

  // The daemon's reply for a just-started sub-agent: openSubagentTab,
  // status running, then the (empty) transcript replay.
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
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    events: [],
    chat_id: 'chat-live',
    extra: '',
  });
  return {win, all, parentId, taskId, subTabId};
}

/**
 * Stream one recognizable run through the sub-agent's live fan-out,
 * exactly as WebPrinter._fanout_stamped emits it: every event carries
 * the sub-agent's taskId plus the subscribed tab's tabId.
 */
function streamSubagentRun(win, subTabId, taskId, marker) {
  const stamp = ev => Object.assign({taskId: taskId, tabId: subTabId}, ev);
  send(win, stamp({type: 'prompt', text: 'sub task body ' + marker}));
  send(win, stamp({type: 'thinking_start'}));
  send(win, stamp({type: 'thinking_delta', text: 'pondering ' + marker}));
  send(win, stamp({type: 'thinking_end'}));
  send(win, stamp({type: 'text_delta', text: 'live-words-' + marker}));
  send(win, stamp({type: 'text_end', text: 'live-words-' + marker}));
  send(
    win,
    stamp({
      type: 'tool_call',
      name: 'Bash',
      tool_input: {command: 'echo ' + marker},
    }),
  );
  send(win, stamp({type: 'tool_result', content: 'tool-output-' + marker}));
}

function activateTab(win, tabId) {
  const el = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"]`,
  );
  assert.ok(el, 'tab ' + tabId + ' must exist in the tab bar');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function outputText(win) {
  const out = win.document.getElementById('output');
  return out ? out.textContent || '' : '';
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

// The reported bug: an agent spawns a sub-task, the client opens the
// sub-agent tab -- and the tab never shows the sub-agent's events.
function testLiveEventsReachTheBackgroundSubagentTab(makeClient, label) {
  const {win, taskId, subTabId} = bootLiveSubagent(makeClient);
  const marker = 'bg';

  // The parent tab is on screen; the sub-agent streams in the background.
  streamSubagentRun(win, subTabId, taskId, marker);

  assert.ok(
    outputText(win).indexOf('live-words-' + marker) === -1,
    label + ': the parent transcript must not show sub-agent output',
  );

  activateTab(win, subTabId);
  const text = outputText(win);
  for (const needle of [
    'live-words-' + marker,
    'pondering ' + marker,
    'tool-output-' + marker,
  ]) {
    assert.ok(
      text.indexOf(needle) !== -1,
      label +
        ': the sub-agent tab must show its streamed events; missing "' +
        needle +
        '" in: ' +
        JSON.stringify(text.slice(0, 400)),
    );
  }
}

// Same stream, but the user is ALREADY standing on the sub-agent tab
// when the events arrive (they clicked it the moment it opened).
function testLiveEventsReachTheActiveSubagentTab(makeClient, label) {
  const {win, taskId, subTabId} = bootLiveSubagent(makeClient);
  const marker = 'fg';

  activateTab(win, subTabId);
  streamSubagentRun(win, subTabId, taskId, marker);

  const text = outputText(win);
  for (const needle of [
    'live-words-' + marker,
    'pondering ' + marker,
    'tool-output-' + marker,
  ]) {
    assert.ok(
      text.indexOf(needle) !== -1,
      label +
        ': the active sub-agent tab must show its streamed events; ' +
        'missing "' +
        needle +
        '" in: ' +
        JSON.stringify(text.slice(0, 400)),
    );
  }
}

// A transcript head replayed by `task_events` and a live tail must
// BOTH be visible: the daemon replays whatever the sub-agent persisted
// before the tab subscribed and streams the rest live.
function testReplayedHeadAndLiveTailBothVisible(makeClient, label) {
  const {win, taskId, subTabId} = bootLiveSubagent(makeClient);

  // A late replay (e.g. the async writer flushed while the round-trip
  // was in flight) re-delivers the head...
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    events: [
      {type: 'prompt', text: 'sub task body head'},
      {type: 'text_delta', text: 'head-words'},
      {type: 'text_end', text: 'head-words'},
    ],
    chat_id: 'chat-live',
    extra: '',
  });
  // ... and the live tail keeps streaming afterwards.
  streamSubagentRun(win, subTabId, taskId, 'tail');

  activateTab(win, subTabId);
  const text = outputText(win);
  for (const needle of ['head-words', 'live-words-tail']) {
    assert.ok(
      text.indexOf(needle) !== -1,
      label +
        ': replayed head and live tail must both render; missing "' +
        needle +
        '" in: ' +
        JSON.stringify(text.slice(0, 400)),
    );
  }
}

const tests = [
  testLiveEventsReachTheBackgroundSubagentTab,
  testLiveEventsReachTheActiveSubagentTab,
  testReplayedHeadAndLiveTailBothVisible,
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
console.log('all subagentLiveEventsStream tests passed');
// pretendToBeVisual keeps a rAF scheduler alive; exit explicitly.
process.exit(0);
