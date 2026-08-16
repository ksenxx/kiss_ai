// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Every client must mint the SAME tab id for the same run_parallel
// sub-agent.
//
// Tab ids are global across clients: the shared tab registry mirrors
// one tab to every window.  Before this invariant existed, each client
// answered the sub-agent's `new_tab` broadcast by minting its own
// RANDOM tab id and resuming the sub-agent on it.  With two or more
// clients connected (the VS Code extension next to the remote webapp),
// the daemon then announced one `openSubagentTab` per client-minted id,
// every client "deduped" the other clients' ids via retagSubagentTab ->
// api.closeTab(oldId), and the daemon's cleanup_tab() unsubscribed
// EVERY id from the sub-agent's event stream: the sub-agent tabs froze
// on the head of the transcript (system prompt + prompt, nothing else).
//
// The fix: media/main.js mints the deterministic id
// `${parentTabId}__sub_${taskId}` (the daemon's own replay-id
// convention from VSCodeServer._open_persisted_subagent_tabs), so all
// clients converge on ONE shared tab id — no retag, no closeTab, one
// live subscription.
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
  loadWebviewScripts(win);
  const sock = sockets[0];
  assert.ok(sock, 'the webapp shim must open a WebSocket');
  sock.onopen();
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

function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

/** Boot a client, start the parent task, and return the parent tab id. */
function bootParent(client) {
  const posted = client.drain ? client.drain() : client.posted;
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'the client must post ready with a tabId');
  const parentId = ready.tabId;
  send(client.win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
  });
  send(client.win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 1'])},
  });
  return parentId;
}

function all(client) {
  return client.drain ? client.drain() : client.posted;
}

// ---------------------------------------------------------------------------
// 1. new_tab must resume the sub-agent on the deterministic shared id.
// ---------------------------------------------------------------------------

function testDeterministicId(makeClient) {
  const client = makeClient();
  const parentId = bootParent(client);
  const taskId = 'sub-task-det-1';

  send(client.win, {
    type: 'new_tab',
    task_id: taskId,
    parent_tab_id: parentId,
    taskId: '',
  });

  const resume = all(client)
    .filter(m => m.type === 'resumeSession' && m.taskId === taskId)
    .pop();
  assert.ok(resume, 'new_tab must make the client resume the sub-agent');
  assert.strictEqual(
    resume.tabId,
    parentId + '__sub_' + taskId,
    client.name +
      ': the sub-agent must be resumed on the deterministic ' +
      'shared tab id, not a client-local random id',
  );
  const tabs = subagentTabEls(client.win);
  assert.strictEqual(tabs.length, 1, 'exactly one sub-agent tab');
  assert.strictEqual(
    tabs[0].dataset.tabId,
    parentId + '__sub_' + taskId,
    client.name + ': the open tab must carry the deterministic id',
  );
  console.log('PASS deterministic new_tab id (' + client.name + ')');
}

// ---------------------------------------------------------------------------
// 2. Another client's announcement of the SAME deterministic id must not
//    trigger the retag/closeTab dance that used to kill the event
//    stream for every client.
// ---------------------------------------------------------------------------

function testNoCloseTabOnConvergedAnnouncement(makeClient) {
  const client = makeClient();
  const parentId = bootParent(client);
  const taskId = 'sub-task-det-2';
  const sharedId = parentId + '__sub_' + taskId;

  send(client.win, {
    type: 'new_tab',
    task_id: taskId,
    parent_tab_id: parentId,
    taskId: '',
  });
  // The daemon's ready round-trip for the OTHER client announces the
  // same sub-agent.  Both clients minted the same deterministic id, so
  // the announcement names a tab this client already has.
  send(client.win, {
    type: 'openSubagentTab',
    tab_id: sharedId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: taskId,
    taskIndex: 0,
    isSubagentTab: true,
  });
  send(client.win, {
    type: 'task_events',
    tabId: sharedId,
    task: 'sub 1',
    task_id: taskId,
    events: [],
  });

  const closes = all(client).filter(m => m.type === 'closeTab');
  assert.strictEqual(
    closes.length,
    0,
    client.name +
      ': a converged openSubagentTab must not close any tab id ' +
      '(a closeTab here unsubscribes the shared id for EVERY client): ' +
      JSON.stringify(closes),
  );
  const tabs = subagentTabEls(client.win);
  assert.strictEqual(tabs.length, 1, 'still exactly one sub-agent tab');
  assert.strictEqual(tabs[0].dataset.tabId, sharedId);

  // Live fan-out events for the shared id must land in the (single,
  // still-subscribed) tab: the tab keeps a task panel and stays open.
  send(client.win, {
    type: 'tool_call',
    name: 'Bash',
    tabId: sharedId,
    extras: {command: 'echo sub-live'},
  });
  assert.strictEqual(
    subagentTabEls(client.win).length,
    1,
    client.name + ': the sub-agent tab must survive live events',
  );
  console.log('PASS converged announcement keeps stream (' + client.name + ')');
}

// ---------------------------------------------------------------------------
// 3. A parent-less spawn still mints a deterministic (task-keyed) id.
// ---------------------------------------------------------------------------

function testParentlessDeterministicId(makeClient) {
  const client = makeClient();
  const taskId = 'sub-task-det-3';
  send(client.win, {
    type: 'new_tab',
    task_id: taskId,
    parent_tab_id: '',
    taskId: '',
  });
  const resume = all(client)
    .filter(m => m.type === 'resumeSession' && m.taskId === taskId)
    .pop();
  assert.ok(resume, 'a parent-less new_tab must still resume the sub-agent');
  assert.strictEqual(
    resume.tabId,
    'task__sub_' + taskId,
    client.name + ': parent-less spawns must also converge across clients',
  );
  console.log('PASS parent-less deterministic id (' + client.name + ')');
}

for (const makeClient of [makeExtensionClient, makeWebappClient]) {
  testDeterministicId(makeClient);
  testNoCloseTabOnConvergedAnnouncement(makeClient);
  testParentlessDeterministicId(makeClient);
}
console.log('subagentTabDeterministicId.test.js: all tests passed');
// JSDOM windows keep timers alive (running-task UI timer, shim
// reconnect); an assertion failure above throws and exits non-zero, so
// reaching this line means success.
process.exit(0);
