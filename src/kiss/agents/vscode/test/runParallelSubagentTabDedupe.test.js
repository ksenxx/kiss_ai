// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// One sub-agent => at most ONE tab on a client.
//
// The daemon addresses a sub-agent's tab by three different ids over
// the life of a fan-out:
//
//   * the live fan-out id minted by ChatSorcarAgent._run_tasks_parallel
//     ("task-<parentTaskId>__sub_<idx>"),
//   * the deterministic replay id minted by
//     VSCodeServer._open_persisted_subagent_tabs
//     ("<parentTabId>__sub_<subTaskId>"), and
//   * whatever tab id the webview itself asked to resume the sub-agent
//     on (media/main.js mints a fresh one when a collapsed
//     run_parallel panel is expanded again).
//
// All three name the SAME sub-agent -- the sub-agent's task id is the
// only stable identity.  A client that keys tabs on the tab id alone
// therefore stacks a second (third, ...) tab for one sub-agent.  These
// tests pin the invariant: whatever id the daemon uses, a sub-agent
// task never occupies more than one open tab, and the run_parallel
// panel's collapsed state still decides whether that one tab exists.
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
  // Complete the shim's real handshake: onopen makes it send `auth`,
  // and `auth_ok` marks the socket authenticated and flushes the
  // commands the app queued while it was connecting.
  sock.onopen();
  const authFrame = JSON.parse(sock.sent[0]);
  assert.strictEqual(
    authFrame.type,
    'auth',
    'the shim must authenticate before sending commands',
  );
  sock.onmessage({data: JSON.stringify({type: 'auth_ok'})});
  const posted = [];
  // Drain the socket into the same shape the extension test uses.  The
  // shim buffers pre-auth frames, so read everything it has sent so
  // far each time the assertions look at `posted`.
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

function runParallelPanel(win) {
  const headers = win.document.querySelectorAll('#output .ev.tc .tc-h');
  for (const h of headers) {
    const txt = (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt.startsWith('run_parallel')) return h.closest('.ev.tc');
  }
  return null;
}

function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

function togglePanel(win, panel) {
  panel
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/**
 * Boot a client and run a fan-out of *declared* tasks of which the
 * first *n* have already spawned (declared defaults to n).
 */
function bootFanOut(makeClient, n, declared) {
  const client = makeClient();
  const {win} = client;
  const posted = client.drain ? client.drain() : client.posted;
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'the client must post ready with a tabId');
  const parentId = ready.tabId;

  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
  });
  const names = [];
  const declaredCount = declared === undefined ? n : declared;
  for (let i = 0; i < declaredCount; i++) names.push('sub ' + (i + 1));
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(names)},
  });
  const panel = runParallelPanel(win);
  assert.ok(panel, 'run_parallel must render a tool-call panel');

  const taskIds = [];
  const liveTabIds = [];
  for (let i = 0; i < n; i++) {
    const taskId = 'sub-task-' + (i + 1);
    taskIds.push(taskId);
    send(win, {
      type: 'new_tab',
      task_id: taskId,
      parent_tab_id: parentId,
      taskId: '',
    });
    const all = client.drain ? client.drain() : client.posted;
    const resume = all
      .filter(m => m.type === 'resumeSession' && m.taskId === taskId)
      .pop();
    assert.ok(resume, 'new_tab must make the client resume the sub-agent');
    liveTabIds.push(resume.tabId);
    send(win, {
      type: 'openSubagentTab',
      tab_id: resume.tabId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
    });
    // The daemon always follows openSubagentTab with the sub-agent's
    // transcript, which is what binds the tab to its task id.
    send(win, {
      type: 'task_events',
      tabId: resume.tabId,
      task: 'sub ' + (i + 1),
      task_id: taskId,
      events: [],
    });
  }
  assert.strictEqual(
    subagentTabEls(win).length,
    n,
    'each spawned sub-agent must get exactly one tab',
  );
  return Object.assign({}, client, {
    parentId,
    panel,
    taskIds,
    liveTabIds,
    all: () => (client.drain ? client.drain() : client.posted),
  });
}

/** Every open sub-agent tab id, in tab-bar order. */
function openSubTabIds(win) {
  return subagentTabEls(win).map(el => el.dataset.tabId);
}

function assertOneTabPerSubagent(win, taskIds, where) {
  assert.strictEqual(
    subagentTabEls(win).length,
    taskIds.length,
    'DUPLICATE SUB-AGENT TABS (' +
      where +
      '): ' +
      taskIds.length +
      ' sub-agents but ' +
      subagentTabEls(win).length +
      ' open sub-agent tabs: ' +
      JSON.stringify(openSubTabIds(win)),
  );
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

// The parent tab is replayed (history click / reconnect) while the
// fan-out's tabs are open.  _open_persisted_subagent_tabs addresses
// each sub-agent by its deterministic replay id, which differs from
// the live id the tabs already carry.
function testPersistedReplayIdsDoNotDuplicate(makeClient, label) {
  const {win, parentId, taskIds, liveTabIds} = bootFanOut(makeClient, 2);

  taskIds.forEach((taskId, i) => {
    const replayId = parentId + '__sub_' + taskId;
    assert.notStrictEqual(replayId, liveTabIds[i], 'ids must differ');
    send(win, {
      type: 'openSubagentTab',
      tab_id: replayId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
      isDone: false,
    });
    send(win, {
      type: 'task_events',
      tabId: replayId,
      task: 'sub ' + (i + 1),
      task_id: taskId,
      events: [],
    });
  });

  assertOneTabPerSubagent(win, taskIds, label + ': persisted replay ids');
  win.close();
  console.log('  ok - [' + label + '] persisted replay ids reuse one tab');
}

// Collapse (tabs close) then expand (the webview mints fresh tab ids
// and resumes) and only then does the daemon's replay burst arrive
// under the deterministic ids.
function testReplayAfterExpandDoesNotDuplicate(makeClient, label) {
  const scenario = bootFanOut(makeClient, 2);
  const {win, panel, parentId, taskIds} = scenario;

  togglePanel(win, panel);
  assert.strictEqual(subagentTabEls(win).length, 0, 'collapse closes tabs');
  togglePanel(win, panel);
  assert.strictEqual(subagentTabEls(win).length, 2, 'expand reopens tabs');

  const reopenedIds = openSubTabIds(win);
  taskIds.forEach((taskId, i) => {
    const replayId = parentId + '__sub_' + taskId;
    assert.ok(
      !reopenedIds.includes(replayId),
      'sanity: the reopened tab uses a client-minted id',
    );
    send(win, {
      type: 'openSubagentTab',
      tab_id: replayId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
    });
    send(win, {
      type: 'task_events',
      tabId: replayId,
      task: 'sub ' + (i + 1),
      task_id: taskId,
      events: [],
    });
  });

  assertOneTabPerSubagent(win, taskIds, label + ': replay after expand');
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the panel must stay uncollapsed while its sub-agent tabs are open',
  );
  win.close();
  console.log('  ok - [' + label + '] replay after expand reuses one tab');
}

// A sub-agent spawned while the panel is collapsed is remembered, and
// a later duplicate `new_tab` for a sub-agent that already has a tab
// must not open a second one.
function testRepeatedNewTabDoesNotDuplicate(makeClient, label) {
  const scenario = bootFanOut(makeClient, 2);
  const {win, parentId, taskIds} = scenario;

  for (const taskId of taskIds) {
    send(win, {
      type: 'new_tab',
      task_id: taskId,
      parent_tab_id: parentId,
      taskId: '',
    });
  }
  assertOneTabPerSubagent(win, taskIds, label + ': repeated new_tab');
  win.close();
  console.log('  ok - [' + label + '] a repeated new_tab reuses one tab');
}

// Expanding a panel that learned about the same sub-agent twice (once
// as a live tab, once while collapsed) must still open one tab each.
function testExpandAfterMixedRegistrationDoesNotDuplicate(makeClient, label) {
  const scenario = bootFanOut(makeClient, 2);
  const {win, panel, parentId, taskIds} = scenario;

  togglePanel(win, panel);
  assert.strictEqual(subagentTabEls(win).length, 0, 'collapse closes tabs');
  // While collapsed the daemon keeps addressing the sub-agents, under
  // both id schemes.
  taskIds.forEach((taskId, i) => {
    send(win, {
      type: 'new_tab',
      task_id: taskId,
      parent_tab_id: parentId,
      taskId: '',
    });
    send(win, {
      type: 'openSubagentTab',
      tab_id: parentId + '__sub_' + taskId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
    });
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'nothing may open a tab while the run_parallel panel is collapsed',
  );

  togglePanel(win, panel);
  assertOneTabPerSubagent(win, taskIds, label + ': expand after mixed regs');
  win.close();
  console.log('  ok - [' + label + '] mixed registrations expand to one tab');
}

// The daemon may re-announce a live sub-agent tab under a THIRD id
// (e.g. the parent tab is reopened in a new window and replays), while
// the panel is collapsed: still no tab, and expanding yields one each.
function testCollapsedStaysClosedAcrossIdSchemes(makeClient, label) {
  const scenario = bootFanOut(makeClient, 2);
  const {win, panel, parentId, taskIds} = scenario;

  togglePanel(win, panel);
  taskIds.forEach((taskId, i) => {
    send(win, {
      type: 'openSubagentTab',
      tab_id: 'task-parent__sub_' + i,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
    });
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'a collapsed run_parallel panel must keep every sub-agent tab closed',
  );

  togglePanel(win, panel);
  assertOneTabPerSubagent(win, taskIds, label + ': expand after 3rd id');
  win.close();
  console.log('  ok - [' + label + '] collapsed panel ignores every id form');
}

// Closing one sub-agent tab by hand keeps the siblings open (lenient
// manual close), and the daemon re-announcing the closed sub-agent
// under any id must not resurrect it or duplicate a sibling.
function testManualCloseThenReplayStaysClosed(makeClient, label) {
  const scenario = bootFanOut(makeClient, 3);
  const {win, panel, parentId, taskIds, liveTabIds} = scenario;

  win.document
    .querySelector(
      `#tab-list .chat-tab[data-tab-id="${liveTabIds[0]}"] .chat-tab-close`,
    )
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'closing one sub-agent tab must leave its two siblings open',
  );

  taskIds.forEach((taskId, i) => {
    send(win, {
      type: 'openSubagentTab',
      tab_id: parentId + '__sub_' + taskId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
    });
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'DUPLICATE/RESURRECTED SUB-AGENT TABS (' +
      label +
      '): expected the two surviving sub-agent tabs, got ' +
      JSON.stringify(openSubTabIds(win)),
  );
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the panel stays uncollapsed while sub-agent tabs are open',
  );
  win.close();
  console.log('  ok - [' + label + '] hand-closed sub tab is never revived');
}

// Nothing above may weaken the collapse/expand contract itself.
function testCollapseExpandContract(makeClient, label) {
  const scenario = bootFanOut(makeClient, 2);
  const {win, panel, taskIds} = scenario;
  const all = scenario.all;

  const before = all().length;
  togglePanel(win, panel);
  assert.ok(panel.classList.contains('collapsed'), 'panel collapses');
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'a collapsed run_parallel panel must have no open sub-agent tabs',
  );
  const closes = all()
    .slice(before)
    .filter(m => m.type === 'closeTab')
    .map(m => m.tabId);
  for (const id of scenario.liveTabIds) {
    assert.ok(
      closes.includes(id),
      'the host must be told to close sub-agent tab ' + id,
    );
  }

  const beforeExpand = all().length;
  togglePanel(win, panel);
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'an uncollapsed run_parallel panel must show every sub-agent tab',
  );
  for (const taskId of taskIds) {
    assert.ok(
      all()
        .slice(beforeExpand)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      'each reopened sub-agent tab must resume its task ' + taskId,
    );
  }
  win.close();
  console.log('  ok - [' + label + '] collapse/expand contract holds');
}

// A sub-agent tab the user closed by hand must stay closed even when the
// daemon re-delivers the spawn itself (`new_tab`), not just an
// `openSubagentTab` announcement.
function testNewTabDoesNotRevivHandClosedSubagent(makeClient, label) {
  const scenario = bootFanOut(makeClient, 2);
  const {win, panel, parentId, taskIds, liveTabIds} = scenario;

  win.document
    .querySelector(
      `#tab-list .chat-tab[data-tab-id="${liveTabIds[0]}"] .chat-tab-close`,
    )
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'closing one of two sub-agent tabs must leave the sibling open',
  );

  send(win, {
    type: 'new_tab',
    task_id: taskIds[0],
    parent_tab_id: parentId,
    taskId: '',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'RESURRECTED SUB-AGENT TAB (' +
      label +
      '): a re-delivered new_tab reopened the sub-agent tab the user ' +
      'closed by hand: ' +
      JSON.stringify(openSubTabIds(win)),
  );
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the panel must stay uncollapsed while a sibling tab is open',
  );

  // Collapsing and expanding is the way back: it reopens the whole
  // fan-out, hand-closed sub-agents included.
  togglePanel(win, panel);
  togglePanel(win, panel);
  assertOneTabPerSubagent(win, taskIds, label + ': expand after hand close');
  win.close();
  console.log('  ok - [' + label + '] new_tab cannot revive a closed sub tab');
}

// Two run_parallel calls in one task: a sub-agent spawned late belongs
// to the call that requested it, so a collapsed FIRST panel must keep
// its sub-agents tabless even while the second panel is uncollapsed.
function testLateSpawnHonoursItsOwnPanel(makeClient, label) {
  // The first call fanned out to three tasks but only two have started.
  const scenario = bootFanOut(makeClient, 2, 3);
  const {win, panel, parentId} = scenario;
  const all = scenario.all;

  togglePanel(win, panel);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'collapsing the first fan-out closes its sub-agent tabs',
  );

  // A second run_parallel call renders a second, uncollapsed panel.
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 3'])},
  });
  const panels = Array.from(
    win.document.querySelectorAll('#output .ev.tc.tc-run-parallel'),
  );
  assert.strictEqual(panels.length, 2, 'the task must show two fan-outs');
  const second = panels[1];
  assert.ok(
    !second.classList.contains('collapsed'),
    'the second fan-out starts uncollapsed',
  );

  const before = all().length;
  // The first fan-out's third sub-agent starts only now.
  send(win, {
    type: 'new_tab',
    task_id: 'sub-task-late',
    parent_tab_id: parentId,
    taskId: '',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'INVARIANT VIOLATED (' +
      label +
      '): a sub-agent spawned into a fan-out whose panel is collapsed ' +
      'opened a tab: ' +
      JSON.stringify(openSubTabIds(win)),
  );
  assert.ok(
    !all()
      .slice(before)
      .some(m => m.type === 'resumeSession'),
    'no sub-agent may be resumed while its own panel is collapsed',
  );

  // The second fan-out's own sub-agent does get a tab.
  send(win, {
    type: 'new_tab',
    task_id: 'sub-task-3',
    parent_tab_id: parentId,
    taskId: '',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'the uncollapsed second fan-out must open its own sub-agent tab',
  );
  win.close();
  console.log('  ok - [' + label + '] a late spawn honours its own panel');
}

// Renaming a sub-agent's tab must tell the host to release the retired
// id (the daemon opened the new id as an extra viewer of the same
// sub-agent, so the old one would otherwise keep host state alive).
function testRetagReleasesTheOldTabId(makeClient, label) {
  const scenario = bootFanOut(makeClient, 1);
  const {win, parentId, taskIds, liveTabIds} = scenario;
  const all = scenario.all;

  const before = all().length;
  const replayId = parentId + '__sub_' + taskIds[0];
  send(win, {
    type: 'openSubagentTab',
    tab_id: replayId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: taskIds[0],
    taskIndex: 0,
    isSubagentTab: true,
  });
  assert.deepStrictEqual(
    openSubTabIds(win),
    [replayId],
    'the sub-agent must move onto the id the daemon now uses',
  );
  assert.ok(
    all()
      .slice(before)
      .some(m => m.type === 'closeTab' && m.tabId === liveTabIds[0]),
    'STALE HOST STATE (' +
      label +
      '): the host was never told to release the retired sub-agent tab ' +
      'id ' +
      liveTabIds[0],
  );
  assert.ok(
    !all()
      .slice(before)
      .some(m => m.type === 'closeTab' && m.tabId === replayId),
    'the surviving sub-agent tab must not be closed',
  );
  win.close();
  console.log('  ok - [' + label + '] a retag releases the retired tab id');
}

// The host is told which CHAT tab is on screen even while the user
// looks at a content tab (it decides which chat may take over the
// editor).  Renaming that chat tab must re-report it, or the host keeps
// naming a tab that no longer exists.
function testRetagReReportsTheChatTabBehindAContentTab(makeClient, label) {
  const scenario = bootFanOut(makeClient, 1);
  const {win, parentId, taskIds, liveTabIds} = scenario;
  const all = scenario.all;

  // Stand on the sub-agent tab: the host now names it as the chat on
  // screen ...
  win.document
    .querySelector(`#tab-list .chat-tab[data-tab-id="${liveTabIds[0]}"]`)
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    all().some(m => m.type === 'activeTabChanged' && m.tabId === liveTabIds[0]),
    'sanity: activating the sub-agent tab reports it to the host',
  );
  // ... and keeps naming it while a content tab is on screen.
  send(win, {
    type: 'fileContent',
    name: 'note.txt',
    path: 'note.txt',
    content: 'hello',
  });
  const contentTab = win.document.querySelector(
    '#tab-list .chat-tab.content-tab',
  );
  assert.ok(contentTab, 'sanity: fileContent must open a content tab');

  const before = all().length;
  const replayId = parentId + '__sub_' + taskIds[0];
  send(win, {
    type: 'openSubagentTab',
    tab_id: replayId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: taskIds[0],
    taskIndex: 0,
    isSubagentTab: true,
  });
  assert.deepStrictEqual(
    openSubTabIds(win),
    [replayId],
    'the sub-agent must move onto the id the daemon now uses',
  );
  assert.ok(
    all()
      .slice(before)
      .some(m => m.type === 'activeTabChanged' && m.tabId === replayId),
    'STALE CHAT TAB REPORTED (' +
      label +
      '): after renaming the sub-agent tab the host still names the ' +
      'retired id ' +
      liveTabIds[0],
  );
  win.close();
  console.log('  ok - [' + label + '] a retag re-reports the chat tab');
}

const SCENARIOS = [
  testPersistedReplayIdsDoNotDuplicate,
  testReplayAfterExpandDoesNotDuplicate,
  testRepeatedNewTabDoesNotDuplicate,
  testExpandAfterMixedRegistrationDoesNotDuplicate,
  testCollapsedStaysClosedAcrossIdSchemes,
  testManualCloseThenReplayStaysClosed,
  testNewTabDoesNotRevivHandClosedSubagent,
  testLateSpawnHonoursItsOwnPanel,
  testRetagReleasesTheOldTabId,
  testRetagReReportsTheChatTabBehindAContentTab,
  testCollapseExpandContract,
];

const HOSTS = [
  [makeExtensionClient, 'extension'],
  [makeWebappClient, 'webapp'],
];

function main() {
  for (const [makeClient, label] of HOSTS) {
    for (const scenario of SCENARIOS) scenario(makeClient, label);
  }
  console.log('runParallelSubagentTabDedupe.test.js: all tests passed');
}

main();
