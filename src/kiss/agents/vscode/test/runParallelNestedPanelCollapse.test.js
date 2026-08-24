// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Invariant under test (chat webview, media/main.js -- shared verbatim by
// the VS Code extension webview and the remote webapp):
//
//   While the run_parallel tool event panel is COLLAPSED, every tab of
//   the sub-agents that tool created MUST be closed.
//
// runParallelPanelTabsSync.test.js covers the panel's OWN chevron. This
// suite covers the panels that collapse a run_parallel panel by
// swallowing it: the `summary` tool panel adopts the event panels that
// precede it into a `.summary-sub` child and collapses itself, and
// `.tc.collapsed > :not(.tc-h, .panel-copy-btn) {display:none}`
// (media/main.css) then hides the adopted run_parallel panel. The
// sub-agent tabs of a run_parallel panel the user can no longer see --
// let alone reach the chevron of -- must be closed just like those of a
// panel collapsed by hand.
//
// Every test runs twice: once against the VS Code extension host
// (`acquireVsCodeApi` stub) and once against the remote webapp (the real
// `_WS_SHIM_JS` from src/kiss/server/web_server.py driven over a stub
// WebSocket), because the invariant must hold on both surfaces.

/* global require, __dirname, console, process */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM, VirtualConsole} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const WEB_SERVER_PY = path.join(
  __dirname,
  '..',
  '..',
  '..',
  'server',
  'web_server.py',
);

/**
 * The remote webapp's `acquireVsCodeApi` shim, sliced out of its Python
 * host module so the test always runs the shipped source.
 *
 * @returns {string} The shim JavaScript source.
 */
function webappShimSource() {
  const py = fs.readFileSync(WEB_SERVER_PY, 'utf8');
  const marker = '_WS_SHIM_JS = r"""';
  const at = py.indexOf(marker);
  assert.ok(at >= 0, 'web_server.py must define _WS_SHIM_JS');
  const from = at + marker.length;
  const to = py.indexOf('\n"""', from);
  assert.ok(to > from, '_WS_SHIM_JS must be a closed triple-quoted string');
  return py.slice(from, to);
}

/**
 * Install a WebSocket stub on *win* that records every frame the shim
 * sends and hands the live socket to the caller.
 *
 * @param {object} win The jsdom window.
 * @param {Array<string>} frames Array that receives sent frames.
 * @returns {function(): object} Getter for the current socket stub.
 */
function installWebSocketStub(win, frames) {
  let socket = null;
  class WebSocketStub {
    constructor() {
      this.readyState = WebSocketStub.OPEN;
      this.onopen = null;
      this.onmessage = null;
      this.onclose = null;
      this.onerror = null;
      socket = this;
    }
    send(data) {
      frames.push(data);
    }
    close() {
      this.readyState = WebSocketStub.CLOSED;
    }
  }
  WebSocketStub.CONNECTING = 0;
  WebSocketStub.OPEN = 1;
  WebSocketStub.CLOSING = 2;
  WebSocketStub.CLOSED = 3;
  win.WebSocket = WebSocketStub;
  return () => socket;
}

/**
 * Boot the chat webview in jsdom.
 *
 * @param {string} mode 'extension' or 'webapp'.
 * @param {boolean} quiet Swallow page errors, for the scenario that
 *     feeds the webview a broken transcript on purpose.
 * @returns {object} {win, posted, deliver} for the booted webview.
 */
function makeWebview(mode, quiet) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const domOpts = {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  };
  // A VirtualConsole nobody listens to keeps the deliberate page error
  // of the broken-transcript scenario out of the test output.
  if (quiet) domOpts.virtualConsole = new VirtualConsole();
  const dom = new JSDOM(html, domOpts);
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let deliver;
  if (mode === 'webapp') {
    const frames = [];
    const currentSocket = installWebSocketStub(win, frames);
    win.eval(webappShimSource());
    const ws = currentSocket();
    assert.ok(ws, 'the webapp shim must open a WebSocket on load');
    ws.onopen();
    assert.deepStrictEqual(
      JSON.parse(frames.shift()),
      {type: 'auth', password: ''},
      'the webapp shim must start with an auth handshake',
    );
    ws.onmessage({data: JSON.stringify({type: 'auth_ok'})});
    frames.length = 0;
    // Every command the app sends now leaves as a JSON frame on the
    // socket; mirror them into `posted` so both modes assert alike.
    const drain = () => {
      while (frames.length) posted.push(JSON.parse(frames.shift()));
    };
    deliver = data => {
      currentSocket().onmessage({data: JSON.stringify(data)});
      drain();
    };
    win.__drain = drain;
  } else {
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
    deliver = data => {
      win.dispatchEvent(new win.MessageEvent('message', {data}));
    };
  }

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  // The real parser fires DOMContentLoaded once every body script has
  // run; the webapp shim defers app-bound events until then so none
  // are lost while main.js is still being fetched.  jsdom keeps
  // readyState 'loading' until a later macrotask, so fire it here to
  // flush the shim's queue synchronously (harmless in extension mode).
  win.document.dispatchEvent(new win.Event('DOMContentLoaded', {bubbles: true}));
  if (win.__drain) win.__drain();

  return {win, posted, deliver};
}

function runParallelPanel(win) {
  const headers = win.document.querySelectorAll('#output .ev.tc .tc-h');
  for (const h of headers) {
    const txt = (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt.startsWith('run_parallel')) return h.closest('.ev.tc');
  }
  return null;
}

function summaryPanel(win) {
  return win.document.querySelector('#output .ev.tc.tc-summary');
}

function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

function togglePanel(win, panel, drain) {
  panel
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  if (drain) drain();
}

/**
 * Start a task that fanned out *n* sub-agents through run_parallel, each
 * with its own open sub-agent tab.
 *
 * @param {string} mode 'extension' or 'webapp'.
 * @param {number} n How many sub-agents to spawn.
 * @param {boolean} [quiet] Swallow deliberate page errors.
 * @returns {object} Boot state for the assertions.
 */
function bootParallelRun(mode, n, quiet) {
  const {win, posted, deliver} = makeWebview(mode, quiet);
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const parentId = ready.tabId;
  const drain = win.__drain || null;

  deliver({type: 'status', running: true, tabId: parentId, startTs: Date.now()});
  const taskNames = [];
  for (let i = 0; i < n; i++) taskNames.push('sub ' + (i + 1));
  deliver({
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(taskNames)},
  });
  const panel = runParallelPanel(win);
  assert.ok(panel, 'run_parallel tool_call must render a .ev.tc panel');

  const taskIds = [];
  const subTabIds = [];
  for (let i = 0; i < n; i++) {
    const taskId = 'sub-task-' + (i + 1);
    taskIds.push(taskId);
    const before = posted.length;
    deliver({
      type: 'new_tab',
      task_id: taskId,
      parent_tab_id: parentId,
      taskId: '',
    });
    const resume = posted
      .slice(before)
      .find(m => m.type === 'resumeSession' && m.taskId === taskId);
    assert.ok(resume, 'new_tab must make the webview post resumeSession');
    subTabIds.push(resume.tabId);
    deliver({
      type: 'openSubagentTab',
      tab_id: resume.tabId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
    });
  }
  assert.strictEqual(
    subagentTabEls(win).length,
    n,
    'each spawned sub-agent must get its own tab',
  );
  return {win, posted, deliver, drain, parentId, panel, taskIds, subTabIds};
}

/**
 * Make the agent call the `summary` tool, which swallows every event
 * panel before it -- the live run_parallel panel included.
 *
 * @param {object} st Boot state from bootParallelRun.
 */
function sendSummary(st) {
  st.deliver({
    type: 'tool_call',
    name: 'summary',
    tabId: st.parentId,
    description: 'progress so far',
  });
  const sum = summaryPanel(st.win);
  assert.ok(sum, 'the summary tool must render a .tc-summary panel');
  assert.ok(
    sum.classList.contains('collapsed'),
    'the summary panel must render collapsed',
  );
  assert.ok(
    st.panel.closest('.summary-sub'),
    'the summary panel must adopt the preceding run_parallel panel',
  );
  return sum;
}

/**
 * Open a fresh chat through the tab bar's "+" button -- the control both
 * surfaces share -- pushing the current chat into the background.
 *
 * @param {object} st Boot state from bootParallelRun.
 */
function openNewChat(st) {
  const addBtn = st.win.document.querySelector('#tab-bar .chat-tab-add');
  assert.ok(addBtn, 'the tab bar must offer a "+" new-chat button');
  addBtn.dispatchEvent(new st.win.MouseEvent('click', {bubbles: true}));
  if (st.drain) st.drain();
  const active = st.win.document.querySelector('#tab-list .chat-tab.active');
  assert.ok(
    active && active.dataset.tabId !== st.parentId,
    'the "+" button must open and activate a new chat tab',
  );
}

/**
 * Bring the summary-nested fan-out back on screen with its sub-agent
 * tabs open, whatever state the summary adoption left it in.
 *
 * @param {object} st Boot state from bootParallelRun.
 * @param {Element} sum The summary panel that adopted the fan-out.
 */
function openFanOutInsideSummary(st, sum) {
  if (sum.classList.contains('collapsed')) togglePanel(st.win, sum, st.drain);
  assert.ok(!sum.classList.contains('collapsed'), 'summary panel expanded');
  if (st.panel.classList.contains('collapsed'))
    togglePanel(st.win, st.panel, st.drain);
  assert.ok(
    !st.panel.classList.contains('collapsed'),
    'the nested run_parallel panel must be expanded',
  );
  assert.strictEqual(
    subagentTabEls(st.win).length,
    2,
    'sanity: the fan-out is on screen inside the expanded summary, so ' +
      'its sub-agent tabs must be open',
  );
}

// A run_parallel panel swallowed by a collapsed summary panel is off
// screen: its sub-agent tabs must go with it.
function testSummaryAdoptionClosesSubagentTabs(mode) {
  const st = bootParallelRun(mode, 2);
  sendSummary(st);

  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'INVARIANT VIOLATED (' +
      mode +
      '): the summary panel collapsed the run_parallel panel into its ' +
      'hidden .summary-sub, but the fan-out sub-agent tabs are still open',
  );
  assert.ok(
    st.panel.classList.contains('collapsed'),
    'a run_parallel panel hidden inside a collapsed panel must itself ' +
      'be marked collapsed, so its chevron can reopen the fan-out',
  );
  for (const id of st.subTabIds) {
    assert.ok(
      st.posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close sub-agent tab ' + id,
    );
  }
  st.win.close();
  console.log('  ok [' + mode + '] summary adoption closes sub-agent tabs');
}

// Reopening the fan-out from inside an expanded summary must work, and
// re-collapsing the summary must close those tabs again.
function testReCollapsingSummaryClosesReopenedTabs(mode) {
  const st = bootParallelRun(mode, 2);
  const sum = sendSummary(st);

  togglePanel(st.win, sum, st.drain);
  assert.ok(!sum.classList.contains('collapsed'), 'summary panel expanded');
  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'expanding the summary panel alone must not reopen the fan-out: the ' +
      'nested run_parallel panel is still collapsed',
  );

  const beforeResume = st.posted.length;
  togglePanel(st.win, st.panel, st.drain);
  assert.strictEqual(
    subagentTabEls(st.win).length,
    2,
    'expanding the nested run_parallel panel must reopen its sub-agent tabs',
  );
  for (const taskId of st.taskIds) {
    assert.ok(
      st.posted
        .slice(beforeResume)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      'a reopened sub-agent tab must resume backend task ' + taskId,
    );
  }

  const reopenedIds = subagentTabEls(st.win).map(el => el.dataset.tabId);
  togglePanel(st.win, sum, st.drain);
  assert.ok(sum.classList.contains('collapsed'), 'summary panel re-collapsed');
  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'INVARIANT VIOLATED (' +
      mode +
      '): re-collapsing the summary panel hid the run_parallel panel ' +
      'again while its sub-agent tabs stayed open',
  );
  for (const id of reopenedIds) {
    assert.ok(
      st.posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close reopened sub-agent tab ' + id,
    );
  }
  st.win.close();
  console.log('  ok [' + mode + '] re-collapsing the summary closes sub tabs');
}

// A sub-agent that the daemon announces after the summary swallowed its
// fan-out panel must not open a tab.
function testSpawnAfterAdoptionOpensNoTab(mode) {
  const st = bootParallelRun(mode, 2);
  sendSummary(st);

  const before = st.posted.length;
  st.deliver({
    type: 'new_tab',
    task_id: 'sub-task-3',
    parent_tab_id: st.parentId,
    taskId: '',
  });
  st.deliver({
    type: 'openSubagentTab',
    tab_id: st.parentId + '__sub_sub-task-3',
    parent_tab_id: st.parentId,
    description: 'sub 3',
    task_id: 'sub-task-3',
    taskIndex: 2,
  });
  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'INVARIANT VIOLATED (' +
      mode +
      '): a sub-agent spawned after the summary hid its run_parallel ' +
      'panel opened a tab',
  );
  assert.ok(
    !st.posted.slice(before).some(m => m.type === 'resumeSession'),
    'no resumeSession may be posted while the fan-out panel is hidden',
  );

  const beforeExpand = st.posted.length;
  togglePanel(st.win, summaryPanel(st.win), st.drain);
  togglePanel(st.win, st.panel, st.drain);
  assert.strictEqual(
    subagentTabEls(st.win).length,
    3,
    'expanding the nested run_parallel panel must open the deferred tab too',
  );
  assert.ok(
    st.posted
      .slice(beforeExpand)
      .some(m => m.type === 'resumeSession' && m.taskId === 'sub-task-3'),
    'the deferred sub-agent must be resumed when the panel is expanded',
  );
  st.win.close();
  console.log('  ok [' + mode + '] spawn after adoption opens no tab');
}

// The task-end collapse pass must not skip a run_parallel panel just
// because a summary panel adopted it.
function testTaskEndClosesNestedSubagentTabs(mode) {
  const st = bootParallelRun(mode, 2);
  openFanOutInsideSummary(st, sendSummary(st));

  st.deliver({
    type: 'tool_call',
    name: 'finish',
    tabId: st.parentId,
    extras: {summary: 'done'},
  });
  st.deliver({
    type: 'result',
    tabId: st.parentId,
    summary: 'done',
    success: true,
  });
  st.deliver({type: 'status', running: false, tabId: st.parentId});
  st.deliver({type: 'usage_info', tabId: st.parentId});

  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'INVARIANT VIOLATED (' +
      mode +
      '): the task-end collapse pass left the fan-out tabs of a ' +
      'summary-nested run_parallel panel open',
  );
  assert.ok(
    st.panel.classList.contains('collapsed'),
    'the task-end pass must collapse the nested run_parallel panel',
  );
  st.win.close();
  console.log('  ok [' + mode + '] task end closes nested fan-out tabs');
}

// The parent chat's task can also end while the user is reading another
// chat: the background-tab collapse pass (collapseAllExceptResult over
// the tab's detached fragment) must close a summary-nested fan-out too,
// even though it deliberately leaves a live fan-out panel alone -- the
// summary it hides behind has already gone off screen.
function testBackgroundTaskEndClosesNestedSubagentTabs(mode) {
  const st = bootParallelRun(mode, 2);
  openFanOutInsideSummary(st, sendSummary(st));

  // The user opens a fresh chat with the tab bar's "+" button (the one
  // control both surfaces share), so the fan-out's parent chat -- and
  // the panels of this test -- move into a background tab.
  openNewChat(st);
  assert.ok(
    !st.win.document.getElementById('output').contains(st.panel),
    'sanity: the fan-out panels must have left #output for the ' +
      "background tab's detached fragment",
  );
  assert.strictEqual(
    subagentTabEls(st.win).length,
    2,
    'opening a new chat must not disturb the fan-out tabs',
  );

  st.deliver({
    type: 'result',
    tabId: st.parentId,
    summary: 'done',
    success: true,
  });

  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'INVARIANT VIOLATED (' +
      mode +
      '): the background-tab collapse pass hid the summary that owns ' +
      'the fan-out panel but left its sub-agent tabs open',
  );
  for (const id of st.subTabIds) {
    assert.ok(
      st.posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close sub-agent tab ' + id,
    );
  }
  st.win.close();
  console.log('  ok [' + mode + '] background task end closes nested tabs');
}

// The daemon re-sends a whole transcript (task_events) whenever it
// replays a task. For a chat the user is not looking at, the replay
// renders into a fresh fragment that replaces the tab's own: the
// finished run_parallel panel of that replacement is collapsed by the
// replay's collapse pass, so it must hand its sub-agent tabs in -- the
// panel that owned them no longer exists.
function testBackgroundReplayCollapseClosesSubagentTabs(mode) {
  const st = bootParallelRun(mode, 2);
  openNewChat(st);

  st.deliver({
    type: 'task_events',
    tabId: st.parentId,
    task: 'parent replay',
    task_id: 'parent-task',
    events: [
      {
        type: 'tool_call',
        name: 'run_parallel',
        tabId: st.parentId,
        extras: {tasks: JSON.stringify(['sub 1', 'sub 2'])},
      },
      {type: 'tool_result', tabId: st.parentId, content: 'sub-agents done'},
      {type: 'result', tabId: st.parentId, summary: 'done', success: true},
    ],
  });

  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'INVARIANT VIOLATED (' +
      mode +
      "): replaying a background chat's transcript collapsed its " +
      'replacement run_parallel panel but left the sub-agent tabs of ' +
      'the panel it replaced open',
  );
  for (const id of st.subTabIds) {
    assert.ok(
      st.posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close sub-agent tab ' + id,
    );
  }
  st.win.close();
  console.log('  ok [' + mode + '] background replay collapse closes sub tabs');
}

// Collapsing a fan-out panel closes its sub-agent tabs, and closing a
// sub-agent tab takes that sub-agent's OWN fan-out tabs with it. Those
// grandchildren must be forgotten as thoroughly as the children: a
// later announcement for one of them may not resurrect a tab whose
// owning panel went away with its chat.
function testCollapseForgetsGrandchildSubagentTabs(mode) {
  const st = bootParallelRun(mode, 1);
  const childId = st.subTabIds[0];

  // The sub-agent fans out itself: its run_parallel panel and the
  // grandchild's tab both belong to the sub-agent's own chat.
  st.deliver({
    type: 'tool_call',
    name: 'run_parallel',
    tabId: childId,
    extras: {tasks: JSON.stringify(['deep'])},
  });
  let before = st.posted.length;
  st.deliver({
    type: 'new_tab',
    task_id: 'deep-task',
    parent_tab_id: childId,
    taskId: '',
  });
  const deepResume = st.posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === 'deep-task');
  assert.ok(deepResume, "the sub-agent's own fan-out must open a tab");
  const grandchildId = deepResume.tabId;
  assert.strictEqual(
    subagentTabEls(st.win).length,
    2,
    'sanity: the sub-agent and its own sub-agent both have tabs',
  );

  togglePanel(st.win, st.panel, st.drain);
  assert.ok(st.panel.classList.contains('collapsed'), 'panel collapsed');
  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'collapsing the fan-out must close the sub-agent tab and the ' +
      'grandchild tab that hangs off it',
  );
  for (const id of [childId, grandchildId]) {
    assert.ok(
      st.posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close tab ' + id,
    );
  }

  before = st.posted.length;
  togglePanel(st.win, st.panel, st.drain);
  const reopened = subagentTabEls(st.win).map(el => el.dataset.tabId);
  assert.strictEqual(
    reopened.length,
    1,
    'expanding the fan-out must reopen its own sub-agent only -- the ' +
      "grandchild belongs to that sub-agent's transcript, which has " +
      'not been replayed yet (open tabs: ' +
      JSON.stringify(reopened) +
      ')',
  );
  const newChildId = reopened[0];

  // The daemon still has the grandchild's old tab id in flight.
  st.deliver({
    type: 'openSubagentTab',
    tab_id: grandchildId,
    parent_tab_id: newChildId,
    description: 'deep',
    task_id: 'deep-task',
    taskIndex: 0,
  });
  const after = subagentTabEls(st.win).map(el => el.dataset.tabId);
  assert.deepStrictEqual(
    after,
    [newChildId],
    'INVARIANT VIOLATED (' +
      mode +
      '): a late announcement resurrected a grandchild sub-agent tab ' +
      'whose run_parallel panel went away when its chat was closed ' +
      '(open tabs: ' +
      JSON.stringify(after) +
      ')',
  );
  st.win.close();
  console.log('  ok [' + mode + '] collapse forgets grandchild sub tabs');
}

// Starting a new task in a chat ("clear") or resetting it to the welcome
// screen throws that chat's whole transcript away -- the fan-out panel
// included. Sub-agent tabs must not outlive the panel that owned them:
// nothing would be left to close or reopen them.
function testTranscriptWipeClosesSubagentTabs(mode) {
  for (const wipe of ['clear', 'showWelcome']) {
    for (const where of ['active', 'background']) {
      const st = bootParallelRun(mode, 2);
      const label = wipe + '/' + where + ' (' + mode + ')';
      if (where === 'background') openNewChat(st);

      st.deliver({type: wipe, tabId: st.parentId, chat_id: 7});

      assert.strictEqual(
        subagentTabEls(st.win).length,
        0,
        'INVARIANT VIOLATED ' +
          label +
          ': the run_parallel panel was thrown away with the ' +
          'transcript but its sub-agent tabs are still open',
      );
      for (const id of st.subTabIds) {
        assert.ok(
          st.posted.some(m => m.type === 'closeTab' && m.tabId === id),
          label + ': the backend must be told to close sub-agent tab ' + id,
        );
      }
      // A second wipe finds no transcript left and must be harmless.
      st.deliver({type: wipe, tabId: st.parentId, chat_id: 7});
      assert.strictEqual(
        subagentTabEls(st.win).length,
        0,
        label + ': a repeated wipe must keep the sub-agent tabs closed',
      );
      st.win.close();
    }
  }
  console.log('  ok [' + mode + '] transcript wipe closes sub-agent tabs');
}

// Closing sub-agent tabs is what a collapse does, and a replay may
// collapse a fan-out whose sub-agent tab the user is looking at right
// now. Closing the tab on screen moves the user to the parent chat, so
// the closes must wait until the replayed transcript is complete --
// otherwise the parent is shown a half-written transcript and the rest
// of the replay is appended to it after it left the screen.
function testReplayWhileViewingSubagentKeepsWholeTranscript(mode) {
  const st = bootParallelRun(mode, 2);

  const subTabEl = st.win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${st.subTabIds[0]}"]`,
  );
  assert.ok(subTabEl, 'the sub-agent tab must be in the tab bar');
  subTabEl.dispatchEvent(new st.win.MouseEvent('click', {bubbles: true}));
  if (st.drain) st.drain();
  assert.strictEqual(
    st.win.document.querySelector('#tab-list .chat-tab.active').dataset.tabId,
    st.subTabIds[0],
    'the sub-agent tab must be the one on screen',
  );

  st.deliver({
    type: 'task_events',
    tabId: st.parentId,
    task: 'parent replay',
    task_id: 'parent-task',
    events: [
      {
        type: 'tool_call',
        name: 'run_parallel',
        tabId: st.parentId,
        extras: {tasks: JSON.stringify(['sub 1', 'sub 2'])},
      },
      {type: 'tool_result', tabId: st.parentId, content: 'sub-agents done'},
      {
        type: 'tool_call',
        name: 'summary',
        tabId: st.parentId,
        description: 'what happened',
      },
      {
        type: 'tool_call',
        name: 'finish',
        tabId: st.parentId,
        extras: {summary: 'done'},
      },
      {type: 'result', tabId: st.parentId, summary: 'all done', success: true},
    ],
  });

  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'the replayed summary hid the fan-out panel, so its sub-agent tabs ' +
      'must be closed',
  );
  const active = st.win.document.querySelector('#tab-list .chat-tab.active');
  assert.strictEqual(
    active.dataset.tabId,
    st.parentId,
    'closing the sub-agent tab on screen must move the user to the ' +
      'parent chat',
  );
  const out = st.win.document.getElementById('output');
  const headers = Array.from(out.querySelectorAll('.ev.tc > .tc-h')).map(h =>
    (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim(),
  );
  assert.ok(
    headers.some(h => h.startsWith('summary')),
    'the replayed transcript on screen must include the summary panel ' +
      '(headers: ' +
      JSON.stringify(headers) +
      ')',
  );
  assert.ok(
    headers.some(h => h.startsWith('finish')),
    'BUG: the events replayed AFTER the summary closed the tab on ' +
      'screen were lost from the transcript (headers: ' +
      JSON.stringify(headers) +
      ')',
  );
  assert.ok(
    out.querySelector('.rc'),
    "BUG: the replayed task's result panel was lost from the transcript",
  );
  st.win.close();
  console.log('  ok [' + mode + '] replay while viewing sub-agent keeps all');
}

// A replay that blows up half way (a malformed transcript) must not
// take the invariant down with it: the chat keeps the transcript it had,
// and later collapses still close their sub-agent tabs at once instead
// of queueing them behind a replay that never finished.
function testFailedReplayDoesNotStrandLaterCloses(mode) {
  const st = bootParallelRun(mode, 2, true);
  openNewChat(st);

  try {
    st.deliver({
      type: 'task_events',
      tabId: st.parentId,
      task: 'broken replay',
      task_id: 'parent-task',
      events: [null],
    });
  } catch (_e) {
    // The webapp shim hands the throw straight back to the caller,
    // while the extension host's listener reports it to the page.
  }
  if (st.drain) st.drain();

  const parentTabEl = st.win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${st.parentId}"]`,
  );
  assert.ok(parentTabEl, 'the parent chat must still be in the tab bar');
  parentTabEl.dispatchEvent(new st.win.MouseEvent('click', {bubbles: true}));
  if (st.drain) st.drain();
  const out = st.win.document.getElementById('output');
  assert.ok(
    out.contains(st.panel),
    'BUG: the failed replay threw the chat\u2019s transcript away',
  );
  assert.strictEqual(
    subagentTabEls(st.win).length,
    2,
    'the failed replay must not have closed anything',
  );

  const before = st.posted.length;
  togglePanel(st.win, st.panel, st.drain);
  assert.strictEqual(
    subagentTabEls(st.win).length,
    0,
    'INVARIANT VIOLATED (' +
      mode +
      '): after a failed replay, collapsing a run_parallel panel no ' +
      'longer closes its sub-agent tabs -- the closes are stuck in the ' +
      "replay's queue",
  );
  for (const id of st.subTabIds) {
    assert.ok(
      st.posted
        .slice(before)
        .some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close sub-agent tab ' + id,
    );
  }
  st.win.close();
  console.log('  ok [' + mode + '] failed replay strands no later closes');
}

// A neighbouring task's transcript replays into the same #output and
// may well contain its own run_parallel + summary pair. Those panels own
// no tab of the conversation on screen, so hiding them must leave this
// task's live fan-out -- panel and tabs alike -- exactly as it was.
function testAdjacentTaskSummaryLeavesLiveFanOutAlone(mode) {
  const st = bootParallelRun(mode, 2);

  st.deliver({
    type: 'adjacent_task_events',
    direction: 'prev',
    task: 'the previous task',
    task_id: 'other-task',
    tabId: st.parentId,
    events: [
      {
        type: 'tool_call',
        name: 'run_parallel',
        tabId: st.parentId,
        extras: {tasks: JSON.stringify(['other sub 1'])},
      },
      {
        type: 'tool_call',
        name: 'summary',
        tabId: st.parentId,
        description: 'what the previous task did',
      },
    ],
  });

  const adjacent = st.win.document.querySelector('#output .adjacent-task');
  assert.ok(adjacent, "the neighbouring task's transcript must render");
  const adjacentFanOut = adjacent.querySelector('.tc-run-parallel');
  assert.ok(
    adjacentFanOut && adjacentFanOut.closest('.summary-sub'),
    "the neighbour's summary must adopt the neighbour's fan-out panel",
  );

  assert.ok(
    !st.panel.classList.contains('collapsed'),
    "a neighbouring task's collapsed summary must not collapse this " +
      "conversation's live run_parallel panel",
  );
  assert.strictEqual(
    subagentTabEls(st.win).length,
    2,
    "BUG: a neighbouring task's replayed transcript closed this " +
      "conversation's live sub-agent tabs",
  );
  for (const id of st.subTabIds) {
    assert.ok(
      !st.posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must NOT be told to close live sub-agent tab ' + id,
    );
  }
  st.win.close();
  console.log('  ok [' + mode + '] adjacent task summary leaves fan-out alone');
}

async function main() {
  const tests = [
    testSummaryAdoptionClosesSubagentTabs,
    testReCollapsingSummaryClosesReopenedTabs,
    testSpawnAfterAdoptionOpensNoTab,
    testTaskEndClosesNestedSubagentTabs,
    testBackgroundTaskEndClosesNestedSubagentTabs,
    testBackgroundReplayCollapseClosesSubagentTabs,
    testCollapseForgetsGrandchildSubagentTabs,
    testTranscriptWipeClosesSubagentTabs,
    testReplayWhileViewingSubagentKeepsWholeTranscript,
    testFailedReplayDoesNotStrandLaterCloses,
    testAdjacentTaskSummaryLeavesLiveFanOutAlone,
  ];
  // RP_ONLY runs a single scenario, which is how each test here was
  // shown to fail on its own before the fix (the suite otherwise stops
  // at the first failure).
  const only = process.env.RP_ONLY || '';
  for (const mode of ['extension', 'webapp']) {
    for (const t of tests) {
      if (only && t.name !== only) continue;
      await t(mode);
    }
  }
  console.log('runParallelNestedPanelCollapse.test.js: all tests passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
