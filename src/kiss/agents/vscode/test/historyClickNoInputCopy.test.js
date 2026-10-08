// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Clicking a row in the Task History sidebar must NEVER copy the task text
// into the chat input textarea (#task-input).  The input belongs to the user:
// it holds their draft for the NEXT prompt.  Writing the historical task text
// there makes the user re-send an old task by accident and silently destroys
// whatever they had typed.
//
// The task panel that opens the transcript (and the tab's label) is the
// correct place to echo the resumed task and must keep working.
//
// media/main.js is served both to the VS Code extension webview and — by
// src/kiss/server/web_server.py, which ships the identical media/chat.html +
// media/main.js — to the remote web app, so every scenario below runs twice:
// once as the extension webview and once as the remote webview
// (<body class="remote-chat">).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  const {remote = false} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) {
    html = html.replace('<body', '<body class="remote-chat"');
  }

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  // The remote app docks the sidebar permanently on wide screens, where
  // closeSidebar() is a no-op by design.  Pin both modes to the narrow
  // (drawer) layout so the SAME sidebar assertions hold for both.
  win.matchMedia = function (query) {
    return {
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function input(win) {
  return win.document.getElementById('task-input');
}

/** The task text the transcript on screen opens with, '' when none. */
function taskPanelText(win) {
  const text = win.document.querySelector('#output .task-panel-text');
  return text ? text.textContent : '';
}

// The open chat tabs (no row renders them any more: the chat on
// screen is the one picked in the Chats panel, the others are hidden).
function chatTabs(win) {
  return win._testApi
    .openTabs()
    .filter(t => !t.isSubagentTab && !t.isContentTab);
}

/** The active tab's label: it names the task the user is looking at. */
function activeTabLabel(win) {
  const active = win._testApi.getActiveTabId();
  const tab = win._testApi.openTabs().find(t => t.id === active);
  return tab ? tab.title : '';
}

function historyRows(win) {
  return Array.from(win.document.querySelectorAll('#history-list .sidebar-item'));
}

function sidebarOpen(win) {
  return win.document.getElementById('sidebar').classList.contains('open');
}

function countMessages(posted, type) {
  return posted.filter(msg => msg && msg.type === type).length;
}

function lastMessage(posted, type) {
  const msgs = posted.filter(msg => msg && msg.type === type);
  return msgs.length ? msgs[msgs.length - 1] : null;
}

function disableWorkspaceFilter(win) {
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  const ws = win.document.getElementById('hf-workspace');
  if (ws && ws.checked) {
    ws.checked = false;
    ws.dispatchEvent(new win.Event('change', {bubbles: true}));
  }
}

function openSidebar(win) {
  const btn = win.document.getElementById('menu-btn');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(sidebarOpen(win), 'burger click must open the history sidebar');
}

function session(overrides) {
  return Object.assign(
    {
      id: 'chat-1',
      task_id: 1,
      title: 'Refactor the payment gateway retries',
      preview: 'Refactor the payment gateway retries',
      has_events: true,
      failed: false,
      is_running: false,
      tokens: 0,
      cost: 0,
      steps: 0,
      is_favorite: false,
      timestamp: 1_700_000_000,
      work_dir: '',
    },
    overrides || {},
  );
}

// renderHistory drops replies whose generation is stale, and opening the
// sidebar bumps it, so always answer the generation the webview last asked
// for.
function sendHistory(win, posted, sessions) {
  const req = lastMessage(posted, 'getHistory');
  assert.ok(req, 'the open sidebar must have requested history');
  send(win, {
    type: 'history',
    offset: 0,
    generation: req.generation,
    sessions,
  });
}

function clickOnlyRow(win) {
  const rows = historyRows(win);
  assert.strictEqual(rows.length, 1, 'exactly one history row must render');
  rows[0].click();
}

// A history row for a chat that is not open lands in a FRESH tab, and
// the idle new chat the user left behind is retired at the same time
// (nothing was running in it and nothing was typed), so it is the only
// chat tab left.
function assertOpenedInFreshTab(win, mode, tabBefore, what) {
  const active = win._testApi.getActiveTabId();
  assert.notStrictEqual(
    active,
    tabBefore,
    `${mode.name}: ${what} must create a new tab`,
  );
  assert.strictEqual(
    chatTabs(win).map(t => t.id).join(','),
    active,
    `${mode.name}: the idle chat left behind is retired`,
  );
}

// --- scenarios -------------------------------------------------------------

// Branch 1: has_events -> resume.  The task text belongs in the read-only task
// panel only; the input must stay empty.
function testResumeBranchLeavesInputEmpty(mode) {
  const {win, posted} = makeWebview({remote: mode.remote});
  disableWorkspaceFilter(win);
  openSidebar(win);

  const tabBefore = win._testApi.getActiveTabId();
  const s = session({id: 'chat-resume', task_id: 7});
  sendHistory(win, posted, [s]);
  clickOnlyRow(win);

  assert.strictEqual(
    input(win).value,
    '',
    `${mode.name}: clicking a resumable history row must NOT copy the task ` +
      `text into the chat input textbox (#task-input) — got ` +
      `${JSON.stringify(input(win).value)}`,
  );
  assert.strictEqual(
    taskPanelText(win),
    s.preview,
    `${mode.name}: the read-only task panel must still show the resumed task`,
  );
  assertOpenedInFreshTab(win, mode, tabBefore, 'resuming an unopened chat');
  const resume = lastMessage(posted, 'resumeSession');
  assert.ok(
    resume,
    `${mode.name}: clicking a resumable row must post resumeSession`,
  );
  assert.strictEqual(resume.id, 'chat-resume');
  assert.strictEqual(resume.taskId, 7);
  assert.strictEqual(
    resume.tabId,
    win._testApi.getActiveTabId(),
    `${mode.name}: the resume targets the fresh tab`,
  );
  assert.ok(!sidebarOpen(win), `${mode.name}: the sidebar must close on click`);

  win.close();
  console.log(`  ok - ${mode.name}: resume branch leaves the input empty`);
}

// Branch 2: no events -> plain new tab.  There is nothing to resume, but the
// row still knows the task text, so it belongs in the read-only task panel —
// and still never in the input.
function testFallbackBranchLeavesInputEmpty(mode) {
  const {win, posted} = makeWebview({remote: mode.remote});
  disableWorkspaceFilter(win);
  openSidebar(win);

  const tabBefore = win._testApi.getActiveTabId();
  const resumeBefore = countMessages(posted, 'resumeSession');
  sendHistory(win, posted, [
    session({
      id: 'chat-empty',
      task_id: 8,
      has_events: false,
      title: 'Never ran anything',
      preview: 'Never ran anything',
    }),
  ]);
  clickOnlyRow(win);

  assert.strictEqual(
    input(win).value,
    '',
    `${mode.name}: clicking an event-less history row must NOT copy the task ` +
      `text into the chat input textbox (#task-input) — got ` +
      `${JSON.stringify(input(win).value)}`,
  );
  assert.strictEqual(
    taskPanelText(win),
    'Never ran anything',
    `${mode.name}: the row still knows the task text, so the read-only task ` +
      `panel must show it even when there is nothing to resume`,
  );
  assertOpenedInFreshTab(win, mode, tabBefore, 'the fallback branch');
  assert.strictEqual(
    countMessages(posted, 'resumeSession'),
    resumeBefore,
    `${mode.name}: an event-less row must not resume anything`,
  );
  assert.ok(!sidebarOpen(win), `${mode.name}: the sidebar must close on click`);

  win.close();
  console.log(`  ok - ${mode.name}: fallback branch leaves the input empty`);
}

// Branch 3: the chat is already open -> just switch tabs, touch nothing else.
function testSwitchBranchLeavesInputEmpty(mode) {
  const {win, posted} = makeWebview({remote: mode.remote});
  disableWorkspaceFilter(win);

  const ready = posted.find(msg => msg && msg.type === 'ready');
  assert.ok(ready && ready.tabId, 'main.js must announce the initial tab id');
  send(win, {
    type: 'task_events',
    tabId: ready.tabId,
    chat_id: 'chat-open',
    task_id: 9,
    task: 'Already open task',
    events: [],
  });
  // The chat's task is still running: "+" leaves it open in the
  // background (an idle chat would be retired on leaving it).
  send(win, {type: 'status', running: true, tabId: ready.tabId});
  win.document.querySelector('#new-chat-btn').click();
  assert.strictEqual(chatTabs(win).length, 2, 'sanity: two tabs are open');
  assert.strictEqual(activeTabLabel(win), 'new chat', 'sanity: new tab active');

  openSidebar(win);
  const resumeBefore = countMessages(posted, 'resumeSession');
  sendHistory(win, posted, [
    session({
      id: 'chat-open',
      task_id: 9,
      title: 'Already open task',
      preview: 'Already open task',
    }),
  ]);
  clickOnlyRow(win);

  assert.strictEqual(
    input(win).value,
    '',
    `${mode.name}: switching to an already-open chat must NOT write the task ` +
      `text into the chat input textbox (#task-input) — got ` +
      `${JSON.stringify(input(win).value)}`,
  );
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    ready.tabId,
    `${mode.name}: the already-open chat tab must become active`,
  );
  assert.strictEqual(
    activeTabLabel(win),
    'Already open task',
    `${mode.name}: the already-open chat tab must become active`,
  );
  // No duplicate tab for the chat; the empty new chat left behind is
  // retired, so the running chat is the only one open.
  assert.strictEqual(
    chatTabs(win).map(t => t.id).join(','),
    ready.tabId,
    `${mode.name}: switching must not create a duplicate tab`,
  );
  assert.strictEqual(
    countMessages(posted, 'resumeSession'),
    resumeBefore,
    `${mode.name}: switching tabs must not resume the session again`,
  );
  assert.ok(!sidebarOpen(win), `${mode.name}: the sidebar must close on click`);

  win.close();
  console.log(`  ok - ${mode.name}: switch branch leaves the input empty`);
}

// The user's half-written prompt is data.  Browsing history must not eat it.
function testResumeBranchPreservesTypedDraft(mode) {
  const {win, posted} = makeWebview({remote: mode.remote});
  disableWorkspaceFilter(win);

  const draft = 'now also add a retry budget of 3 attempts';
  input(win).value = draft;
  input(win).dispatchEvent(new win.Event('input', {bubbles: true}));

  openSidebar(win);
  sendHistory(win, posted, [session({id: 'chat-draft-resume', task_id: 11})]);
  clickOnlyRow(win);

  assert.strictEqual(
    input(win).value,
    draft,
    `${mode.name}: a history click on a resumable row must preserve the ` +
      `draft the user already typed — got ${JSON.stringify(input(win).value)}`,
  );

  win.close();
  console.log(`  ok - ${mode.name}: resume branch preserves a typed draft`);
}

function testFallbackBranchPreservesTypedDraft(mode) {
  const {win, posted} = makeWebview({remote: mode.remote});
  disableWorkspaceFilter(win);

  const draft = 'draft that must survive browsing the history';
  input(win).value = draft;
  input(win).dispatchEvent(new win.Event('input', {bubbles: true}));

  openSidebar(win);
  sendHistory(win, posted, [
    session({
      id: 'chat-draft-empty',
      task_id: 12,
      has_events: false,
      title: 'Nothing ran here',
      preview: 'Nothing ran here',
    }),
  ]);
  clickOnlyRow(win);

  assert.strictEqual(
    input(win).value,
    draft,
    `${mode.name}: a history click on an event-less row must preserve the ` +
      `draft the user already typed — got ${JSON.stringify(input(win).value)}`,
  );
  assert.strictEqual(
    taskPanelText(win),
    'Nothing ran here',
    `${mode.name}: the read-only task panel still shows the row's task text`,
  );

  win.close();
  console.log(`  ok - ${mode.name}: fallback branch preserves a typed draft`);
}

// A task row is broadcast as soon as it is inserted, before its first event is
// persisted, so a *running* row can legitimately arrive with has_events:false
// (src/kiss/server/server.py).  The server's _replay_session reattaches the
// live chat in exactly that case, so such a row must resume — not dump the
// user into an unrelated blank tab.
function testRunningRowWithoutEventsResumes(mode) {
  const {win, posted} = makeWebview({remote: mode.remote});
  disableWorkspaceFilter(win);
  openSidebar(win);

  const tabBefore = win._testApi.getActiveTabId();
  const s = session({
    id: 'chat-running',
    task_id: 21,
    has_events: false,
    is_running: true,
    title: 'Just started, no events yet',
    preview: 'Just started, no events yet',
  });
  sendHistory(win, posted, [s]);
  clickOnlyRow(win);

  const resume = lastMessage(posted, 'resumeSession');
  assert.ok(
    resume,
    `${mode.name}: a running row without persisted events must still post ` +
      `resumeSession so the live chat gets reattached`,
  );
  assert.strictEqual(resume.id, 'chat-running');
  assert.strictEqual(resume.taskId, 21);
  assert.strictEqual(
    taskPanelText(win),
    s.preview,
    `${mode.name}: the read-only task panel must show the running task`,
  );
  assert.strictEqual(
    input(win).value,
    '',
    `${mode.name}: resuming a running row must NOT copy the task text into ` +
      `the chat input textbox (#task-input) — got ` +
      `${JSON.stringify(input(win).value)}`,
  );
  assertOpenedInFreshTab(win, mode, tabBefore, 'resuming a running row');
  assert.ok(!sidebarOpen(win), `${mode.name}: the sidebar must close on click`);

  win.close();
  console.log(`  ok - ${mode.name}: running row without events resumes`);
}

function testRunningRowWithoutEventsPreservesTypedDraft(mode) {
  const {win, posted} = makeWebview({remote: mode.remote});
  disableWorkspaceFilter(win);

  const draft = 'draft typed while a task is still starting up';
  input(win).value = draft;
  input(win).dispatchEvent(new win.Event('input', {bubbles: true}));

  openSidebar(win);
  sendHistory(win, posted, [
    session({
      id: 'chat-running-draft',
      task_id: 22,
      has_events: false,
      is_running: true,
      title: 'Running with a draft in the composer',
      preview: 'Running with a draft in the composer',
    }),
  ]);
  clickOnlyRow(win);

  assert.strictEqual(
    input(win).value,
    draft,
    `${mode.name}: resuming a running row must preserve the draft the user ` +
      `already typed — got ${JSON.stringify(input(win).value)}`,
  );
  const resume = lastMessage(posted, 'resumeSession');
  assert.ok(resume, `${mode.name}: the running row must still resume`);
  assert.strictEqual(resume.id, 'chat-running-draft');
  assert.strictEqual(resume.taskId, 22);

  win.close();
  console.log(`  ok - ${mode.name}: running row keeps a typed draft`);
}

// `s.preview || s.title || ''`: fall through to the title when the preview is
// missing, and to the empty string when both are.
function testTaskTextFallsBackToTitle(mode) {
  const {win, posted} = makeWebview({remote: mode.remote});
  disableWorkspaceFilter(win);
  openSidebar(win);

  sendHistory(win, posted, [
    session({
      id: 'chat-title-only',
      task_id: 31,
      title: 'Only a title survived',
      preview: '',
    }),
  ]);
  clickOnlyRow(win);

  assert.strictEqual(
    taskPanelText(win),
    'Only a title survived',
    `${mode.name}: with no preview the task panel must fall back to the title`,
  );
  assert.strictEqual(
    input(win).value,
    '',
    `${mode.name}: the title must not reach the chat input textbox — got ` +
      `${JSON.stringify(input(win).value)}`,
  );

  win.close();
  console.log(`  ok - ${mode.name}: task text falls back to the title`);
}

function testTaskTextEmptyWithoutPreviewOrTitle(mode) {
  const {win, posted} = makeWebview({remote: mode.remote});
  disableWorkspaceFilter(win);
  openSidebar(win);

  sendHistory(win, posted, [
    session({
      id: 'chat-no-text',
      task_id: 32,
      has_events: false,
      title: '',
      preview: '',
    }),
  ]);
  clickOnlyRow(win);

  assert.strictEqual(
    taskPanelText(win),
    '',
    `${mode.name}: a row with neither preview nor title shows no task text`,
  );
  assert.strictEqual(
    input(win).value,
    '',
    `${mode.name}: and the chat input textbox stays empty — got ` +
      `${JSON.stringify(input(win).value)}`,
  );

  win.close();
  console.log(`  ok - ${mode.name}: missing preview and title yield no text`);
}

// The backend's setTaskText message is the legitimate way to show the task —
// it targets the read-only panel and must never reach the input.
function testSetTaskTextMessageNeverWritesInput(mode) {
  const {win} = makeWebview({remote: mode.remote});

  const draft = 'user draft kept while the backend announces the task';
  input(win).value = draft;
  input(win).dispatchEvent(new win.Event('input', {bubbles: true}));

  send(win, {type: 'setTaskText', text: 'Backend announced task title'});

  assert.strictEqual(
    activeTabLabel(win),
    'Backend announced task title',
    `${mode.name}: setTaskText must name the task on the tab`,
  );
  assert.strictEqual(
    input(win).value,
    draft,
    `${mode.name}: the setTaskText message must never write the chat input ` +
      `textbox — got ${JSON.stringify(input(win).value)}`,
  );

  win.close();
  console.log(`  ok - ${mode.name}: setTaskText names the tab, not the input`);
}

const SCENARIOS = [
  testResumeBranchLeavesInputEmpty,
  testFallbackBranchLeavesInputEmpty,
  testSwitchBranchLeavesInputEmpty,
  testResumeBranchPreservesTypedDraft,
  testFallbackBranchPreservesTypedDraft,
  testRunningRowWithoutEventsResumes,
  testRunningRowWithoutEventsPreservesTypedDraft,
  testTaskTextFallsBackToTitle,
  testTaskTextEmptyWithoutPreviewOrTitle,
  testSetTaskTextMessageNeverWritesInput,
];

const MODES = [
  {name: 'extension webview', remote: false},
  {name: 'remote webview', remote: true},
];

function main() {
  MODES.forEach(mode => {
    console.log(`[${mode.name}]`);
    SCENARIOS.forEach(scenario => scenario(mode));
  });
  console.log('historyClickNoInputCopy.test.js: all assertions passed.');
}

main();
