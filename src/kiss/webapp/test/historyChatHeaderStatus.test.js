// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The collapsible chat panel's header shows the status icon of the
// chat's LAST task: the same "?", spinner, cross or tick that task's
// own row shows, and nothing when that task settled before the page
// loaded.  Runs with: node test/historyChatHeaderStatus.test.js

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {inlineDesignTokens} = require('./designTokens');

const MEDIA = path.join(__dirname, '..', 'media');

function makePage() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
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
  win.acquireTransport = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  const styleEl = win.document.createElement('style');
  styleEl.textContent = inlineDesignTokens(
    fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8'),
  );
  win.document.head.appendChild(styleEl);
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Let main.js's end-of-tick history refetch coalescing settle. */
function tick() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

function openSidebar(win) {
  win.document.getElementById('sidebar').classList.add('open');
}

function sendEmptyWorkDir(win) {
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
}

function makeRow(overrides) {
  return Object.assign(
    {
      id: 'chat-' + overrides.task_id,
      task_id: overrides.task_id,
      title: 'task ' + overrides.task_id,
      timestamp: 1_700_000_000,
      preview: 'task ' + overrides.task_id,
      has_events: false,
      failed: false,
      is_running: false,
      awaiting_answer: false,
      tokens: 1,
      cost: 0,
      steps: 1,
      is_favorite: false,
      work_dir: '',
      startTs: 1_700_000_000_000,
      endTs: 0,
    },
    overrides,
  );
}

/** Reply to the latest getHistory request (or the first load). */
function sendHistory(win, posted, sessions, offset) {
  const req = posted.filter(m => m && m.type === 'getHistory').pop();
  send(win, {
    type: 'history',
    offset: offset || 0,
    generation: req ? req.generation : undefined,
    sessions,
  });
}

function group(win, chatId) {
  const g = win.document.querySelector(
    '#history-list .history-chat-group[data-chat-id="' + chatId + '"]',
  );
  assert.ok(g, 'chat panel for ' + chatId);
  return g;
}

function header(win, chatId) {
  return group(win, chatId).querySelector(':scope > .history-chat-header');
}

function headerMark(win, chatId) {
  return header(win, chatId).querySelector(':scope > .history-chat-status');
}

function rowByTask(win, taskId) {
  const rows = win.document.querySelectorAll('#history-list .sidebar-item');
  for (const r of rows) {
    const t = r.querySelector('.sidebar-item-text');
    if (t && t.textContent === 'task ' + taskId) return r;
  }
  assert.fail('no row for task ' + taskId);
}

function rowMark(win, taskId) {
  return rowByTask(win, taskId).querySelector(
    '.sidebar-item-running, .sidebar-item-failed, .sidebar-item-completed',
  );
}

/** The header icon mirrors the row icon of task *taskId* class for class. */
function assertMirrors(win, chatId, taskId, where) {
  const h = headerMark(win, chatId);
  const r = rowMark(win, taskId);
  assert.ok(r, 'the row of task ' + taskId + ' shows an icon ' + where);
  assert.ok(h, 'the header of ' + chatId + ' shows an icon ' + where);
  const rowClasses = Array.from(r.classList).sort();
  const headerClasses = Array.from(h.classList)
    .filter(c => c !== 'history-chat-status')
    .sort();
  assert.deepStrictEqual(headerClasses, rowClasses, 'same icon ' + where);
  assert.strictEqual(h.textContent, r.textContent, 'same glyph ' + where);
  assert.strictEqual(
    h.dataset.tooltip,
    r.dataset.tooltip,
    'same tooltip ' + where,
  );
  assert.strictEqual(
    h.getAttribute('aria-label'),
    r.getAttribute('aria-label'),
    'same label ' + where,
  );
  // Before the title; the action strip (Show details, last launched)
  // comes last.
  const btn = header(win, chatId);
  assert.strictEqual(btn.children[0], h, 'icon first ' + where);
  assert.strictEqual(
    btn.children[1].classList.contains('history-chat-title'),
    true,
    'title second ' + where,
  );
  assert.strictEqual(
    btn.children[2].classList.contains('history-chat-actions'),
    true,
    'action strip third ' + where,
  );
  assert.strictEqual(
    win.getComputedStyle(h).marginRight,
    '6px',
    'the icon is spaced by its own header margin, not the row margin ' + where,
  );
}

function testHeaderShowsLastTaskIcon() {
  const {win, posted} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);
  // Newest first, as the daemon sends them.  Each chat's newest task
  // decides its header's icon regardless of its older tasks.
  sendHistory(win, posted, [
    // chat A: last task running, older one failed.
    makeRow({
      id: 'A',
      task_id: 1,
      is_running: true,
      startTs: 1_700_000_900_000,
    }),
    makeRow({id: 'A', task_id: 2, failed: true, startTs: 1_700_000_800_000}),
    // chat B: last task waiting for an answer.
    makeRow({
      id: 'B',
      task_id: 3,
      is_running: true,
      awaiting_answer: true,
      startTs: 1_700_000_700_000,
    }),
    // chat C: last task failed, older one running.
    makeRow({id: 'C', task_id: 4, failed: true, startTs: 1_700_000_600_000}),
    makeRow({
      id: 'C',
      task_id: 5,
      is_running: true,
      startTs: 1_700_000_500_000,
    }),
    // chat D: last task settled before the page loaded: no icon.
    makeRow({id: 'D', task_id: 6, startTs: 1_700_000_400_000}),
    makeRow({id: 'D', task_id: 7, failed: true, startTs: 1_700_000_300_000}),
  ]);
  assertMirrors(win, 'A', 1, 'chat A (running)');
  assertMirrors(win, 'B', 3, 'chat B (asking)');
  assertMirrors(win, 'C', 4, 'chat C (failed)');
  assert.strictEqual(headerMark(win, 'D'), null, 'chat D shows no icon');
  assert.strictEqual(rowMark(win, 6), null, 'nor does its last row');
  assert.ok(rowMark(win, 7), 'the older failed row keeps its cross');
  assert.strictEqual(
    header(win, 'D').children.length,
    2,
    'title and action strip only when there is no icon',
  );
  // Clicking the icon (part of the header) opens the chat, as the rest
  // of the header does; only the Show details chevron folds the panel.
  const a = group(win, 'A');
  const before = a.classList.contains('collapsed');
  posted.length = 0;
  headerMark(win, 'A').click();
  assert.strictEqual(
    a.classList.contains('collapsed'),
    before,
    'the icon never folds the panel',
  );
  const resumes = byType(posted, 'resumeSession');
  assert.strictEqual(resumes.length, 1, 'the icon opens the chat');
  assert.strictEqual(resumes[0].id, 'A');
  win.close();
  console.log("  ok - the chat header shows its last task's icon");
}

async function testHeaderFollowsTaskStatus() {
  const {win, posted} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);
  sendHistory(win, posted, [
    makeRow({
      id: 'A',
      task_id: 1,
      is_running: true,
      startTs: 1_700_000_900_000,
    }),
  ]);
  assertMirrors(win, 'A', 1, 'while running');

  // The task asks a question: the header shows the "?" too.
  // Every nudge below waits a tick first: a refresh in the same tick as
  // the previous one folds into its end-of-tick refetch.
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({
      id: 'A',
      task_id: 1,
      is_running: true,
      awaiting_answer: true,
      startTs: 1_700_000_900_000,
    }),
  ]);
  assertMirrors(win, 'A', 1, 'while asking');

  // Answered and finished: the just-completed tick, on both.
  await tick();
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, startTs: 1_700_000_900_000, endTs: 1}),
  ]);
  assertMirrors(win, 'A', 1, 'just completed');
  assert.ok(headerMark(win, 'A').classList.contains('status-tick'), 'a tick');

  // A newer task launched in the same chat fails: the header follows
  // the new last task, not the completed older one.
  await tick();
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 2, failed: true, startTs: 1_700_000_950_000}),
    makeRow({id: 'A', task_id: 1, startTs: 1_700_000_900_000, endTs: 1}),
  ]);
  assertMirrors(win, 'A', 2, 'after a newer task failed');
  assert.ok(headerMark(win, 'A').classList.contains('status-cross'), 'a cross');
  win.close();
  console.log(
    '  ok - the header icon follows the last task through its states',
  );
}

function testNewestLaunchWinsAcrossPagesAndOrder() {
  const {win, posted} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);
  // A full first page (50 rows) whose chat A's newest task is running.
  const page = [
    makeRow({
      id: 'A',
      task_id: 1,
      is_running: true,
      startTs: 1_700_000_900_000,
    }),
  ];
  for (let i = 2; i <= 50; i++) {
    page.push(
      makeRow({id: 'X' + i, task_id: i, startTs: 1_700_000_900_000 - i * 1000}),
    );
  }
  sendHistory(win, posted, page);
  assertMirrors(win, 'A', 1, 'from the first page');

  // The second page brings an OLDER failed task of chat A: the header
  // keeps the newer task's spinner.
  const list = win.document.getElementById('history-list');
  Object.defineProperty(list, 'scrollHeight', {
    value: 1000,
    configurable: true,
  });
  Object.defineProperty(list, 'clientHeight', {value: 100, configurable: true});
  list.scrollTop = 900;
  list.dispatchEvent(new win.Event('scroll'));
  const req = posted.filter(m => m && m.type === 'getHistory').pop();
  assert.ok(req && req.offset === 50, 'the scroll asked for the next page');
  send(win, {
    type: 'history',
    offset: 50,
    generation: req.generation,
    sessions: [
      makeRow({id: 'A', task_id: 60, failed: true, startTs: 1_700_000_100_000}),
    ],
  });
  assert.ok(
    rowMark(win, 60).classList.contains('status-cross'),
    'the older row shows its cross',
  );
  assertMirrors(win, 'A', 1, 'after the older page arrived');

  // Rows out of order (an older row first): the newer launch still
  // decides the header, and an undated row never displaces a dated one.
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'B', task_id: 70, failed: true, startTs: 1_700_000_100_000}),
    makeRow({
      id: 'B',
      task_id: 71,
      is_running: true,
      startTs: 1_700_000_200_000,
    }),
    makeRow({id: 'B', task_id: 72, failed: true, startTs: 0, timestamp: null}),
    makeRow({id: 'C', task_id: 80, failed: true, startTs: 0, timestamp: null}),
    makeRow({
      id: 'C',
      task_id: 81,
      is_running: true,
      startTs: 1_700_000_200_000,
    }),
    // Same launch instant: the row seen first keeps the header.
    makeRow({
      id: 'D',
      task_id: 90,
      is_running: true,
      startTs: 1_700_000_200_000,
    }),
    makeRow({id: 'D', task_id: 91, failed: true, startTs: 1_700_000_200_000}),
    // An older failed row first, then the newer settled last task: the
    // cross put up for the older row comes down again.
    makeRow({id: 'E', task_id: 95, failed: true, startTs: 1_700_000_100_000}),
    makeRow({id: 'E', task_id: 96, startTs: 1_700_000_200_000}),
  ]);
  assertMirrors(win, 'B', 71, 'the newer launch wins out of order');
  assertMirrors(win, 'C', 81, 'a dated row displaces an undated one');
  assertMirrors(win, 'D', 90, 'the first row wins a tie');
  assert.strictEqual(
    headerMark(win, 'E'),
    null,
    'a settled newer task clears the icon',
  );
  assert.ok(
    rowMark(win, 95).classList.contains('status-cross'),
    'the older row keeps its cross',
  );
  win.close();
  console.log(
    '  ok - the newest launch decides the header across pages and order',
  );
}

function testLegacyFlatViewHasNoHeaders() {
  const {win, posted} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);
  win.document.getElementById('history-view-toggle').click();
  sendHistory(win, posted, [makeRow({id: 'A', task_id: 1, is_running: true})]);
  assert.strictEqual(
    win.document.querySelector('#history-list .history-chat-status'),
    null,
    'the flat list has no chat headers to carry an icon',
  );
  assert.ok(
    rowMark(win, 1).classList.contains('status-spinner'),
    'the row keeps its spinner',
  );
  win.close();
  console.log('  ok - the flat view is unchanged');
}

// The open chat tabs (chat tabs have no row of their own: the Chats
// panel picks the one on screen; the rest stay open, hidden).
function chatTabs(win) {
  return win._testApi
    .openTabs()
    .filter(t => !t.isSubagentTab && !t.isContentTab);
}

function hasTab(win, tabId) {
  return win._testApi.openTabs().some(t => t.id === tabId);
}

function byType(posted, type) {
  return posted.filter(m => m && m.type === type);
}

/** The chat panel's "Show details" chevron. */
function chevron(win, chatId) {
  const btn = header(win, chatId).querySelector(
    '.history-chat-actions > .history-chat-collapse',
  );
  assert.ok(btn, 'Show details chevron for ' + chatId);
  return btn;
}

function testChevronFoldsAndHeaderOpensLastTask() {
  const {win, posted} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);
  const sidebar = win.document.getElementById('sidebar');
  // Settled chats: their panels start collapsed.  Chat A's last task
  // (newest launch) is task 2, listed after an older row to prove the
  // header opens the LAST task, not the first row.
  sendHistory(win, posted, [
    makeRow({
      id: 'A',
      task_id: 1,
      has_events: true,
      startTs: 1_700_000_800_000,
      endTs: 1,
    }),
    makeRow({
      id: 'A',
      task_id: 2,
      has_events: true,
      startTs: 1_700_000_900_000,
      endTs: 1,
    }),
    makeRow({
      id: 'B',
      task_id: 3,
      has_events: true,
      startTs: 1_700_000_700_000,
      endTs: 1,
    }),
  ]);
  const a = group(win, 'A');
  assert.ok(
    a.classList.contains('collapsed'),
    'a settled chat starts collapsed',
  );
  const tabsBefore = chatTabs(win).length;
  const idleTab = win.kissActiveTabId();
  posted.length = 0;

  // The chevron only shows / hides the chat's task panels; it reads
  // like a task panel's and opens nothing.
  const chev = chevron(win, 'A');
  assert.strictEqual(
    chev.dataset.tooltip,
    'Show details',
    'folded: Show details',
  );
  assert.strictEqual(chev.getAttribute('aria-expanded'), 'false');
  chev.click();
  assert.ok(
    !a.classList.contains('collapsed'),
    'the chevron unfolds the panel',
  );
  assert.strictEqual(
    chev.dataset.tooltip,
    'Hide details',
    'unfolded: Hide details',
  );
  assert.strictEqual(chev.getAttribute('aria-expanded'), 'true');
  assert.ok(sidebar.classList.contains('open'), 'the tasks stay in view');
  assert.strictEqual(
    win.kissActiveTabId(),
    idleTab,
    'the chevron opens no tab',
  );
  assert.strictEqual(
    byType(posted, 'resumeSession').length,
    0,
    'and resumes nothing',
  );
  chev.click();
  assert.ok(a.classList.contains('collapsed'), 'folded again');
  assert.strictEqual(
    byType(posted, 'resumeSession').length,
    0,
    'no resume on fold',
  );

  // The header itself never folds the panel: it brings the chat on
  // screen, as a task row click does.  A fresh tab shows the chat;
  // the idle chat left behind is retired, so the count does not grow.
  posted.length = 0;
  header(win, 'A').click();
  assert.ok(
    a.classList.contains('collapsed'),
    'the header click leaves the panel folded',
  );
  assert.ok(
    !sidebar.classList.contains('open'),
    'the sidebar closes as for a row click',
  );
  assert.notStrictEqual(win.kissActiveTabId(), idleTab, 'a new tab is active');
  assert.ok(!hasTab(win, idleTab), 'the idle chat left behind is retired');
  assert.strictEqual(chatTabs(win).length, tabsBefore, 'one tab for the chat');
  let resumes = byType(posted, 'resumeSession');
  assert.strictEqual(resumes.length, 1, 'the chat is resumed once');
  assert.strictEqual(resumes[0].id, 'A', 'at chat A');
  assert.strictEqual(resumes[0].taskId, 2, 'at its last task');
  assert.strictEqual(
    resumes[0].tabId,
    win.kissActiveTabId(),
    'in the fresh, now active, tab',
  );

  // The daemon binds the new tab to chat A; a second header click
  // finds that tab showing the chat and switches to it.
  const chatATab = win.kissActiveTabId();
  send(win, {
    type: 'task_events',
    tabId: chatATab,
    chat_id: 'A',
    task_id: 2,
    task: 'task 2',
    events: [],
    extra: JSON.stringify({
      startTs: 1_700_000_900_000,
      endTs: 1_700_000_900_001,
    }),
  });
  // Moving to the chat highlights its panel (setHistoryActiveTask)
  // but does not unfold it: only the user folds or unfolds.
  assert.ok(
    a.classList.contains('history-active-chat'),
    'the chat on screen is highlighted',
  );
  assert.ok(
    a.classList.contains('collapsed'),
    'the chat on screen stays folded',
  );
  // Chat A starts a new task so "+" keeps its tab open while moving
  // to a fresh chat.
  send(win, {
    type: 'status',
    running: true,
    tabId: chatATab,
    startTs: 1_700_000_950_000,
  });
  win.document.getElementById('new-chat-btn').click();
  const freshTab = win.kissActiveTabId();
  assert.notStrictEqual(freshTab, chatATab, 'a fresh chat is active');
  assert.ok(hasTab(win, chatATab), 'the running chat A stays open');
  openSidebar(win);
  posted.length = 0;
  header(win, 'A').click();
  assert.strictEqual(
    win.kissActiveTabId(),
    chatATab,
    'the header switches to chat A',
  );
  assert.ok(
    !hasTab(win, freshTab),
    'the idle fresh chat left behind is retired',
  );
  assert.strictEqual(
    byType(posted, 'resumeSession').length,
    0,
    'chat A is already on screen',
  );
  assert.ok(
    a.classList.contains('collapsed'),
    'and leaves the panel as it was',
  );

  // Clicking the icon in the header behaves like the header.
  openSidebar(win);
  posted.length = 0;
  sendHistory(win, posted, [
    makeRow({
      id: 'C',
      task_id: 9,
      is_running: true,
      has_events: true,
      startTs: 1_700_000_950_000,
    }),
  ]);
  const c = group(win, 'C');
  assert.ok(
    c.classList.contains('collapsed'),
    'a running chat starts folded too',
  );
  posted.length = 0;
  headerMark(win, 'C').click();
  assert.ok(
    c.classList.contains('collapsed'),
    'the icon leaves the panel folded',
  );
  resumes = byType(posted, 'resumeSession');
  assert.strictEqual(resumes.length, 1, 'and opens the running chat');
  assert.strictEqual(resumes[0].id, 'C');
  assert.strictEqual(resumes[0].taskId, 9);
  assert.notStrictEqual(win.kissActiveTabId(), chatATab, 'in a fresh tab');
  assert.ok(hasTab(win, chatATab), 'the running chat A stays open');
  win.close();
  console.log(
    '  ok - the chevron folds the chat panel; the header opens its last task',
  );
}

function testEventlessLastTaskResumesByChatId() {
  const {win, posted} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);
  sendHistory(win, posted, [
    makeRow({
      id: 'E',
      task_id: 5,
      has_events: false,
      startTs: 1_700_000_900_000,
      endTs: 1,
    }),
  ]);
  const tabsBefore = chatTabs(win).length;
  const idleTab = win.kissActiveTabId();
  posted.length = 0;
  header(win, 'E').click();
  // The idle chat left behind is retired, so the count does not grow.
  assert.notStrictEqual(win.kissActiveTabId(), idleTab, 'a new tab is active');
  assert.ok(!hasTab(win, idleTab), 'the idle chat left behind is retired');
  assert.strictEqual(chatTabs(win).length, tabsBefore, 'one tab for chat E');
  const resumes = byType(posted, 'resumeSession');
  assert.strictEqual(
    resumes.length,
    1,
    'resumed by chat id, not shown read-only',
  );
  assert.strictEqual(resumes[0].id, 'E');
  assert.strictEqual(resumes[0].taskId, 5);
  // The daemon binds the tab to the chat on replay: a second header
  // click finds it and opens nothing more.
  send(win, {
    type: 'task_events',
    tabId: resumes[0].tabId,
    chat_id: 'E',
    task_id: 5,
    task: 'task 5',
    events: [],
    extra: JSON.stringify({
      startTs: 1_700_000_900_000,
      endTs: 1_700_000_900_001,
    }),
  });
  openSidebar(win);
  posted.length = 0;
  header(win, 'E').click();
  assert.strictEqual(
    chatTabs(win).length,
    tabsBefore,
    'no second tab for chat E',
  );
  assert.strictEqual(
    byType(posted, 'resumeSession').length,
    0,
    'no second resume',
  );
  // A row click on the same task still shows it read-only in a fresh
  // tab when no tab is bound to the chat; the settled chat E left
  // behind is retired.
  const chatETab = win.kissActiveTabId();
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({
      id: 'F',
      task_id: 6,
      has_events: false,
      startTs: 1_700_000_900_000,
      endTs: 1,
    }),
  ]);
  posted.length = 0;
  rowByTask(win, 6).click();
  assert.notStrictEqual(
    win.kissActiveTabId(),
    chatETab,
    'a fresh read-only tab',
  );
  assert.ok(!hasTab(win, chatETab), 'chat E is retired');
  assert.strictEqual(
    chatTabs(win).length,
    tabsBefore,
    'only the read-only tab',
  );
  assert.strictEqual(
    byType(posted, 'resumeSession').length,
    0,
    'nothing to resume',
  );
  win.close();
  console.log(
    '  ok - an eventless last task is resumed by chat id from the chat header',
  );
}

function headerTitle(win, chatId) {
  return header(win, chatId).querySelector(':scope > .history-chat-title');
}

function keyIn(win, el, key) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {key, bubbles: true, cancelable: true}),
  );
}

/**
 * The pencil on a chat panel's header (and a double-click on its
 * title) turns the summary into a text box: Enter saves the new text
 * through setChatSummary and shows it at once, Escape cancels, and the
 * box's own clicks never open the chat.  Until the daemon echoes the
 * new summary, history rows still carrying the old one do not paint
 * over it; once echoed, the daemon's copy is what shows.
 */
function testPencilEditsTheChatSummary() {
  const {win, posted} = makePage();
  openSidebar(win);
  sendEmptyWorkDir(win);
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, chat_summary: 'Fix the login bug'}),
    makeRow({id: 'A', task_id: 2, chat_summary: 'Fix the login bug'}),
  ]);
  assert.strictEqual(headerTitle(win, 'A').textContent, 'Fix the login bug');
  const pencil = header(win, 'A').querySelector('.history-chat-rename');
  assert.ok(pencil, 'the header carries a pencil');
  assert.strictEqual(pencil.getAttribute('aria-label'), 'Edit the chat summary');

  const tabsBefore = chatTabs(win).length;
  pencil.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  let input = headerTitle(win, 'A').querySelector('input');
  assert.ok(input, 'the pencil opens a text box in the title');
  assert.strictEqual(input.value, 'Fix the login bug');
  assert.ok(group(win, 'A').classList.contains('renaming'));
  // A click inside the box is not a click on the header.
  input.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(chatTabs(win).length, tabsBefore, 'the chat did not open');
  // Escape: nothing saved, the old title is back.
  input.value = 'typo';
  keyIn(win, input, 'Escape');
  assert.strictEqual(headerTitle(win, 'A').querySelector('input'), null);
  assert.strictEqual(headerTitle(win, 'A').textContent, 'Fix the login bug');
  assert.strictEqual(byType(posted, 'setChatSummary').length, 0);

  // Double-clicking the title edits too; Enter saves.
  headerTitle(win, 'A').dispatchEvent(
    new win.MouseEvent('dblclick', {bubbles: true}),
  );
  input = headerTitle(win, 'A').querySelector('input');
  assert.ok(input, 'double-click opens the text box');
  input.value = '  Login fix, round two  ';
  keyIn(win, input, 'Enter');
  assert.deepStrictEqual(
    JSON.parse(JSON.stringify(byType(posted, 'setChatSummary'))),
    [{type: 'setChatSummary', chatId: 'A', summary: 'Login fix, round two'}],
  );
  assert.strictEqual(headerTitle(win, 'A').textContent, 'Login fix, round two');
  assert.strictEqual(
    header(win, 'A').getAttribute('aria-label'),
    'Login fix, round two',
  );
  assert.ok(!group(win, 'A').classList.contains('renaming'));

  // A refresh whose rows still carry the old summary leaves the new
  // title alone ...
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, chat_summary: 'Fix the login bug'}),
  ]);
  assert.strictEqual(headerTitle(win, 'A').textContent, 'Login fix, round two');
  // ... and once the daemon echoes it, its copy is followed again.
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, chat_summary: 'Login fix, round two'}),
  ]);
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, chat_summary: 'Login fix (daemon)'}),
  ]);
  assert.strictEqual(headerTitle(win, 'A').textContent, 'Login fix (daemon)');

  // Saving the same text again sends nothing (the refreshes rebuilt
  // the panel: its pencil is a new one).
  const pencilNow = () => header(win, 'A').querySelector('.history-chat-rename');
  pencilNow().dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  input = headerTitle(win, 'A').querySelector('input');
  keyIn(win, input, 'Enter');
  assert.strictEqual(byType(posted, 'setChatSummary').length, 1);
  // Leaving the box (blur) saves like Enter does.
  pencilNow().dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  input = headerTitle(win, 'A').querySelector('input');
  input.value = 'Blurred title';
  input.dispatchEvent(new win.Event('blur'));
  assert.strictEqual(byType(posted, 'setChatSummary').length, 2);
  assert.strictEqual(headerTitle(win, 'A').textContent, 'Blurred title');
  win.close();
  console.log('  ok - the pencil edits the chat summary');
}

async function main() {
  testHeaderShowsLastTaskIcon();
  await testHeaderFollowsTaskStatus();
  testNewestLaunchWinsAcrossPagesAndOrder();
  testLegacyFlatViewHasNoHeaders();
  testChevronFoldsAndHeaderOpensLastTask();
  testEventlessLastTaskResumesByChatId();
  testPencilEditsTheChatSummary();
  console.log('\n7 passed, 0 failed');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
