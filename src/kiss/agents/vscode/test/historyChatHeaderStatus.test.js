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

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs || '');
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

function openSidebar(win) {
  win.document.getElementById('sidebar').classList.add('open');
}

function uncheckWorkspaceFilter(win) {
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  const ws = win.document.getElementById('hf-workspace');
  if (ws && ws.checked) {
    ws.checked = false;
    ws.dispatchEvent(new win.Event('change', {bubbles: true}));
  }
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
  assert.strictEqual(h.dataset.tooltip, r.dataset.tooltip, 'same tooltip ' + where);
  assert.strictEqual(
    h.getAttribute('aria-label'),
    r.getAttribute('aria-label'),
    'same label ' + where,
  );
  // Between the chevron and the title.
  const btn = header(win, chatId);
  assert.strictEqual(
    btn.children[0].classList.contains('history-chat-chevron'),
    true,
    'chevron first ' + where,
  );
  assert.strictEqual(btn.children[1], h, 'icon second ' + where);
  assert.strictEqual(
    btn.children[2].classList.contains('history-chat-title'),
    true,
    'title third ' + where,
  );
  assert.strictEqual(
    win.getComputedStyle(h).marginRight,
    '0px',
    'the header gap spaces the icon, not the row margin ' + where,
  );
}

function testHeaderShowsLastTaskIcon() {
  const {win, posted} = makeWebview();
  openSidebar(win);
  uncheckWorkspaceFilter(win);
  // Newest first, as the daemon sends them.  Each chat's newest task
  // decides its header's icon regardless of its older tasks.
  sendHistory(win, posted, [
    // chat A: last task running, older one failed.
    makeRow({id: 'A', task_id: 1, is_running: true, startTs: 1_700_000_900_000}),
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
    makeRow({id: 'C', task_id: 5, is_running: true, startTs: 1_700_000_500_000}),
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
    'chevron and title only when there is no icon',
  );
  // Clicking the icon toggles the panel like the rest of the header.
  const a = group(win, 'A');
  const before = a.classList.contains('collapsed');
  headerMark(win, 'A').click();
  assert.strictEqual(a.classList.contains('collapsed'), !before, 'the icon toggles the panel');
  win.close();
  console.log('  ok - the chat header shows its last task\'s icon');
}

function testHeaderFollowsTaskStatus() {
  const {win, posted} = makeWebview();
  openSidebar(win);
  uncheckWorkspaceFilter(win);
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, is_running: true, startTs: 1_700_000_900_000}),
  ]);
  assertMirrors(win, 'A', 1, 'while running');

  // The task asks a question: the header shows the "?" too.
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
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, startTs: 1_700_000_900_000, endTs: 1}),
  ]);
  assertMirrors(win, 'A', 1, 'just completed');
  assert.ok(headerMark(win, 'A').classList.contains('status-tick'), 'a tick');

  // A newer task launched in the same chat fails: the header follows
  // the new last task, not the completed older one.
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 2, failed: true, startTs: 1_700_000_950_000}),
    makeRow({id: 'A', task_id: 1, startTs: 1_700_000_900_000, endTs: 1}),
  ]);
  assertMirrors(win, 'A', 2, 'after a newer task failed');
  assert.ok(headerMark(win, 'A').classList.contains('status-cross'), 'a cross');
  win.close();
  console.log('  ok - the header icon follows the last task through its states');
}

function testNewestLaunchWinsAcrossPagesAndOrder() {
  const {win, posted} = makeWebview();
  openSidebar(win);
  uncheckWorkspaceFilter(win);
  // A full first page (50 rows) whose chat A's newest task is running.
  const page = [
    makeRow({id: 'A', task_id: 1, is_running: true, startTs: 1_700_000_900_000}),
  ];
  for (let i = 2; i <= 50; i++) {
    page.push(makeRow({id: 'X' + i, task_id: i, startTs: 1_700_000_900_000 - i * 1000}));
  }
  sendHistory(win, posted, page);
  assertMirrors(win, 'A', 1, 'from the first page');

  // The second page brings an OLDER failed task of chat A: the header
  // keeps the newer task's spinner.
  const list = win.document.getElementById('history-list');
  Object.defineProperty(list, 'scrollHeight', {value: 1000, configurable: true});
  Object.defineProperty(list, 'clientHeight', {value: 100, configurable: true});
  list.scrollTop = 900;
  list.dispatchEvent(new win.Event('scroll'));
  const req = posted.filter(m => m && m.type === 'getHistory').pop();
  assert.ok(req && req.offset === 50, 'the scroll asked for the next page');
  send(win, {
    type: 'history',
    offset: 50,
    generation: req.generation,
    sessions: [makeRow({id: 'A', task_id: 60, failed: true, startTs: 1_700_000_100_000})],
  });
  assert.ok(rowMark(win, 60).classList.contains('status-cross'), 'the older row shows its cross');
  assertMirrors(win, 'A', 1, 'after the older page arrived');

  // Rows out of order (an older row first): the newer launch still
  // decides the header, and an undated row never displaces a dated one.
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'B', task_id: 70, failed: true, startTs: 1_700_000_100_000}),
    makeRow({id: 'B', task_id: 71, is_running: true, startTs: 1_700_000_200_000}),
    makeRow({id: 'B', task_id: 72, failed: true, startTs: 0, timestamp: null}),
    makeRow({id: 'C', task_id: 80, failed: true, startTs: 0, timestamp: null}),
    makeRow({id: 'C', task_id: 81, is_running: true, startTs: 1_700_000_200_000}),
    // Same launch instant: the row seen first keeps the header.
    makeRow({id: 'D', task_id: 90, is_running: true, startTs: 1_700_000_200_000}),
    makeRow({id: 'D', task_id: 91, failed: true, startTs: 1_700_000_200_000}),
    // An older failed row first, then the newer settled last task: the
    // cross put up for the older row comes down again.
    makeRow({id: 'E', task_id: 95, failed: true, startTs: 1_700_000_100_000}),
    makeRow({id: 'E', task_id: 96, startTs: 1_700_000_200_000}),
  ]);
  assertMirrors(win, 'B', 71, 'the newer launch wins out of order');
  assertMirrors(win, 'C', 81, 'a dated row displaces an undated one');
  assertMirrors(win, 'D', 90, 'the first row wins a tie');
  assert.strictEqual(headerMark(win, 'E'), null, 'a settled newer task clears the icon');
  assert.ok(rowMark(win, 95).classList.contains('status-cross'), 'the older row keeps its cross');
  win.close();
  console.log('  ok - the newest launch decides the header across pages and order');
}

function testLegacyFlatViewHasNoHeaders() {
  const {win, posted} = makeWebview();
  openSidebar(win);
  uncheckWorkspaceFilter(win);
  win.document.getElementById('history-view-toggle').click();
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, is_running: true}),
  ]);
  assert.strictEqual(
    win.document.querySelector('#history-list .history-chat-status'),
    null,
    'the flat list has no chat headers to carry an icon',
  );
  assert.ok(rowMark(win, 1).classList.contains('status-spinner'), 'the row keeps its spinner');
  win.close();
  console.log('  ok - the flat view is unchanged');
}

function chatTabs(win) {
  return win.document.querySelectorAll('#main-tab-list .chat-tab');
}

function byType(posted, type) {
  return posted.filter(m => m && m.type === type);
}

function testExpandingPanelOpensLastTask() {
  const {win, posted} = makeWebview();
  openSidebar(win);
  uncheckWorkspaceFilter(win);
  const sidebar = win.document.getElementById('sidebar');
  // Settled chats: their panels start collapsed.  Chat A's last task
  // (newest launch) is task 2, listed after an older row to prove the
  // header opens the LAST task, not the first row.
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, has_events: true, startTs: 1_700_000_800_000, endTs: 1}),
    makeRow({id: 'A', task_id: 2, has_events: true, startTs: 1_700_000_900_000, endTs: 1}),
    makeRow({id: 'B', task_id: 3, has_events: true, startTs: 1_700_000_700_000, endTs: 1}),
  ]);
  const a = group(win, 'A');
  assert.ok(a.classList.contains('collapsed'), 'a settled chat starts collapsed');
  const tabsBefore = chatTabs(win).length;
  posted.length = 0;

  header(win, 'A').click();
  assert.ok(!a.classList.contains('collapsed'), 'the click expands the panel');
  assert.ok(sidebar.classList.contains('open'), 'the tasks stay in view');
  assert.strictEqual(chatTabs(win).length, tabsBefore + 1, 'one new tab for the chat');
  let resumes = byType(posted, 'resumeSession');
  assert.strictEqual(resumes.length, 1, 'the chat is resumed once');
  assert.strictEqual(resumes[0].id, 'A', 'at chat A');
  assert.strictEqual(resumes[0].taskId, 2, 'at its last task');
  assert.strictEqual(
    resumes[0].tabId,
    win.kissActiveTabId(),
    'in the fresh, now active, tab',
  );

  // Collapsing opens nothing.
  posted.length = 0;
  header(win, 'A').click();
  assert.ok(a.classList.contains('collapsed'), 'collapsed again');
  assert.strictEqual(byType(posted, 'resumeSession').length, 0, 'no resume on collapse');
  assert.strictEqual(chatTabs(win).length, tabsBefore + 1, 'no new tab on collapse');

  // The daemon binds the new tab to chat A; expanding again finds that
  // tab showing the chat and opens nothing more.
  send(win, {
    type: 'task_events',
    tabId: win.kissActiveTabId(),
    chat_id: 'A',
    task_id: 2,
    task: 'task 2',
    events: [],
    extra: JSON.stringify({startTs: 1_700_000_900_000, endTs: 1_700_000_900_001}),
  });
  // Moving to the chat unfolds its panel (setHistoryActiveTask); fold
  // it by hand, then expand it once more.
  assert.ok(!a.classList.contains('collapsed'), 'the chat on screen unfolds its panel');
  header(win, 'A').click();
  assert.ok(a.classList.contains('collapsed'), 'folded by hand');
  posted.length = 0;
  header(win, 'A').click();
  assert.ok(!a.classList.contains('collapsed'), 'expanded again');
  assert.strictEqual(byType(posted, 'resumeSession').length, 0, 'chat A is already on screen');
  assert.strictEqual(chatTabs(win).length, tabsBefore + 1, 'no duplicate tab for chat A');

  // Clicking the icon in the header behaves like the header.
  posted.length = 0;
  sendHistory(win, posted, [
    makeRow({id: 'C', task_id: 9, is_running: true, has_events: true, startTs: 1_700_000_950_000}),
  ]);
  const c = group(win, 'C');
  assert.ok(!c.classList.contains('collapsed'), 'a running chat starts open');
  header(win, 'C').click();
  assert.ok(c.classList.contains('collapsed'), 'folded by the click');
  posted.length = 0;
  headerMark(win, 'C').click();
  assert.ok(!c.classList.contains('collapsed'), 'the icon expands the panel');
  resumes = byType(posted, 'resumeSession');
  assert.strictEqual(resumes.length, 1, 'and opens the running chat');
  assert.strictEqual(resumes[0].id, 'C');
  assert.strictEqual(resumes[0].taskId, 9);
  win.close();
  console.log('  ok - expanding a chat panel loads its last task unless a tab shows the chat');
}

function testExpandingPanelInEditorTabsModeAsksHost() {
  const {win, posted} = makeWebview(' class="editor-tab-mode history-panel-mode"');
  openSidebar(win);
  uncheckWorkspaceFilter(win);
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, has_events: true, startTs: 1_700_000_900_000, endTs: 1}),
    makeRow({id: 'A', task_id: 2, has_events: true, startTs: 1_700_000_800_000, endTs: 1}),
  ]);
  const a = group(win, 'A');
  assert.ok(a.classList.contains('collapsed'), 'starts collapsed');
  posted.length = 0;
  header(win, 'A').click();
  assert.ok(!a.classList.contains('collapsed'), 'expanded');
  let opens = byType(posted, 'openChatPanel');
  assert.strictEqual(opens.length, 1, 'the host is asked to open the chat');
  assert.strictEqual(opens[0].chatId, 'A');
  assert.strictEqual(opens[0].taskId, 1, 'at the last task');
  assert.strictEqual(
    opens[0].onlyIfMissing,
    true,
    'only when no editor panel shows the chat yet',
  );
  assert.strictEqual(byType(posted, 'resumeSession').length, 0, 'no in-panel resume');
  assert.ok(
    win.document.getElementById('sidebar').classList.contains('open'),
    'the history panel stays',
  );
  // A task row's click is an ordinary open: the host reveals the
  // chat's panel and moves it to the clicked task.
  posted.length = 0;
  rowByTask(win, 2).click();
  opens = byType(posted, 'openChatPanel');
  assert.strictEqual(opens.length, 1, 'a row click asks the host too');
  assert.strictEqual(opens[0].taskId, 2);
  assert.strictEqual(opens[0].onlyIfMissing, false, 'and always lands on the task');

  // A chat whose last task has no persisted events: a row click opens
  // it read-only (no chat id), but an expand keeps the chat id so the
  // host can see the chat's panel is already open.
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'E', task_id: 5, has_events: false, startTs: 1_700_000_900_000, endTs: 1}),
  ]);
  posted.length = 0;
  header(win, 'E').click();
  opens = byType(posted, 'openChatPanel');
  assert.strictEqual(opens.length, 1, 'the expand asks the host');
  assert.strictEqual(opens[0].chatId, 'E', 'with the chat id even without events');
  assert.strictEqual(opens[0].taskId, 5);
  assert.strictEqual(opens[0].onlyIfMissing, true);
  posted.length = 0;
  rowByTask(win, 5).click();
  opens = byType(posted, 'openChatPanel');
  assert.strictEqual(opens.length, 1, 'the row click asks the host');
  assert.strictEqual(opens[0].chatId, undefined, 'read-only: no chat to resume');
  win.close();
  console.log('  ok - in editor-tabs mode expanding a panel asks the host for the chat');
}

function testEventlessLastTaskResumesByChatId() {
  const {win, posted} = makeWebview();
  openSidebar(win);
  uncheckWorkspaceFilter(win);
  sendHistory(win, posted, [
    makeRow({id: 'E', task_id: 5, has_events: false, startTs: 1_700_000_900_000, endTs: 1}),
  ]);
  const tabsBefore = chatTabs(win).length;
  posted.length = 0;
  header(win, 'E').click();
  assert.strictEqual(chatTabs(win).length, tabsBefore + 1, 'one tab for chat E');
  let resumes = byType(posted, 'resumeSession');
  assert.strictEqual(resumes.length, 1, 'resumed by chat id, not shown read-only');
  assert.strictEqual(resumes[0].id, 'E');
  assert.strictEqual(resumes[0].taskId, 5);
  // The daemon binds the tab to the chat on replay: a second expand
  // finds it and opens nothing more.
  send(win, {
    type: 'task_events',
    tabId: resumes[0].tabId,
    chat_id: 'E',
    task_id: 5,
    task: 'task 5',
    events: [],
    extra: JSON.stringify({startTs: 1_700_000_900_000, endTs: 1_700_000_900_001}),
  });
  header(win, 'E').click();
  assert.ok(group(win, 'E').classList.contains('collapsed'), 'folded');
  posted.length = 0;
  header(win, 'E').click();
  assert.strictEqual(chatTabs(win).length, tabsBefore + 1, 'no second tab for chat E');
  assert.strictEqual(byType(posted, 'resumeSession').length, 0, 'no second resume');
  // A row click on the same task still shows it read-only in a fresh
  // tab when no tab is bound to the chat.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'F', task_id: 6, has_events: false, startTs: 1_700_000_900_000, endTs: 1}),
  ]);
  posted.length = 0;
  rowByTask(win, 6).click();
  assert.strictEqual(chatTabs(win).length, tabsBefore + 2, 'a fresh read-only tab');
  assert.strictEqual(byType(posted, 'resumeSession').length, 0, 'nothing to resume');
  win.close();
  console.log('  ok - an eventless last task is resumed by chat id on expand');
}

function testSidebarOpenFromHistoryHonoursOnlyIfMissing() {
  const {win, posted} = makeWebview();
  // Bind the first tab to chat A, then move to a fresh second tab.
  send(win, {
    type: 'task_events',
    tabId: win.kissActiveTabId(),
    chat_id: 'A',
    task_id: 2,
    task: 'task 2',
    events: [],
    extra: JSON.stringify({startTs: 1_700_000_900_000, endTs: 1_700_000_900_001}),
  });
  const chatATab = win.kissActiveTabId();
  win.document.getElementById('new-chat-btn').click();
  const freshTab = win.kissActiveTabId();
  assert.notStrictEqual(freshTab, chatATab, 'a second tab is active');

  // The host relays an expanded history panel: chat A has a tab, so
  // nothing moves.
  posted.length = 0;
  send(win, {
    type: 'openChatFromHistory',
    chatId: 'A',
    taskId: 2,
    title: 'task 2',
    onlyIfMissing: true,
  });
  assert.strictEqual(win.kissActiveTabId(), freshTab, 'the active tab is kept');
  assert.strictEqual(byType(posted, 'resumeSession').length, 0, 'nothing resumed');
  assert.strictEqual(chatTabs(win).length, 2, 'no tab opened');

  // A row click relayed the same way switches to chat A's tab.
  send(win, {
    type: 'openChatFromHistory',
    chatId: 'A',
    taskId: 2,
    title: 'task 2',
    onlyIfMissing: false,
  });
  assert.strictEqual(win.kissActiveTabId(), chatATab, 'a row click switches to the chat');
  assert.strictEqual(chatTabs(win).length, 2, 'still no new tab');

  // Expanding a chat with no tab resumes it in a fresh tab.
  posted.length = 0;
  send(win, {
    type: 'openChatFromHistory',
    chatId: 'B',
    taskId: 5,
    title: 'task 5',
    onlyIfMissing: true,
  });
  assert.strictEqual(chatTabs(win).length, 3, 'a tab for chat B');
  const resumes = byType(posted, 'resumeSession');
  assert.strictEqual(resumes.length, 1, 'chat B is resumed');
  assert.strictEqual(resumes[0].id, 'B');
  assert.strictEqual(resumes[0].taskId, 5);
  win.close();
  console.log('  ok - the sidebar chat view leaves an open chat alone on a panel expand');
}

try {
  testHeaderShowsLastTaskIcon();
  testHeaderFollowsTaskStatus();
  testNewestLaunchWinsAcrossPagesAndOrder();
  testLegacyFlatViewHasNoHeaders();
  testExpandingPanelOpensLastTask();
  testExpandingPanelInEditorTabsModeAsksHost();
  testSidebarOpenFromHistoryHonoursOnlyIfMissing();
  testEventlessLastTaskResumesByChatId();
  console.log('\n8 passed, 0 failed');
} catch (e) {
  console.error(e);
  process.exit(1);
}
