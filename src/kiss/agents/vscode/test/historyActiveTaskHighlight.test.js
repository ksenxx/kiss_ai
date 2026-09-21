// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the history list's ACTIVE-TASK
// highlight in media/main.js: the row of the task shown by the chat
// webview the user is looking at carries `.history-active-task`, its
// chat panel stays open by default, and the row is scrolled into view.
//
// Covered behavior:
//  - history-panel mode: the host's `activeTask` relay highlights the
//    row of that very task, opens its chat panel (the other panels
//    stay collapsed), and scrolls the row into view once;
//  - a task without a loaded row falls back to its chat's newest row;
//  - a relay that lands BEFORE the history does is applied (and
//    scrolled to) when the rows arrive;
//  - the identical-refresh fast path and a changed-data rebuild both
//    keep the highlight; the rebuild scrolls again (the emptied list
//    lands at the top);
//  - a named task without a loaded row highlights nothing (another
//    task must not pass for it);
//  - a chat panel the user folded while looking at it stays folded
//    (the header stands in as scroll target); moving away and back
//    unfolds it so the row shows;
//  - a row the filters hide is marked but its scroll waits for the
//    filter change that shows it;
//  - a relay with empty ids clears the highlight and the panel falls
//    back to collapsed;
//  - the remote webapp paints its in-page list from its own visible
//    tab (task_settings, tab switches) and never posts `activeTask`;
//  - a VS Code chat surface posts `activeTask` to the host once per
//    change of the shown task.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const HISTORY_ATTRS =
  ' class="editor-tab-mode history-panel-mode"' +
  ' data-kiss-tab-id="history-panel"';
const PANEL_ROOT = 'panel-tab-1';
const PANEL_ATTRS = ` class="editor-tab-mode" data-kiss-tab-id="${PANEL_ROOT}"`;
const REMOTE_ATTRS = ' class="remote-chat"';

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs);
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  // Surfaces whose chat.html lacks the body placeholder.
  if (!/<body[^>]*class=/.test(html)) {
    html = html.replace('<body', '<body' + bodyAttrs);
  }

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  // jsdom has no scrollIntoView: record every call and its target.
  const scrolled = [];
  win.Element.prototype.scrollIntoView = function () {
    scrolled.push(this);
  };
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=history-active-main.js',
  );

  return {win, posted, scrolled};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byType(posted, type) {
  return posted.filter(m => m && m.type === type);
}

function lastMessage(posted, type) {
  const msgs = byType(posted, type);
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

// Two chats, three tasks, newest first as the daemon sends them.
function threeSessions() {
  return [
    session({
      id: 'chat-2',
      task_id: 3,
      title: 'chat two',
      preview: 'chat two',
      timestamp: 1_700_000_300,
    }),
    session({
      id: 'chat-1',
      task_id: 2,
      title: 'chat one, second',
      preview: 'chat one, second',
      timestamp: 1_700_000_200,
    }),
    session({
      id: 'chat-1',
      task_id: 1,
      title: 'chat one, first',
      preview: 'chat one, first',
      timestamp: 1_700_000_100,
    }),
  ];
}

// renderHistory drops replies whose generation is stale, so always
// answer the generation the webview last asked for.
function sendHistory(win, posted, sessions) {
  const req = lastMessage(posted, 'getHistory');
  assert.ok(req, 'the webview must have requested history');
  send(win, {
    type: 'history',
    offset: 0,
    generation: req.generation,
    sessions,
  });
}

function rows(win) {
  return Array.from(
    win.document.querySelectorAll('#history-list .running-item'),
  );
}

// Rows carry no ids of their own: they sit under their chat's group in
// the daemon's order (newest first), the same order as threeSessions().
function rowFor(win, chatId, taskId) {
  const ownTasks = threeSessions()
    .filter(s => s.id === chatId)
    .map(s => s.task_id);
  const idx = ownTasks.indexOf(taskId);
  const row = group(win, chatId).querySelectorAll('.running-item')[idx];
  assert.ok(idx >= 0 && row, `row for ${chatId}/${taskId} must exist`);
  return row;
}

function highlighted(win) {
  return rows(win).filter(r => r.classList.contains('history-active-task'));
}

function group(win, chatId) {
  const g = win.document.querySelector(
    `#history-list .history-chat-group[data-chat-id="${chatId}"]`,
  );
  assert.ok(g, `group for ${chatId} must exist`);
  return g;
}

function collapsed(win, chatId) {
  return group(win, chatId).classList.contains('collapsed');
}

function testRelayHighlightsOpensAndScrolls() {
  const {win, posted, scrolled} = makeWebview(HISTORY_ATTRS);
  disableWorkspaceFilter(win);
  sendHistory(win, posted, threeSessions());
  assert.strictEqual(highlighted(win).length, 0, 'nothing highlighted yet');
  assert.ok(
    collapsed(win, 'chat-1') && collapsed(win, 'chat-2'),
    'idle chats start collapsed',
  );
  scrolled.length = 0;

  send(win, {type: 'activeTask', chatId: 'chat-1', taskId: '2'});
  const row = rowFor(win, 'chat-1', 2);
  assert.deepStrictEqual(
    highlighted(win),
    [row],
    'only the shown task is highlighted',
  );
  assert.ok(!collapsed(win, 'chat-1'), 'the shown chat opens');
  assert.ok(collapsed(win, 'chat-2'), 'other chats stay collapsed');
  assert.deepStrictEqual(scrolled, [row], 'the row is scrolled into view once');

  // The same relay again changes nothing and scrolls nothing.
  send(win, {type: 'activeTask', chatId: 'chat-1', taskId: '2'});
  assert.strictEqual(
    scrolled.length,
    1,
    'a repeated relay does not scroll again',
  );

  // A named task without a loaded row: no other row passes for it,
  // but its chat opens and the previous chat falls back to collapsed.
  send(win, {type: 'activeTask', chatId: 'chat-2', taskId: '999'});
  assert.strictEqual(
    highlighted(win).length,
    0,
    'no stand-in for a named task',
  );
  assert.ok(!collapsed(win, 'chat-2'), 'the shown chat still opens');
  assert.ok(collapsed(win, 'chat-1'), 'the previous chat collapses again');
  assert.strictEqual(scrolled.length, 1, 'nothing to scroll to');

  // A chat that names no task yet: its newest row stands in.
  send(win, {type: 'activeTask', chatId: 'chat-2', taskId: ''});
  const newest = rowFor(win, 'chat-2', 3);
  assert.deepStrictEqual(highlighted(win), [newest]);
  assert.deepStrictEqual(scrolled.slice(1), [newest]);

  // Empty ids clear the highlight.
  send(win, {type: 'activeTask', chatId: '', taskId: ''});
  assert.strictEqual(highlighted(win).length, 0);
  assert.ok(collapsed(win, 'chat-2'), 'no shown chat: default collapsed');
  assert.strictEqual(scrolled.length, 2, 'nothing to scroll to');
  win.close();
  console.log('PASS relay highlights the row, opens its chat and scrolls once');
}

function testRelayBeforeHistoryAppliesOnRender() {
  const {win, posted, scrolled} = makeWebview(HISTORY_ATTRS);
  disableWorkspaceFilter(win);
  send(win, {type: 'activeTask', chatId: 'chat-1', taskId: '1'});
  assert.strictEqual(rows(win).length, 0, 'no rows yet');
  scrolled.length = 0;
  sendHistory(win, posted, threeSessions());
  const row = rowFor(win, 'chat-1', 1);
  assert.deepStrictEqual(highlighted(win), [row]);
  assert.ok(!collapsed(win, 'chat-1'));
  assert.deepStrictEqual(
    scrolled,
    [row],
    'the pending scroll lands with the rows',
  );
  win.close();
  console.log('PASS a relay ahead of the history is applied on render');
}

function testRefreshAndRebuildKeepHighlight() {
  const {win, posted, scrolled} = makeWebview(HISTORY_ATTRS);
  disableWorkspaceFilter(win);
  sendHistory(win, posted, threeSessions());
  send(win, {type: 'activeTask', chatId: 'chat-1', taskId: '1'});
  const before = rowFor(win, 'chat-1', 1);
  scrolled.length = 0;

  // Identical data: the fast path keeps the very same row elements.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, threeSessions());
  assert.strictEqual(
    rowFor(win, 'chat-1', 1),
    before,
    'fast path kept the row',
  );
  assert.deepStrictEqual(highlighted(win), [before]);
  assert.strictEqual(
    scrolled.length,
    0,
    'an identical refresh does not scroll',
  );

  // Changed data: the list is rebuilt (scroll offset lost), so the
  // fresh row is highlighted and scrolled to again.
  const changed = threeSessions();
  changed[0].tokens = 4242;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, changed);
  const after = rowFor(win, 'chat-1', 1);
  assert.notStrictEqual(after, before, 'the row was rebuilt');
  assert.deepStrictEqual(highlighted(win), [after]);
  assert.deepStrictEqual(
    scrolled,
    [after],
    'a rebuild scrolls the fresh row into view',
  );
  win.close();
  console.log('PASS refreshes and rebuilds keep the highlight');
}

function testFoldedChat() {
  const {win, posted, scrolled} = makeWebview(HISTORY_ATTRS);
  disableWorkspaceFilter(win);
  sendHistory(win, posted, threeSessions());
  send(win, {type: 'activeTask', chatId: 'chat-1', taskId: '2'});
  assert.ok(!collapsed(win, 'chat-1'));
  const header = () =>
    group(win, 'chat-1').querySelector(':scope > .history-chat-header');
  header().dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(collapsed(win, 'chat-1'), 'the user folded the chat they look at');

  // A rebuild keeps that fold; the rebuild's scroll lands on the
  // header, the highlighted row being hidden inside.
  scrolled.length = 0;
  const changed = threeSessions();
  changed[2].cost = 1.5;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, changed);
  assert.ok(collapsed(win, 'chat-1'), 'the fold survives a rebuild');
  assert.deepStrictEqual(highlighted(win), [rowFor(win, 'chat-1', 2)]);
  assert.deepStrictEqual(
    scrolled,
    [header()],
    'the header stands in for the hidden row',
  );

  // Moving to another chat and back unfolds it: the row must show.
  send(win, {type: 'activeTask', chatId: 'chat-2', taskId: '3'});
  assert.ok(collapsed(win, 'chat-1'));
  scrolled.length = 0;
  send(win, {type: 'activeTask', chatId: 'chat-1', taskId: '1'});
  assert.ok(!collapsed(win, 'chat-1'), 'moving to the folded chat unfolds it');
  const row = rowFor(win, 'chat-1', 1);
  assert.deepStrictEqual(highlighted(win), [row]);
  assert.deepStrictEqual(
    scrolled,
    [row],
    'the row itself is scrolled into view',
  );

  // The dropped fold stays dropped across a rebuild.
  changed[2].cost = 2.5;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, changed);
  assert.ok(!collapsed(win, 'chat-1'), 'the unfold survives a rebuild');
  win.close();
  console.log('PASS a folded chat unfolds when moved to, else keeps the fold');
}

function testFilteredRowWaitsForTheFilter() {
  const {win, posted, scrolled} = makeWebview(HISTORY_ATTRS);
  disableWorkspaceFilter(win);
  sendHistory(win, posted, threeSessions());
  const completedBox = win.document.getElementById('hf-completed');
  completedBox.checked = false;
  completedBox.dispatchEvent(new win.Event('change', {bubbles: true}));
  const row = rowFor(win, 'chat-1', 1);
  assert.strictEqual(
    row.style.display,
    'none',
    'the filter hides completed tasks',
  );

  scrolled.length = 0;
  send(win, {type: 'activeTask', chatId: 'chat-1', taskId: '1'});
  assert.deepStrictEqual(highlighted(win), [row], 'the hidden row is marked');
  assert.strictEqual(scrolled.length, 0, 'a hidden row is not scrolled to');

  // Showing completed tasks again lands the pending scroll.
  completedBox.checked = true;
  completedBox.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(row.style.display, '');
  assert.deepStrictEqual(scrolled, [row], 'the scroll waited for the filter');
  win.close();
  console.log(
    'PASS a filtered-out row is scrolled to once the filter shows it',
  );
}

function testRemoteWebappPaintsFromOwnTabs() {
  const {win, posted, scrolled} = makeWebview(REMOTE_ATTRS);
  disableWorkspaceFilter(win);
  const mine = byType(posted, 'ready')[0].tabId;
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: mine, chatId: 'chat-1', title: 'one', workDir: ''},
      {tabId: 'other-1', chatId: 'chat-2', title: 'two', workDir: ''},
    ],
  });
  win.document.getElementById('menu-btn').click();
  sendHistory(win, posted, threeSessions());
  // The visible tab is bound to chat-1 but names no task yet: the
  // chat's newest row stands in.
  assert.deepStrictEqual(highlighted(win), [rowFor(win, 'chat-1', 2)]);

  send(win, {
    type: 'task_settings',
    tabId: mine,
    taskId: '1',
    settings: {model: 'm', chat_id: 'chat-1', task_id: 1, start_ts: 1},
  });
  const own = rowFor(win, 'chat-1', 1);
  assert.deepStrictEqual(
    highlighted(win),
    [own],
    'the tab\u2019s own task takes over',
  );
  assert.ok(scrolled.includes(own), 'the row is scrolled into view');
  assert.ok(!collapsed(win, 'chat-1') && collapsed(win, 'chat-2'));

  // Switching to the other tab follows that chat.
  win.document
    .querySelector('.chat-tab[data-tab-id="other-1"]')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.deepStrictEqual(highlighted(win), [rowFor(win, 'chat-2', 3)]);
  assert.ok(!collapsed(win, 'chat-2') && collapsed(win, 'chat-1'));

  assert.strictEqual(
    byType(posted, 'activeTask').length,
    0,
    'the remote webapp never posts activeTask (its postMessage is the daemon)',
  );
  win.close();
  console.log(
    'PASS the remote webapp paints its in-page list from its own tabs',
  );
}

async function testChatPanelPostsActiveTask() {
  const {win, posted} = makeWebview(PANEL_ATTRS);
  const boot = byType(posted, 'activeTask').length;
  send(win, {
    type: 'task_settings',
    tabId: PANEL_ROOT,
    taskId: 'task-77',
    settings: {model: 'm', chat_id: 'chat-9', task_id: 'task-77', start_ts: 1},
  });
  let msgs = byType(posted, 'activeTask').slice(boot);
  // Messages are built in the jsdom realm: compare by value, not prototype.
  assert.strictEqual(
    JSON.stringify(msgs[msgs.length - 1]),
    JSON.stringify({type: 'activeTask', chatId: 'chat-9', taskId: 'task-77'}),
    'the panel reports its chat and task ids',
  );
  const count = byType(posted, 'activeTask').length;

  // A repaint with the same ids posts nothing new.
  send(win, {type: 'configData', config: {work_dir: '/x'}, apiKeys: {}});
  assert.strictEqual(
    byType(posted, 'activeTask').length,
    count,
    'no post without a change',
  );

  // The next task of the same chat is reported.
  send(win, {
    type: 'task_settings',
    tabId: PANEL_ROOT,
    taskId: 'task-78',
    settings: {model: 'm', chat_id: 'chat-9', task_id: 'task-78', start_ts: 2},
  });
  msgs = byType(posted, 'activeTask');
  assert.strictEqual(
    JSON.stringify(msgs[msgs.length - 1]),
    JSON.stringify({type: 'activeTask', chatId: 'chat-9', taskId: 'task-78'}),
  );
  win.close();
  console.log('PASS a chat editor panel posts activeTask once per change');
}

function testHistoryPanelIgnoresOwnPlaceholderTab() {
  // The history panel's own placeholder tab must never paint a
  // highlight: only the host relay decides.
  const {win, posted} = makeWebview(HISTORY_ATTRS);
  disableWorkspaceFilter(win);
  sendHistory(win, posted, threeSessions());
  send(win, {
    type: 'task_settings',
    tabId: 'history-panel',
    taskId: '3',
    settings: {model: 'm', chat_id: 'chat-2', task_id: 3, start_ts: 1},
  });
  assert.strictEqual(highlighted(win).length, 0);
  assert.strictEqual(
    byType(posted, 'activeTask').length,
    0,
    'the history panel posts no activeTask',
  );
  win.close();
  console.log('PASS the history panel follows only the host relay');
}

async function main() {
  testRelayHighlightsOpensAndScrolls();
  testRelayBeforeHistoryAppliesOnRender();
  testRefreshAndRebuildKeepHighlight();
  testFoldedChat();
  testFilteredRowWaitsForTheFilter();
  testRemoteWebappPaintsFromOwnTabs();
  await testChatPanelPostsActiveTask();
  testHistoryPanelIgnoresOwnPlaceholderTab();
  console.log('historyActiveTaskHighlight: all tests passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
