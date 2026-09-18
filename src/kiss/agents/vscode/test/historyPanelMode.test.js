// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for HISTORY-PANEL MODE in the chat webview
// (media/main.js): the primary-sidebar history panel of editor-tabs
// mode. The extension host stamps `editor-tab-mode history-panel-mode`
// on <body> (SorcarTab.historyPanelBodyAttrs), so the surface behaves
// like an editor-tab webview — history clicks post `openChatPanel` —
// but shows ONLY the history sidebar, permanently open.
//
// Covered behavior:
//  - boot opens the sidebar and requests history right away;
//  - the boot placeholder tab is never announced to the daemon's tab
//    registry (no `openTab`), so no phantom tab reaches other clients;
//  - a resumable history row click posts `openChatPanel` carrying the
//    chat/task ids, a non-resumable one posts it without a chat id,
//    and the sidebar STAYS open either way (closeSidebar is a no-op);
//  - the panel keeps refreshing: `tasks_updated` broadcasts and the
//    webview becoming visible again both re-request history;
//  - registry snapshots that do not know the panel's root tab must not
//    make it ask its host to close (`closePanel`);
//  - the burger button is hidden by CSS in editor-tab-mode surfaces
//    and everything except the sidebar is hidden in history-panel
//    mode.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const BODY_ATTRS =
  ' class="editor-tab-mode history-panel-mode"' +
  ' data-kiss-tab-id="history-panel"';

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs);
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=history-panel-main.js',
  );

  return {win, posted};
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

function sidebarOpen(win) {
  return win.document.getElementById('sidebar').classList.contains('open');
}

function historyRows(win) {
  return Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item'),
  );
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

// renderHistory drops replies whose generation is stale, so always
// answer the generation the webview last asked for.
function sendHistory(win, posted, sessions) {
  const req = lastMessage(posted, 'getHistory');
  assert.ok(req, 'the history panel must have requested history');
  send(win, {
    type: 'history',
    offset: 0,
    generation: req.generation,
    sessions,
  });
}

function testBootOpensSidebarAndLoadsHistory() {
  const {win, posted} = makeWebview(BODY_ATTRS);
  assert.ok(sidebarOpen(win), 'the history panel is born open');
  assert.ok(
    byType(posted, 'getHistory').length >= 1,
    'boot must request history without any burger click',
  );
  const readies = byType(posted, 'ready');
  assert.strictEqual(readies.length, 1, 'exactly one ready');
  assert.strictEqual(
    readies[0].tabId,
    'history-panel',
    'ready announces the fixed history-panel root tab id',
  );
  assert.strictEqual(
    byType(posted, 'openTab').length,
    0,
    'the placeholder tab must NOT be announced to the shared registry',
  );
  win.close();
  console.log('PASS boot opens the sidebar and loads history');
}

function testResumableRowClickPostsOpenChatPanel() {
  const {win, posted} = makeWebview(BODY_ATTRS);
  disableWorkspaceFilter(win);
  sendHistory(win, posted, [session()]);
  const rows = historyRows(win);
  assert.strictEqual(rows.length, 1, 'the session renders one row');
  rows[0].click();
  const open = lastMessage(posted, 'openChatPanel');
  assert.ok(open, 'clicking a history row must ask the host for a panel');
  assert.strictEqual(open.chatId, 'chat-1');
  assert.strictEqual(open.taskId, 1);
  assert.strictEqual(open.title, 'Refactor the payment gateway retries');
  assert.ok(
    sidebarOpen(win),
    'the panel IS the sidebar: a row click must not close it',
  );
  assert.strictEqual(
    byType(posted, 'openTab').length,
    0,
    'opening a chat spawns an editor tab, never a registry tab here',
  );
  win.close();
  console.log('PASS a resumable row click posts openChatPanel, stays open');
}

function testNonResumableRowClickOpensFreshPanel() {
  const {win, posted} = makeWebview(BODY_ATTRS);
  disableWorkspaceFilter(win);
  sendHistory(win, posted, [
    session({id: '', task_id: null, has_events: false, is_running: false}),
  ]);
  const rows = historyRows(win);
  assert.strictEqual(rows.length, 1);
  rows[0].click();
  const open = lastMessage(posted, 'openChatPanel');
  assert.ok(open, 'a non-resumable row still opens a panel');
  assert.strictEqual(
    open.chatId,
    undefined,
    'nothing to resume: the panel starts a fresh conversation',
  );
  assert.ok(sidebarOpen(win), 'sidebar still open');
  win.close();
  console.log('PASS a non-resumable row click opens a fresh panel');
}

function testBroadcastsAndVisibilityRefresh() {
  const {win, posted} = makeWebview(BODY_ATTRS);
  const before = byType(posted, 'getHistory').length;

  send(win, {type: 'tasks_updated'});
  const afterBroadcast = byType(posted, 'getHistory').length;
  assert.ok(
    afterBroadcast > before,
    'tasks_updated must refresh the (always open) history panel',
  );

  Object.defineProperty(win.document, 'visibilityState', {
    value: 'hidden',
    configurable: true,
  });
  win.document.dispatchEvent(new win.Event('visibilitychange'));
  assert.strictEqual(
    byType(posted, 'getHistory').length,
    afterBroadcast,
    'going hidden must not refresh',
  );

  Object.defineProperty(win.document, 'visibilityState', {
    value: 'visible',
    configurable: true,
  });
  win.document.dispatchEvent(new win.Event('visibilitychange'));
  assert.ok(
    byType(posted, 'getHistory').length > afterBroadcast,
    'a re-shown webview must re-request history to catch up',
  );
  win.close();
  console.log('PASS broadcasts and re-shown visibility refresh history');
}

function testRegistrySnapshotsNeverCloseThePanel() {
  const {win, posted} = makeWebview(BODY_ATTRS);
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'other-tab', chatId: 'c2', title: 'Other', workDir: ''}],
  });
  send(win, {type: 'tabs_state', tabs: []});
  assert.strictEqual(
    byType(posted, 'closePanel').length,
    0,
    'the history panel is not a registry tab: snapshots must not close it',
  );
  assert.ok(sidebarOpen(win), 'sidebar untouched by snapshots');
  win.close();
  console.log('PASS registry snapshots never close the history panel');
}

// The hiding rules are pure CSS. jsdom applies <style> sheets to
// getComputedStyle for plain (non-@media) selectors, so the
// editor-tab-mode rule is verified end-to-end; the remote-desktop rule
// lives inside a `@media (width >= 900px)` block that jsdom's cascade
// does not evaluate, so its selector is checked textually instead.
function testBurgerAndChromeHiddenByCss() {
  const mainCss = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

  const {win} = makeWebview(BODY_ATTRS);
  const style = win.document.createElement('style');
  style.textContent = mainCss;
  win.document.head.appendChild(style);

  const display = id =>
    win.getComputedStyle(win.document.getElementById(id)).display;
  assert.strictEqual(
    display('menu-btn'),
    'none',
    'editor-tab surfaces have no burger',
  );
  assert.strictEqual(
    display('input-area'),
    'none',
    'history-panel mode hides the composer',
  );
  assert.strictEqual(
    display('output'),
    'none',
    'history-panel mode hides the transcript',
  );
  assert.strictEqual(
    display('sidebar-close'),
    'none',
    'no close button: the panel cannot be dismissed from inside',
  );
  assert.strictEqual(
    display('sidebar-resizer'),
    'none',
    'the hosting VS Code sidebar owns resizing',
  );
  assert.notStrictEqual(
    display('sidebar'),
    'none',
    'the sidebar itself is the panel and stays visible',
  );
  win.close();

  // A plain sidebar-mode webview drops the burger too: its history
  // lives in the PRIMARY sidebar now, exactly like editor-tabs mode.
  const plain = makeWebview(' class=""');
  const plainStyle = plain.win.document.createElement('style');
  plainStyle.textContent = mainCss;
  plain.win.document.head.appendChild(plainStyle);
  assert.strictEqual(
    plain.win
      .getComputedStyle(plain.win.document.getElementById('menu-btn'))
      .display,
    'none',
    'the secondary-sidebar webview has no burger either',
  );
  plain.win.close();

  const remoteCss = fs.readFileSync(
    path.join(MEDIA, 'remote-codex.css'),
    'utf8',
  );
  const desktopBlock = remoteCss.slice(remoteCss.indexOf('@media (width >= 900px)'));
  assert.ok(
    /body\.remote-chat\.remote-desktop #menu-btn,\s*body\.remote-chat\.remote-desktop #sidebar-close \{\s*display: none;/.test(
      desktopBlock,
    ),
    'non-mobile remote hides the burger and the drawer close button',
  );
  console.log('PASS burger and drawer chrome are hidden where required');
}

testBootOpensSidebarAndLoadsHistory();
testResumableRowClickPostsOpenChatPanel();
testNonResumableRowClickOpensFreshPanel();
testBroadcastsAndVisibilityRefresh();
testRegistrySnapshotsNeverCloseThePanel();
testBurgerAndChromeHiddenByCss();
console.log('All historyPanelMode tests passed.');
