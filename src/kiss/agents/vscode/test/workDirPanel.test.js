// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the "Working directory" entry of the "..."
// menu and the panel it opens: a path box with an Open button, a folder
// button, and the directories opened so far (most recently opened
// first, as the daemon reports them in configData.recent_work_dirs).
//
// The panel means something different on each surface:
//   * remote webapp (body.remote-chat): a typed / listed directory is
//     checked through the daemon's listDir ('workdir:<n>' token) and then
//     adopted as the instance's workspace (saveConfig + setWorkDir);
//     the folder button opens the in-page folder browser.
//   * VS Code webview: the pick goes to the extension host (openWorkDir /
//     pickWorkDir), which only checks the folder exists -- it answers
//     workDirPicked (or workDirError for a bad path) and never opens the
//     folder as the window's workspace.  The verified folder becomes the
//     ACTIVE CHAT TAB's working directory: its next task runs there
//     (submit.workDir, plus submit.tabScopeWorkDir = the workspace when
//     the folder lies outside it so the tab stays in this window); the
//     window, the workspace and the other tabs are untouched.

/* global require, __dirname, console, process */

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
  if (remote) html = html.replace('<body', '<body class="remote-chat"');

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function byId(win, id) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  return node;
}

function typeInto(win, id, value) {
  const node = byId(win, id);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function pressEnter(win, id) {
  byId(win, id).dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

function msgs(posted, type) {
  return posted.filter(m => m && m.type === type);
}

// Messages are created inside the JSDOM realm, whose Object prototype is
// not Node's, so structural equality goes through JSON.
function assertJsonEqual(actual, expected) {
  assert.strictEqual(JSON.stringify(actual), JSON.stringify(expected));
}

function panelOpen(win) {
  return byId(win, 'workdir-panel').classList.contains('open');
}

/** Open the "..." menu and click its "Working directory" item. */
function openPanelViaMenu(win) {
  click(win, byId(win, 'more-btn'));
  assert.ok(
    byId(win, 'more-menu').classList.contains('open'),
    'the "..." button must open its menu',
  );
  click(win, byId(win, 'workdir-btn'));
}

const NOW = Date.now() / 1000;
const RECENTS = [
  {path: '/home/u/older', ts: NOW - 3 * 3600},
  {path: '/home/u/newest', ts: NOW - 30},
  {path: '/home/u/middle', ts: NOW - 2 * 86400},
];

function rowPaths(win) {
  return Array.from(
    win.document.querySelectorAll('#workdir-list .workdir-item-path'),
  ).map(el => el.textContent);
}

// ---------------------------------------------------------------------------

function testMenuItemIsFirstAndOpensPanel(remote) {
  const {win, posted} = makeWebview({remote});
  const items = win.document.querySelectorAll('#more-menu .more-menu-item');
  assert.ok(items.length >= 2, 'the "..." menu must list several items');
  assert.strictEqual(items[0].id, 'workdir-btn');
  assert.strictEqual(
    items[0].querySelector('.more-item-label').textContent.trim(),
    'Working directory',
  );
  assert.ok(!panelOpen(win), 'the panel starts closed');

  const before = msgs(posted, 'getConfig').length;
  openPanelViaMenu(win);
  assert.ok(panelOpen(win), 'clicking the item opens the panel');
  assert.ok(
    byId(win, 'workdir-overlay').classList.contains('open'),
    'the backdrop opens with the panel',
  );
  assert.ok(
    !byId(win, 'more-menu').classList.contains('open'),
    'the menu closes once an item is clicked',
  );
  assert.strictEqual(
    msgs(posted, 'getConfig').length,
    before + 1,
    'opening the panel refreshes the list from the daemon',
  );

  // The panel's controls in order: path box, Open, folder button, list.
  const panel = byId(win, 'workdir-panel');
  const order = Array.from(
    panel.querySelectorAll(
      '#workdir-input, #workdir-open-btn, #workdir-pick-btn, #workdir-list',
    ),
  ).map(el => el.id);
  assert.deepStrictEqual(order, [
    'workdir-input',
    'workdir-open-btn',
    'workdir-pick-btn',
    'workdir-list',
  ]);
  assert.strictEqual(byId(win, 'workdir-input').value, '');
  assert.ok(byId(win, 'workdir-open-btn').disabled, 'Open waits for text');
  typeInto(win, 'workdir-input', '  ');
  assert.ok(byId(win, 'workdir-open-btn').disabled, 'blank text is no path');
  typeInto(win, 'workdir-input', '/tmp/x');
  assert.ok(!byId(win, 'workdir-open-btn').disabled);

  // Empty list before any configData carries directories.
  assert.ok(byId(win, 'workdir-empty'), 'the empty state is shown');

  // Close button, then overlay, both close it.
  click(win, byId(win, 'workdir-panel-close'));
  assert.ok(!panelOpen(win));
  openPanelViaMenu(win);
  assert.strictEqual(
    byId(win, 'workdir-input').value,
    '',
    'reopening clears the path box',
  );
  click(win, byId(win, 'workdir-overlay'));
  assert.ok(!panelOpen(win));
}

function testRecentsRenderNewestFirst(remote) {
  const {win} = makeWebview({remote});
  send(win, {type: 'configData', config: {recent_work_dirs: RECENTS}});
  openPanelViaMenu(win);
  assert.deepStrictEqual(rowPaths(win), [
    '/home/u/newest',
    '/home/u/older',
    '/home/u/middle',
  ]);
  const agos = Array.from(
    win.document.querySelectorAll('#workdir-list .workdir-item-ago'),
  ).map(el => el.textContent);
  assert.deepStrictEqual(agos, [
    'opened just now',
    'opened 3 hours ago',
    'opened 2 days ago',
  ]);
  assert.strictEqual(
    win.document.getElementById('workdir-empty'),
    null,
    'no empty state once rows exist',
  );

  // A later configData (another window opened a folder) repaints the
  // open panel; junk rows are dropped.
  send(win, {
    type: 'configData',
    config: {
      recent_work_dirs: [
        {path: '/home/u/brand-new', ts: NOW},
        {path: '', ts: NOW},
        {path: '/home/u/no-ts'},
        'junk',
        RECENTS[1],
      ],
    },
  });
  assert.deepStrictEqual(rowPaths(win), [
    '/home/u/brand-new',
    '/home/u/newest',
  ]);

  // A configData without the list leaves the rows alone.
  send(win, {type: 'configData', config: {}});
  assert.deepStrictEqual(rowPaths(win), [
    '/home/u/brand-new',
    '/home/u/newest',
  ]);
}

function testRemoteTypedPathIsCheckedThenAdopted() {
  const {win, posted} = makeWebview({remote: true});
  openPanelViaMenu(win);
  typeInto(win, 'workdir-input', ' /srv/project ');
  pressEnter(win, 'workdir-input');
  let listDirs = msgs(posted, 'listDir');
  assert.strictEqual(listDirs.length, 1, 'Enter asks the daemon to list it');
  assert.strictEqual(listDirs[0].path, '/srv/project');
  assert.strictEqual(listDirs[0].workDir, '/srv/project');
  assert.strictEqual(listDirs[0].token, 'workdir:1');
  assert.ok(panelOpen(win), 'the panel waits for the answer');

  // A stale token (some other listing) is ignored.
  send(win, {type: 'dirListing', token: 'workdir:0', error: 'nope'});
  assert.ok(byId(win, 'workdir-error').hidden);

  // Not a folder: the error shows inside the panel, nothing is saved.
  send(win, {
    type: 'dirListing',
    token: 'workdir:1',
    error: 'Not a directory: /srv/project',
  });
  assert.ok(panelOpen(win));
  assert.ok(!byId(win, 'workdir-error').hidden);
  assert.strictEqual(
    byId(win, 'workdir-error').textContent,
    'Not a directory: /srv/project',
  );
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);

  // Second try through the Open button: the daemon lists it (with its
  // canonical spelling), so it is adopted as the workspace.
  typeInto(win, 'workdir-input', '/srv/project2/');
  click(win, byId(win, 'workdir-open-btn'));
  listDirs = msgs(posted, 'listDir');
  assert.strictEqual(listDirs.length, 2);
  assert.strictEqual(listDirs[1].token, 'workdir:2');
  send(win, {
    type: 'dirListing',
    token: 'workdir:2',
    path: '/srv/project2',
    entries: [],
  });
  const saves = msgs(posted, 'saveConfig');
  assert.strictEqual(saves.length, 1);
  assertJsonEqual(saves[0].config, {work_dir: '/srv/project2'});
  const pins = msgs(posted, 'setWorkDir');
  assert.strictEqual(pins[pins.length - 1].workDir, '/srv/project2');
  assert.ok(!panelOpen(win), 'a successful open closes the panel');
  assert.ok(byId(win, 'workdir-error').hidden, 'the old error is gone');

  // Nothing is posted to the VS Code host on the remote surface.
  assert.strictEqual(msgs(posted, 'openWorkDir').length, 0);
}

function testRemoteRecentRowAndRootGuard() {
  const {win, posted} = makeWebview({remote: true});
  send(win, {type: 'configData', config: {recent_work_dirs: RECENTS}});
  openPanelViaMenu(win);

  // Clicking a row checks that row's directory.
  const rows = win.document.querySelectorAll('#workdir-list .workdir-item');
  click(win, rows[1].querySelector('.workdir-item-path'));
  let listDirs = msgs(posted, 'listDir');
  assert.strictEqual(listDirs[listDirs.length - 1].path, '/home/u/older');

  // Keyboard: Space on a focused row does the same; other keys do not.
  rows[2].dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Tab', bubbles: true}),
  );
  assert.strictEqual(msgs(posted, 'listDir').length, listDirs.length);
  rows[2].dispatchEvent(
    new win.KeyboardEvent('keydown', {key: ' ', bubbles: true}),
  );
  listDirs = msgs(posted, 'listDir');
  assert.strictEqual(listDirs[listDirs.length - 1].path, '/home/u/middle');

  // The reply to the superseded first click is ignored; the second's
  // adopts its folder.
  send(win, {
    type: 'dirListing',
    token: listDirs[0].token,
    path: '/home/u/older',
  });
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
  send(win, {
    type: 'dirListing',
    token: listDirs[listDirs.length - 1].token,
    path: '/home/u/middle',
  });
  assertJsonEqual(msgs(posted, 'saveConfig')[0].config, {
    work_dir: '/home/u/middle',
  });
  assert.ok(!panelOpen(win));

  // A file-system root is refused before asking the daemon.
  openPanelViaMenu(win);
  const before = msgs(posted, 'listDir').length;
  typeInto(win, 'workdir-input', '/');
  pressEnter(win, 'workdir-input');
  assert.strictEqual(msgs(posted, 'listDir').length, before);
  assert.ok(!byId(win, 'workdir-error').hidden);
  assert.ok(/root/.test(byId(win, 'workdir-error').textContent));

  // Closing the panel abandons a pending check: its late reply must not
  // adopt anything.
  typeInto(win, 'workdir-input', '/home/u/late');
  pressEnter(win, 'workdir-input');
  const pending = msgs(posted, 'listDir').pop();
  click(win, byId(win, 'workdir-panel-close'));
  send(win, {type: 'dirListing', token: pending.token, path: '/home/u/late'});
  assert.strictEqual(msgs(posted, 'saveConfig').length, 1);
}

function testRemoteFolderButtonOpensPicker() {
  const {win, posted} = makeWebview({remote: true});
  openPanelViaMenu(win);
  assert.strictEqual(win.document.getElementById('folder-picker'), null);
  click(win, byId(win, 'workdir-pick-btn'));
  const picker = byId(win, 'folder-picker');
  assert.ok(!picker.hidden, 'the folder browser opens');
  assert.strictEqual(
    picker.querySelector('#folder-picker-title').textContent,
    'Open Folder as Working Directory',
  );
  assert.strictEqual(msgs(posted, 'pickWorkDir').length, 0);
  // The browser lists through the daemon, never through the host.
  const listing = msgs(posted, 'listDir').pop();
  assert.ok(/^picker:/.test(listing.token));

  // A pick made in the browser closes both dialogs.
  send(win, {
    type: 'dirListing',
    token: listing.token,
    path: listing.path,
    entries: [{name: 'app', isDir: true}],
  });
  click(win, picker.querySelector('.folder-picker-item'));
  click(win, picker.querySelector('.folder-picker-select'));
  assert.ok(picker.hidden, 'the folder browser closes on a pick');
  assert.ok(!panelOpen(win), 'the Working directory panel closes too');
  const saves = msgs(posted, 'saveConfig');
  assert.ok(saves.length >= 1);
  assert.ok(/\/app$/.test(saves[saves.length - 1].config.work_dir));
}

function testVsCodeAsksTheHost() {
  const {win, posted} = makeWebview({remote: false});
  send(win, {type: 'configData', config: {recent_work_dirs: RECENTS}});
  openPanelViaMenu(win);
  const tabId = win._testApi.getActiveTabId();

  // Typed path: the host checks the folder (no daemon listDir); the
  // request names the tab so the reply lands on it.
  typeInto(win, 'workdir-input', '/work/repo');
  pressEnter(win, 'workdir-input');
  assertJsonEqual(msgs(posted, 'openWorkDir'), [
    {type: 'openWorkDir', path: '/work/repo', tabId},
  ]);
  assert.strictEqual(msgs(posted, 'listDir').length, 0);
  assert.ok(panelOpen(win), 'the panel stays until the host answers');

  // The host's failure lands in the panel.
  send(win, {type: 'workDirError', text: 'Not a directory: /work/repo'});
  assert.ok(!byId(win, 'workdir-error').hidden);
  assert.strictEqual(
    byId(win, 'workdir-error').textContent,
    'Not a directory: /work/repo',
  );

  // A recent row asks the host for that folder and clears the error.
  click(win, win.document.querySelector('#workdir-list .workdir-item'));
  const opens = msgs(posted, 'openWorkDir');
  assert.strictEqual(opens.length, 2);
  assert.strictEqual(opens[1].path, '/home/u/newest');
  assert.ok(byId(win, 'workdir-error').hidden);

  // The folder button uses the editor's own dialog, not the in-page one.
  click(win, byId(win, 'workdir-pick-btn'));
  assertJsonEqual(msgs(posted, 'pickWorkDir'), [{type: 'pickWorkDir', tabId}]);
  assert.strictEqual(win.document.getElementById('folder-picker'), null);

  // A root is refused locally.
  typeInto(win, 'workdir-input', '/');
  click(win, byId(win, 'workdir-open-btn'));
  assert.strictEqual(msgs(posted, 'openWorkDir').length, 2);
  assert.ok(/root/.test(byId(win, 'workdir-error').textContent));

  // Nothing settings-related is saved from this surface.
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
}

/** Submit *text* from the composer; the posted `submit` message. */
function submitPrompt(win, posted, text) {
  const before = msgs(posted, 'submit').length;
  byId(win, 'task-input').value = text;
  click(win, byId(win, 'send-btn'));
  const sent = msgs(posted, 'submit');
  assert.strictEqual(sent.length, before + 1, 'one submit is posted');
  return sent[sent.length - 1];
}

/** The tab ids the tab bar currently shows. */
function shownTabIds(win) {
  return Array.from(win.document.querySelectorAll('.chat-tab')).map(
    el => el.dataset.tabId,
  );
}

/** Pick *dir* for the active tab through the host round trip. */
function pickForActiveTab(win, posted, dir) {
  const tabId = win._testApi.getActiveTabId();
  const asked = msgs(posted, 'openWorkDir').length;
  openPanelViaMenu(win);
  typeInto(win, 'workdir-input', dir);
  pressEnter(win, 'workdir-input');
  const opens = msgs(posted, 'openWorkDir');
  assert.strictEqual(opens.length, asked + 1);
  assert.strictEqual(opens[asked].tabId, tabId, 'the request names the tab');
  send(win, {type: 'workDirPicked', path: dir, tabId});
  assert.ok(!panelOpen(win), 'a successful pick closes the panel');
  return tabId;
}

function testVsCodePickChangesOnlyTheTab() {
  const {win, posted} = makeWebview({remote: false});
  // The window's folder, as the host reports it.
  send(win, {
    type: 'configData',
    config: {work_dir: '/work/ws', recent_work_dirs: RECENTS},
  });
  openPanelViaMenu(win);
  assert.strictEqual(
    byId(win, 'workdir-current').textContent,
    'Current: /work/ws',
    'the panel names the folder the next task would run in',
  );
  click(win, byId(win, 'workdir-panel-close'));

  // The host verified the typed folder: the active chat is pinned to
  // it, the panel closes, and nothing about the window changes.
  const firstTab = pickForActiveTab(win, posted, '/elsewhere/repo');
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  openPanelViaMenu(win);
  assert.strictEqual(
    byId(win, 'workdir-current').textContent,
    'Current: /elsewhere/repo',
  );
  click(win, byId(win, 'workdir-panel-close'));

  // The next task runs there; the tab stays scoped to this window's
  // workspace, which the picked folder lies outside of.
  let sub = submitPrompt(win, posted, 'list the files');
  assert.strictEqual(sub.tabId, firstTab);
  assert.strictEqual(sub.workDir, '/elsewhere/repo');
  assert.strictEqual(sub.tabScopeWorkDir, '/work/ws');

  // A pick inside the workspace needs no scope override.
  send(win, {type: 'status', running: false, tabId: firstTab});
  pickForActiveTab(win, posted, '/work/ws/sub');
  sub = submitPrompt(win, posted, 'and again');
  assert.strictEqual(sub.workDir, '/work/ws/sub');
  assert.strictEqual(sub.tabScopeWorkDir, undefined);

  // Another chat tab is untouched by the first tab's pin.
  win._testApi.createNewTab();
  const secondTab = win._testApi.getActiveTabId();
  assert.notStrictEqual(secondTab, firstTab);
  sub = submitPrompt(win, posted, 'third');
  assert.strictEqual(sub.tabId, secondTab);
  assert.strictEqual(sub.workDir, undefined);
  assert.strictEqual(sub.tabScopeWorkDir, undefined);

  // A running tab keeps its directory: the panel refuses locally and
  // asks the host nothing.
  send(win, {type: 'status', running: true, tabId: secondTab});
  const asked = msgs(posted, 'openWorkDir').length;
  openPanelViaMenu(win);
  typeInto(win, 'workdir-input', '/elsewhere/other');
  pressEnter(win, 'workdir-input');
  assert.strictEqual(msgs(posted, 'openWorkDir').length, asked);
  assert.ok(/running task/.test(byId(win, 'workdir-error').textContent));
  click(win, byId(win, 'workdir-pick-btn'));
  assert.strictEqual(msgs(posted, 'pickWorkDir').length, 0);
  assert.ok(panelOpen(win));
  // A late host answer for a tab that started running meanwhile is
  // refused the same way.
  send(win, {
    type: 'workDirPicked',
    path: '/elsewhere/other',
    tabId: secondTab,
  });
  assert.ok(panelOpen(win), 'the panel stays with the refusal');
  assert.ok(/running task/.test(byId(win, 'workdir-error').textContent));
  send(win, {type: 'status', running: false, tabId: secondTab});
  sub = submitPrompt(win, posted, 'fourth');
  assert.strictEqual(sub.workDir, undefined, 'the refused pick left no pin');
}

function testVsCodeLateReplyLandsOnTheTabThatAsked() {
  const {win, posted} = makeWebview({remote: false});
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  const tabA = win._testApi.getActiveTabId();
  openPanelViaMenu(win);
  click(win, byId(win, 'workdir-pick-btn'));
  assertJsonEqual(msgs(posted, 'pickWorkDir'), [
    {type: 'pickWorkDir', tabId: tabA},
  ]);

  // The editor's folder dialog is slow; the user opens another tab
  // meanwhile.  The answer still pins tab A, not the now-active tab B.
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  send(win, {type: 'workDirPicked', path: '/picked/for-a', tabId: tabA});
  assert.ok(!panelOpen(win));
  let sub = submitPrompt(win, posted, 'from b');
  assert.strictEqual(sub.tabId, tabB);
  assert.strictEqual(sub.workDir, undefined, 'tab B was not pinned');

  click(win, win.document.querySelector(`.chat-tab[data-tab-id="${tabA}"]`));
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);
  sub = submitPrompt(win, posted, 'from a');
  assert.strictEqual(sub.workDir, '/picked/for-a', 'tab A runs where it asked');
  assert.strictEqual(sub.tabScopeWorkDir, '/work/ws');

  // A reply for a tab that no longer exists changes nothing.
  send(win, {type: 'workDirPicked', path: '/picked/gone', tabId: 'no-such'});
  send(win, {type: 'status', running: false, tabId: tabA});
  sub = submitPrompt(win, posted, 'again');
  assert.strictEqual(sub.workDir, undefined, 'the pin was consumed by the run');
}

function testVsCodePinSurvivesReplayAndKeepsTheTabVisible() {
  const {win, posted} = makeWebview({remote: false});
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  const tabA = pickForActiveTab(win, posted, '/outside/repo');

  // The pin lies outside the workspace, yet the (not yet registered)
  // tab stays in this window's tab bar -- also after another tab
  // re-renders the strip.
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.deepStrictEqual(shownTabIds(win).sort(), [tabA, tabB].sort());
  click(win, win.document.querySelector(`.chat-tab[data-tab-id="${tabA}"]`));
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);

  // A replay of the chat's previous task (its own work dir in `extra`)
  // reaches the tab before the user submits: the pick still wins.
  send(win, {
    type: 'task_events',
    tabId: tabA,
    chat_id: 'chat-a',
    task_id: 41,
    task: 'the earlier task',
    events: [],
    extra: JSON.stringify({work_dir: '/old/task/dir', startTs: 1, endTs: 2}),
  });
  openPanelViaMenu(win);
  assert.strictEqual(
    byId(win, 'workdir-current').textContent,
    'Current: /outside/repo',
  );
  click(win, byId(win, 'workdir-panel-close'));
  let sub = submitPrompt(win, posted, 'run it');
  assert.strictEqual(sub.workDir, '/outside/repo');
  assert.strictEqual(sub.tabScopeWorkDir, '/work/ws');

  // The pick is consumed: the tab now carries the replayed task's dir.
  send(win, {type: 'status', running: false, tabId: tabA});
  openPanelViaMenu(win);
  assert.strictEqual(
    byId(win, 'workdir-current').textContent,
    'Current: /old/task/dir',
    'without a pin the tab shows its last task directory',
  );
  click(win, byId(win, 'workdir-panel-close'));
  sub = submitPrompt(win, posted, 'once more');
  assert.strictEqual(sub.workDir, '/old/task/dir');
}

function testRemoteCanonicalRootIsRefused() {
  const {win, posted} = makeWebview({remote: true});
  openPanelViaMenu(win);
  // "/tmp/.." passes the lexical root check; the daemon's canonical
  // spelling of it is "/", which must not be adopted.
  typeInto(win, 'workdir-input', '/tmp/..');
  pressEnter(win, 'workdir-input');
  const check = msgs(posted, 'listDir').pop();
  assert.strictEqual(check.path, '/tmp/..');
  send(win, {type: 'dirListing', token: check.token, path: '/', entries: []});
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  assert.ok(panelOpen(win));
  assert.ok(/root/.test(byId(win, 'workdir-error').textContent));
}

function testClosedSheetIsInertAndEscapeCloses(remote) {
  const {win} = makeWebview({remote});
  const panel = byId(win, 'workdir-panel');
  assert.ok(panel.hasAttribute('inert'), 'the closed sheet starts inert');
  assert.strictEqual(panel.getAttribute('aria-hidden'), 'true');

  openPanelViaMenu(win);
  assert.ok(!panel.hasAttribute('inert'));
  assert.strictEqual(panel.getAttribute('aria-hidden'), 'false');
  byId(win, 'workdir-input').focus();
  assert.strictEqual(win.document.activeElement.id, 'workdir-input');

  // Escape closes it and hands focus back to the "..." button.
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
  );
  assert.ok(!panelOpen(win), 'Escape closes the sheet');
  assert.ok(panel.hasAttribute('inert'));
  assert.strictEqual(panel.getAttribute('aria-hidden'), 'true');
  assert.strictEqual(win.document.activeElement.id, 'more-btn');

  // A second Escape with the sheet closed is nobody's business.
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
  );
  assert.ok(!panelOpen(win));

  // The Close button leaves focus alone when it was elsewhere.
  openPanelViaMenu(win);
  byId(win, 'task-input').focus();
  click(win, byId(win, 'workdir-panel-close'));
  assert.strictEqual(win.document.activeElement.id, 'task-input');
}

function testRemoteEscapeDefersToOpenFolderBrowser() {
  const {win} = makeWebview({remote: true});
  openPanelViaMenu(win);
  click(win, byId(win, 'workdir-pick-btn'));
  const picker = byId(win, 'folder-picker');
  assert.ok(!picker.hidden);
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
  );
  assert.ok(panelOpen(win), 'Escape over the folder browser is not ours');
  click(win, picker.querySelector('.folder-picker-close'));
  assert.ok(picker.hidden);
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
  );
  assert.ok(!panelOpen(win));
}

const tests = [
  [
    'menu item first + panel opens (remote)',
    () => testMenuItemIsFirstAndOpensPanel(true),
  ],
  [
    'menu item first + panel opens (vscode)',
    () => testMenuItemIsFirstAndOpensPanel(false),
  ],
  ['recents newest first (remote)', () => testRecentsRenderNewestFirst(true)],
  ['recents newest first (vscode)', () => testRecentsRenderNewestFirst(false)],
  [
    'remote: typed path checked then adopted',
    testRemoteTypedPathIsCheckedThenAdopted,
  ],
  [
    'remote: recent row, root guard, stale replies',
    testRemoteRecentRowAndRootGuard,
  ],
  [
    'remote: folder button opens the in-page browser',
    testRemoteFolderButtonOpensPicker,
  ],
  ['vscode: host opens / picks the folder', testVsCodeAsksTheHost],
  [
    'vscode: a pick changes only the active tab and its next task',
    testVsCodePickChangesOnlyTheTab,
  ],
  [
    'vscode: a late host reply pins the tab that asked',
    testVsCodeLateReplyLandsOnTheTabThatAsked,
  ],
  [
    'vscode: the pin survives a task replay and keeps the tab shown',
    testVsCodePinSurvivesReplayAndKeepsTheTabVisible,
  ],
  [
    'remote: canonical root spelling is refused',
    testRemoteCanonicalRootIsRefused,
  ],
  [
    'closed sheet inert + Escape (remote)',
    () => testClosedSheetIsInertAndEscapeCloses(true),
  ],
  [
    'closed sheet inert + Escape (vscode)',
    () => testClosedSheetIsInertAndEscapeCloses(false),
  ],
  [
    'remote: Escape defers to the folder browser',
    testRemoteEscapeDefersToOpenFolderBrowser,
  ],
];

let failed = 0;
for (const [name, fn] of tests) {
  try {
    fn();
    console.log('ok - ' + name);
  } catch (e) {
    failed += 1;
    console.log('not ok - ' + name);
    console.log(e && e.stack ? e.stack : String(e));
  }
}
if (failed) {
  console.log(`${failed} of ${tests.length} tests failed`);
  process.exit(1);
}
console.log(`all ${tests.length} workDirPanel tests passed`);
