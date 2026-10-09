// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the "Working directory" entry of the "..."
// menu and the panel it opens: a path box with an Open button, a folder
// button, and the directories opened so far (most recently opened
// first, as the daemon reports them in configData.recent_work_dirs).
//
// The working directory is ONE global value (config.json work_dir) that
// every task on every surface runs in; the panel's "Current:" line shows
// it and a pick changes it for every tab.  Only the check differs per
// surface:
//   * remote webapp (body.remote-chat): a typed / listed directory is
//     checked through the daemon's listDir ('workdir:<n>' token) and then
//     sent as setWorkDir (never saveConfig); the folder button opens the
//     in-page folder browser.
//   * VS Code webview: the pick goes to the extension host (openWorkDir /
//     pickWorkDir, no tabId), which checks the folder exists, sends it to
//     the daemon itself and answers workDirPicked {path} (or workDirError
//     for a bad path); the webview posts no setWorkDir of its own and the
//     window's workspace is never changed.
// Either way `submit` carries no workDir (the daemon runs the task in the
// global value), a running task never blocks a pick (it keeps the folder
// it started in), and the daemon's workDirChanged broadcast re-scopes the
// webview to a pick made on another surface.

/* global require, __dirname, console, process */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// Every window a test opened; closed after the test so the webview's
// timers (the 1 s meta-panel poll a desktop layout starts, Monaco's
// load timeout) do not keep this node process alive once the tests
// are done.
const openWindows = [];

function makeWebview(opts) {
  const {remote = false, desktop = false} = opts || {};
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
  openWindows.push(win);
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
  if (desktop) {
    // The remote desktop layout (the activity bar's Explorer / Source
    // Control views) is gated on this media query.
    win.requestAnimationFrame = cb => {
      cb();
      return 0;
    };
    win.cancelAnimationFrame = () => {};
    win.matchMedia = query => ({
      matches: query === '(min-width: 900px)',
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    });
  }
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

/** The panel's "Current:" line ('' while it is hidden). */
function currentLine(win) {
  const el = byId(win, 'workdir-current');
  return el.hidden ? '' : el.textContent;
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
  assert.strictEqual(currentLine(win), '', 'no directory known yet');
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

  // Not a folder: the error shows inside the panel, nothing is sent.
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
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);

  // Second try through the Open button: the daemon lists it (with its
  // canonical spelling), so it becomes the global working directory
  // through setWorkDir -- the settings file is not written from here.
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
  // api.js spreads the fields before `type`, hence the key order.
  assertJsonEqual(msgs(posted, 'setWorkDir'), [
    {workDir: '/srv/project2', type: 'setWorkDir'},
  ]);
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
  assert.ok(!panelOpen(win), 'a successful open closes the panel');
  assert.ok(byId(win, 'workdir-error').hidden, 'the old error is gone');

  // The client re-scopes at once, without waiting for the broadcast,
  // and the next task carries no directory of its own.
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /srv/project2');
  click(win, byId(win, 'workdir-panel-close'));
  const sub = submitPrompt(win, posted, 'list the files');
  assert.strictEqual(sub.workDir, undefined);

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
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  send(win, {
    type: 'dirListing',
    token: listDirs[listDirs.length - 1].token,
    path: '/home/u/middle',
  });
  assertJsonEqual(msgs(posted, 'setWorkDir'), [
    {workDir: '/home/u/middle', type: 'setWorkDir'},
  ]);
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
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 1);
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
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
  const sets = msgs(posted, 'setWorkDir');
  assert.ok(sets.length >= 1);
  assert.ok(/\/app$/.test(sets[sets.length - 1].workDir));
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
}

function testVsCodeAsksTheHost() {
  const {win, posted} = makeWebview({remote: false});
  send(win, {type: 'configData', config: {recent_work_dirs: RECENTS}});
  openPanelViaMenu(win);

  // Typed path: the host checks the folder (no daemon listDir).  The
  // request names no tab: the pick is global, not the tab's.
  typeInto(win, 'workdir-input', '/work/repo');
  pressEnter(win, 'workdir-input');
  assertJsonEqual(msgs(posted, 'openWorkDir'), [
    {type: 'openWorkDir', path: '/work/repo'},
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
  assert.strictEqual(opens[1].tabId, undefined);
  assert.ok(byId(win, 'workdir-error').hidden);

  // The folder button uses the editor's own dialog, not the in-page one.
  click(win, byId(win, 'workdir-pick-btn'));
  assertJsonEqual(msgs(posted, 'pickWorkDir'), [{type: 'pickWorkDir'}]);
  assert.strictEqual(win.document.getElementById('folder-picker'), null);

  // A root is refused locally.
  typeInto(win, 'workdir-input', '/');
  click(win, byId(win, 'workdir-open-btn'));
  assert.strictEqual(msgs(posted, 'openWorkDir').length, 2);
  assert.ok(/root/.test(byId(win, 'workdir-error').textContent));

  // The host's verified answer closes the panel and is the directory
  // the panel names from then on.
  send(win, {type: 'workDirPicked', path: '/home/u/newest'});
  assert.ok(!panelOpen(win), 'a verified pick closes the panel');
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /home/u/newest');

  // The host talks to the daemon itself: this webview sends neither
  // setWorkDir nor saveConfig.
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
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

/** The ids of every open tab (a background chat has no strip entry, so
 *  the registry, not the DOM, lists them). */
function openTabIds(win) {
  return Array.from(win._testApi.openTabs(), t => t.id);
}

/** Bring a background chat on screen the way the Chats panel's pick does. */
function switchToChat(win, tabId) {
  assert.ok(openTabIds(win).includes(tabId), 'tab ' + tabId + ' is open');
  win._testApi.switchToTab(tabId);
}

/** Pick *dir* through the VS Code host round trip (panel -> host -> panel). */
function pickViaHost(win, posted, dir) {
  const asked = msgs(posted, 'openWorkDir').length;
  openPanelViaMenu(win);
  typeInto(win, 'workdir-input', dir);
  pressEnter(win, 'workdir-input');
  const opens = msgs(posted, 'openWorkDir');
  assert.strictEqual(opens.length, asked + 1);
  assertJsonEqual(opens[asked], {type: 'openWorkDir', path: dir});
  send(win, {type: 'workDirPicked', path: dir});
  assert.ok(!panelOpen(win), 'a successful pick closes the panel');
}

function testVsCodePickIsGlobal() {
  const {win, posted} = makeWebview({remote: false});
  // The daemon's global directory arrives untouched in configData.
  send(win, {
    type: 'configData',
    config: {work_dir: '/work/ws', recent_work_dirs: RECENTS},
  });
  openPanelViaMenu(win);
  assert.strictEqual(
    currentLine(win),
    'Current: /work/ws',
    'the panel names the directory the next task would run in',
  );
  click(win, byId(win, 'workdir-panel-close'));
  const firstTab = win._testApi.getActiveTabId();

  // The host verified the typed folder: it is the global directory
  // now; the webview itself writes nothing to the daemon.
  pickViaHost(win, posted, '/elsewhere/repo');
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /elsewhere/repo');
  click(win, byId(win, 'workdir-panel-close'));

  // The next task carries no directory: the daemon runs it in the
  // global value.  No workspace scope travels with it either.
  let sub = submitPrompt(win, posted, 'list the files');
  assert.strictEqual(sub.tabId, firstTab);
  assert.strictEqual(sub.workDir, undefined);
  assert.strictEqual(sub.tabScopeWorkDir, undefined);

  // Every other tab sees the same directory: it is not the tab's.
  win._testApi.createNewTab();
  const secondTab = win._testApi.getActiveTabId();
  assert.notStrictEqual(secondTab, firstTab);
  assert.deepStrictEqual(openTabIds(win).sort(), [firstTab, secondTab].sort());
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /elsewhere/repo');
  click(win, byId(win, 'workdir-panel-close'));
  sub = submitPrompt(win, posted, 'third');
  assert.strictEqual(sub.tabId, secondTab);
  assert.strictEqual(sub.workDir, undefined);

  // A running task does not block a pick: it keeps the folder it
  // started in, the next task runs in the new one.
  send(win, {type: 'status', running: true, tabId: secondTab});
  openPanelViaMenu(win);
  typeInto(win, 'workdir-input', '/elsewhere/other');
  pressEnter(win, 'workdir-input');
  let opens = msgs(posted, 'openWorkDir');
  assertJsonEqual(opens[opens.length - 1], {
    type: 'openWorkDir',
    path: '/elsewhere/other',
  });
  assert.ok(byId(win, 'workdir-error').hidden, 'no refusal');
  click(win, byId(win, 'workdir-pick-btn'));
  assertJsonEqual(msgs(posted, 'pickWorkDir'), [{type: 'pickWorkDir'}]);
  send(win, {type: 'workDirPicked', path: '/elsewhere/other'});
  assert.ok(!panelOpen(win), 'the pick lands while the task runs');
  send(win, {type: 'status', running: false, tabId: secondTab});
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /elsewhere/other');
  click(win, byId(win, 'workdir-panel-close'));
  switchToChat(win, firstTab);
  assert.strictEqual(win._testApi.getActiveTabId(), firstTab);
  openPanelViaMenu(win);
  assert.strictEqual(
    currentLine(win),
    'Current: /elsewhere/other',
    'the first tab follows the same global directory',
  );
  click(win, byId(win, 'workdir-panel-close'));
  opens = msgs(posted, 'openWorkDir');
  assert.ok(
    opens.every(m => m.tabId === undefined),
    'no request ever named a tab',
  );
}

function testVsCodeLateReplyRescopesEveryTab() {
  const {win, posted} = makeWebview({remote: false});
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  const tabA = win._testApi.getActiveTabId();
  openPanelViaMenu(win);
  click(win, byId(win, 'workdir-pick-btn'));
  assertJsonEqual(msgs(posted, 'pickWorkDir'), [{type: 'pickWorkDir'}]);

  // The editor's folder dialog is slow; the user opens another tab
  // meanwhile (the panel stays up across the switch).  The answer is
  // the global directory, so it closes the waiting panel and both
  // tabs run there.
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.ok(panelOpen(win));
  send(win, {type: 'workDirPicked', path: '/picked/late'});
  assert.ok(!panelOpen(win), 'the waiting panel closes on the answer');
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /picked/late');
  click(win, byId(win, 'workdir-panel-close'));
  let sub = submitPrompt(win, posted, 'from b');
  assert.strictEqual(sub.tabId, tabB);
  assert.strictEqual(sub.workDir, undefined);

  switchToChat(win, tabA);
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /picked/late');
  click(win, byId(win, 'workdir-panel-close'));
  sub = submitPrompt(win, posted, 'from a');
  assert.strictEqual(sub.workDir, undefined);

  // A blank or root answer is nobody's directory.
  send(win, {type: 'workDirPicked', path: '   '});
  send(win, {type: 'workDirPicked', path: '/'});
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /picked/late');
}

function testGlobalDirWinsOverReplayedTaskDir() {
  const {win, posted} = makeWebview({remote: false});
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  const tabA = win._testApi.getActiveTabId();

  // A replay of the chat's previous task (its own work dir in `extra`)
  // reaches the tab: the global directory still wins for the next run.
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
  assert.strictEqual(currentLine(win), 'Current: /work/ws');
  click(win, byId(win, 'workdir-panel-close'));
  let sub = submitPrompt(win, posted, 'run it');
  assert.strictEqual(sub.workDir, undefined);

  // Only while the daemon reports no global directory does the panel
  // fall back to the tab's last task directory.
  send(win, {type: 'status', running: false, tabId: tabA});
  send(win, {type: 'configData', config: {}});
  openPanelViaMenu(win);
  assert.strictEqual(
    currentLine(win),
    'Current: /old/task/dir',
    'without a global directory the tab shows its last task directory',
  );
  click(win, byId(win, 'workdir-panel-close'));
  // A root as the global value is no directory to run in either.
  send(win, {type: 'configData', config: {work_dir: '/'}});
  openPanelViaMenu(win);
  assert.strictEqual(
    currentLine(win),
    'Current: /old/task/dir',
    'a root-valued global directory falls back the same way',
  );
  click(win, byId(win, 'workdir-panel-close'));

  // A pick made on another surface arrives as the daemon's broadcast
  // and re-scopes this window too.
  send(win, {type: 'workDirChanged', workDir: '/picked/elsewhere'});
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /picked/elsewhere');
  click(win, byId(win, 'workdir-panel-close'));
  sub = submitPrompt(win, posted, 'once more');
  assert.strictEqual(sub.workDir, undefined);
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);
}

function testRemoteWorkDirChangedRescopes() {
  const {win, posted} = makeWebview({remote: true});
  // A directory an older build left in sessionStorage is not this
  // browser tab's own working directory any more: the daemon's value
  // wins, and configData is not echoed back as setWorkDir.
  win.sessionStorage.setItem('sorcar-work-dir', '/stale/per-tab');
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0, 'no echo');
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /work/ws');
  click(win, byId(win, 'workdir-pick-btn'));
  const picker = byId(win, 'folder-picker');
  assert.ok(!picker.hidden);

  // Another surface picked a folder: the daemon's broadcast re-scopes
  // this client -- the "Current:" line follows and the folder browser,
  // which was browsing for a pick that is now moot, closes.  Nothing
  // is echoed back to the daemon (no setWorkDir loop, no saveConfig).
  send(win, {type: 'workDirChanged', workDir: '/picked/on/vscode'});
  assert.strictEqual(currentLine(win), 'Current: /picked/on/vscode');
  assert.ok(picker.hidden, 'the folder browser closes');
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0);
  assert.strictEqual(msgs(posted, 'saveConfig').length, 0);

  // The same value again, or a non-string, changes nothing.
  click(win, byId(win, 'workdir-panel-close'));
  openPanelViaMenu(win);
  click(win, byId(win, 'workdir-pick-btn'));
  assert.ok(!picker.hidden);
  send(win, {type: 'workDirChanged', workDir: '/picked/on/vscode'});
  send(win, {type: 'workDirChanged', workDir: 42});
  assert.ok(!picker.hidden, 'an unchanged value leaves the browser alone');
  assert.strictEqual(currentLine(win), 'Current: /picked/on/vscode');
  click(win, picker.querySelector('.folder-picker-close'));
  click(win, byId(win, 'workdir-panel-close'));
  const sub = submitPrompt(win, posted, 'go');
  assert.strictEqual(sub.workDir, undefined);
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

function testConfigDataRepaintsCurrentLine() {
  // A configData that carries a new work_dir (a reconnect, a settings
  // save) repaints the open panel's "Current:" line, exactly like the
  // workDirChanged broadcast does.
  const {win} = makeWebview({remote: true});
  send(win, {type: 'configData', config: {work_dir: '/old'}});
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /old');
  send(win, {type: 'configData', config: {work_dir: '/new'}});
  assert.strictEqual(currentLine(win), 'Current: /new');
  assert.strictEqual(byId(win, 'meta-workdir').textContent, '/new');
}

function testOrphanedFileTabBrowsesTheGlobalDir() {
  // A file tab remembers the folder of the chat it was opened from so
  // the views keep browsing it after that chat closes -- but only as a
  // fallback: once the daemon reports a global directory, the Explorer
  // and Source Control views of the orphaned file tab browse THAT.
  const {win, posted} = makeWebview({remote: true, desktop: true});
  send(win, {type: 'configData', config: {work_dir: '/old'}});
  const owner = win._testApi.getActiveTabId();
  send(win, {
    type: 'fileContent',
    path: '/old/a.txt',
    name: 'a.txt',
    content: 'x',
    tabId: owner,
  });
  // The split layout lists content tabs on the content pane's own row.
  const fileTab = win.document.querySelector(
    '#content-tab-list .chat-tab.content-tab',
  );
  assert.ok(fileTab, 'the file opened as a content tab');
  click(win, fileTab);
  click(
    win,
    win.document.querySelector(
      '#tab-list .chat-tab[data-tab-id="' + owner + '"] .chat-tab-close',
    ),
  );
  assert.ok(!openTabIds(win).includes(owner), 'the owning chat is closed');
  // Opening the file slid the task-info panel off the desktop layout;
  // its views reload the moment the drawer brings the panel back.
  assert.ok(
    win.document.body.classList.contains('meta-hidden'),
    'the file open hid the panel',
  );
  const before = msgs(posted, 'listDir').length;
  send(win, {type: 'workDirChanged', workDir: '/new'});
  assert.strictEqual(
    msgs(posted, 'listDir').length,
    before,
    'a hidden Explorer is only marked stale',
  );
  // The Explorer and Source Control sections of the task-info panel
  // follow the new workspace on their own.
  click(win, byId(win, 'meta-drawer'));
  const listing = msgs(posted, 'listDir').pop();
  assert.strictEqual(listing.path, '/new');
  assert.strictEqual(listing.workDir, '/new');
  const status = msgs(posted, 'gitStatus').pop();
  assert.strictEqual(status.workDir, '/new');
  // The desktop layout polls the task info every 5 s for as long as the
  // page lives; closing the window drops that timer so node can exit.
  win.close();
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
    'remote: typed path checked then sent as setWorkDir',
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
    'vscode: a pick is the global directory of every tab',
    testVsCodePickIsGlobal,
  ],
  [
    'vscode: a late host reply re-scopes every tab',
    testVsCodeLateReplyRescopesEveryTab,
  ],
  [
    'vscode: the global directory wins over a replayed task directory',
    testGlobalDirWinsOverReplayedTaskDir,
  ],
  [
    'remote: workDirChanged from the daemon re-scopes the client',
    testRemoteWorkDirChangedRescopes,
  ],
  [
    'remote: canonical root spelling is refused',
    testRemoteCanonicalRootIsRefused,
  ],
  [
    'remote: configData repaints the "Current:" line',
    testConfigDataRepaintsCurrentLine,
  ],
  [
    'remote: an orphaned file tab browses the global directory',
    testOrphanedFileTabBrowsesTheGlobalDir,
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
  } finally {
    while (openWindows.length) openWindows.pop().close();
  }
}
if (failed) {
  console.log(`${failed} of ${tests.length} tests failed`);
  process.exit(1);
}
console.log(`all ${tests.length} workDirPanel tests passed`);
process.exit(0);
