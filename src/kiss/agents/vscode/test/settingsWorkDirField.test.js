// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the Settings panel having NO "Working
// directory" (and no "Memory directory") field, and for what the config
// reply still does without it.
//
// The same media/ files are served to two very different clients:
//
//   * In a VS Code webview the working directory IS the workspace folder
//     open in that window, so a settings box for it had nothing to edit;
//     the form leaves `work_dir` out of the `saveConfig` payload so three
//     windows open on three projects never overwrite one another's stored
//     value.
//   * In the standalone web client the folder is chosen in the "Working
//     directory" panel of the "..." menu, which saves it AND re-pins THIS
//     browser tab (sessionStorage `sorcar-work-dir`, written by the WS
//     shim when the page posts `setWorkDir`).  The config reply still
//     honours that pin -- a second browser tab pointed elsewhere does not
//     drag this one along -- and a fresh tab with no pin yet adopts the
//     stored value as its own.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// `remote` picks which of the two clients is being tested: the served web
// page carries body.remote-chat (web_server.py injects the class), the
// VS Code webview does not.
function makeWebview(opts) {
  const {remote = false, pinnedWorkDir = ''} = opts || {};
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

  // The pin a previous page instance left behind. In production the WS
  // shim writes this key; here the same real sessionStorage is seeded
  // before main.js ever reads it.
  if (pinnedWorkDir) {
    win.sessionStorage.setItem('sorcar-work-dir', pinnedWorkDir);
  }

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

function openSettings(win) {
  const gear = win.document.querySelector('#settings-btn');
  assert.ok(gear, 'the tab bar must render the settings gear');
  gear.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function closeSettings(win) {
  win.document
    .getElementById('settings-panel-close')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

// Typing for real: set the value and fire the event the browser fires.
function typeInto(win, id, value) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function assertNoDirectoryFields(win) {
  assert.strictEqual(
    win.document.getElementById('cfg-work-dir'),
    null,
    'the settings panel must not have a working-directory field',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-memory-dir'),
    null,
    'the settings panel must not have a memory-directory field',
  );
  const labels = Array.from(
    win.document.querySelectorAll('#config-form .config-label'),
  ).map(l => l.textContent.trim());
  assert.ok(
    !labels.some(t => /^(Working|Memory) directory/.test(t)),
    'no settings label may read "Working directory" or "Memory directory"',
  );
}

function tabBarIds(win) {
  return Array.from(win.document.querySelectorAll('.chat-tab'))
    .filter(el => !!el.dataset.tabId)
    .map(el => el.dataset.tabId);
}

function tabEntry(tabId, workDir) {
  return {tabId, chatId: '', title: tabId, workDir};
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

function setWorkDirs(posted) {
  return posted.filter(m => m && m.type === 'setWorkDir').map(m => m.workDir);
}

// --- the standalone web client -------------------------------------------

function testRemoteSettingsHaveNoDirectoryFieldsAndSaveNoWorkDir() {
  const {win, posted} = makeWebview({remote: true});
  openSettings(win);
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});
  assertNoDirectoryFields(win);

  typeInto(win, 'cfg-max-budget', '77');
  closeSettings(win);

  const saved = lastMsg(posted, 'saveConfig');
  assert.ok(saved, 'closing the settings panel must save the form');
  assert.strictEqual(saved.config.max_budget, 77, 'the edit is saved');
  assert.ok(
    !('work_dir' in saved.config) && !('memory_dir' in saved.config),
    'the form has no directory boxes, so it must not invent either key ' +
      '(the daemon merges the payload; a blank would wipe the stored value)',
  );
  win.close();
  console.log('  ok - the web client settings carry no directory fields');
}

function testRemoteBlankWhenNothingIsStored() {
  const {win, posted} = makeWebview({remote: true});
  openSettings(win);
  send(win, {type: 'configData', config: {}});
  closeSettings(win);

  const saved = lastMsg(posted, 'saveConfig');
  assert.ok(saved, 'closing the settings panel must save the form');
  assert.ok(!('work_dir' in saved.config), 'nothing to save as work_dir');
  assert.deepStrictEqual(
    setWorkDirs(posted),
    [],
    'there is nothing to adopt and nothing to pin',
  );
  win.close();
  console.log('  ok - a blank stored working directory pins nothing');
}

function testRemoteInstancePrefersItsOwnPin() {
  const {win, posted} = makeWebview({
    remote: true,
    pinnedWorkDir: '/srv/mine',
  });
  // Another browser tab has since saved its own folder globally.
  send(win, {type: 'configData', config: {work_dir: '/srv/other-instance'}});
  assert.ok(
    !setWorkDirs(posted).includes('/srv/other-instance'),
    'a page that already pinned a folder must not re-adopt the one ' +
      'another instance happened to store last',
  );
  // And it scopes its shared tabs by its OWN pin, not the stored value.
  send(win, {
    type: 'tabs_state',
    tabs: [
      tabEntry('mine-1', '/srv/mine'),
      tabEntry('other-1', '/srv/other-instance'),
    ],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['mine-1'],
    'the tab bar shows the tabs of the pinned folder only',
  );
  win.close();
  console.log('  ok - a pinned web client keeps its own working directory');
}

function testRemoteInstanceAdoptsStoredWorkDirWhenUnpinned() {
  const {win, posted} = makeWebview({remote: true});
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});
  assert.deepStrictEqual(
    setWorkDirs(posted),
    ['/srv/project'],
    'a fresh page claims the stored working directory as its own pin, so ' +
      'its tasks run where the daemon says they do',
  );
  send(win, {
    type: 'tabs_state',
    tabs: [tabEntry('p-1', '/srv/project'), tabEntry('q-1', '/srv/other')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['p-1'], 'and scopes by it');
  win.close();
  console.log('  ok - an unpinned web client adopts the stored directory');
}

function testRemoteWorkDirChangesThroughThePanel() {
  const {win, posted} = makeWebview({
    remote: true,
    pinnedWorkDir: '/srv/project',
  });
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});

  // The "..." menu's "Working directory" panel is the one place to
  // change the folder now that the settings box is gone.
  click(win, win.document.getElementById('more-btn'));
  click(win, win.document.getElementById('workdir-btn'));
  const panel = win.document.getElementById('workdir-panel');
  assert.ok(panel.classList.contains('open'), 'the panel opens');
  typeInto(win, 'workdir-input', '/srv/elsewhere');
  click(win, win.document.getElementById('workdir-open-btn'));

  const check = lastMsg(posted, 'listDir');
  assert.ok(
    check && String(check.token).startsWith('workdir:'),
    'the daemon is asked to list the typed folder first',
  );
  assert.strictEqual(check.path, '/srv/elsewhere');
  send(win, {
    type: 'dirListing',
    token: check.token,
    path: '/srv/elsewhere',
    root: '/srv/elsewhere',
    entries: [],
  });

  const saved = lastMsg(posted, 'saveConfig');
  assert.ok(
    saved && saved.config.work_dir === '/srv/elsewhere',
    'a real folder is stored as the working directory',
  );
  assert.ok(
    setWorkDirs(posted).includes('/srv/elsewhere'),
    'and re-pins this page, otherwise the panel would show one folder ' +
      'while the tasks kept running in another',
  );
  assert.ok(!panel.classList.contains('open'), 'the panel closes');
  win.close();
  console.log('  ok - the web client changes its folder through the panel');
}

// --- the VS Code webview -------------------------------------------------

function testWebviewSettingsHaveNoDirectoryFieldsAndSaveNoWorkDir() {
  const {win, posted} = makeWebview({remote: false});
  openSettings(win);
  send(win, {
    type: 'configData',
    config: {work_dir: '/home/user/ws_a', max_budget: 42},
  });
  assertNoDirectoryFields(win);

  // The user changes something else entirely.
  typeInto(win, 'cfg-max-budget', '77');
  closeSettings(win);

  const saved = lastMsg(posted, 'saveConfig');
  assert.ok(saved, 'closing the settings panel must save the form');
  assert.strictEqual(
    saved.config.max_budget,
    77,
    'the edit the user actually made is saved',
  );
  assert.ok(
    !('work_dir' in saved.config) && !('memory_dir' in saved.config),
    'a VS Code window must leave work_dir out of what it saves: with ' +
      'three windows open on three projects, whichever closed its ' +
      'settings panel last would otherwise own the stored value',
  );
  assert.deepStrictEqual(
    setWorkDirs(posted),
    [],
    'and the settings form must not announce a folder either -- the ' +
      'extension announces the workspace folder itself on connect',
  );
  win.close();
  console.log('  ok - the VS Code settings carry no directory fields');
}

function main() {
  testRemoteSettingsHaveNoDirectoryFieldsAndSaveNoWorkDir();
  testRemoteBlankWhenNothingIsStored();
  testRemoteInstancePrefersItsOwnPin();
  testRemoteInstanceAdoptsStoredWorkDirWhenUnpinned();
  testRemoteWorkDirChangesThroughThePanel();
  testWebviewSettingsHaveNoDirectoryFieldsAndSaveNoWorkDir();
  console.log('settingsWorkDirField.test.js: all tests passed');
}

main();
