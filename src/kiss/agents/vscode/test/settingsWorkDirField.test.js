// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the Settings panel having NO "Working
// directory" (and no "Memory directory") field, and for how the config
// reply carries the ONE global working directory instead.
//
// The working directory is a single daemon-wide value (config.json
// work_dir) that every task on every surface runs in.  The same media/
// files are served to the VS Code webview and to the standalone web
// client, and both treat `configData.config.work_dir` the same way:
//
//   * it is adopted as-is -- no per-browser-tab sessionStorage pin, no
//     `setWorkDir` echo back to the daemon (that used to re-pin the
//     connection; there is no connection pin any more);
//   * the Settings form never carries `work_dir` in `saveConfig`, because
//     the folder is changed only through the "Working directory" panel of
//     the "..." menu (remote: a `listDir` check, then `setWorkDir`; the
//     daemon persists it and broadcasts `workDirChanged`);
//   * a `workDirChanged` broadcast from the daemon (a pick made on another
//     surface) re-scopes this client without any echo.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// `remote` picks which of the two clients is being tested: the served web
// page carries body.remote-chat (web_server.py injects the class), the
// VS Code webview does not.  `stalePin` seeds the sessionStorage key an
// older WS shim used to write, to prove main.js no longer reads it.
function makeWebview(opts) {
  const {remote = false, stalePin = ''} = opts || {};
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

  if (stalePin) {
    win.sessionStorage.setItem('sorcar-work-dir', stalePin);
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
  // The chat whose group is on screen sits on the main row and on the
  // group strip under it; count each tab once.
  const ids = Array.from(win.document.querySelectorAll('.chat-tab'))
    .filter(el => !!el.dataset.tabId)
    .map(el => el.dataset.tabId);
  return ids.filter((id, i) => ids.indexOf(id) === i);
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

// The directory the active tab's daemon-bound commands actually carry:
// typing an @-mention posts getFiles stamped with workDirForTab().
function mentionWorkDir(win, posted) {
  const before = posted.length;
  typeInto(win, 'task-input', '@readm');
  const msg = lastMsg(posted.slice(before), 'getFiles');
  assert.ok(msg, 'typing an @-mention must post a getFiles command');
  typeInto(win, 'task-input', '');
  return msg.workDir;
}

// The "Current:" line of the "..." menu's "Working directory" panel.
function openPanelCurrentLine(win) {
  const panel = win.document.getElementById('workdir-panel');
  if (!panel.classList.contains('open')) {
    click(win, win.document.getElementById('more-btn'));
    click(win, win.document.getElementById('workdir-btn'));
  }
  const el = win.document.getElementById('workdir-current');
  return el && !el.hidden ? el.textContent : '';
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
  assert.deepStrictEqual(setWorkDirs(posted), [], 'nothing to announce');
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '',
    'with no global directory yet, commands carry none and the daemon ' +
      'resolves its own fallback',
  );
  win.close();
  console.log('  ok - a blank stored working directory announces nothing');
}

function testRemoteAdoptsGlobalWorkDirWithoutEcho() {
  const {win, posted} = makeWebview({remote: true});
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});
  assert.deepStrictEqual(
    setWorkDirs(posted),
    [],
    'the config reply reports the global value; echoing it back as ' +
      'setWorkDir would persist and re-broadcast what the daemon just said',
  );
  assert.strictEqual(
    win.sessionStorage.getItem('sorcar-work-dir'),
    null,
    'no per-browser-tab pin is written: every page uses the one global value',
  );
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/srv/project',
    'commands from this page run in the global working directory',
  );
  assert.strictEqual(openPanelCurrentLine(win), 'Current: /srv/project');
  send(win, {
    type: 'tabs_state',
    tabs: [tabEntry('p-1', '/srv/project'), tabEntry('q-1', '/srv/other')],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['p-1', 'q-1'],
    'the tab bar still shows every registry tab: the directory is where ' +
      'tasks run, not a filter',
  );
  win.close();
  console.log('  ok - the web client adopts the global directory without echo');
}

function testRemoteIgnoresStaleSessionPin() {
  const {win, posted} = makeWebview({remote: true, stalePin: '/srv/mine'});
  send(win, {type: 'configData', config: {work_dir: '/srv/global'}});
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/srv/global',
    'a sessionStorage pin left by an older page must not override the ' +
      'global working directory',
  );
  assert.deepStrictEqual(setWorkDirs(posted), [], 'and nothing is echoed');
  send(win, {
    type: 'tabs_state',
    tabs: [tabEntry('mine-1', '/srv/mine'), tabEntry('g-1', '/srv/global')],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['mine-1', 'g-1'],
    'the tab bar shows every registry tab whatever folder it ran in',
  );
  win.close();
  console.log(
    '  ok - a stale per-tab pin is ignored in favour of the global value',
  );
}

function testRemoteWorkDirChangesThroughThePanel() {
  const {win, posted} = makeWebview({remote: true});
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
  assert.strictEqual(
    setWorkDirs(posted).length,
    0,
    'nothing is sent before the daemon confirms the folder exists',
  );
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/srv/project',
    'and nothing is adopted locally either: commands keep the old folder',
  );
  send(win, {
    type: 'dirListing',
    token: check.token,
    path: '/srv/elsewhere',
    root: '/srv/elsewhere',
    entries: [],
  });

  assert.deepStrictEqual(
    setWorkDirs(posted),
    ['/srv/elsewhere'],
    'a real folder is sent as setWorkDir: the daemon persists it as the ' +
      'global value and broadcasts workDirChanged to every other client',
  );
  assert.strictEqual(
    lastMsg(posted, 'saveConfig'),
    null,
    'the panel does not go through saveConfig; setWorkDir is the one path',
  );
  assert.ok(!panel.classList.contains('open'), 'the panel closes');
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/srv/elsewhere',
    'this page re-scopes at once instead of waiting for the broadcast',
  );
  win.close();
  console.log(
    '  ok - the web client changes the global folder through the panel',
  );
}

function testRemoteFollowsWorkDirChangedBroadcast() {
  const {win, posted} = makeWebview({remote: true});
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});
  // A pick made on another surface (a VS Code window, another browser
  // tab) reaches this page as the daemon's broadcast.
  send(win, {type: 'workDirChanged', workDir: '/srv/picked-elsewhere'});
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/srv/picked-elsewhere',
    'every client follows the daemon broadcast so all surfaces agree',
  );
  assert.strictEqual(
    openPanelCurrentLine(win),
    'Current: /srv/picked-elsewhere',
  );
  assert.deepStrictEqual(
    setWorkDirs(posted),
    [],
    'a broadcast is adopted, never echoed back as another setWorkDir',
  );
  win.close();
  console.log(
    '  ok - the web client follows workDirChanged from other surfaces',
  );
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
    'the settings form must leave work_dir out of what it saves: the ' +
      'global directory is changed only through the Working directory panel',
  );
  assert.deepStrictEqual(
    setWorkDirs(posted),
    [],
    'and the webview must not echo the config value as setWorkDir: the ' +
      'host seeds the daemon with the workspace folder (ifUnset) itself',
  );
  win.close();
  console.log('  ok - the VS Code settings carry no directory fields');
}

function testWebviewAdoptsGlobalWorkDirAndBroadcast() {
  const {win, posted} = makeWebview({remote: false});
  send(win, {type: 'configData', config: {work_dir: '/home/user/ws_a'}});
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/home/user/ws_a',
    'the host passes config.work_dir through untouched and the webview ' +
      'runs its commands there, whatever folder the window has open',
  );
  send(win, {type: 'workDirChanged', workDir: '/home/user/ws_b'});
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/home/user/ws_b',
    'a pick on another surface re-scopes this window too',
  );
  assert.deepStrictEqual(setWorkDirs(posted), [], 'never echoed');
  win.close();
  console.log(
    '  ok - the VS Code webview adopts the global directory and its changes',
  );
}

function main() {
  testRemoteSettingsHaveNoDirectoryFieldsAndSaveNoWorkDir();
  testRemoteBlankWhenNothingIsStored();
  testRemoteAdoptsGlobalWorkDirWithoutEcho();
  testRemoteIgnoresStaleSessionPin();
  testRemoteWorkDirChangesThroughThePanel();
  testRemoteFollowsWorkDirChangedBroadcast();
  testWebviewSettingsHaveNoDirectoryFieldsAndSaveNoWorkDir();
  testWebviewAdoptsGlobalWorkDirAndBroadcast();
  console.log('settingsWorkDirField.test.js: all tests passed');
}

main();
