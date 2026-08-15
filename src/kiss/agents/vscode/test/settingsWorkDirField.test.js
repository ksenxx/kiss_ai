// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the "Working directory" field of the
// Settings panel.
//
// The same media/ files are served to two very different clients, and the
// field means something different in each:
//
//   * In a VS Code webview the working directory is not a preference --
//     it IS the workspace folder open in that window. The field is there
//     to be read, so it is read-only, and it is left out of the
//     `saveConfig` payload entirely: three windows open on three
//     projects must not take turns overwriting one another's stored
//     work_dir.
//   * In the standalone web client there is no workspace folder, so the
//     field is the only way to say where tasks should run. It is
//     editable, it is saved, and saving it also re-pins THIS browser tab
//     (sessionStorage `sorcar-work-dir`, written by the WS shim when the
//     page posts `setWorkDir`) so a second browser tab pointed elsewhere
//     does not drag this one along. A fresh tab with no pin yet adopts
//     the stored value as its own.

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
  const gear = win.document.querySelector('.chat-tab-settings');
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

function workDirField(win) {
  const node = win.document.getElementById('cfg-work-dir');
  assert.ok(node, 'the settings panel must have a working-directory field');
  return node;
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

function testRemoteFieldIsEditableAndSaved() {
  const {win, posted} = makeWebview({remote: true});
  openSettings(win);
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});

  const field = workDirField(win);
  assert.strictEqual(
    field.value,
    '/srv/project',
    'the field shows the stored working directory',
  );
  assert.strictEqual(
    field.readOnly,
    false,
    'the web client has no workspace folder, so the field is the only way ' +
      'to set the working directory and must stay editable',
  );

  typeInto(win, 'cfg-work-dir', '  /srv/elsewhere  ');
  closeSettings(win);

  const saved = lastMsg(posted, 'saveConfig');
  assert.ok(saved, 'closing the settings panel must save the form');
  assert.strictEqual(
    saved.config.work_dir,
    '/srv/elsewhere',
    'the edited working directory is saved, with surrounding blanks removed',
  );
  win.close();
  console.log('  ok - the web client field is editable and saved');
}

function testRemoteBlankWhenNothingIsStored() {
  const {win, posted} = makeWebview({remote: true});
  openSettings(win);
  send(win, {type: 'configData', config: {}});

  assert.strictEqual(
    workDirField(win).value,
    '',
    'a config without a work_dir leaves the field blank rather than ' +
      'inventing a path',
  );

  closeSettings(win);
  const saved = lastMsg(posted, 'saveConfig');
  assert.ok(saved, 'closing the settings panel must save the form');
  assert.strictEqual(
    saved.config.work_dir,
    '',
    'an untouched blank field saves as blank',
  );
  assert.deepStrictEqual(
    setWorkDirs(posted),
    [],
    'there is nothing to adopt and nothing to pin',
  );
  win.close();
  console.log('  ok - a blank stored working directory stays blank');
}

function testRemoteInstancePrefersItsOwnPin() {
  const {win, posted} = makeWebview({
    remote: true,
    pinnedWorkDir: '/srv/mine',
  });
  openSettings(win);
  // Another browser tab has since saved its own folder globally.
  send(win, {type: 'configData', config: {work_dir: '/srv/other-instance'}});

  assert.strictEqual(
    workDirField(win).value,
    '/srv/mine',
    'a page that already pinned a folder keeps showing its own, not the ' +
      'one another instance happened to store last',
  );
  assert.ok(
    !setWorkDirs(posted).includes('/srv/other-instance'),
    'and it must not re-adopt that other folder',
  );
  win.close();
  console.log('  ok - a pinned web client keeps its own working directory');
}

function testRemoteInstanceAdoptsStoredWorkDirWhenUnpinned() {
  const {win, posted} = makeWebview({remote: true});
  openSettings(win);
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});

  assert.strictEqual(
    workDirField(win).value,
    '/srv/project',
    'a fresh page shows the stored working directory',
  );
  assert.deepStrictEqual(
    setWorkDirs(posted),
    ['/srv/project'],
    'and claims it as its own pin, so its tasks run where the settings ' +
      'panel says they do',
  );
  win.close();
  console.log('  ok - an unpinned web client adopts the stored directory');
}

function testRemoteSaveRepinsTheEditedWorkDir() {
  const {win, posted} = makeWebview({
    remote: true,
    pinnedWorkDir: '/srv/project',
  });
  openSettings(win);
  send(win, {type: 'configData', config: {work_dir: '/srv/project'}});

  typeInto(win, 'cfg-work-dir', '/srv/elsewhere');
  closeSettings(win);

  assert.ok(
    setWorkDirs(posted).includes('/srv/elsewhere'),
    'saving an edited working directory re-pins this page as well as ' +
      'storing it, otherwise the panel would show one folder while the ' +
      'tasks kept running in another',
  );
  win.close();
  console.log('  ok - saving an edited working directory re-pins the page');
}

// --- the VS Code webview -------------------------------------------------

function testWebviewFieldIsReadOnlyAndNotSaved() {
  const {win, posted} = makeWebview({remote: false});
  openSettings(win);
  send(win, {type: 'configData', config: {work_dir: '/home/user/ws_a'}});

  const field = workDirField(win);
  assert.strictEqual(
    field.value,
    '/home/user/ws_a',
    "the field shows this window's own workspace folder",
  );
  assert.strictEqual(
    field.readOnly,
    true,
    'in VS Code the working directory is the workspace folder, so the ' +
      'field is there to be read',
  );
  assert.ok(
    field.title,
    'and it says so, since a field that refuses to be typed into owes the ' +
      'reader an explanation',
  );

  closeSettings(win);
  const saved = lastMsg(posted, 'saveConfig');
  assert.ok(saved, 'closing the settings panel must save the form');
  assert.ok(
    !('work_dir' in saved.config),
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
  console.log('  ok - the VS Code field is read-only and never saved');
}

function testWebviewSaveDoesNotClobberAnotherWindowsWorkDir() {
  const {win, posted} = makeWebview({remote: false});
  openSettings(win);
  send(win, {
    type: 'configData',
    config: {work_dir: '/home/user/ws_a', max_budget: 42},
  });

  // The user changes something else entirely.
  typeInto(win, 'cfg-max-budget', '77');
  closeSettings(win);

  const saved = lastMsg(posted, 'saveConfig');
  assert.strictEqual(
    saved.config.max_budget,
    77,
    'the edit the user actually made is saved',
  );
  assert.ok(
    !('work_dir' in saved.config),
    'while the read-only working directory rides along with nothing',
  );
  win.close();
  console.log('  ok - an unrelated VS Code save carries no work_dir');
}

function main() {
  testRemoteFieldIsEditableAndSaved();
  testRemoteBlankWhenNothingIsStored();
  testRemoteInstancePrefersItsOwnPin();
  testRemoteInstanceAdoptsStoredWorkDirWhenUnpinned();
  testRemoteSaveRepinsTheEditedWorkDir();
  testWebviewFieldIsReadOnlyAndNotSaved();
  testWebviewSaveDoesNotClobberAnotherWindowsWorkDir();
  console.log('settingsWorkDirField.test.js: all tests passed');
}

main();
