// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the OTHER half of the
// ``settingsEditedFields`` guard.
//
// ``settingsPanelStaleConfig.test.js`` proves the guard does its job:
// a field the user is editing is not painted over by a ``configData``
// that lands mid-edit.  A guard like that has a natural failure mode in
// the opposite direction -- marks that outlive the edit and then
// suppress a value the server legitimately changed, leaving the panel
// showing a number the daemon no longer holds and re-saving it on
// close.
//
// The marks are dropped in two places (``openSettingsPanel`` and
// ``closeSettingsPanel``), and one mark is set with the panel CLOSED:
// the welcome screen mirrors its password box into ``#cfg-remote-
// password`` by assignment, which fires no ``input`` event, so it calls
// ``markSettingsFieldEdited`` by hand.  That mark belongs to no editing
// session and is the one most likely to get stuck.
//
// These tests pin the release valve rather than the guard.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
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

function typeInto(win, id, value) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function valueOf(win, id) {
  return win.document.getElementById(id).value;
}

const STORED = {
  max_budget: 100,
  custom_endpoint: 'https://one.example/v1',
  custom_api_key: '',
  custom_headers: 'X-One: 1',
  remote_password: 'pw-one',
  auto_commit_mode: true,
  is_worktree: true,
};

// A push that arrives with the panel shut belongs to nobody's edit, so
// it must land -- otherwise the next time the panel opens it shows a
// value the daemon has not held for minutes.
function testServerChangeWhilePanelClosedRepaints() {
  const {win} = makeWebview();
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {}});
  typeInto(win, 'cfg-custom-headers', 'X-Mine: 2');
  closeSettings(win);

  // Another VS Code window (or the user hand-editing config.json, which
  // the host polls every two seconds) changes the same setting.
  send(win, {
    type: 'configData',
    config: {...STORED, custom_headers: 'X-Other: 9'},
    apiKeys: {},
  });
  assert.strictEqual(
    valueOf(win, 'cfg-custom-headers'),
    'X-Other: 9',
    'a field edited in a FINISHED session must accept the next server ' +
      'value: closeSettingsPanel() drops the edited marks, so nothing ' +
      'may still be suppressing the repaint',
  );
  win.close();
  console.log('  ok - a server change with the panel closed repaints');
}

// Reopening asks for the config again; that reply must win even though
// the user never touched the field in the NEW session.
function testReopenedPanelShowsTheServerValueNotTheOldEdit() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {}});
  typeInto(win, 'cfg-custom-endpoint', 'https://mine.example/v1');
  closeSettings(win);

  openSettings(win);
  assert.ok(
    lastMsg(posted, 'getConfig'),
    'reopening the panel must re-request the config',
  );
  // The daemon rejected/normalised the edit and reports something else.
  send(win, {
    type: 'configData',
    config: {...STORED, custom_endpoint: 'https://server.example/v1'},
    apiKeys: {},
  });
  assert.strictEqual(
    valueOf(win, 'cfg-custom-endpoint'),
    'https://server.example/v1',
    'a reopened panel must show what the daemon actually holds, not the ' +
      'value this window typed in the previous session',
  );

  // ...and closing it must save the server value back unchanged rather
  // than resurrecting the stale edit.
  const before = posted.length;
  closeSettings(win);
  const save = posted
    .slice(before)
    .filter(m => m && m.type === 'saveConfig')
    .pop();
  assert.ok(save, 'closing a populated panel must save the form');
  assert.strictEqual(
    save.config.custom_endpoint,
    'https://server.example/v1',
    'the repainted server value is what gets saved back',
  );
  win.close();
  console.log('  ok - a reopened panel shows and re-saves the server value');
}

// The welcome-screen mirror is the only place that marks a field edited
// with the settings panel closed. If that mark could outlive the
// welcome screen, the password box would be frozen for the rest of the
// session.
function testWelcomeMirrorMarkDoesNotFreezeThePasswordField() {
  const {win} = makeWebview();
  const welcomePw = win.document.getElementById('welcome-cfg-remote-password');
  assert.ok(welcomePw, 'the welcome screen must have a password box');

  welcomePw.value = 'welcome-secret';
  welcomePw.dispatchEvent(new win.Event('input', {bubbles: true}));
  assert.strictEqual(
    valueOf(win, 'cfg-remote-password'),
    'welcome-secret',
    'the welcome box must mirror into the settings field',
  );

  // The daemon confirms a DIFFERENT password (another window set one).
  // While the mirror mark is live the settings field must hold what was
  // just typed -- that is the guard working.
  send(win, {
    type: 'configData',
    config: {...STORED, remote_password: 'pw-from-other-window'},
    apiKeys: {},
  });
  assert.strictEqual(
    valueOf(win, 'cfg-remote-password'),
    'welcome-secret',
    'the just-typed password must not be painted over',
  );

  // Opening the settings panel starts a fresh session and re-requests
  // the config: now the server value must win, or the field is stuck
  // for good.
  openSettings(win);
  send(win, {
    type: 'configData',
    config: {...STORED, remote_password: 'pw-from-other-window'},
    apiKeys: {},
  });
  assert.strictEqual(
    valueOf(win, 'cfg-remote-password'),
    'pw-from-other-window',
    'openSettingsPanel() must clear the welcome mirror mark too, or the ' +
      'password field can never be repainted again in this session',
  );
  win.close();
  console.log('  ok - the welcome mirror mark is released, not sticky');
}

// The run toggles are seeded by configData for every session, so a mark
// stuck on one of them would silently run tasks with the wrong flags.
function testToggleMarksAreReleasedBetweenSessions() {
  const {win} = makeWebview();
  const doc = win.document;
  openSettings(win);
  send(win, {type: 'configData', config: STORED, apiKeys: {}});

  const wt = doc.getElementById('cfg-use-worktree');
  wt.checked = false;
  wt.dispatchEvent(new win.Event('change', {bubbles: true}));
  closeSettings(win);

  // The daemon reports worktrees back ON (another window re-enabled it).
  send(win, {
    type: 'configData',
    config: {...STORED, is_worktree: true},
    apiKeys: {},
  });
  assert.strictEqual(
    wt.checked,
    true,
    'a toggle edited in a finished session must accept the next server ' +
      'value; a stuck mark would run every later task with the wrong flag',
  );
  win.close();
  console.log('  ok - toggle marks are released between sessions');
}

function main() {
  testServerChangeWhilePanelClosedRepaints();
  testReopenedPanelShowsTheServerValueNotTheOldEdit();
  testWelcomeMirrorMarkDoesNotFreezeThePasswordField();
  testToggleMarksAreReleasedBetweenSessions();
  console.log('settingsEditedGuardRepaint.test.js: all tests passed');
}

main();
