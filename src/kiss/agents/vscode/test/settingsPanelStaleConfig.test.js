// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the Settings panel vs. an incoming
// ``configData``.
//
// openSettingsPanel() makes the form editable immediately and only THEN
// asks for the config, and ``configData`` is not a one-shot reply: the
// VS Code host polls ~/.kiss/config.json every two seconds and
// re-requests the config on every daemon reconnect, so an unsolicited
// copy can land at any moment -- including one triggered by ANOTHER
// window saving its own settings.
//
// populateConfigForm() used to repaint every field unconditionally, so
//   A) a key pasted before the reply arrived was silently replaced by
//      the stored value, and the stale value was then saved on close;
//   B) closing the panel before the reply arrived dropped the edit
//      entirely, because configFormPopulated was still false.
//
// Every other async reply in main.js is checked against what the user is
// doing (files vs. the typed prefix, ghost/completions vs. the input
// value, history vs. its generation). This makes the config form
// consistent with them.

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

// Typing for real: set the value and fire the event the browser fires.
function typeInto(win, id, value) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

const STORED = {
  max_budget: 100,
  custom_endpoint: 'https://old.example/v1',
  custom_api_key: 'old-custom',
  custom_headers: '',
  remote_password: 'old-password',
  auto_commit_mode: true,
  is_worktree: true,
};

function testLateConfigDataDoesNotClobberTypedKey() {
  const {win, posted} = makeWebview();
  openSettings(win);
  assert.ok(
    lastMsg(posted, 'getConfig'),
    'opening the settings panel must request the config',
  );

  // The user pastes a key before the reply comes back.
  typeInto(win, 'cfg-key-ANTHROPIC_API_KEY', 'sk-user-typed');
  typeInto(win, 'cfg-custom-endpoint', 'https://new.example/v1');

  send(win, {
    type: 'configData',
    config: STORED,
    apiKeys: {ANTHROPIC_API_KEY: 'sk-old', OPENAI_API_KEY: 'sk-openai'},
  });

  assert.strictEqual(
    win.document.getElementById('cfg-key-ANTHROPIC_API_KEY').value,
    'sk-user-typed',
    'a configData arriving after the user typed must not paint over the ' +
      'field being edited',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-custom-endpoint').value,
    'https://new.example/v1',
    'the same holds for every field the user touched',
  );
  // Untouched fields must still be filled in from the reply.
  assert.strictEqual(
    win.document.getElementById('cfg-key-OPENAI_API_KEY').value,
    'sk-openai',
    'untouched fields must still be populated from configData',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-remote-password').value,
    'old-password',
    'untouched fields must still be populated from configData',
  );

  closeSettings(win);
  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'closing the panel must save the form');
  assert.strictEqual(
    save.apiKeys.ANTHROPIC_API_KEY,
    'sk-user-typed',
    'the saved key must be the one the user typed',
  );
  assert.strictEqual(
    save.config.custom_endpoint,
    'https://new.example/v1',
    'the saved endpoint must be the one the user typed',
  );
  assert.strictEqual(
    save.config.remote_password,
    'old-password',
    'untouched fields must be saved with their loaded values',
  );
  win.close();
  console.log('  ok - a late configData does not clobber a typed field');
}

function testClosingBeforeTheReplyStillSavesTheEdit() {
  const {win, posted} = makeWebview();
  openSettings(win);
  typeInto(win, 'cfg-remote-password', 'new-secret');
  // No configData ever arrives (slow link) -- the user closes the panel.
  closeSettings(win);

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(
    save,
    'an edit made before the config reply arrived must still be saved, ' +
      'not silently dropped',
  );
  assert.strictEqual(
    save.config.remote_password,
    'new-secret',
    'the typed password must be what is saved',
  );
  // Only the touched field may be sent: the rest of the form is still
  // blank, and the daemon merges what it receives, so sending blanks
  // would wipe the stored settings.
  assert.deepStrictEqual(
    Object.keys(save.config).sort(),
    ['remote_password'],
    'a form that was never populated must send only what was touched, ' +
      `got ${JSON.stringify(save.config)}`,
  );
  win.close();
  console.log('  ok - an edit made before the reply is saved, not dropped');
}

function testUntouchedPanelStillSavesNothing() {
  const {win, posted} = makeWebview();
  openSettings(win);
  const before = posted.length;
  closeSettings(win);
  assert.ok(
    !posted.slice(before).some(m => m.type === 'saveConfig'),
    'closing an untouched, unpopulated panel must not save anything',
  );
  win.close();
  console.log('  ok - an untouched unpopulated panel saves nothing');
}

// The fix must key on "this field is being edited", not on suppressing
// configData: with the panel closed the reply is what seeds the run
// toggles for a fresh session.
function testClosedPanelIsStillRepainted() {
  const {win} = makeWebview();
  win.document.getElementById('cfg-auto-commit').checked = true;
  win.document.getElementById('cfg-use-worktree').checked = true;
  send(win, {
    type: 'configData',
    config: {...STORED, auto_commit_mode: false, is_worktree: false},
    apiKeys: {ANTHROPIC_API_KEY: 'sk-seeded'},
  });
  assert.strictEqual(
    win.document.getElementById('cfg-auto-commit').checked,
    false,
    'with the panel closed configData must still seed the run toggles',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-use-worktree').checked,
    false,
    'with the panel closed configData must still seed the run toggles',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-key-ANTHROPIC_API_KEY').value,
    'sk-seeded',
    'with the panel closed configData must still fill the form',
  );
  win.close();
  console.log('  ok - a closed panel is still repainted by configData');
}

// Reopening the panel starts a new editing session: what was typed
// before has been saved, so the fresh reply must win again.
function testReopeningClearsTheEditedMarks() {
  const {win} = makeWebview();
  openSettings(win);
  typeInto(win, 'cfg-custom-headers', 'X-A: 1');
  closeSettings(win);

  openSettings(win);
  send(win, {
    type: 'configData',
    config: {...STORED, custom_headers: 'X-Stored: 1'},
    apiKeys: {},
  });
  assert.strictEqual(
    win.document.getElementById('cfg-custom-headers').value,
    'X-Stored: 1',
    'a new editing session must accept the stored value again',
  );
  win.close();
  console.log('  ok - reopening the panel clears the edited marks');
}

// The welcome screen mirrors its password box into the settings field
// programmatically (no input event), so that path must be marked edited
// too or the very first password a user sets is dropped.
function testWelcomePasswordMirrorIsSaved() {
  const {win, posted} = makeWebview();
  const welcomePw = win.document.getElementById('welcome-cfg-remote-password');
  assert.ok(welcomePw, 'the welcome screen must have a password box');
  welcomePw.value = 'welcome-secret';
  welcomePw.dispatchEvent(new win.Event('input', {bubbles: true}));
  welcomePw.dispatchEvent(new win.Event('change', {bubbles: true}));

  const save = lastMsg(posted, 'saveConfig');
  assert.ok(save, 'the welcome password box must save what it mirrors');
  assert.strictEqual(
    save.config.remote_password,
    'welcome-secret',
    'the mirrored password must be the one saved',
  );
  win.close();
  console.log('  ok - the welcome password mirror is saved');
}

function main() {
  testLateConfigDataDoesNotClobberTypedKey();
  testClosingBeforeTheReplyStillSavesTheEdit();
  testUntouchedPanelStillSavesNothing();
  testClosedPanelIsStillRepainted();
  testReopeningClearsTheEditedMarks();
  testWelcomePasswordMirrorIsSaved();
  console.log('settingsPanelStaleConfig.test.js: all tests passed');
}

main();
