// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';
// Settings > Remote password (main.js against the real chat.html):
//   * the box is a plain, editable text input showing the password
//     `configData` carries; there is no Rotate button and no eye toggle;
//   * a typed password is saved with the rest of the form when Settings
//     closes (`saveConfig` with `config.remote_password`, trimmed), and
//     a `configData` arriving mid-edit does not overwrite what was typed;
//   * clearing the box saves an empty password (remote access off);
//   * the `configData` after the save repaints the box;
//   * nothing posts the retired `rotateRemotePassword`.
// The transport's adoption of the new password for its own frames is
// covered by transportFramePassword.test.js; the daemon side by
// src/kiss/tests/server/test_save_config_remote_password.py.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {JSDOM} = require('jsdom');

const TAB_ID = 'tab-remote-password-edit-test';

function makeDomPage() {
  const mediaDir = path.join(__dirname, '..', 'media');
  let html = fs.readFileSync(path.join(mediaDir, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  const posted = [];
  let state = {
    tabs: [{title: 'new chat', chatId: TAB_ID, backendChatId: '', parentTabId: ''}],
    activeTabIndex: 0,
    chatId: TAB_ID,
  };
  win.acquireTransport = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  for (const name of ['panelCopy.js', 'api.js', 'main.js']) {
    vm.runInContext(
      fs.readFileSync(path.join(mediaDir, name), 'utf8'),
      dom.getInternalVMContext(),
    );
  }
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: TAB_ID, chatId: '', title: 'new chat', workDir: ''}],
  });
  return {win, posted, close: () => win.close()};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {bubbles: true}),
  );
}

function byId(win, id) {
  const el = win.document.getElementById(id);
  assert.ok(el, '#' + id + ' must exist');
  return el;
}

function typeInto(win, id, value) {
  const node = byId(win, id);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function openSettings(wv, stored) {
  click(byId(wv.win, 'settings-btn'));
  assert.ok(
    wv.posted.some(m => m.type === 'getConfig'),
    'opening Settings must request the config',
  );
  send(wv.win, {
    type: 'configData',
    config: {remote_password: stored, max_budget: 5},
  });
}

function closeSettings(wv) {
  click(byId(wv.win, 'settings-panel-close'));
}

function lastSave(wv) {
  const saves = wv.posted.filter(m => m.type === 'saveConfig');
  return saves[saves.length - 1];
}

function testBoxIsPlainAndEditable() {
  const wv = makeDomPage();
  openSettings(wv, 'Old-Password-1');
  const box = byId(wv.win, 'cfg-remote-password');
  assert.strictEqual(box.type, 'text', 'the password is shown in plain text');
  assert.strictEqual(box.readOnly, false, 'the box is editable');
  assert.strictEqual(box.value, 'Old-Password-1', 'the box shows the stored password');
  assert.strictEqual(
    wv.win.document.getElementById('cfg-rotate-password-btn'),
    null,
    'there is no Rotate button',
  );
  assert.strictEqual(
    wv.win.document.getElementById('cfg-remote-password-toggle'),
    null,
    'there is no eye toggle on the password box',
  );
  closeSettings(wv);
  const save = lastSave(wv);
  assert.ok(save, 'closing Settings saves the form');
  assert.strictEqual(
    save.config.remote_password,
    'Old-Password-1',
    'an untouched box echoes the stored password',
  );
  wv.close();
  console.log('  ok - the remote password box is a plain editable text input');
}

function testTypedPasswordIsSavedAndKeptOverALateConfigData() {
  const wv = makeDomPage();
  openSettings(wv, 'Old-Password-1');
  typeInto(wv.win, 'cfg-remote-password', '  New-Password-2  ');
  // The daemon re-pushes configData (another window saved something):
  // the edit in progress must stand.
  send(wv.win, {
    type: 'configData',
    config: {remote_password: 'Old-Password-1', max_budget: 5},
  });
  assert.strictEqual(
    byId(wv.win, 'cfg-remote-password').value,
    '  New-Password-2  ',
    'a configData arriving mid-edit must not overwrite the typed password',
  );
  closeSettings(wv);
  const save = lastSave(wv);
  assert.ok(save, 'closing Settings saves the form');
  assert.strictEqual(
    save.config.remote_password,
    'New-Password-2',
    'the typed password is saved, trimmed',
  );
  // The daemon's configData after the save repaints the box.
  send(wv.win, {
    type: 'configData',
    config: {remote_password: 'New-Password-2', max_budget: 5},
  });
  click(byId(wv.win, 'settings-btn'));
  assert.strictEqual(
    byId(wv.win, 'cfg-remote-password').value,
    'New-Password-2',
    'the box shows the saved password',
  );
  assert.ok(
    !wv.posted.some(m => m.type === 'rotateRemotePassword'),
    'nothing posts the retired rotateRemotePassword',
  );
  wv.close();
  console.log('  ok - a typed remote password is saved with the form');
}

function testClearingTheBoxSavesAnEmptyPassword() {
  const wv = makeDomPage();
  openSettings(wv, 'Old-Password-1');
  typeInto(wv.win, 'cfg-remote-password', '');
  closeSettings(wv);
  const save = lastSave(wv);
  assert.ok(save, 'closing Settings saves the form');
  assert.strictEqual(
    save.config.remote_password,
    '',
    'an emptied box saves an empty password (remote access disabled)',
  );
  wv.close();
  console.log('  ok - clearing the box saves an empty remote password');
}

testBoxIsPlainAndEditable();
testTypedPasswordIsSavedAndKeptOverALateConfigData();
testClearingTheBoxSavesAnEmptyPassword();
console.log('\nAll remotePasswordEdit tests passed.');
