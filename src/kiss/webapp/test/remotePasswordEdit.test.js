// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';
// Settings > Remote password (main.js against the real chat.html):
//   * the box is an editable input, masked until its eye toggle reveals
//     the password `configData` carries; there is no daemon-side Rotate
//     button (the refresh button that fills in a random password is
//     covered by remotePasswordRefresh.test.js);
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
const {
  makePreloadedPage,
  send,
  click,
  byId,
  lastMsg,
} = require('./simplify2_harness');
const {
  openSettings,
  closeSettings,
  typeInto,
  openSettingsWith,
} = require('./settings_helpers');

const TAB_ID = 'tab-remote-password-edit-test';
const STORED = {remote_password: 'Old-Password-1', max_budget: 5};

// That the box is an editable, masked input whose untouched value is
// saved back is pinned by settingsPanelStaleConfig.test.js; here: the eye
// toggle and the absence of a daemon-side Rotate button.
function testEyeToggleRevealsAndHides() {
  const wv = makePreloadedPage(TAB_ID);
  openSettingsWith(wv, STORED);
  assert.ok(
    wv.posted.some(m => m.type === 'getConfig'),
    'opening Settings must request the config',
  );
  const box = byId(wv.win, 'cfg-remote-password');
  assert.strictEqual(
    box.value,
    'Old-Password-1',
    'the box shows the stored password',
  );
  assert.strictEqual(
    wv.win.document.getElementById('cfg-rotate-password-btn'),
    null,
    'there is no daemon-side Rotate button',
  );
  const toggle = byId(wv.win, 'cfg-remote-password-toggle');
  assert.strictEqual(toggle.getAttribute('aria-label'), 'Show password');
  click(wv.win, toggle);
  assert.strictEqual(box.type, 'text', 'the eye toggle reveals the password');
  assert.strictEqual(toggle.getAttribute('aria-label'), 'Hide password');
  click(wv.win, toggle);
  assert.strictEqual(box.type, 'password', 'the eye toggle hides it again');
  wv.close();
  console.log('  ok - the eye toggle reveals and hides the remote password');
}

function testTypedPasswordIsSavedAndKeptOverALateConfigData() {
  const wv = makePreloadedPage(TAB_ID);
  openSettingsWith(wv, STORED);
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
  closeSettings(wv.win);
  const save = lastMsg(wv.posted, 'saveConfig');
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
  openSettings(wv.win);
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
  const wv = makePreloadedPage(TAB_ID);
  openSettingsWith(wv, STORED);
  typeInto(wv.win, 'cfg-remote-password', '');
  closeSettings(wv.win);
  const save = lastMsg(wv.posted, 'saveConfig');
  assert.ok(save, 'closing Settings saves the form');
  assert.strictEqual(
    save.config.remote_password,
    '',
    'an emptied box saves an empty password (remote access disabled)',
  );
  wv.close();
  console.log('  ok - clearing the box saves an empty remote password');
}

testEyeToggleRevealsAndHides();
testTypedPasswordIsSavedAndKeptOverALateConfigData();
testClearingTheBoxSavesAnEmptyPassword();
console.log('\nAll remotePasswordEdit tests passed.');
