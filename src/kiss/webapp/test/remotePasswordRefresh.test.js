// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';
// Settings > Remote password > refresh button (main.js against the real
// chat.html):
//   * the box sits in a .config-password-wrap with the eye toggle AND a
//     refresh button (#cfg-remote-password-refresh), both inside the
//     wrap so they share the box's row;
//   * a click fills the box with a fresh random password of exactly 20
//     letters / digits (the daemon's own alphabet), reveals it, focuses
//     the box, and counts as a user edit: a configData arriving
//     afterwards does not overwrite it and closing Settings saves it;
//   * a second click replaces it with a different one; a later manual
//     edit of the box still wins;
//   * the refresh posts nothing to the daemon by itself (no
//     rotateRemotePassword), and clicking it does not toggle the eye
//     back to masked when the box is already revealed.
// generateRemotePassword's rejection sampling (a byte at or above 248
// is redrawn) has no deterministic trigger from the page; 200 clicks
// draw thousands of bytes, so both arms of its `if` run with
// probability indistinguishable from 1 and every output is checked
// against the alphabet.

const assert = require('assert');
const {
  makePreloadedPage,
  send,
  click,
  byId,
  lastMsg,
} = require('./simplify2_harness');
const {closeSettings, typeInto, openSettingsWith} = require('./settings_helpers');

const TAB_ID = 'tab-remote-password-refresh-test';
const ALPHABET_RE = /^[A-Za-z0-9]{20}$/;

/** Open Settings with `stored` as the remote password. */
function openWithPassword(wv, stored) {
  openSettingsWith(wv, {remote_password: stored, max_budget: 5});
}

function testRefreshButtonIsALabelledIconButton() {
  const wv = makePreloadedPage(TAB_ID);
  openWithPassword(wv, 'Old-Password-1');
  const refresh = byId(wv.win, 'cfg-remote-password-refresh');
  assert.strictEqual(
    refresh.type,
    'button',
    'the refresh button never submits',
  );
  assert.strictEqual(
    refresh.getAttribute('aria-label'),
    'Generate a new random password',
  );
  assert.ok(refresh.querySelector('svg'), 'the refresh button shows an icon');
  wv.close();
  console.log('  ok - the refresh button is a labelled icon button');
}

function testClickFillsRevealsAndSavesARandomPassword() {
  const wv = makePreloadedPage(TAB_ID);
  openWithPassword(wv, 'Old-Password-1');
  const box = byId(wv.win, 'cfg-remote-password');
  const refresh = byId(wv.win, 'cfg-remote-password-refresh');
  const toggle = byId(wv.win, 'cfg-remote-password-toggle');
  assert.strictEqual(box.type, 'password', 'masked before the click');
  click(wv.win, refresh);
  const first = box.value;
  assert.ok(
    ALPHABET_RE.test(first),
    'a 20-character letters/digits password: ' + first,
  );
  assert.notStrictEqual(
    first,
    'Old-Password-1',
    'the stored password is replaced',
  );
  assert.strictEqual(box.type, 'text', 'the new password is revealed');
  assert.strictEqual(toggle.getAttribute('aria-label'), 'Hide password');
  assert.strictEqual(wv.win.document.activeElement, box, 'the box is focused');
  // A configData re-push (another window saved) must not overwrite the
  // generated password: the refresh counts as a user edit.
  send(wv.win, {
    type: 'configData',
    config: {remote_password: 'Old-Password-1', max_budget: 5},
  });
  assert.strictEqual(
    box.value,
    first,
    'a later configData keeps the generated password',
  );
  // A second click makes another one, and does not hide the box again.
  click(wv.win, refresh);
  const second = box.value;
  assert.ok(
    ALPHABET_RE.test(second),
    'the second password is well-formed: ' + second,
  );
  assert.notStrictEqual(
    second,
    first,
    'the second click makes a different password',
  );
  assert.strictEqual(box.type, 'text', 'the box stays revealed');
  assert.ok(
    !wv.posted.some(m => m.type === 'rotateRemotePassword'),
    'the refresh asks nothing of the daemon',
  );
  assert.ok(
    !lastMsg(wv.posted, 'saveConfig'),
    'nothing is saved until Settings closes',
  );
  closeSettings(wv.win);
  const save = lastMsg(wv.posted, 'saveConfig');
  assert.ok(save, 'closing Settings saves the form');
  assert.strictEqual(
    save.config.remote_password,
    second,
    'the generated password is saved',
  );
  wv.close();
  console.log(
    '  ok - a click fills, reveals and saves a random 20-character password',
  );
}

function testManualEditAfterRefreshWins() {
  const wv = makePreloadedPage(TAB_ID);
  openWithPassword(wv, 'Old-Password-1');
  click(wv.win, byId(wv.win, 'cfg-remote-password-refresh'));
  typeInto(wv.win, 'cfg-remote-password', 'Typed-After-3');
  closeSettings(wv.win);
  assert.strictEqual(
    lastMsg(wv.posted, 'saveConfig').config.remote_password,
    'Typed-After-3',
    'an edit after the refresh is what gets saved',
  );
  wv.close();
  console.log('  ok - a manual edit after a refresh wins');
}

function testEveryGeneratedPasswordIsWellFormed() {
  const wv = makePreloadedPage(TAB_ID);
  openWithPassword(wv, '');
  const box = byId(wv.win, 'cfg-remote-password');
  const refresh = byId(wv.win, 'cfg-remote-password-refresh');
  const seen = new Set();
  for (let i = 0; i < 200; i++) {
    click(wv.win, refresh);
    assert.ok(ALPHABET_RE.test(box.value), 'malformed password: ' + box.value);
    seen.add(box.value);
  }
  assert.strictEqual(
    seen.size,
    200,
    'two hundred clicks give two hundred different passwords',
  );
  wv.close();
  console.log(
    '  ok - every generated password is 20 letters/digits and unique',
  );
}

testRefreshButtonIsALabelledIconButton();
testClickFillsRevealsAndSavesARandomPassword();
testManualEditAfterRefreshWins();
testEveryGeneratedPasswordIsWellFormed();
console.log('\nAll remotePasswordRefresh tests passed.');
