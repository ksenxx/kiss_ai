// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The Enter that picks a Japanese / Chinese / Korean candidate arrives
// as a keydown with `isComposing` set, or with keyCode 229 (Safari sends
// only the latter).  It belongs to the IME, so it must not submit:
// * the sign-in window's paste box (Enter hands the pasted redirect
//   address or code to the daemon, `signinPaste`);
// * the folder picker's path box (Enter lists the typed folder).
// A plain Enter still does both.

const assert = require('assert');
const h = require('./ui_antipattern_harness');
const {makePage, send} = require('./simplify2_harness');

const {test, report} = h.makeRunner();
// The two forms of an IME Enter, and a key that is no Enter at all.
const NOT_SUBMITS = [{isComposing: true}, {keyCode: 229}, {key: 'a'}];

function enter(win, el, extra) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'Enter',
      bubbles: true,
      cancelable: true,
      ...extra,
    }),
  );
}

/** A page whose sign-in window shows the paste box for Google. */
function signinPage() {
  const {win, posted} = makePage({
    beforeScripts: w => {
      w.open = () => ({closed: false, location: {href: ''}, close() {}});
    },
  });
  send(win, {
    type: 'signin',
    title: 'Sign in',
    providers: [{id: 'google', label: 'Login with Google'}],
  });
  win.document.querySelector('#signin-modal-buttons button').click();
  send(win, {
    type: 'signinUrl',
    provider: 'google',
    url: 'https://accounts.example/auth',
  });
  const row = win.document.getElementById('signin-modal-paste');
  assert.notStrictEqual(row.style.display, 'none', 'the paste row shows');
  return {win, posted};
}

test('an IME Enter in the sign-in paste box sends nothing', () => {
  const {win, posted} = signinPage();
  const input = win.document.getElementById('signin-modal-paste-input');
  input.value = 'コード-1234';
  for (const form of NOT_SUBMITS) enter(win, input, form);
  assert.strictEqual(h.ofType(posted, 'signinPaste').length, 0);
  enter(win, input);
  assert.deepStrictEqual(
    h.ofType(posted, 'signinPaste').map(m => m.text),
    ['コード-1234'],
    'a plain Enter sends the paste',
  );
  win.close();
});

test('an IME Enter in the folder picker path box lists nothing', () => {
  const {win, posted} = h.makePage();
  h.openExplorer(win, posted, []);
  h.click(win, h.byId(win, 'explorer-add-folder'));
  const picker = h.byId(win, 'folder-picker');
  assert.ok(picker && !picker.hidden, 'the folder picker is open');
  const input = picker.querySelector('.folder-picker-input');
  const listings = () =>
    h
      .ofType(posted, 'listDir')
      .filter(m => String(m.token).startsWith('picker:'))
      .map(m => m.path);
  const before = listings().length;
  input.value = '/データ';
  for (const form of NOT_SUBMITS) enter(win, input, form);
  assert.strictEqual(listings().length, before, 'only a plain Enter lists');
  enter(win, input);
  assert.deepStrictEqual(
    listings().slice(before),
    ['/データ'],
    'a plain Enter lists the typed folder',
  );
  win.close();
});

report('imeSigninFolderPicker_wave2');
