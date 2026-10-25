// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Shared page builder and drop-down helpers of the work-directory suites
// (chatWorkDirSelect.test.js, chatMachineWorkDirs.test.js): a remote-chat
// page over a recording transport plus readers and a picker for the
// `#chat-workdir` select.

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');

// Every window built by makePage, so that main.js's timers (the
// elapsed-time ticker, the daemon-status spinner, the completions-load
// timeout) do not keep the node process alive once the tests are done.
const openWindows = [];

/**
 * Build a remote-chat jsdom page running panelCopy.js + api.js + main.js
 * over a recording transport.  `desktop` enables the remote desktop
 * layout (the activity bar's Explorer / Source Control views), which
 * main.js gates on the `(min-width: 900px)` media query.
 *
 * @returns {{win: object, posted: Array<object>}}
 */
function makePage(opts) {
  const {desktop = false} = opts || {};
  const page = harness.makePage({
    bodyClass: 'remote-chat',
    beforeScripts(win) {
      openWindows.push(win);
      if (desktop) {
        harness.stubRaf(win);
        harness.fakeMatchMedia(win, true);
      }
    },
  });
  return {win: page.win, posted: page.posted};
}

/** Close every window makePage built. */
function closeWindows() {
  while (openWindows.length) openWindows.pop().close();
}

const {send, byId, ofType: msgs} = harness;

// Messages are created inside the JSDOM realm, whose Object prototype is
// not Node's, so structural equality goes through JSON.
function assertJsonEqual(actual, expected) {
  assert.strictEqual(JSON.stringify(actual), JSON.stringify(expected));
}

// pick()'s argument for the "Choose ..." entry (the id of its option).
const CHOOSE = 'chat-workdir-choose';

/** The labels of the drop-down's options, in order. */
function optionLabels(win) {
  return Array.from(byId(win, 'chat-workdir').options, o => o.textContent);
}

/** The directory the drop-down shows as current ('' = none). */
function currentDir(win) {
  return byId(win, 'chat-workdir').value;
}

/**
 * Pick *value* (a directory, or CHOOSE for "Choose ...") in the
 * drop-down the way a user does: the option becomes the selected one
 * and the select fires `change` -- which a real select only does for
 * a different option, so picking the selected one is asserted against.
 */
function pick(win, value) {
  const select = byId(win, 'chat-workdir');
  const option =
    value === CHOOSE
      ? byId(win, CHOOSE)
      : Array.from(select.options).find(o => o.value === value);
  assert.ok(option, 'the drop-down lists ' + value);
  assert.ok(!option.selected, value + ' is not the selected option already');
  assert.ok(!option.disabled, value + ' can be picked');
  option.selected = true;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/** The text of every error notification on screen. */
function errorNotifications(win) {
  return Array.from(
    win.document.querySelectorAll('.kiss-notification-error'),
    el => el.textContent,
  );
}

module.exports = {
  makePage,
  closeWindows,
  send,
  byId,
  msgs,
  assertJsonEqual,
  CHOOSE,
  optionLabels,
  currentDir,
  pick,
  errorNotifications,
};
