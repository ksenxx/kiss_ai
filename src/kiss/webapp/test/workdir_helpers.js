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
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body class="remote-chat"');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  openWindows.push(win);
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state;
  win.acquireTransport = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  if (desktop) {
    win.requestAnimationFrame = cb => {
      cb();
      return 0;
    };
    win.cancelAnimationFrame = () => {};
    win.matchMedia = query => ({
      matches: query === '(min-width: 900px)',
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    });
  }
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

/** Close every window makePage built. */
function closeWindows() {
  while (openWindows.length) openWindows.pop().close();
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byId(win, id) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  return node;
}

function msgs(posted, type) {
  return posted.filter(m => m && m.type === type);
}

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
