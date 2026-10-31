// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The persisted "don't show tips again" choice lives on the daemon
// ($KISS_HOME/TIPS_DISABLED) and reaches a browser through the `tipsData`
// event the daemon answers every `ready` with.  For an opted-out user
// server/tips.py `tips_data` sends `{tips, show: false, optOut: true,
// version}` (pinned by tests/server/test_tips_opt_out_bootstrap_wave2.py).
// main.js hands the event to tips.js.  On a browser that never stored the
// choice itself the tips window must stay closed and, opened from the
// Tips button, show the box ticked.
//
// Real chat.html, main.js and tips.js on jsdom, browser storage empty.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {test, report} = require('./ui_antipattern_harness').makeRunner();

const TIPS = ['# One\n\nfirst', '# Two\n\nsecond'];

/** The chat page with tips.js and no opt-out stored in this browser. */
function freshBrowserPage() {
  const page = makePage({extraScripts: ['tips.js']});
  page.win.localStorage.clear();
  return page.win;
}

function tipsWindow(win) {
  return win.document.body.querySelector('kiss-tips-panel');
}

/** Open the tips from the Tips button: the opt-out checkbox's state. */
function boxTicked(win) {
  win.document.getElementById('tips-btn').click();
  const host = tipsWindow(win);
  assert.ok(host, 'the Tips button opens the tips window');
  const ticked = host.shadowRoot.querySelector('.tips-optout-input').checked;
  host.shadowRoot.querySelector('.tips-close').click();
  return ticked;
}

test("an opted-out daemon's tipsData opens nothing and ticks the box", () => {
  const win = freshBrowserPage();
  send(win, {
    type: 'tipsData',
    tips: TIPS,
    show: false,
    optOut: true,
    version: '9.9',
  });
  assert.strictEqual(tipsWindow(win), null, 'the tips window stays closed');
  assert.strictEqual(boxTicked(win), true);
  win.close();
});

test("the daemon's opt-out holds even when the event asks to show", () => {
  const win = freshBrowserPage();
  send(win, {
    type: 'tipsData',
    tips: TIPS,
    show: true,
    optOut: true,
    version: '9.9',
  });
  assert.strictEqual(tipsWindow(win), null, 'the tips window stays closed');
  win.close();
});

test('without the opt-out the same event opens the tips once per version', () => {
  const win = freshBrowserPage();
  const event = {type: 'tipsData', tips: TIPS, show: true, version: '9.9'};
  send(win, event);
  const host = tipsWindow(win);
  assert.ok(host, 'the tips window opens on its own');
  assert.strictEqual(
    host.shadowRoot.querySelector('.tips-optout-input').checked,
    false,
  );
  host.shadowRoot.querySelector('.tips-close').click();
  send(win, event);
  assert.strictEqual(tipsWindow(win), null, 'not again for the same version');
  win.close();
});

report('tipsDaemonOptOutNoAutoOpen_wave3');
