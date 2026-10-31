// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The daemon answers every `ready` with a `tipsData` event carrying the
// user's persisted "Don't show tips automatically" choice as `optOut`
// (server/tips.py `tips_data`).  main.js hands the event to tips.js
// (window.__kissApplyTipsConfig); dropping `optOut` there would leave
// the tips window's checkbox unticked on a device whose browser never
// stored the choice itself, although the daemon keeps the tips closed.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {test, report} = require('./ui_antipattern_harness').makeRunner();

const TIPS = ['# One\n\nfirst', '# Two\n\nsecond'];

/** The chat page with tips.js, its browser storage empty. */
function tipsPage() {
  const page = makePage({extraScripts: ['tips.js']});
  page.win.localStorage.clear();
  return page.win;
}

/** Open the tips from the Tips button: the opt-out checkbox's state. */
function optOutTicked(win) {
  win.document.getElementById('tips-btn').click();
  const host = win.document.body.querySelector('kiss-tips-panel');
  assert.ok(host, 'the Tips button opens the tips window');
  const box = host.shadowRoot.querySelector('.tips-optout-input');
  const ticked = box.checked;
  host.shadowRoot.querySelector('.tips-close').click();
  return ticked;
}

test("the daemon's opt-out in tipsData ticks the box", () => {
  const win = tipsPage();
  send(win, {
    type: 'tipsData',
    tips: TIPS,
    show: false,
    version: '2.0',
    optOut: true,
  });
  assert.strictEqual(optOutTicked(win), true);
  assert.strictEqual(win.__TIPS__.optOut, true);
  win.close();
});

test('a tipsData without the opt-out leaves the box unticked', () => {
  const win = tipsPage();
  send(win, {type: 'tipsData', tips: TIPS, show: false, version: '2.0'});
  assert.strictEqual(optOutTicked(win), false);
  send(win, {
    type: 'tipsData',
    tips: TIPS,
    show: false,
    version: '2.0',
    optOut: 'yes',
  });
  assert.strictEqual(optOutTicked(win), false, 'only true opts out');
  win.close();
});

report('tipsDataOptOut_wave2');
