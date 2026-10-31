// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The tips window's "Don't show tips automatically" checkbox
// (media/tips.js) shows the daemon's persisted opt-out, not only this
// browser's localStorage copy.  The daemon's copy is the durable one
// (server/tips.py TIPS_DISABLED): on a second device, or after the
// site data was cleared, the tips correctly stay closed on load, and
// opening them from the Tips button must show the box ticked.  The
// daemon reports the choice as `optOut` in the tips bootstrap
// (window.__TIPS__, page load) and in every `tipsData` event (main.js
// passes it to window.__kissApplyTipsConfig).

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {newDom, MEDIA} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {test, report} = makeRunner();

const TIPS = ['# One\n\nfirst', '# Two\n\nsecond'];

/** The chat page with only tips.js run over `window.__TIPS__ = cfg`. */
function loadTipsPage(cfg, storage) {
  const win = newDom('outside-only');
  win.localStorage.clear();
  for (const k of Object.keys(storage || {})) {
    win.localStorage.setItem(k, storage[k]);
  }
  win.eval(`window.__TIPS__ = ${JSON.stringify(cfg)};`);
  win.eval(fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf8'));
  return win;
}

/** Open the tips from the Tips button and return the opt-out checkbox. */
function openOptOutBox(win) {
  win.document.getElementById('tips-btn').click();
  const host = win.document.body.querySelector('kiss-tips-panel');
  assert.ok(host, 'the Tips button opens the tips window');
  return host.shadowRoot.querySelector('.tips-optout-input');
}

/** Close the open tips window with its close button. */
function closeTips(win) {
  const host = win.document.body.querySelector('kiss-tips-panel');
  host.shadowRoot.querySelector('.tips-close').click();
  assert.strictEqual(win.document.body.querySelector('kiss-tips-panel'), null);
}

test('the page-load opt-out of the daemon ticks the box in a browser without one', () => {
  const win = loadTipsPage({
    tips: TIPS,
    show: false,
    version: '1.0',
    optOut: true,
  });
  assert.strictEqual(
    win.document.body.querySelector('kiss-tips-panel'),
    null,
    'the tips stay closed on load',
  );
  assert.strictEqual(win.localStorage.getItem('kissTipsOptOut'), null);
  assert.strictEqual(openOptOutBox(win).checked, true);
});

test('a tipsData event carrying the opt-out of the daemon ticks the box', () => {
  const win = loadTipsPage(
    {tips: TIPS, show: true, version: '1.0'},
    {
      kissTipsSeenVersion: '1.0',
    },
  );
  win.__kissApplyTipsConfig({
    tips: TIPS,
    show: false,
    version: '1.0',
    optOut: true,
  });
  assert.strictEqual(openOptOutBox(win).checked, true);
});

test('unticking the box of a daemon opt-out holds when the tips are reopened', () => {
  const win = loadTipsPage({
    tips: TIPS,
    show: false,
    version: '1.0',
    optOut: true,
  });
  const relayed = [];
  win.addEventListener('kiss-tips-opt-out', e =>
    relayed.push(JSON.parse(JSON.stringify(e.detail))),
  );
  const box = openOptOutBox(win);
  box.checked = false;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.deepStrictEqual(relayed, [{type: 'tipsOptOut', optOut: false}]);
  closeTips(win);
  assert.strictEqual(openOptOutBox(win).checked, false);
});

test('without a daemon opt-out the box follows this browser alone', () => {
  const daemonNo = loadTipsPage({
    tips: TIPS,
    show: false,
    version: '1.0',
    optOut: false,
  });
  assert.strictEqual(openOptOutBox(daemonNo).checked, false);
  const localYes = loadTipsPage(
    {tips: TIPS, show: false, version: '1.0', optOut: false},
    {kissTipsOptOut: '1'},
  );
  assert.strictEqual(openOptOutBox(localYes).checked, true);
});

test('ticking the box in a page with no bootstrap still records the choice', () => {
  const win = newDom('outside-only');
  win.localStorage.clear();
  win.eval(fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf8'));
  const box = openOptOutBox(win);
  assert.strictEqual(box.checked, false);
  box.checked = true;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(win.localStorage.getItem('kissTipsOptOut'), '1');
  assert.strictEqual(win.__TIPS__, undefined);
});

report('tipsOptOutDaemon_sweep1030');
