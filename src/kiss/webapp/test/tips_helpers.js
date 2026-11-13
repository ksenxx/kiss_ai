// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Helpers shared by the tips-window suites (tipsWindow,
// tipsSettingsButton, tipsOptOutDaemon_sweep1030,
// tipsDaemonOptOutNoAutoOpen_wave3): the parts of the mounted
// <kiss-tips-panel>, the opt-out box behind the Tips button and a
// three-tip Markdown fixture.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {newDom, MEDIA} = require('./simplify2_harness');

/** The bare chat page running only tips.js with cfg and optional stored values. */
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

/** The mounted tips host, or null when closed. */
function tipsHost(win) {
  return win.document.body.querySelector('kiss-tips-panel');
}

/**
 * The mounted tips window and its controls: `{host, root, body,
 * counter, prev, next, close}`, `root` its shadow root.  Fails when no
 * panel is mounted on document.body or it has no shadow root.
 */
function panelParts(win) {
  const host = tipsHost(win);
  assert.ok(host, 'kiss-tips-panel must be mounted on document.body');
  const root = host.shadowRoot;
  assert.ok(root, 'kiss-tips-panel must use shadow DOM');
  return {
    host,
    root,
    body: root.querySelector('.tips-body'),
    counter: root.querySelector('.tips-counter'),
    prev: root.querySelector('.tips-prev'),
    next: root.querySelector('.tips-next'),
    close: root.querySelector('.tips-close'),
  };
}

/** Open the tips from the Tips button and return the opt-out checkbox. */
function openOptOutBox(win) {
  win.document.getElementById('tips-btn').click();
  const host = tipsHost(win);
  assert.ok(host, 'the Tips button opens the tips window');
  return host.shadowRoot.querySelector('.tips-optout-input');
}

/** Close the open tips window with its close button. */
function closeTips(win) {
  tipsHost(win).shadowRoot.querySelector('.tips-close').click();
  assert.strictEqual(tipsHost(win), null);
}

const THREE_TIPS = [
  '## First tip\n\n- alpha\n- beta',
  'Second tip with **bold** text.',
  '### Third tip\n\n`code` here.',
];

module.exports = {
  loadTipsPage,
  tipsHost,
  panelParts,
  openOptOutBox,
  closeTips,
  THREE_TIPS,
};
