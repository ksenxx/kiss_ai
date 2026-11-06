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

/**
 * The mounted tips window and its controls: `{host, root, body,
 * counter, prev, next, close}`, `root` its shadow root.  Fails when no
 * panel is mounted on document.body or it has no shadow root.
 */
function panelParts(win) {
  const host = win.document.body.querySelector('kiss-tips-panel');
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

const THREE_TIPS = [
  '## First tip\n\n- alpha\n- beta',
  'Second tip with **bold** text.',
  '### Third tip\n\n`code` here.',
];

module.exports = {panelParts, openOptOutBox, closeTips, THREE_TIPS};
