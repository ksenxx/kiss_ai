// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The daemon answers the page's `ready` with a `tipsData` event.  The
// page opens the tips window on it exactly once per
// version (tips.js's once-per-version guard), centered over the chat,
// and a reconnect that re-sends tipsData does not open a second one.

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {makePage, stubRaf, fakeMatchMedia} = require('./simplify2_harness');
const {test, report} = makeRunner();

function loadChatDom() {
  const {win} = makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js', 'tips.js'],
    sourceUrl: 'tips-only-in-chat-main.js',
    beforeScripts(win) {
      win.localStorage.clear();
      stubRaf(win);
      win.setInterval = function () {
        return 0;
      };
      win.clearInterval = function () {};
      win.ResizeObserver = class {
        observe() {}
        disconnect() {}
      };
      fakeMatchMedia(win, true);
      win.__TIPS__ = {tips: [], show: false, version: ''};
    },
  });
  return win;
}

const TIPS_DATA = {
  type: 'tipsData',
  tips: ['# One\n\nfirst', '# Two\n\nsecond'],
  show: true,
  version: '2026.10.1',
};

function sendTips(win) {
  win.dispatchEvent(new win.MessageEvent('message', {data: TIPS_DATA}));
}

function tipsPanels(win) {
  return win.document.body.querySelectorAll('kiss-tips-panel').length;
}

test('the page opens the tips window once, centered over the chat', () => {
  const win = loadChatDom();
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 1);
  assert.strictEqual(
    win.localStorage.getItem('kissTipsSeenVersion'),
    '2026.10.1',
  );
  const host = win.document.body.querySelector('kiss-tips-panel');
  const css = host.shadowRoot.querySelector('style').textContent;
  const overlay = css.split('.tips-overlay {')[1].split('}')[0];
  assert.ok(/position:\s*fixed/.test(overlay), 'overlay covers the page');
  assert.ok(/align-items:\s*center/.test(overlay), 'panel centered vertically');
  assert.ok(
    /justify-content:\s*center/.test(overlay),
    'panel centered horizontally',
  );
  // A reconnect re-sends tipsData: still one window.
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 1);
});

report('tipsOnlyInChatPage');
