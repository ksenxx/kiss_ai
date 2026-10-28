// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The daemon answers the page's `ready` with a `tipsData` event.  The
// page opens the tips window on it exactly once per version (tips.js's
// once-per-version guard) and a reconnect that re-sends tipsData does
// not open a second one.  The window itself (centering, markdown,
// paging) is tipsWindow.test.js's; this only covers main.js's
// message path into it.

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  makePage,
  stubRaf,
  fakeIntervals,
  fakeMatchMedia,
} = require('./simplify2_harness');
const {test, report} = makeRunner();

function loadChatDom() {
  const {win} = makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js', 'tips.js'],
    sourceUrl: 'tips-only-in-chat-main.js',
    beforeScripts(win) {
      win.localStorage.clear();
      stubRaf(win);
      fakeIntervals(win);
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

test('the tipsData message opens the tips window once', () => {
  const win = loadChatDom();
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 1);
  assert.strictEqual(
    win.localStorage.getItem('kissTipsSeenVersion'),
    '2026.10.1',
  );
  // A reconnect re-sends tipsData: still one window.
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 1);
});

report('tipsOnlyInChatPage');
