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
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function loadChatDom() {
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
  win.localStorage.clear();
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};
  win.setInterval = function () {
    return 0;
  };
  win.clearInterval = function () {};
  win.ResizeObserver = class {
    observe() {}
    disconnect() {}
  };
  win.acquireTransport = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.matchMedia = function (query) {
    return {
      matches: query === '(min-width: 900px)',
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  // Same script order as chat.html: panelCopy.js, tips.js, api.js,
  // then main.js.
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval('window.__TIPS__ = {"tips":[],"show":false,"version":""};');
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=tips-only-in-chat-main.js',
  );
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

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  not ok - ${name}`);
  }
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

console.log(`\n${passed} passed, ${failures.length} failed`);
for (const {name, err} of failures) {
  console.log(`\n${name}\n${err && err.stack}`);
}
if (failures.length) process.exit(1);
