// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The daemon answers every webview's `ready` with a `tipsData` event.
// In editor-tabs mode three webviews connect at once: the primary-
// sidebar History view, the secondary-sidebar Task Info view and the
// chat editor tab.  Each VS Code webview type has its own origin and
// therefore its own localStorage, so tips.js's once-per-version guard
// cannot dedupe across them: the tips window used to open in BOTH
// sidebars.  Only chat surfaces may open it: the chat editor tab, the
// sidebar chat view and the remote webapp page.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function loadChatDom(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body' + bodyAttrs);
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
  win.acquireVsCodeApi = function () {
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
  // Same script order as chat.html: the empty VS Code bootstrap, then
  // panelCopy.js, tips.js, then main.js.
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

const CHAT_EDITOR_TAB = ' class="editor-tab-mode" data-kiss-tab-id="panel-1"';
const SIDEBAR_CHAT = '';
const REMOTE = ' class="remote-chat"';
const HISTORY_VIEW =
  ' class="editor-tab-mode history-panel-mode" data-kiss-tab-id="history-panel"';
const TASK_INFO_VIEW =
  ' class="editor-tab-mode meta-panel-mode" data-kiss-tab-id="meta-panel"';

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

test('the History sidebar view never opens the tips window on tipsData', () => {
  const win = loadChatDom(HISTORY_VIEW);
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 0);
  assert.strictEqual(
    win.localStorage.getItem('kissTipsSeenVersion'),
    null,
    'the History view must not claim the version either',
  );
});

test('the Task Info sidebar view never opens the tips window on tipsData', () => {
  const win = loadChatDom(TASK_INFO_VIEW);
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 0);
  assert.strictEqual(win.localStorage.getItem('kissTipsSeenVersion'), null);
});

test('the chat editor tab opens the tips window once, centered over the chat', () => {
  const win = loadChatDom(CHAT_EDITOR_TAB);
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 1);
  assert.strictEqual(
    win.localStorage.getItem('kissTipsSeenVersion'),
    '2026.10.1',
  );
  const host = win.document.body.querySelector('kiss-tips-panel');
  const css = host.shadowRoot.querySelector('style').textContent;
  const overlay = css.split('.tips-overlay {')[1].split('}')[0];
  assert.ok(/position:\s*fixed/.test(overlay), 'overlay covers the webview');
  assert.ok(/align-items:\s*center/.test(overlay), 'panel centered vertically');
  assert.ok(
    /justify-content:\s*center/.test(overlay),
    'panel centered horizontally',
  );
  // A reconnect re-sends tipsData: still one window.
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 1);
});

test('the sidebar chat view (editor tabs off) opens the tips window', () => {
  const win = loadChatDom(SIDEBAR_CHAT);
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 1);
});

test('the remote webapp page opens the tips window', () => {
  const win = loadChatDom(REMOTE);
  sendTips(win);
  assert.strictEqual(tipsPanels(win), 1);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
for (const {name, err} of failures) {
  console.log(`\n${name}\n${err && err.stack}`);
}
if (failures.length) process.exit(1);
