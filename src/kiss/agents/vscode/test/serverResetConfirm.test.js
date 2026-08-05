// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {JSDOM} = require('jsdom');

const PRELOADED_TAB_ID = 'tab-server-reset-test';

function makeDomWebview() {
  const mediaDir = path.join(__dirname, '..', 'media');
  let html = fs.readFileSync(path.join(mediaDir, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state = {
    tabs: [
      {
        title: 'new chat',
        chatId: PRELOADED_TAB_ID,
        backendChatId: '',
        parentTabId: '',
      },
    ],
    activeTabIndex: 0,
    chatId: PRELOADED_TAB_ID,
  };
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'panelCopy.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'api.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'main.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  return {win, posted, close: () => win.close()};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {bubbles: true}),
  );
}

async function waitFor(predicate, message) {
  for (let i = 0; i < 100; i++) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

function isFloatingModalOpen(win) {
  const modal = win.document.getElementById('server-reset-confirm-modal');
  return !!(modal && modal.classList.contains('open'));
}

async function runTests() {
  {
    const wv = makeDomWebview();
    try {
      const settingsPanel =
        wv.win.document.getElementById('settings-panel');
      const modal = wv.win.document.getElementById(
        'server-reset-confirm-modal',
      );
      assert.ok(modal, '#server-reset-confirm-modal must exist');
      assert.ok(
        settingsPanel && settingsPanel.contains(modal),
        'the floating confirmation box must live INSIDE #settings-panel',
      );
      assert.ok(
        wv.win.document.getElementById('server-reset-confirm-ok'),
        'modal must expose #server-reset-confirm-ok',
      );
      assert.ok(
        wv.win.document.getElementById('server-reset-confirm-cancel'),
        'modal must expose #server-reset-confirm-cancel',
      );
      assert.strictEqual(
        isFloatingModalOpen(wv.win),
        false,
        'floating modal must start closed',
      );
    } finally {
      wv.close();
    }
  }

  {
    const wv = makeDomWebview();
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      assert.ok(btn, 'settings panel must expose #cfg-server-reset-btn');

      click(btn);
      const msg = await waitFor(
        () => wv.posted.find(m => m.type === 'serverReset'),
        'no-agent click must post {type: "serverReset"} to the extension',
      );
      assert.strictEqual(msg.type, 'serverReset');
      assert.strictEqual(
        isFloatingModalOpen(wv.win),
        false,
        'no-agent click must NOT open the floating confirmation box',
      );
      assert.strictEqual(
        wv.win.document.querySelectorAll('.kiss-notification').length,
        0,
        'the webview must NOT render any in-webview confirmation toast',
      );
    } finally {
      wv.close();
    }
  }

  {
    const wv = makeDomWebview();
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      click(btn);
      await waitFor(
        () => isFloatingModalOpen(wv.win),
        'agent-running click must open the floating confirmation box',
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'serverReset').length,
        0,
        'no serverReset post must be made until the user clicks OK',
      );

      click(wv.win.document.getElementById('server-reset-confirm-cancel'));
      await waitFor(
        () => !isFloatingModalOpen(wv.win),
        'Cancel must close the floating confirmation box',
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'serverReset').length,
        0,
        'Cancel must NOT post serverReset',
      );

      click(btn);
      await waitFor(
        () => isFloatingModalOpen(wv.win),
        'a fresh click after Cancel must re-open the dialog',
      );
      click(wv.win.document.getElementById('server-reset-confirm-ok'));
      const msg = await waitFor(
        () => wv.posted.find(m => m.type === 'serverReset'),
        'OK must post serverReset',
      );
      assert.strictEqual(msg.type, 'serverReset');
      assert.strictEqual(
        isFloatingModalOpen(wv.win),
        false,
        'OK must close the floating confirmation box',
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'serverReset').length,
        1,
        'OK must produce exactly ONE serverReset post',
      );
    } finally {
      wv.close();
    }
  }

  {
    const wv = makeDomWebview();
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: false,
      });

      click(btn);
      const msg = await waitFor(
        () => wv.posted.find(m => m.type === 'serverReset'),
        'click after the agent stops must fast-path serverReset',
      );
      assert.strictEqual(msg.type, 'serverReset');
      assert.strictEqual(
        isFloatingModalOpen(wv.win),
        false,
        'fast path must NOT open the floating dialog',
      );
    } finally {
      wv.close();
    }
  }

  {
    const wv = makeDomWebview();
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      click(btn);
      await waitFor(
        () => isFloatingModalOpen(wv.win),
        'first click must open the floating dialog',
      );
      click(btn);
      assert.strictEqual(
        isFloatingModalOpen(wv.win),
        true,
        'second click while the dialog is open must be a no-op',
      );
      click(wv.win.document.getElementById('server-reset-confirm-ok'));
      await waitFor(
        () => !isFloatingModalOpen(wv.win),
        'OK must close the dialog',
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'serverReset').length,
        1,
        'double-click + OK must produce exactly ONE serverReset post',
      );
    } finally {
      wv.close();
    }
  }

  {
    const wv = makeDomWebview();
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });
      click(btn);
      await waitFor(
        () => isFloatingModalOpen(wv.win),
        'click must open the floating dialog',
      );
      wv.win.document.dispatchEvent(
        new wv.win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
      );
      await waitFor(
        () => !isFloatingModalOpen(wv.win),
        'Escape must close the floating dialog',
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'serverReset').length,
        0,
        'Escape must NOT post serverReset',
      );
    } finally {
      wv.close();
    }
  }
}

runTests().then(
  () => {
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
