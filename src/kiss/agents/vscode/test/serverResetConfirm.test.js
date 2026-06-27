// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Webview-side contract test for the settings-panel "Server reset"
// button.
//
// Background
// ----------
// The "Server reset" button (``#cfg-server-reset-btn``) in the
// settings panel asks the kiss-web daemon to SIGTERM itself so the
// supervising LaunchAgent/systemd unit respawns a fresh daemon
// process.  When a server-reset click happens while ANY tab still
// has a running agent, the webview MUST tell the extension about it
// so the extension can surface a native VS Code modal dialog asking
// the user to confirm — running agents would otherwise be killed
// mid-task without any prompt.
//
// The webview's responsibility is therefore narrow but exact:
//   * On every click, post ``{type: 'serverReset', agentRunning}``.
//   * ``agentRunning`` must be ``true`` iff any persisted tab has
//     ``isRunning === true``.
//   * The webview must NOT render any in-webview confirmation toast;
//     the dialog is the extension's job (a real VS Code modal, not
//     a webview toast).
//
// The integration test in ``serverResetDialog.test.js`` covers the
// dialog half — this test pins the webview half so a future refactor
// of ``main.js`` cannot silently drop the ``agentRunning`` flag.

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

async function runTests() {
  // ---------- Case A: no agent running.
  // The webview must report agentRunning=false so the extension can
  // fast-path the reset, and must NOT render any in-webview toast.
  {
    const wv = makeDomWebview();
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      assert.ok(btn, 'settings panel must expose #cfg-server-reset-btn');

      click(btn);
      const msg = await waitFor(
        () => wv.posted.find(m => m.type === 'serverReset'),
        'click must post {type: "serverReset"} to the extension',
      );
      assert.strictEqual(
        msg.agentRunning,
        false,
        'with no running agent the webview must post agentRunning=false',
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

  // ---------- Case B: agent running.
  // The webview must still post {type:'serverReset', agentRunning:true}
  // — the dialog is the extension's job.  It must NOT render any
  // in-webview confirmation toast and must NOT swallow the post.
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
      const msg = await waitFor(
        () => wv.posted.find(m => m.type === 'serverReset'),
        'click with a running agent must still post serverReset (with agentRunning=true)',
      );
      assert.strictEqual(
        msg.agentRunning,
        true,
        'with a running agent the webview must post agentRunning=true so the extension can raise the dialog',
      );
      assert.strictEqual(
        wv.win.document.querySelectorAll('.kiss-notification').length,
        0,
        'the webview must NOT render any in-webview confirmation toast — the dialog is a real VS Code modal',
      );
    } finally {
      wv.close();
    }
  }

  // ---------- Case C: agent finished.
  // After ``running:false`` arrives the agentRunning flag must reset
  // — no stale "still running" reports on subsequent clicks.
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
        'click after the agent stops must post serverReset',
      );
      assert.strictEqual(
        msg.agentRunning,
        false,
        'after the agent stops, agentRunning must be false on subsequent clicks',
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
