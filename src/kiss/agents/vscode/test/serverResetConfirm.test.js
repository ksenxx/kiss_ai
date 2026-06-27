// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the settings-panel "Server reset"
// button.
//
// Bug reproduced
// --------------
// The "Server reset" button (``#cfg-server-reset-btn``) in the
// settings panel asks the kiss-web daemon to SIGTERM itself so the
// supervising LaunchAgent/systemd unit respawns a fresh daemon
// process.  Before this fix the click handler unconditionally posted
// ``{type: 'serverReset'}`` to the extension — even when the user's
// currently active (or any) tab still owned a running agent.  Running
// agents would therefore be killed mid-task without any prompt or
// chance for the user to cancel the destructive action.
//
// The fix: when a server-reset click happens while ANY tab has
// ``isRunning === true``, the webview must NOT post the reset
// immediately.  Instead it must surface a sticky confirmation
// notification asking the user whether to forcefully restart the
// server.  Only after the user activates the "Forcefully restart"
// affordance does ``{type: 'serverReset'}`` get posted.  Clicking
// "Cancel" must dismiss the prompt without posting anything.
//
// When no agent is running the click must still post the reset
// immediately (preserving the existing fast path).
//
// This test drives the real ``media/main.js`` against a JSDOM-rendered
// ``media/chat.html`` and asserts the contract above.

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
  // Strip any inline <script>…</script> blocks the template carries —
  // panelCopy.js + main.js are evaluated by hand below so the JSDOM
  // sandbox starts from a known, identical baseline to the other
  // webview tests in this folder.
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
  // Preload a single-tab persisted state with a deterministic chatId
  // so the test can target the active tab by id when broadcasting
  // status updates.
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

async function waitForFalsy(predicate, message) {
  for (let i = 0; i < 100; i++) {
    if (!predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitForFalsy timed out');
}

function findActionButton(toast, labelMatcher) {
  const buttons = toast.querySelectorAll('.kiss-notification-action');
  for (const btn of buttons) {
    const label = (btn.textContent || '').trim().toLowerCase();
    const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
    if (labelMatcher(label, aria)) return btn;
  }
  return null;
}

async function runTests() {
  // ---------- Case A: no agent running → reset goes through immediately.
  {
    const wv = makeDomWebview();
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      assert.ok(btn, 'settings panel must expose #cfg-server-reset-btn');

      click(btn);

      // The fast path is unchanged: the click posts the reset right
      // away with NO confirmation notification interposed.
      await waitFor(
        () => wv.posted.some(m => m.type === 'serverReset'),
        'click without a running agent must post {type: "serverReset"} immediately',
      );
      assert.strictEqual(
        wv.win.document.querySelectorAll('.kiss-notification').length,
        0,
        'fast path must not surface a confirmation notification',
      );
    } finally {
      wv.close();
    }
  }

  // ---------- Case B: agent running → click shows confirmation prompt;
  // ``Cancel`` dismisses it without posting; a second click + ``Forcefully
  // restart`` posts the reset.
  {
    const wv = makeDomWebview();
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      assert.ok(btn);

      // Mark the (single, preloaded) active tab as running, exactly as
      // a real backend ``status`` event would.
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      // First click — must NOT immediately post serverReset.  Must
      // raise a sticky confirmation toast with two action buttons.
      click(btn);

      const toast = await waitFor(
        () => wv.win.document.querySelector('.kiss-notification'),
        'click with a running agent must raise a confirmation notification',
      );

      // The toast must be sticky: the click is asking the user to
      // confirm a destructive action; a self-dismissing toast would
      // make the prompt fire-and-forget.
      assert.strictEqual(
        toast.getAttribute('data-notification-sticky'),
        'true',
        'confirmation toast must be sticky',
      );

      // Body text must mention that an agent is still running so the
      // user understands WHY they are being asked.
      const body = (toast.textContent || '').toLowerCase();
      assert.ok(
        body.includes('agent') && body.includes('running'),
        'confirmation must explain that an agent is still running, got: ' +
          JSON.stringify(toast.textContent),
      );

      // No serverReset must have been posted yet — that is the bug
      // this test guards against.
      assert.ok(
        !wv.posted.some(m => m.type === 'serverReset'),
        'serverReset must NOT be posted before the user confirms',
      );

      // The toast must expose at least a Cancel and a Forcefully-
      // restart affordance.
      const cancelBtn = findActionButton(
        toast,
        label => label.includes('cancel'),
      );
      assert.ok(
        cancelBtn,
        'confirmation must expose a Cancel button',
      );
      const forceBtn = findActionButton(
        toast,
        (label, aria) =>
          label.includes('forcefully') ||
          label.includes('force') ||
          aria.includes('forcefully'),
      );
      assert.ok(
        forceBtn,
        'confirmation must expose a "Forcefully restart" button',
      );

      // Clicking Cancel must close the toast and NOT post serverReset.
      click(cancelBtn);
      await waitForFalsy(
        () => wv.win.document.querySelector('.kiss-notification'),
        'Cancel must dismiss the confirmation notification',
      );
      assert.ok(
        !wv.posted.some(m => m.type === 'serverReset'),
        'Cancel must not post {type: "serverReset"}',
      );

      // Click again → confirmation reappears.  Click "Forcefully
      // restart" — this time the reset must be posted.
      click(btn);
      const toast2 = await waitFor(
        () => wv.win.document.querySelector('.kiss-notification'),
        'second click while running must re-raise the confirmation',
      );
      const forceBtn2 = findActionButton(
        toast2,
        (label, aria) =>
          label.includes('forcefully') ||
          label.includes('force') ||
          aria.includes('forcefully'),
      );
      assert.ok(forceBtn2);
      click(forceBtn2);
      await waitFor(
        () => wv.posted.some(m => m.type === 'serverReset'),
        '"Forcefully restart" must post {type: "serverReset"}',
      );
      await waitForFalsy(
        () => wv.win.document.querySelector('.kiss-notification'),
        '"Forcefully restart" must dismiss the confirmation notification',
      );
    } finally {
      wv.close();
    }
  }

  // ---------- Case C: agent stops running → subsequent reset bypasses
  // the prompt again (no stale "still running" state).
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
      await waitFor(
        () => wv.posted.some(m => m.type === 'serverReset'),
        'after the agent stops, click must post the reset immediately',
      );
      assert.strictEqual(
        wv.win.document.querySelectorAll('.kiss-notification').length,
        0,
        'no prompt should appear once no agent is running',
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
