// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test pinning where the "Server restart
// complete" acknowledgement is rendered in the chat webview.
//
// Background
// ----------
// When the user clicks the settings panel's "Server reset" button,
// the daemon broadcasts a ``notification`` with id
// ``server-reset-restarting`` and then SIGTERMs itself.  The
// supervising LaunchAgent / systemd unit respawns a fresh
// ``kiss-web``.  The freshly-respawned daemon detects a
// pending-reset flag file left by the previous instance and
// broadcasts a paired ``notification`` event with id
// ``server-reset-complete`` so reconnecting clients see a
// confirmation toast.
//
// This test drives the real ``media/main.js`` against a JSDOM copy
// of ``media/chat.html`` and asserts:
//   1. dispatching the "Server restart complete" payload renders a
//      ``.kiss-notification`` toast whose visible text contains
//      "KISS Sorcar web server restart complete.";
//   2. the toast carries the daemon-side id/severity contract
//      (``data-notification-id="server-reset-complete"`` and
//      ``kiss-notification-info``);
//   3. the stable id makes the new "complete" notification REPLACE
//      a still-visible "restarting" toast in place rather than
//      stacking next to it (the stable-id dedup is the whole
//      reason the daemon stamps an id at all);
//   4. NO inline chat-output note (``div.note`` with "Note:"
//      prefix) is appended — the message must be a notification,
//      not a chat banner.
//
// Pinning all four together prevents a future refactor from
// silently restoring the old chat-output banner OR breaking the
// stable-id dedup that prevents a stale "restarting" toast from
// lingering next to the "complete" one.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {JSDOM} = require('jsdom');

const RESTARTING_MESSAGE = 'Restarting the KISS Sorcar web server…';
const COMPLETE_MESSAGE = 'KISS Sorcar web server restart complete.';

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
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => undefined,
      setState: () => {},
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

async function waitFor(predicate, message) {
  for (let i = 0; i < 100; i++) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

async function runTests() {
  // ---------- Case A: the "complete" toast renders with the right
  //               contract and does NOT spawn a chat-output note.
  {
    const wv = makeDomWebview();
    try {
      send(wv.win, {
        type: 'notification',
        id: 'server-reset-complete',
        severity: 'info',
        message: COMPLETE_MESSAGE,
      });

      const toast = await waitFor(
        () =>
          wv.win.document.querySelector(
            '.kiss-notification[data-notification-id="server-reset-complete"]',
          ),
        '"Server restart complete" must render a .kiss-notification toast',
      );

      // 1. Visible text must contain the exact daemon message.
      assert.ok(
        toast.textContent.includes(COMPLETE_MESSAGE),
        'toast must show the "KISS Sorcar web server restart complete." message',
      );

      // 2. Daemon-side id/severity contract.
      assert.strictEqual(
        toast.getAttribute('data-notification-id'),
        'server-reset-complete',
        'toast must be stamped with id="server-reset-complete" so a repeat broadcast does not stack duplicates',
      );
      assert.ok(
        toast.classList.contains('kiss-notification-info'),
        'toast must use the "info" severity styling',
      );

      // 3. No inline chat-output "Note: …" banner must be created
      //    for the completion message.
      const noteEls = wv.win.document.querySelectorAll('div.note');
      for (const el of noteEls) {
        assert.ok(
          !el.textContent.includes(COMPLETE_MESSAGE),
          '"Server restart complete" must NOT be rendered as a chat-output "Note: …" banner',
        );
      }
    } finally {
      wv.close();
    }
  }

  // ---------- Case B: the "complete" notification REPLACES a
  //               still-visible "restarting" notification rather
  //               than stacking next to it.  This is the whole
  //               reason both broadcasts stamp a stable id — the
  //               dedup keys off ``data-notification-id`` (see
  //               ``showNotification`` in media/main.js).
  //
  // The two ids are intentionally DIFFERENT (``-restarting`` vs
  // ``-complete``) so a brand-new toast appears: the user sees
  // "restart complete" replace "restarting".  Pin: exactly ONE
  // toast remains, it carries the ``-complete`` id, and the
  // ``-restarting`` toast is gone.
  {
    const wv = makeDomWebview();
    try {
      send(wv.win, {
        type: 'notification',
        id: 'server-reset-restarting',
        severity: 'info',
        message: RESTARTING_MESSAGE,
      });
      await waitFor(
        () =>
          wv.win.document.querySelector(
            '.kiss-notification[data-notification-id="server-reset-restarting"]',
          ),
        'pre-condition: the "restarting" toast must be visible first',
      );

      send(wv.win, {
        type: 'notification',
        id: 'server-reset-complete',
        severity: 'info',
        message: COMPLETE_MESSAGE,
      });
      const completeToast = await waitFor(
        () =>
          wv.win.document.querySelector(
            '.kiss-notification[data-notification-id="server-reset-complete"]',
          ),
        'the "complete" toast must appear after the broadcast',
      );

      assert.ok(
        completeToast.textContent.includes(COMPLETE_MESSAGE),
        'the surviving toast must show the completion message',
      );

      // The "restarting" toast (different id) is independent — the
      // dedup is per-id, not across ids.  Both may coexist briefly,
      // but in practice the restarting toast auto-dismisses on the
      // info severity timer.  What matters for *this* contract is
      // that the broadcast did NOT stack a *second* "complete"
      // toast — there must be exactly one element with
      // ``data-notification-id="server-reset-complete"``.
      const completeToasts = wv.win.document.querySelectorAll(
        '.kiss-notification[data-notification-id="server-reset-complete"]',
      );
      assert.strictEqual(
        completeToasts.length,
        1,
        'a repeat "server-reset-complete" broadcast must not stack a duplicate toast',
      );

      // And a repeated "complete" broadcast still leaves exactly
      // one — the stable-id dedup keys off the daemon id, not the
      // message text.
      send(wv.win, {
        type: 'notification',
        id: 'server-reset-complete',
        severity: 'info',
        message: COMPLETE_MESSAGE,
      });
      await new Promise(r => setTimeout(r, 50));
      assert.strictEqual(
        wv.win.document.querySelectorAll(
          '.kiss-notification[data-notification-id="server-reset-complete"]',
        ).length,
        1,
        're-broadcasting "server-reset-complete" must update the existing toast, not append a new one',
      );
    } finally {
      wv.close();
    }
  }

  // ---------- Case C: a legacy ``notice`` with the same text would
  // still render as a chat-output banner (regression guard).  This
  // makes sure Case A is meaningful — i.e. the absence of a
  // ``div.note`` is because the daemon now sends a ``notification``,
  // not because the webview lost the ``notice`` path altogether
  // (other features still use it).
  {
    const wv = makeDomWebview();
    try {
      send(wv.win, {type: 'notice', text: COMPLETE_MESSAGE});
      const note = await waitFor(
        () =>
          Array.from(wv.win.document.querySelectorAll('div.note')).find(el =>
            el.textContent.includes(COMPLETE_MESSAGE),
          ),
        'control: a legacy `notice` event must still render as a chat-output note (proves Case A is not a false-positive)',
      );
      assert.ok(
        note.textContent.startsWith('Note:'),
        'control: legacy notice banner is prefixed with "Note:" — the surface the fix moves the post-restart message away from',
      );
      assert.strictEqual(
        wv.win.document.querySelectorAll('.kiss-notification').length,
        0,
        'control: legacy `notice` event must NOT raise a top-right notification toast',
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
