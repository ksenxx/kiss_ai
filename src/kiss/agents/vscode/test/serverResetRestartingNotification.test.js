// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test pinning where the "Restarting the KISS
// Sorcar web server…" acknowledgement is rendered in the chat
// webview.
//
// Bug reproduced
// --------------
// When the user clicks the settings panel's "Server reset" button,
// the kiss-web daemon used to broadcast a ``notice`` event back to
// the clicking window, which the webview rendered as an inline
// "Note: …" banner inside the chat output area
// (``addNotice(text)``).  The note got buried under the chat
// transcript, never grabbed user focus, and disappeared together
// with the socket when the daemon SIGTERM-ed itself a few hundred
// milliseconds later.
//
// The fix changes the daemon to broadcast a ``notification`` event
// instead, so the existing top-right webview notification stack
// (``updateNotification(ev)``) renders the acknowledgement as a
// real ``.kiss-notification`` toast — the same visual surface the
// user already sees for "update available", "worktree_result", and
// other transient status events.
//
// This test drives the real ``media/main.js`` against a JSDOM copy
// of ``media/chat.html`` and asserts:
//   1. dispatching the new server-reset payload renders a
//      ``.kiss-notification`` toast whose visible text contains
//      "Restarting the KISS Sorcar web server…";
//   2. the toast carries the daemon-side id/severity contract
//      (``data-notification-id="server-reset-restarting"`` and
//      ``kiss-notification-info``);
//   3. NO inline chat-output note (``div.note`` with "Note:" prefix)
//      is appended — the message must be a notification, not a note.
//
// Pinning all three together prevents a future refactor from
// silently restoring the old chat-output banner.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {JSDOM} = require('jsdom');

const SERVER_RESET_MESSAGE = 'Restarting the KISS Sorcar web server…';

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
  // ---------- Case A: notification renders, no chat-output note.
  {
    const wv = makeDomWebview();
    try {
      send(wv.win, {
        type: 'notification',
        id: 'server-reset-restarting',
        severity: 'info',
        message: SERVER_RESET_MESSAGE,
      });

      const toast = await waitFor(
        () => wv.win.document.querySelector('.kiss-notification'),
        'server-reset acknowledgement must render a .kiss-notification toast',
      );

      // 1. The toast's visible text must contain the exact daemon message.
      assert.ok(
        toast.textContent.includes(SERVER_RESET_MESSAGE),
        'toast must show the "Restarting the KISS Sorcar web server…" message',
      );

      // 2. The toast must carry the daemon-side id/severity contract.
      assert.strictEqual(
        toast.getAttribute('data-notification-id'),
        'server-reset-restarting',
        'toast must be stamped with id="server-reset-restarting" so a repeat broadcast does not stack duplicates',
      );
      assert.ok(
        toast.classList.contains('kiss-notification-info'),
        'toast must use the "info" severity styling',
      );

      // 3. No inline chat-output "Note: …" banner must be created.
      //    ``addNotice`` builds a ``div.ev.tr.note`` whose innerHTML
      //    starts with "<strong>Note:</strong> …".  Pin both the
      //    selector and the message body so a future regression that
      //    routes the event through ``addNotice`` is caught here.
      const noteEls = wv.win.document.querySelectorAll('div.note');
      for (const el of noteEls) {
        assert.ok(
          !el.textContent.includes('web server') &&
            !el.textContent.toLowerCase().includes('restart'),
          'server-reset message must NOT be rendered as a chat-output "Note: …" banner',
        );
      }
    } finally {
      wv.close();
    }
  }

  // ---------- Case B: a legacy ``notice`` with the same text would
  // still render as a chat-output banner (regression guard).  This
  // makes sure the test in Case A is meaningful — i.e. the absence
  // of a ``div.note`` is because the daemon now sends a
  // ``notification``, not because the webview lost the ``notice``
  // path altogether (other features still use it).
  {
    const wv = makeDomWebview();
    try {
      send(wv.win, {type: 'notice', text: SERVER_RESET_MESSAGE});
      const note = await waitFor(
        () =>
          Array.from(wv.win.document.querySelectorAll('div.note')).find(el =>
            el.textContent.includes(SERVER_RESET_MESSAGE),
          ),
        'control: a legacy `notice` event must still render as a chat-output note (proves Case A is not a false-positive)',
      );
      assert.ok(
        note.textContent.startsWith('Note:'),
        'control: legacy notice banner is prefixed with "Note:" — the surface the fix moves the server-reset message away from',
      );
      // And critically — a legacy ``notice`` event must NOT raise a
      // top-right notification toast either, so Case A's toast is
      // unambiguously caused by the new ``notification`` event.
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
