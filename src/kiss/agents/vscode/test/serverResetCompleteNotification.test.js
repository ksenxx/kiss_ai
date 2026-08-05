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

async function waitFor(predicate, message) {
  for (let i = 0; i < 100; i++) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

async function runTests() {
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

      assert.ok(
        toast.textContent.includes(COMPLETE_MESSAGE),
        'toast must show the "KISS Sorcar web server restart complete." message',
      );

      assert.strictEqual(
        toast.getAttribute('data-notification-id'),
        'server-reset-complete',
        'toast must be stamped with id="server-reset-complete" so a repeat broadcast does not stack duplicates',
      );
      assert.ok(
        toast.classList.contains('kiss-notification-info'),
        'toast must use the "info" severity styling',
      );

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

      const completeToasts = wv.win.document.querySelectorAll(
        '.kiss-notification[data-notification-id="server-reset-complete"]',
      );
      assert.strictEqual(
        completeToasts.length,
        1,
        'a repeat "server-reset-complete" broadcast must not stack a duplicate toast',
      );

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
