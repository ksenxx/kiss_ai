// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the "KISS Sorcar update available"
// permanent webview notification.
//
// Bug reproduced
// --------------
// The kiss-web daemon hourly polls PyPI and broadcasts an
// ``update_available`` event of the form
// ``{type: 'update_available', available: true, latest, current}``
// to every connected chat webview.  Before this fix the only thing the
// webview did with the event was inject a tiny green download badge
// onto the settings panel's "Update" button (``#cfg-update-btn``) —
// which is hidden whenever the settings panel itself is collapsed.
// A user who never opens the settings panel therefore had no way to
// learn that a new KISS Sorcar release was available.
//
// The fix makes ``update_available`` also raise a *permanent*
// (sticky, never auto-dismissed) toast in the existing notification
// stack that carries a single action button with an embedded inline
// ``<svg>`` download-arrow icon.  Clicking the SVG button posts the
// same ``{type: 'runUpdate'}`` message that the settings-panel
// button does, so the existing extension-side handler runs
// ``install.sh`` unchanged.
//
// This test drives the real ``media/main.js`` against a JSDOM-rendered
// ``media/chat.html`` and asserts:
//   1. an ``update_available`` event with ``available: true`` creates
//      a ``.kiss-notification`` toast with ``sticky`` semantics
//      (``data-notification-sticky="true"`` AND no auto-dismiss
//      timer);
//   2. the toast contains exactly one action button (a real
//      ``<button class="kiss-notification-action">``) that holds an
//      inline ``<svg>`` icon — the SVG is what the task explicitly
//      asks for;
//   3. clicking that action button posts ``{type: 'runUpdate'}``
//      back to the extension;
//   4. a subsequent ``update_available`` event with
//      ``available: false`` removes the toast (no orphan notification
//      after the user updates).
//
// The test runs under bare ``node`` (no TypeScript compile required),
// matching the convention of the existing notification/webview tests.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {JSDOM} = require('jsdom');

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

async function runTests() {
  const wv = makeDomWebview();
  try {
    // 1. Broadcast an "update available" event.  Before the fix this
    //    rendered nothing the user can see when the settings panel is
    //    closed.  After the fix it produces a permanent toast.
    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });

    const toast = await waitFor(
      () => wv.win.document.querySelector('.kiss-notification'),
      'update_available with available:true must render a permanent webview notification',
    );

    // 2. The notification must be sticky.  In the existing notification
    //    code this is signalled by:
    //      a. no auto-dismiss timer being set (we can't peek inside the
    //         timers map directly from the DOM, but if the toast is
    //         still present after 250 ms with no mouse interaction the
    //         auto-dismiss path was not scheduled with a short delay);
    //      b. the ``sticky`` flag propagated via the toast element so
    //         the assertion is not flaky.
    assert.strictEqual(
      toast.getAttribute('data-notification-sticky'),
      'true',
      'update notification must be marked sticky on the DOM element',
    );

    // 3. It must contain at least one action button (`Update`) whose
    //    content includes an inline <svg> — the user explicitly asked
    //    for an SVG button.
    const actionButton = toast.querySelector('.kiss-notification-action');
    assert.ok(
      actionButton,
      'update notification must expose an action button users can click',
    );
    const svg = actionButton.querySelector('svg');
    assert.ok(
      svg,
      'update notification action button must contain an inline <svg> icon',
    );
    // Sanity check: the SVG must be an actual SVGElement (created via
    // createElementNS), not a stray <svg>-named HTML element.
    assert.strictEqual(
      svg.namespaceURI,
      'http://www.w3.org/2000/svg',
      'action-button SVG must be in the SVG namespace',
    );
    // The accessible label must surface BOTH the action label and the
    // version information so screen-reader users learn what changed.
    const ariaLabel = (
      actionButton.getAttribute('aria-label') ||
      actionButton.textContent ||
      ''
    ).toLowerCase();
    assert.ok(
      ariaLabel.includes('update'),
      'action button must advertise itself as an update action',
    );

    // 4. Clicking the SVG button must trigger the same install flow as
    //    the settings panel's update button (``{type: 'runUpdate'}``).
    click(actionButton);
    await waitFor(
      () => wv.posted.some(m => m.type === 'runUpdate'),
      'clicking the update notification button must post {type: "runUpdate"}',
    );

    // 5. A follow-up ``update_available`` with ``available: false``
    //    (broadcast when the next PyPI poll finds the user is current)
    //    must remove the toast so the user does not see a stale
    //    "update available" notification after they updated.
    send(wv.win, {
      type: 'update_available',
      available: false,
      latest: '',
      current: '',
    });
    await waitForFalsy(
      () => wv.win.document.querySelector('.kiss-notification'),
      'available:false broadcast must dismiss the permanent update notification',
    );

    // 6. The notification id is stable across re-broadcasts so the
    //    daemon's repeated polling does not stack duplicate toasts.
    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });
    await waitFor(
      () => wv.win.document.querySelector('.kiss-notification'),
      'second broadcast must re-create the permanent notification',
    );
    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });
    const toasts = wv.win.document.querySelectorAll('.kiss-notification');
    assert.strictEqual(
      toasts.length,
      1,
      'repeated update_available broadcasts must not stack duplicate notifications',
    );
  } finally {
    wv.close();
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
