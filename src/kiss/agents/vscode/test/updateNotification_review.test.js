// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Review-driven end-to-end regressions for the KISS Sorcar update
// notification work.  These tests exercise the real media/main.js in a
// JSDOM webview and lock down bugs found during the post-change review.

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
  const scheduledTimeouts = [];
  const realSetTimeout = win.setTimeout.bind(win);
  const realClearTimeout = win.clearTimeout.bind(win);
  win.setTimeout = function (callback, delay) {
    const timeout = {callback, delay, cleared: false};
    scheduledTimeouts.push(timeout);
    return timeout;
  };
  win.clearTimeout = function (timeout) {
    if (timeout && typeof timeout === 'object') {
      timeout.cleared = true;
      return;
    }
    realClearTimeout(timeout);
  };

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
  return {
    win,
    posted,
    scheduledTimeouts,
    realSetTimeout,
    close: () => win.close(),
  };
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {bubbles: true}),
  );
}

function timerDelays(wv) {
  return wv.scheduledTimeouts.filter(t => !t.cleared).map(t => t.delay);
}

function latestTimer(wv, delay) {
  for (let i = wv.scheduledTimeouts.length - 1; i >= 0; i--) {
    const timeout = wv.scheduledTimeouts[i];
    if (!timeout.cleared && timeout.delay === delay) return timeout;
  }
  return null;
}

async function waitFor(predicate, message) {
  for (let i = 0; i < 100; i++) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

async function testStickyStateRefreshesAfterReusingNotificationId() {
  const wv = makeDomWebview();
  try {
    send(wv.win, {
      type: 'notification',
      id: 'review-sticky-toggle',
      severity: 'info',
      message: 'Originally sticky',
      sticky: true,
    });
    const toast = await waitFor(
      () => wv.win.document.querySelector('.kiss-notification'),
      'first notification must render',
    );
    assert.strictEqual(
      toast.getAttribute('data-notification-sticky'),
      'true',
      'first render must expose sticky=true',
    );

    send(wv.win, {
      type: 'notification',
      id: 'review-sticky-toggle',
      severity: 'info',
      message: 'Now transient',
      sticky: false,
    });
    assert.strictEqual(
      toast.getAttribute('data-notification-sticky'),
      'false',
      'reused notification id must expose sticky=false after refresh',
    );
    assert.ok(
      timerDelays(wv).includes(10000),
      'transient info notification must schedule a 10s auto-dismiss timer',
    );

    toast.dispatchEvent(new wv.win.MouseEvent('mouseenter', {bubbles: true}));
    assert.ok(
      !timerDelays(wv).includes(10000),
      'hovering the toast must clear the transient auto-dismiss timer',
    );

    toast.dispatchEvent(new wv.win.MouseEvent('mouseleave', {bubbles: true}));
    const dismissTimer = latestTimer(wv, 10000);
    assert.ok(
      dismissTimer,
      'leaving a reused now-transient toast must reschedule auto-dismiss using current sticky state',
    );
    dismissTimer.callback();
    assert.ok(
      !wv.win.document.querySelector('.kiss-notification'),
      'running the rescheduled auto-dismiss timer must remove the toast',
    );
  } finally {
    wv.close();
  }
}

async function testUpdateToastCloseDoesNotPostNotificationAction() {
  const wv = makeDomWebview();
  try {
    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });
    const closeButton = await waitFor(
      () => wv.win.document.querySelector('.kiss-notification-close'),
      'update toast close button must render',
    );

    click(closeButton);
    assert.ok(
      !wv.win.document.querySelector('.kiss-notification'),
      'X button must dismiss the update toast locally',
    );
    assert.deepStrictEqual(
      wv.posted.filter(m => m.type === 'notificationAction'),
      [],
      'dismissing the in-webview update toast must not post notificationAction to the extension',
    );
  } finally {
    wv.close();
  }
}

async function testUpdateRebroadcastRefreshesMessageAndAriaLabel() {
  const wv = makeDomWebview();
  try {
    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });
    const toast = await waitFor(
      () => wv.win.document.querySelector('.kiss-notification'),
      'initial update toast must render',
    );
    const firstButton = toast.querySelector('.kiss-notification-action');
    assert.ok(firstButton, 'initial update toast must include an action');
    assert.ok(
      toast.textContent.includes('9999.0.0'),
      'initial toast message must include the first latest version',
    );
    assert.strictEqual(
      firstButton.getAttribute('aria-label'),
      'Update KISS Sorcar to 9999.0.0',
      'initial action aria-label must include the first latest version',
    );

    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.1.0',
      current: '2026.6.29',
    });
    const updatedToasts = wv.win.document.querySelectorAll(
      '.kiss-notification',
    );
    assert.strictEqual(
      updatedToasts.length,
      1,
      'rebroadcasting an update must reuse the existing toast id, not stack duplicates',
    );
    const updatedButton = updatedToasts[0].querySelector(
      '.kiss-notification-action',
    );
    assert.ok(
      updatedToasts[0].textContent.includes('9999.1.0'),
      'rebroadcast must refresh the visible latest version in the toast message',
    );
    assert.strictEqual(
      updatedButton.getAttribute('aria-label'),
      'Update KISS Sorcar to 9999.1.0',
      'rebroadcast must refresh the update action aria-label',
    );
  } finally {
    wv.close();
  }
}

async function runTests() {
  await testStickyStateRefreshesAfterReusingNotificationId();
  await testUpdateToastCloseDoesNotPostNotificationAction();
  await testUpdateRebroadcastRefreshesMessageAndAriaLabel();
}

runTests().then(
  () => {
    console.log('\nAll review-driven update notification tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
