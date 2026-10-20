// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, waitFor} = require('./simplify2_harness');

function makeDomPage() {
  // Timers are recorded, never armed: a test fires the one it wants.
  const scheduledTimeouts = [];
  const {win, posted} = makePage({
    beforeScripts(win) {
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
    },
  });
  return {
    win,
    posted,
    scheduledTimeouts,
    close: () => win.close(),
  };
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

async function testStickyStateRefreshesAfterReusingNotificationId() {
  const wv = makeDomPage();
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
      timerDelays(wv).includes(5000),
      'transient info notification must schedule a 5s auto-dismiss timer',
    );

    toast.dispatchEvent(new wv.win.MouseEvent('mouseenter', {bubbles: true}));
    assert.ok(
      !timerDelays(wv).includes(5000),
      'hovering the toast must clear the transient auto-dismiss timer',
    );

    toast.dispatchEvent(new wv.win.MouseEvent('mouseleave', {bubbles: true}));
    const dismissTimer = latestTimer(wv, 5000);
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
  const wv = makeDomPage();
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
      'dismissing the update toast must not post notificationAction (a command the daemon never had)',
    );
  } finally {
    wv.close();
  }
}

async function testUpdateRebroadcastRefreshesMessageAndAriaLabel() {
  const wv = makeDomPage();
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
    const updatedToasts =
      wv.win.document.querySelectorAll('.kiss-notification');
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
