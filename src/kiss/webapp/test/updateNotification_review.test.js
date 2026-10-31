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

// A repaint of a transient toast (same id) while the pointer is on it
// must not re-arm the auto-dismiss that the hover cleared; the timer
// comes back when the pointer leaves.
async function testRepaintUnderPointerKeepsDismissCleared() {
  const wv = makeDomPage();
  try {
    const ev = {
      type: 'notification',
      id: 'review-hover-repaint',
      severity: 'info',
      message: 'first',
      sticky: false,
    };
    send(wv.win, ev);
    const toast = await waitFor(
      () => wv.win.document.querySelector('.kiss-notification'),
      'notification must render',
    );
    toast.dispatchEvent(new wv.win.MouseEvent('mouseenter', {bubbles: true}));
    assert.ok(!timerDelays(wv).includes(5000), 'hover clears the timer');

    send(wv.win, Object.assign({}, ev, {message: 'second'}));
    assert.strictEqual(
      wv.win.document.querySelectorAll('.kiss-notification').length,
      1,
      'the repaint reuses the toast',
    );
    assert.ok(
      !timerDelays(wv).includes(5000),
      'a repaint under the pointer must not re-arm the auto-dismiss',
    );

    toast.dispatchEvent(new wv.win.MouseEvent('mouseleave', {bubbles: true}));
    assert.ok(
      latestTimer(wv, 5000),
      'leaving the toast arms the auto-dismiss again',
    );
  } finally {
    wv.close();
  }
}

// A transient toast armed its 5 s auto-dismiss; the same toast is then
// repainted as a sticky error.  That repaint must cancel the transient
// timer: the error stays.  The timers are recorded, so the cancel is
// checked directly instead of sleeping past the 5 s dismiss.
async function testStickyErrorRepaintCancelsTransientDismiss() {
  const wv = makeDomPage();
  const {win} = wv;
  try {
    send(win, {
      type: 'notification',
      id: 'review-sticky',
      severity: 'info',
      message: 'initial',
      sticky: false,
    });
    const toast = await waitFor(
      () => win.document.querySelector('.kiss-notification'),
      'notification must render',
    );
    const transientDismiss = latestTimer(wv, 5000);
    assert.ok(transientDismiss, 'the transient toast armed its 5 s dismiss');

    send(win, {
      type: 'notification',
      id: 'review-sticky',
      severity: 'error',
      message: 'something broke',
      sticky: true,
    });
    assert.strictEqual(
      win.document.querySelectorAll('.kiss-notification').length,
      1,
      'the repaint reuses the toast',
    );
    assert.ok(
      transientDismiss.cleared,
      'the sticky repaint cancels the transient dismiss',
    );
    assert.ok(
      !timerDelays(wv).includes(5000),
      'the sticky repaint arms no new auto-dismiss',
    );
    assert.ok(toast.isConnected, 'the sticky error stays on screen');
    assert.strictEqual(
      toast.getAttribute('data-notification-sticky'),
      'true',
      'the repaint shows the sticky error',
    );
  } finally {
    wv.close();
  }
}

async function runTests() {
  await testStickyStateRefreshesAfterReusingNotificationId();
  await testRepaintUnderPointerKeepsDismissCleared();
  await testStickyErrorRepaintCancelsTransientDismiss();
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
