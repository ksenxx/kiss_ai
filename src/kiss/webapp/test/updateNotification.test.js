// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The sticky update notification's markup (sticky flag, SVG icon, aria
// label) and its lifecycle across `available: false` / re-broadcast.  The
// badge icon is pinned by simplify2_update_badge.test.js and the Update
// click posting `runUpdate` by updateWhenIdleToast.test.js.  main.js
// renders the toast synchronously inside its message handler, so every
// check is direct.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function runTests() {
  const wv = makePage();
  try {
    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });

    const toast = wv.win.document.querySelector('.kiss-notification');
    assert.ok(
      toast,
      'update_available with available:true must render a permanent page notification',
    );

    assert.strictEqual(
      toast.getAttribute('data-notification-sticky'),
      'true',
      'update notification must be marked sticky on the DOM element',
    );

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
    assert.strictEqual(
      svg.namespaceURI,
      'http://www.w3.org/2000/svg',
      'action-button SVG must be in the SVG namespace',
    );
    const ariaLabel = (
      actionButton.getAttribute('aria-label') ||
      actionButton.textContent ||
      ''
    ).toLowerCase();
    assert.ok(
      ariaLabel.includes('update'),
      'action button must advertise itself as an update action',
    );

    send(wv.win, {
      type: 'update_available',
      available: false,
      latest: '',
      current: '',
    });
    assert.ok(
      !wv.win.document.querySelector('.kiss-notification'),
      'available:false broadcast must dismiss the permanent update notification',
    );

    send(wv.win, {
      type: 'update_available',
      available: true,
      latest: '9999.0.0',
      current: '2026.6.29',
    });
    assert.ok(
      wv.win.document.querySelector('.kiss-notification'),
      'second broadcast must re-create the permanent notification',
    );
  } finally {
    wv.win.close();
  }
}

try {
  runTests();
  console.log('\nAll tests passed');
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
