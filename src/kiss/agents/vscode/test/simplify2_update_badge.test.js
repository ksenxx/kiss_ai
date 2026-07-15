// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Regression tests for the 'update_available' event rendering in
// media/main.js: the settings-panel Update-button badge
// (renderUpdateAvailableBadge) and the sticky "update available"
// notification (renderUpdateAvailableNotification).  Locks in the
// exact DOM so the badge's hand-built createElementNS SVG block can be
// replaced by a string constant without any visible change.
//
//     node src/kiss/agents/vscode/test/simplify2_update_badge.test.js
'use strict';

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness');

function run() {
  const {win, posted} = makeWebview();
  const doc = win.document;
  const btn = doc.getElementById('cfg-update-btn');
  assert.ok(btn, 'settings panel has the #cfg-update-btn button');

  // --- Update available: badge icon + tooltip + notification. ---
  send(win, {
    type: 'update_available',
    available: true,
    latest: '9.9.9',
    current: '1.0.0',
  });
  assert.ok(btn.classList.contains('has-update'), 'button gains has-update');
  assert.ok(
    (btn.getAttribute('title') || '').includes('9.9.9'),
    'tooltip mentions the latest version',
  );
  let icons = btn.querySelectorAll('.update-available-icon');
  assert.strictEqual(icons.length, 1, 'exactly one badge icon injected');
  const icon = icons[0];
  assert.strictEqual(icon.tagName.toLowerCase(), 'svg', 'icon is an <svg>');
  assert.strictEqual(icon.getAttribute('width'), '12');
  assert.strictEqual(icon.getAttribute('height'), '12');
  assert.strictEqual(icon.getAttribute('stroke'), 'currentColor');
  assert.ok(icon.querySelector('polyline'), 'icon has the arrow polyline');
  assert.strictEqual(btn.firstChild, icon, 'icon injected before the label');

  const toast = doc.querySelector(
    '[data-notification-id="kiss-update-available"]',
  );
  assert.ok(toast, 'sticky update notification appears');
  assert.ok(
    toast.textContent.includes('9.9.9') && toast.textContent.includes('1.0.0'),
    'notification mentions latest and current versions',
  );
  const action = toast.querySelector('.kiss-notification-action');
  assert.ok(action, 'notification has an Update action button');
  action.click();
  assert.ok(
    posted.some(m => m.type === 'runUpdate'),
    'clicking Update posts {type: runUpdate}',
  );

  // --- Re-broadcast must not stack badges or toasts. ---
  send(win, {
    type: 'update_available',
    available: true,
    latest: '9.9.9',
    current: '1.0.0',
  });
  assert.strictEqual(
    btn.querySelectorAll('.update-available-icon').length,
    1,
    're-broadcast keeps exactly one badge icon',
  );
  assert.strictEqual(
    doc.querySelectorAll('[data-notification-id="kiss-update-available"]')
      .length,
    1,
    're-broadcast keeps exactly one notification',
  );

  // --- No longer available: badge and notification are removed. ---
  send(win, {type: 'update_available', available: false});
  assert.ok(!btn.classList.contains('has-update'), 'has-update removed');
  assert.ok(!btn.getAttribute('title'), 'tooltip removed');
  assert.strictEqual(
    btn.querySelectorAll('.update-available-icon').length,
    0,
    'badge icon removed',
  );

  win.close();
  console.log('  ok - update badge and notification render identically');
}

run();
console.log('simplify2_update_badge.test.js: all tests passed');
