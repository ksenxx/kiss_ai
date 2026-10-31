// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A promptlet Add waits for the daemon's `tricksData` confirmation (or
// its `error` rejection).  An outage swallows that reply, and before
// the fix the wait stayed armed across the reconnect: the NEXT
// unaddressed `error` of the session — an update failure, an unknown
// command, anything — was shown under the Add box as the promptlet's
// rejection.  The disconnect now drops the wait like every other
// swallowed request (see the `daemonStatus` case).
//
// The real media/chat.html + panelCopy.js + api.js + main.js are booted
// in jsdom; nothing under test is mocked.

/* global require, console, process */
'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function addErrorText(doc) {
  const el = doc.getElementById('tricks-add-error');
  return el && !el.hidden ? el.textContent : '';
}

function submitAdd(win, text) {
  const addInput = win.document.getElementById('tricks-add-input');
  addInput.value = text;
  addInput.dispatchEvent(new win.Event('input', {bubbles: true}));
  win.document.getElementById('tricks-add-btn').click();
}

function testOutageDropsThePendingAdd() {
  const {win, posted} = makePage();
  const doc = win.document;
  win.__TRICKS__ = ['Alpha promptlet'];
  doc.getElementById('tricks-btn').click();

  submitAdd(win, 'Delta promptlet');
  assert.strictEqual(
    posted.filter(m => m.type === 'addTrick').length,
    1,
    'the Add click posts addTrick',
  );

  // The reply never comes: the socket drops and comes back.
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    doc.getElementById('tricks-add-input').value,
    'Delta promptlet',
    'the typed promptlet is kept for the user to resend',
  );

  // An unrelated window-level error after the reconnect.
  send(win, {type: 'error', text: 'Update failed: no network'});
  assert.strictEqual(
    addErrorText(doc),
    '',
    'BUG - an unrelated error after the outage was shown under the ' +
      'promptlet Add box as the rejection of the swallowed add',
  );
  assert.ok(
    doc.body.textContent.includes('Update failed: no network'),
    'the error still reaches the transcript banner',
  );

  // Resending works as a fresh add: its own rejection shows under the box.
  submitAdd(win, 'Delta promptlet');
  send(win, {type: 'error', text: 'Promptlet already exists'});
  assert.strictEqual(addErrorText(doc), 'Promptlet already exists');

  win.close();
  console.log('  ok - an outage drops the pending promptlet add');
}

function testRejectionStillShowsWithoutOutage() {
  // Control: the daemon's own rejection of a pending add is still
  // shown under the box when no outage intervened.
  const {win} = makePage();
  const doc = win.document;
  win.__TRICKS__ = [];
  doc.getElementById('tricks-btn').click();
  submitAdd(win, 'Echo promptlet');
  const duplicate = 'That promptlet is already in MY_INJECTION.md';
  send(win, {type: 'error', text: duplicate});
  assert.strictEqual(addErrorText(doc), duplicate);
  win.close();
  console.log('  ok - a rejection without an outage shows under the box');
}

try {
  testOutageDropsThePendingAdd();
  testRejectionStillShowsWithoutOutage();
} catch (e) {
  console.error(e && e.stack ? e.stack : e);
  process.exit(1);
}
