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
const {
  makePage,
  send,
  byId,
  typeInto,
  ofType,
} = require('./simplify2_harness');

function addErrorText(doc) {
  const el = doc.getElementById('tricks-add-error');
  return el && !el.hidden ? el.textContent : '';
}

function submitAdd(win, text) {
  typeInto(win, byId(win, 'tricks-add-input'), text);
  win.document.getElementById('tricks-add-btn').click();
}

function dropAndReconnect(win) {
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  send(win, {type: 'daemonStatus', connected: true});
}

function testOutageDropsThePendingAdd() {
  const {win, posted} = makePage();
  const doc = win.document;
  win.__TRICKS__ = ['Alpha promptlet'];
  doc.getElementById('tricks-btn').click();

  submitAdd(win, 'Delta promptlet');
  assert.strictEqual(
    ofType(posted, 'addTrick').length,
    1,
    'the Add click posts addTrick',
  );

  // The reply never comes: the socket drops and comes back.
  dropAndReconnect(win);
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

function editFirstRow(win, text) {
  const doc = win.document;
  doc.querySelector('#tricks-list .sidebar-item-edit').click();
  doc.querySelector('.tricks-edit-input').value = text;
  doc.querySelector('.tricks-edit-save').click();
}

function openPromptlets(tricks) {
  const page = makePage();
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {type: 'tricksData', tricks, userCount: tricks.length});
  page.win.document.getElementById('tricks-btn').click();
  return page;
}

// Audit 2026-11-11 (W1): an edit's new text lives nowhere else once
// its editor is closed.  The disconnect dropped the pending edits like
// the add's wait, and their texts with them: a change queued behind an
// unanswered edit was never sent, and an edit the daemon had not
// applied was gone from the page.
function testOutageKeepsThePendingEdits() {
  // The first edit reached the disk: the one queued behind it goes out.
  let {win, posted} = openPromptlets(['Original A']);
  editFirstRow(win, 'First B');
  editFirstRow(win, 'Latest C');
  const edits = () => ofType(posted, 'editTrick');
  assert.strictEqual(edits().length, 1, 'setup: the second edit waits for the first answer');
  dropAndReconnect(win);
  send(win, {type: 'tricksData', tricks: ['First B'], userCount: 1});
  assert.deepStrictEqual(
    edits()
      .slice(1)
      .map(m => m.text + ' -> ' + m.newText),
    ['First B -> Latest C'],
    'BUG - the edit queued behind the outage was never sent',
  );
  assert.strictEqual(
    win.document.querySelector('#tricks-list .sidebar-item-text').textContent,
    'Latest C',
  );
  win.close();

  // The edit never reached the disk: its text comes back as a draft,
  // and an unrelated error after the reconnect is not its rejection.
  ({win, posted} = openPromptlets(['Original A']));
  editFirstRow(win, 'First B');
  dropAndReconnect(win);
  send(win, {type: 'error', text: 'Update failed: no network'});
  send(win, {type: 'tricksData', tricks: ['Original A'], userCount: 1});
  const editor = win.document.querySelector('.tricks-edit-input');
  assert.ok(editor, 'BUG - the unsaved edit was dropped with the connection');
  assert.strictEqual(editor.value, 'First B', 'the draft is back to be saved again');
  const shown = win.document.getElementById('tricks-list').textContent;
  assert.ok(shown.includes('connection dropped'), 'with the reason: ' + shown);
  assert.ok(!shown.includes('Update failed'), 'an unrelated error is not its rejection');
  assert.strictEqual(edits().length, 1, 'nothing is sent again on its own');
  win.close();
  console.log('  ok - an outage keeps the pending promptlet edits');
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

function testAnotherCommandsRefusalIsNotTheAdds() {
  // The daemon stamps a refused mutation with the command it answers
  // (commands.py _send_error_to_sender): a custom-model save refused
  // while the add awaits its answer is not the add's rejection, and
  // the add goes on waiting for its own.
  const {win} = makePage();
  const doc = win.document;
  win.__TRICKS__ = [];
  doc.getElementById('tricks-btn').click();
  submitAdd(win, 'Echo promptlet');
  send(win, {
    type: 'error',
    command: 'saveMyModel',
    text: "A custom model named 'x' already exists",
  });
  assert.strictEqual(addErrorText(doc), '', "another command's refusal is not shown under the box");
  const duplicate = 'That promptlet is already in MY_INJECTION.md';
  send(win, {type: 'error', command: 'addTrick', text: duplicate});
  assert.strictEqual(addErrorText(doc), duplicate, 'the add is still awaiting its own');
  win.close();
  console.log("  ok - another command's refusal leaves the pending add waiting");
}

try {
  testOutageDropsThePendingAdd();
  testOutageKeepsThePendingEdits();
  testRejectionStillShowsWithoutOutage();
  testAnotherCommandsRefusalIsNotTheAdds();
} catch (e) {
  console.error(e && e.stack ? e.stack : e);
  process.exit(1);
}
