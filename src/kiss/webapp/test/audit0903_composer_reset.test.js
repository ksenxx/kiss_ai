// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the composer reset after a send in
// media/main.js.
//
// Redundancy: sendMessage() carried two identical 9-line copies of
// the "clear the composer" sequence — one in the append-to-running-
// task branch, one after a fresh submit.  Two copies of the same
// sequence drift; both paths now share resetComposerAfterSend().
// These tests lock the behavior of BOTH paths so the extraction (and
// any future edit to either path) keeps them identical.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  click,
  byId,
  typeInto,
} = require('./simplify2_harness');
const {key} = require('./ui_antipattern_harness');
const {paste, settle, typeAndSend} = require('./attachment_helpers');

function clickSend(win) {
  click(win, byId(win, 'send-btn'));
}

function assertComposerCleared(win, label) {
  const inp = win.document.getElementById('task-input');
  assert.strictEqual(inp.value, '', label + ': the textarea is emptied');
  assert.strictEqual(
    win.document.getElementById('file-chips').children.length,
    0,
    label + ': the attachment chips are gone',
  );
  assert.strictEqual(
    win.document.getElementById('input-clear-btn').style.display,
    'none',
    label + ': the clear button is hidden',
  );
}

/**
 * Fill the composer the way a user does -- attach a PDF and type the
 * prompt (an `input` event, which shows the clear button) -- and check
 * that every part assertComposerCleared reads left its default state.
 */
async function fillComposer(win, text) {
  const chips = win.document.getElementById('file-chips');
  paste(
    win,
    new win.File(['%PDF-1.4'], 'notes.pdf', {type: 'application/pdf'}),
  );
  await settle(() => chips.querySelector('.file-chip:not(.pending)'));
  typeInto(win, byId(win, 'task-input'), text);
  assert.strictEqual(
    chips.children.length,
    1,
    'precondition: one attachment chip',
  );
  assert.strictEqual(
    win.document.getElementById('input-clear-btn').style.display,
    '',
    'precondition: typing shows the clear button',
  );
}

async function testFreshSubmitClearsComposer() {
  const {win, posted} = makePage();
  await fillComposer(win, 'do the thing');
  clickSend(win);
  const submit = posted.find(m => m && m.type === 'submit');
  assert.ok(submit, 'a fresh send posts a submit message');
  assert.strictEqual(submit.prompt, 'do the thing');
  assert.strictEqual(submit.attachments.length, 1, 'the PDF rides along');
  assertComposerCleared(win, 'fresh submit');
  win.close();
  console.log('  ok - a fresh submit clears the composer');
}

async function testAppendToRunningTaskClearsComposer() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId});
  await fillComposer(win, 'also do this');
  clickSend(win);
  const append = posted.find(m => m && m.type === 'appendUserMessage');
  assert.ok(append, 'a send during a running task posts appendUserMessage');
  assert.strictEqual(append.prompt, 'also do this');
  assert.strictEqual(append.attachments.length, 1, 'the PDF rides along');
  assert.ok(
    !posted.some(m => m && m.type === 'submit'),
    'no second task is submitted while one is running',
  );
  assertComposerCleared(win, 'append to running task');
  win.close();
  console.log('  ok - appending to a running task clears the composer');
}

// The composer's x button is a third copy of the same sequence: it must
// reset the prompt-history cursor like a send does, or ArrowDown after
// clearing refills the composer with the next history entry.
function testClearButtonResetsLikeASend() {
  const {win} = makePage();
  const inp = win.document.getElementById('task-input');
  typeAndSend(win, 'first');
  typeAndSend(win, 'second');
  key(win, inp, 'ArrowUp');
  key(win, inp, 'ArrowUp');
  assert.strictEqual(inp.value, 'first', 'ArrowUp walks the history');
  click(win, byId(win, 'input-clear-btn'));
  assertComposerCleared(win, 'clear button');
  key(win, inp, 'ArrowDown');
  assert.strictEqual(inp.value, '', 'ArrowDown after clearing is inert');
  win.close();
  console.log('  ok - the clear button resets the composer like a send');
}

async function main() {
  await testFreshSubmitClearsComposer();
  await testAppendToRunningTaskClearsComposer();
  testClearButtonResetsLikeASend();
  console.log('audit0903_composer_reset.test.js: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
