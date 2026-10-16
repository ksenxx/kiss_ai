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
const {makePage, send} = require('./simplify2_harness');

function clickSend(win) {
  win.document
    .getElementById('send-btn')
    .dispatchEvent(
      new win.MouseEvent('click', {bubbles: true, cancelable: true}),
    );
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

async function testFreshSubmitClearsComposer() {
  const {win, posted} = makePage();
  const inp = win.document.getElementById('task-input');
  inp.value = 'do the thing';
  clickSend(win);
  const submit = posted.find(m => m && m.type === 'submit');
  assert.ok(submit, 'a fresh send posts a submit message');
  assert.strictEqual(submit.prompt, 'do the thing');
  assertComposerCleared(win, 'fresh submit');
  win.close();
  console.log('  ok - a fresh submit clears the composer');
}

async function testAppendToRunningTaskClearsComposer() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId});
  const inp = win.document.getElementById('task-input');
  inp.value = 'also do this';
  clickSend(win);
  const append = posted.find(m => m && m.type === 'appendUserMessage');
  assert.ok(append, 'a send during a running task posts appendUserMessage');
  assert.strictEqual(append.prompt, 'also do this');
  assert.ok(
    !posted.some(m => m && m.type === 'submit'),
    'no second task is submitted while one is running',
  );
  assertComposerCleared(win, 'append to running task');
  win.close();
  console.log('  ok - appending to a running task clears the composer');
}

async function main() {
  await testFreshSubmitClearsComposer();
  await testAppendToRunningTaskClearsComposer();
  console.log('audit0903_composer_reset.test.js: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
