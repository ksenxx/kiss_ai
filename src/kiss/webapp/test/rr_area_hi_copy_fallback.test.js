// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// I-R2: one hidden-textarea execCommand('copy') fallback, exported as
// PanelCopy.fallbackCopyText, backs every same-page copy control. This
// exercises the helper directly and through the copy button of the
// task panel that opens a transcript when the async clipboard API is
// unavailable.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {makePage, send, MEDIA} = require('./simplify2_harness');

function newDom() {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'outside-only',
    url: 'https://localhost/',
  });
  dom.window.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  return dom.window;
}

function testHelperCopiesAndCleansUp() {
  const win = newDom();
  const calls = [];
  win.document.execCommand = cmd => {
    const area = win.document.querySelector('textarea');
    calls.push({cmd, value: area ? area.value : null});
    return true;
  };
  const before = win.document.body.childElementCount;
  const ok = win.PanelCopy.fallbackCopyText('hello clipboard');
  assert.strictEqual(ok, true, 'helper must report execCommand success');
  assert.deepStrictEqual(calls, [{cmd: 'copy', value: 'hello clipboard'}]);
  assert.strictEqual(
    win.document.body.childElementCount,
    before,
    'hidden textarea must be removed again',
  );
  console.log('  ok - fallbackCopyText copies via execCommand and cleans up');
}

function testHelperReportsFailure() {
  const win = newDom();
  win.document.execCommand = () => false;
  assert.strictEqual(win.PanelCopy.fallbackCopyText('x'), false);
  win.document.execCommand = () => {
    throw new Error('denied');
  };
  const before = win.document.body.childElementCount;
  assert.strictEqual(win.PanelCopy.fallbackCopyText('x'), false);
  assert.strictEqual(
    win.document.body.childElementCount,
    before,
    'textarea must be removed even when execCommand throws',
  );
  console.log('  ok - fallbackCopyText fails closed and still cleans up');
}

function testMainTaskPanelUsesSharedFallback() {
  const {win} = makePage();

  const calls = [];
  win.document.execCommand = cmd => {
    // The fallback appends its hidden textarea directly to <body>.
    const area = win.document.querySelector('body > textarea');
    calls.push({cmd, value: area ? area.value : null});
    return true;
  };

  // A task starts: its transcript opens with the task panel, a regular
  // event panel whose copy button copies the task text.
  send(win, {type: 'setTaskText', text: 'the task text'});
  send(win, {type: 'clear'});
  // No navigator.clipboard in this DOM: the button must fall back.
  const panel = win.document.querySelector('#output .task-panel');
  const btn = panel && panel.querySelector(':scope > .panel-copy-btn');
  assert.ok(panel && btn, 'task panel copy controls missing');
  assert.strictEqual(
    panel.querySelector('.task-panel-text').textContent,
    'the task text',
  );
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.deepStrictEqual(
    calls,
    [{cmd: 'copy', value: 'the task text'}],
    'task-panel copy must use the shared execCommand fallback',
  );
  win.close();
  console.log('  ok - main.js task-panel copy uses the shared fallback');
}

testHelperCopiesAndCleansUp();
testHelperReportsFailure();
testMainTaskPanelUsesSharedFallback();
console.log('rr_area_hi_copy_fallback: all tests passed');
process.exit(0);
