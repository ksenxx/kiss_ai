// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// I-R2: one hidden-textarea execCommand('copy') fallback, exported as
// PanelCopy.fallbackCopyText, backs every same-page copy control. This
// exercises the helper directly and through main.js's task-panel copy
// button when the async clipboard API is unavailable.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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

async function testMainTaskPanelUsesSharedFallback() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  const calls = [];
  win.document.execCommand = cmd => {
    // The fallback appends its hidden textarea directly to <body>.
    const area = win.document.querySelector('body > textarea');
    calls.push({cmd, value: area ? area.value : null});
    return true;
  };

  // No navigator.clipboard in this DOM: the button must fall back.
  const text = win.document.getElementById('task-panel-text');
  const btn = win.document.getElementById('task-panel-copy');
  assert.ok(text && btn, 'task panel copy controls missing');
  text.textContent = 'the task text';
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.deepStrictEqual(
    calls,
    [{cmd: 'copy', value: 'the task text'}],
    'task-panel copy must use the shared execCommand fallback',
  );
  console.log('  ok - main.js task-panel copy uses the shared fallback');
}

async function main() {
  testHelperCopiesAndCleansUp();
  testHelperReportsFailure();
  await testMainTaskPanelUsesSharedFallback();
  console.log('rr_area_hi_copy_fallback: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
