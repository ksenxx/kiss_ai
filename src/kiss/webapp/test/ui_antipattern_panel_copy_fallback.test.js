// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// UI anti-pattern fix A9 for the per-panel Copy button
// (media/panelCopy.js): a rejected navigator.clipboard.writeText no longer
// fails silently.  It falls back to the hidden-textarea execCommand('copy')
// path, and when that fails too the button shows a visible "Copy failed"
// state instead of swallowing the error.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {sleep, waitFor} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {test, report} = makeRunner();

const MEDIA = path.join(__dirname, '..', 'media');

// Every page built; closed at the end so the buttons' 1.5 s flash timers
// do not keep node alive after the last assertion.
const windows = [];

function makePage(writeTextImpl) {
  const dom = new JSDOM('<!DOCTYPE html><body><div id="panel"></div></body>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  windows.push(win);
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {writeText: writeTextImpl},
  });
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  const panel = win.document.getElementById('panel');
  panel.textContent = 'panel body text';
  win.PanelCopy.addCopyButton(panel);
  const btn = panel.querySelector('.panel-copy-btn');
  assert.ok(btn, 'addCopyButton must append a copy button');
  return {win, btn};
}

function click(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

function isCheck(btn) {
  return !!btn.querySelector('svg polyline');
}

function isCross(btn) {
  return btn.querySelectorAll('svg line').length === 2;
}


async function main() {
  await test('a rejected writeText falls back to execCommand and reports success', async () => {
    const {win, btn} = makePage(() =>
      Promise.reject(new Error('Write permission denied')),
    );
    const copied = [];
    win.document.execCommand = cmd => {
      if (cmd !== 'copy') return false;
      const ta = win.document.querySelector('textarea');
      copied.push(ta ? ta.value : null);
      return true;
    };
    click(win, btn);
    await sleep(5);
    assert.deepStrictEqual(
      copied,
      ['panel body text'],
      'the fallback copied the panel text',
    );
    assert.ok(isCheck(btn), 'the check mark confirms the copy');
    assert.ok(btn.classList.contains('copied'));
    assert.ok(!btn.classList.contains('copy-failed'));
  });

  await test('when both clipboard paths fail the button shows a visible failure state', async () => {
    const {win, btn} = makePage(() =>
      Promise.reject(new Error('Write permission denied')),
    );
    win.document.execCommand = () => false;
    click(win, btn);
    await sleep(5);
    assert.ok(btn.classList.contains('copy-failed'), btn.className);
    assert.ok(!btn.classList.contains('copied'));
    assert.ok(isCross(btn), 'a cross replaces the copy icon');
    assert.ok(/^Copy failed/.test(btn.title), btn.title);
    assert.ok(
      /select the text and copy it with the keyboard/i.test(btn.title),
      'the title tells the user what to do instead',
    );
    assert.ok(
      !win.document.querySelector('textarea'),
      'the fallback textarea is cleaned up',
    );
    // The failure state reverts like the success flash does (panelCopy.js
    // arms the same 1.5 s timer for both).
    await waitFor(
      () => !btn.classList.contains('copy-failed'),
      'the failure state never reverted',
    );
    assert.strictEqual(btn.title, 'Copy panel text');
    assert.ok(!isCross(btn) && !isCheck(btn), 'the copy icon is back');
  });

  await test('a synchronous execCommand failure with no async clipboard is reported too', async () => {
    const {win, btn} = makePage(undefined);
    Object.defineProperty(win.navigator, 'clipboard', {
      configurable: true,
      value: undefined,
    });
    win.document.execCommand = () => {
      throw new Error('not allowed');
    };
    click(win, btn);
    await sleep(5);
    assert.ok(btn.classList.contains('copy-failed'), btn.className);
    assert.ok(isCross(btn));
  });

  await test('a successful writeText still shows the check mark and a "Copied" title', async () => {
    const {win, btn} = makePage(() => Promise.resolve());
    click(win, btn);
    await sleep(5);
    assert.ok(isCheck(btn));
    assert.strictEqual(btn.title, 'Copied');
    assert.ok(!btn.classList.contains('copy-failed'));
  });
}

main().then(() => {
  windows.forEach(win => win.close());
  report('ui_antipattern_panel_copy_fallback');
});
