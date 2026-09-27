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

const MEDIA = path.join(__dirname, '..', 'media');

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function makePage(writeTextImpl) {
  const dom = new JSDOM('<!DOCTYPE html><body><div id="panel"></div></body>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
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

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  not ok - ${name}`);
  }
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
    // The failure state reverts like the success flash does.
    await sleep(1600);
    assert.ok(!btn.classList.contains('copy-failed'));
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
  console.log(`\n${passed} passed, ${failures.length} failed`);
  for (const f of failures) {
    console.error(`\n${f.name}\n${f.err && f.err.stack}`);
  }
  if (failures.length) process.exit(1);
});
