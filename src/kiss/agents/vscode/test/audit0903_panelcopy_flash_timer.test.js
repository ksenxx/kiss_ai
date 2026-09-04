// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) test for the copy-feedback flash of the panel
// copy button in media/panelCopy.js.
//
// Race: addCopyButton armed a bare 1500 ms setTimeout to revert the
// check-mark back to the copy icon, and never cleared it.  Click the
// button again while a flash is showing and the FIRST click's stale
// timer fires mid-flash, cutting the second flash short (the icon
// reverts after ~300 ms instead of 1500 ms).  The fix tracks the
// timer per button and restarts it on every copy.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function makePage() {
  const dom = new JSDOM('<!DOCTYPE html><body><div id="panel"></div></body>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  const clipboardWrites = [];
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: text => {
        clipboardWrites.push(String(text));
        return Promise.resolve();
      },
    },
  });
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8') +
      '\n//# sourceURL=audit0903-panelcopy.js',
  );
  return {win, clipboardWrites};
}

function isCheck(btn) {
  return !!btn.querySelector('svg polyline');
}

function click(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

async function testStaleTimerDoesNotEndSecondFlash() {
  const {win, clipboardWrites} = makePage();
  const panel = win.document.getElementById('panel');
  panel.textContent = 'panel body text';
  win.PanelCopy.addCopyButton(panel);
  const btn = panel.querySelector('.panel-copy-btn');
  assert.ok(btn, 'addCopyButton must append a copy button');

  click(win, btn);
  await sleep(0);
  assert.deepStrictEqual(clipboardWrites, ['panel body text']);
  assert.ok(isCheck(btn), 'first click must show the check icon');
  assert.ok(btn.classList.contains('copied'));

  // Second click while the first flash is still showing.
  await sleep(1200);
  click(win, btn);
  await sleep(0);
  assert.ok(isCheck(btn), 'second click must show the check icon');

  // 1500 ms after the FIRST click, its stale timer would fire here.
  await sleep(500);
  assert.ok(
    isCheck(btn) && btn.classList.contains('copied'),
    "the first click's stale timer must not end the second flash early",
  );

  // The second flash still reverts on its own.
  await sleep(1200);
  assert.ok(
    !isCheck(btn) && !btn.classList.contains('copied'),
    'the flash must still revert after the second timer',
  );
  win.close();
  console.log('  ok - a stale timer does not end the second flash');
}

async function testSecondPanelFlashesIndependently() {
  const {win} = makePage();
  const doc = win.document;
  const p2 = doc.createElement('div');
  p2.id = 'panel2';
  p2.textContent = 'other panel';
  doc.body.appendChild(p2);
  const p1 = doc.getElementById('panel');
  p1.textContent = 'first panel';
  win.PanelCopy.addCopyButton(p1);
  win.PanelCopy.addCopyButton(p2);
  const b1 = p1.querySelector('.panel-copy-btn');
  const b2 = p2.querySelector('.panel-copy-btn');

  click(win, b1);
  await sleep(0);
  click(win, b2);
  await sleep(0);
  assert.ok(isCheck(b1) && isCheck(b2), 'both buttons flash');
  await sleep(1700);
  assert.ok(
    !isCheck(b1) && !isCheck(b2),
    'both flashes revert on their own timers',
  );
  win.close();
  console.log('  ok - each panel button keeps its own timer');
}

async function testFallbackCopyStillFlashes() {
  const {win} = makePage();
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: undefined,
  });
  const execCalls = [];
  win.document.execCommand = function (cmd) {
    execCalls.push(cmd);
    return true;
  };
  const panel = win.document.getElementById('panel');
  panel.textContent = 'fallback text';
  win.PanelCopy.addCopyButton(panel);
  const btn = panel.querySelector('.panel-copy-btn');
  click(win, btn);
  await sleep(0);
  assert.deepStrictEqual(execCalls, ['copy']);
  assert.ok(isCheck(btn), 'the fallback path must flash too');
  win.close();
  console.log('  ok - the execCommand fallback flashes');
}

async function main() {
  await testStaleTimerDoesNotEndSecondFlash();
  await testSecondPanelFlashesIndependently();
  await testFallbackCopyStillFlashes();
  console.log('audit0903_panelcopy_flash_timer.test.js: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
