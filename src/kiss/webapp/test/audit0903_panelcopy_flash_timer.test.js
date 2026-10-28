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

const {
  sleep,
  click,
  recordTimers,
  fireTimers,
  stubClipboard,
} = require('./simplify2_harness');

// The 1500 ms revert timers are recorded instead of waited for: the
// test fires them by hand, so no wall-clock sleep decides whether the
// stale one lands before or after the second click.

function makePage() {
  const dom = new JSDOM('<!DOCTYPE html><body><div id="panel"></div></body>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  const timers = recordTimers(win, 1500);
  const {writes: clipboardWrites} = stubClipboard(win);
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8') +
      '\n//# sourceURL=audit0903-panelcopy.js',
  );
  return {win, clipboardWrites, timers};
}

function isCheck(btn) {
  return !!btn.querySelector('svg polyline');
}

async function testStaleTimerDoesNotEndSecondFlash() {
  const {win, clipboardWrites, timers} = makePage();
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
  assert.strictEqual(timers.length, 1, 'the first click arms one revert timer');

  // Second click while the first flash is still showing.
  click(win, btn);
  await sleep(0);
  assert.ok(isCheck(btn), 'second click must show the check icon');
  assert.strictEqual(
    timers.length,
    2,
    'the second click arms its own revert timer',
  );

  // 1500 ms after the FIRST click: whatever the first click left armed
  // fires now, before the second click's timer is due.
  fireTimers(timers.slice(0, 1));
  assert.ok(
    isCheck(btn) && btn.classList.contains('copied'),
    "the first click's stale timer must not end the second flash early",
  );

  // The second flash still reverts on its own.
  fireTimers(timers);
  assert.ok(
    !isCheck(btn) && !btn.classList.contains('copied'),
    'the flash must still revert after the second timer',
  );
  win.close();
  console.log('  ok - a stale timer does not end the second flash');
}

async function testSecondPanelFlashesIndependently() {
  const {win, timers} = makePage();
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
  assert.strictEqual(timers.length, 2, 'each button arms its own revert timer');
  fireTimers(timers);
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

// Race: the async clipboard write of a first click settled AFTER a
// faster second click's, so its (stale) outcome overwrote the button:
// a slow failing first copy painted "Copy failed" over the second
// click's check mark.  Each click takes a new generation and only the
// latest one's outcome reaches the button.
async function testStaleClipboardResultIsDropped() {
  const {win} = makePage();
  let rejectFirst = null;
  let calls = 0;
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: () => {
        calls += 1;
        // The first write hangs until the test fails it; the second
        // succeeds at once.
        if (calls === 1)
          return new Promise((_res, rej) => {
            rejectFirst = rej;
          });
        return Promise.resolve();
      },
    },
  });
  // Both the slow write's fallback and the second click's succeed or
  // fail on their own: the fallback of the first is made to fail.
  win.document.execCommand = () => false;
  const panel = win.document.getElementById('panel');
  panel.textContent = 'raced text';
  win.PanelCopy.addCopyButton(panel);
  const btn = panel.querySelector('.panel-copy-btn');

  click(win, btn);
  click(win, btn);
  await sleep(0);
  assert.strictEqual(calls, 2);
  assert.ok(
    isCheck(btn) && btn.classList.contains('copied'),
    'the second click flashed its check',
  );

  // The first click's write now fails (and its fallback too).
  rejectFirst(new Error('denied'));
  await sleep(0);
  await sleep(0);
  assert.ok(
    isCheck(btn) &&
      btn.classList.contains('copied') &&
      !btn.classList.contains('copy-failed'),
    "a stale first-click failure must not overwrite the second click's check mark",
  );
  assert.strictEqual(btn.title, 'Copied');
  win.close();
  console.log('  ok - a stale clipboard outcome is dropped');
}

async function main() {
  await testStaleTimerDoesNotEndSecondFlash();
  await testSecondPanelFlashesIndependently();
  await testFallbackCopyStillFlashes();
  await testStaleClipboardResultIsDropped();
  console.log('audit0903_panelcopy_flash_timer.test.js: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
