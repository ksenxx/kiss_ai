// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) test for the "Copied!" flash of the code-block
// copy button in the tips panel (media/tips.js).
//
// Race: flashCopyResult armed a bare 1500 ms setTimeout to put the
// label back to "Copy" and never cleared it, so a second click while
// a flash was showing had its "Copied!" label wiped early by the
// first click's stale timer.  The fix tracks the timer per button
// and restarts it on every copy.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function makePage(clipboardImpl) {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: clipboardImpl,
  });
  // The real vendored markdown renderer: tips only get a copy button
  // when marked turns ```blocks``` into <pre><code>.
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf8') +
      '\n//# sourceURL=audit0903-tips.js',
  );
  return win;
}

function openTipWithCode(win) {
  const el = win.__kissShowTipsPanel(['A tip:\n\n```\necho hello\n```\n']);
  const btn = el.shadowRoot.querySelector('.tips-copy');
  assert.ok(btn, 'the code block must get a copy button');
  return btn;
}

function click(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

async function testStaleTimerDoesNotEndSecondFlash() {
  const writes = [];
  const win = makePage({
    writeText: t => {
      writes.push(String(t));
      return Promise.resolve();
    },
  });
  const btn = openTipWithCode(win);

  click(win, btn);
  await sleep(20);
  assert.strictEqual(writes.length, 1);
  assert.match(writes[0], /echo hello/);
  assert.strictEqual(btn.textContent, 'Copied!');

  await sleep(1200);
  click(win, btn);
  await sleep(20);
  assert.strictEqual(btn.textContent, 'Copied!');

  // 1500 ms after the FIRST click its stale timer would fire here.
  await sleep(500);
  assert.strictEqual(
    btn.textContent,
    'Copied!',
    "the first click's stale timer must not wipe the second flash early",
  );

  await sleep(1200);
  assert.strictEqual(
    btn.textContent,
    'Copy',
    'the flash must still revert after the second timer',
  );
  win.close();
  console.log('  ok - a stale timer does not end the second flash');
}

async function testFailedCopyFlashesFailed() {
  const win = makePage({
    writeText: () => Promise.reject(new Error('denied')),
  });
  // No document.execCommand in JSDOM: the fallback reports failure.
  const btn = openTipWithCode(win);
  click(win, btn);
  await sleep(20);
  assert.strictEqual(btn.textContent, 'Failed');
  await sleep(1700);
  assert.strictEqual(btn.textContent, 'Copy');
  win.close();
  console.log('  ok - a failed copy flashes and reverts too');
}

// Out-of-order completion: two rapid clicks are two async clipboard
// writes that can settle in either order.  If the NEWER write succeeds
// first and the OLDER one fails afterwards, the stale completion must
// not replace the latest "Copied!" with "Failed" (or restart the revert
// timer from the stale operation).  Each click now takes a per-button
// copy generation and only the latest generation may update the label.
async function testStaleCompletionDoesNotOverwriteLatest() {
  const settlers = [];
  const win = makePage({
    writeText: () =>
      new Promise((resolve, reject) => settlers.push({resolve, reject})),
  });
  const btn = openTipWithCode(win);

  click(win, btn); // click 1 → pending write 1
  click(win, btn); // click 2 → pending write 2
  await sleep(20);
  assert.strictEqual(settlers.length, 2, 'expected two clipboard writes');

  settlers[1].resolve(); // the newest write lands first
  await sleep(20);
  assert.strictEqual(btn.textContent, 'Copied!');

  // The stale older write now fails (no document.execCommand in JSDOM,
  // so its fallback reports failure too).
  settlers[0].reject(new Error('denied'));
  await sleep(20);
  assert.strictEqual(
    btn.textContent,
    'Copied!',
    'a stale clipboard completion overwrote the latest copy result',
  );

  // And the revert timer belongs to the latest click, not the stale one.
  await sleep(1700);
  assert.strictEqual(btn.textContent, 'Copy');
  win.close();
  console.log('  ok - a stale async completion cannot overwrite the latest');
}

async function main() {
  await testStaleTimerDoesNotEndSecondFlash();
  await testFailedCopyFlashesFailed();
  await testStaleCompletionDoesNotOverwriteLatest();
  console.log('audit0903_tips_copy_flash_timer.test.js: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
