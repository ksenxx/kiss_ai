// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) test for the "Copied!" flash of the code-block
// copy button in the tips panel (media/tips.js).
//
// The tips button is wired through PanelCopy.wireCopyButton
// (media/panelCopy.js) with tips.js's label painter.  The per-button
// revert timer (a first click's stale timer once cut a second flash
// short) is pinned by audit0903_panelcopy_flash_timer.test.js, the
// Copy/Copied!/Failed labels by tipsWindow.test.js.  This suite pins
// the out-of-order completion on the tips button, including what the
// panel-copy suite does not check: the stale completion arms no revert
// timer of its own.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const {
  sleep,
  click,
  showTips,
  recordTimers,
  fireTimers,
} = require('./simplify2_harness');

// The 1500 ms revert timers are recorded instead of waited for and the
// clipboard writes are settled by hand, so `sleep(0)` is the exact wait.

function makePage(clipboardImpl) {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  const timers = recordTimers(win, 1500);
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: clipboardImpl,
  });
  // The real vendored markdown renderer: tips only get a copy button
  // when marked turns ```blocks``` into <pre><code>.
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  // tips.js copies through PanelCopy.copyText (panelCopy.js).
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf8'));
  return {win, timers};
}

function openTipWithCode(win) {
  const el = showTips(win, ['A tip:\n\n```\necho hello\n```\n']);
  const btn = el.shadowRoot.querySelector('.tips-copy');
  assert.ok(btn, 'the code block must get a copy button');
  return btn;
}

// Out-of-order completion: two rapid clicks are two async clipboard
// writes that can settle in either order.  If the NEWER write succeeds
// first and the OLDER one fails afterwards, the stale completion must
// not replace the latest "Copied!" with "Failed" (or restart the revert
// timer from the stale operation).  Each click now takes a per-button
// copy generation and only the latest generation may update the label.
async function testStaleCompletionDoesNotOverwriteLatest() {
  const settlers = [];
  const {win, timers} = makePage({
    writeText: () =>
      new Promise((resolve, reject) => settlers.push({resolve, reject})),
  });
  const btn = openTipWithCode(win);

  click(win, btn); // click 1 → pending write 1
  click(win, btn); // click 2 → pending write 2
  await sleep(0);
  assert.strictEqual(settlers.length, 2, 'expected two clipboard writes');

  settlers[1].resolve(); // the newest write lands first
  await sleep(0);
  assert.strictEqual(btn.textContent, 'Copied!');

  // The stale older write now fails (no document.execCommand in JSDOM,
  // so its fallback reports failure too).
  settlers[0].reject(new Error('denied'));
  await sleep(0);
  assert.strictEqual(
    btn.textContent,
    'Copied!',
    'a stale clipboard completion overwrote the latest copy result',
  );

  // And the revert timer belongs to the latest click, not the stale one.
  assert.strictEqual(
    timers.length,
    1,
    'only the latest click arms a revert timer',
  );
  fireTimers(timers);
  assert.strictEqual(btn.textContent, 'Copy');
  win.close();
  console.log('  ok - a stale async completion cannot overwrite the latest');
}

async function main() {
  await testStaleCompletionDoesNotOverwriteLatest();
  console.log('audit0903_tips_copy_flash_timer.test.js: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
