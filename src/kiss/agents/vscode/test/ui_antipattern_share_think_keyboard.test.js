// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Exported share page: the "Thinking" header is a focusable role="button"
// (tabindex="0"), so it must toggle on Enter and Space exactly like a click
// and keep aria-expanded in sync.  On the old share.js the header only had
// the inline onclick, so keyboard users could not open or close it, and
// aria-expanded never changed.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const SHARE_JS = fs.readFileSync(path.join(MEDIA, 'share.js'), 'utf8');

/**
 * Build a share page holding one thinking block with the header markup
 * main.js renders (inline onclick + tabindex/role/aria-expanded) and run
 * share.js in it.
 *
 * @returns {Window} The page's window.
 */
function makeSharePage() {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><head></head><body>' +
      '<div id="app"><div id="output">' +
      '<div class="think">' +
      '<div class="lbl" onclick="toggleThink(this)" tabindex="0" ' +
      'role="button" aria-expanded="true">' +
      '<span class="arrow">▾</span> Thinking</div>' +
      '<div class="cnt">deep thoughts</div>' +
      '</div>' +
      '<p class="txt" tabindex="0">not a header</p>' +
      '</div></div>' +
      '</body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true, url: 'https://x/'},
  );
  const win = dom.window;
  win.eval(SHARE_JS + '\n//# sourceURL=share.js');
  return win;
}

function keydown(el, key) {
  const ev = new el.ownerDocument.defaultView.KeyboardEvent('keydown', {
    key,
    bubbles: true,
    cancelable: true,
  });
  el.dispatchEvent(ev);
  return ev;
}

function run() {
  const win = makeSharePage();
  const lbl = win.document.querySelector('.think > .lbl');
  const cnt = win.document.querySelector('.think > .cnt');
  const arrow = lbl.querySelector('.arrow');

  let ev = keydown(lbl, 'Enter');
  assert.ok(cnt.classList.contains('hidden'), 'Enter collapses the block');
  assert.ok(arrow.classList.contains('collapsed'), 'Enter rotates the arrow');
  assert.strictEqual(lbl.getAttribute('aria-expanded'), 'false');
  assert.ok(ev.defaultPrevented, 'Enter is consumed (no page scroll/submit)');
  console.log(
    '  ok - Enter collapses the thinking block and syncs aria-expanded',
  );

  ev = keydown(lbl, ' ');
  assert.ok(!cnt.classList.contains('hidden'), 'Space expands it again');
  assert.ok(!arrow.classList.contains('collapsed'), 'Space resets the arrow');
  assert.strictEqual(lbl.getAttribute('aria-expanded'), 'true');
  assert.ok(ev.defaultPrevented, 'Space is consumed (no page scroll)');
  console.log(
    '  ok - Space expands the thinking block and syncs aria-expanded',
  );

  ev = keydown(lbl, 'Escape');
  assert.ok(!cnt.classList.contains('hidden'), 'other keys are ignored');
  assert.ok(!ev.defaultPrevented, 'other keys keep their default');

  const txt = win.document.querySelector('.txt');
  ev = keydown(txt, 'Enter');
  assert.ok(!cnt.classList.contains('hidden'), 'Enter elsewhere does nothing');
  assert.ok(!ev.defaultPrevented, 'Enter elsewhere is not consumed');
  console.log('  ok - other keys and other targets are left alone');

  // The inline onclick path (a mouse click) keeps aria-expanded in sync too.
  win.toggleThink(lbl);
  assert.ok(cnt.classList.contains('hidden'));
  assert.strictEqual(lbl.getAttribute('aria-expanded'), 'false');
  win.toggleThink(lbl);
  assert.strictEqual(lbl.getAttribute('aria-expanded'), 'true');
  console.log('  ok - toggleThink itself syncs aria-expanded');

  // A header without a .cnt sibling (or without a parent) must not throw.
  const orphan = win.document.createElement('div');
  orphan.className = 'lbl';
  win.toggleThink(orphan);
  const bare = win.document.createElement('div');
  bare.className = 'think';
  const bareLbl = win.document.createElement('div');
  bareLbl.className = 'lbl';
  bare.appendChild(bareLbl);
  win.toggleThink(bareLbl);
  assert.ok(!bareLbl.hasAttribute('aria-expanded'), 'no .cnt: nothing to sync');
  console.log('  ok - toggleThink tolerates a detached or content-less header');

  win.close();
}

try {
  run();
  console.log('\nAll tests passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err);
  process.exit(1);
}
