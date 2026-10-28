// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// UI anti-pattern fixes A5 / A2 / A10 / A15 in chat.html and main.css:
// every icon-only control has an accessible name, the destructive
// restart-server button names its outcome instead of "OK", the clear
// buttons are 24x24 hit targets, the model pill and model search keep a
// visible focus ring, and disabled menu items stay legible.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {makePage, loadMainCss} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {cssRule} = require('./designTokens');
const {test, report} = makeRunner();

const MEDIA = path.join(__dirname, '..', 'media');


const {win} = makePage({beforeScripts: loadMainCss});
const doc = win.document;

function label(id) {
  const el = doc.getElementById(id);
  assert.ok(el, `#${id} must exist`);
  return (el.getAttribute('aria-label') || '').trim();
}

test('icon-only buttons have an accessible name', () => {
  const expected = {
    'input-clear-btn': 'Clear input',
    'sidebar-close': 'Close sidebar',
    'model-search-clear': 'Clear search',
    'history-search-clear': 'Clear search',
  };
  for (const id of Object.keys(expected)) {
    assert.strictEqual(label(id), expected[id], `#${id} aria-label`);
  }
  for (const id of ['tricks-panel-close', 'settings-panel-close']) {
    assert.ok(/^Close /.test(label(id)), `#${id} must be named "Close ..."`);
  }
});

test('#menu-btn has both a tooltip and an accessible name that say what it does', () => {
  const btn = doc.getElementById('menu-btn');
  const tip = btn.getAttribute('data-tooltip') || '';
  assert.ok(tip.length > 0, 'data-tooltip must be set');
  assert.strictEqual(btn.getAttribute('aria-label'), tip);
  // The panel it toggles is the Chats panel (formerly "History").
  assert.ok(/chats/i.test(tip), `must mention chats, got: ${tip}`);
  assert.notStrictEqual(
    tip,
    doc.getElementById('more-btn').getAttribute('aria-label'),
    'must not collide with the "More actions" button',
  );
});

test('every data-tooltip-only button carries a matching aria-label', () => {
  for (const id of [
    'tricks-btn',
    'voice-btn',
    'share-btn',
    'upload-btn',
    'settings-btn',
    'model-btn',
    'send-btn',
    'stop-btn',
  ]) {
    const el = doc.getElementById(id);
    assert.ok(el, `#${id} must exist`);
    assert.strictEqual(
      el.getAttribute('aria-label'),
      el.getAttribute('data-tooltip'),
      `#${id} aria-label must equal its data-tooltip`,
    );
  }
  assert.strictEqual(label('autocommit-btn'), 'Git commit');
});

test('search inputs are named by aria-label, not only by placeholder', () => {
  assert.strictEqual(label('model-search'), 'Search models');
  assert.strictEqual(label('history-search'), 'Search chats');
});

test('restart-server confirm names the outcome and keeps Cancel first', () => {
  const ok = doc.getElementById('server-reset-confirm-ok');
  const cancel = doc.getElementById('server-reset-confirm-cancel');
  assert.strictEqual(ok.textContent.trim(), 'Restart and abort task');
  assert.strictEqual(cancel.textContent.trim(), 'Cancel');
  assert.strictEqual(
    cancel.compareDocumentPosition(ok) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    win.Node.DOCUMENT_POSITION_FOLLOWING,
    'Cancel must come before the destructive button',
  );
});

const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

test('clear buttons are at least 24x24 px hit targets', () => {
  for (const sel of ['#input-clear-btn', '.search-clear-btn']) {
    const style = win.getComputedStyle(doc.querySelector(sel));
    assert.ok(parseFloat(style.width) >= 24, `${sel} width >= 24, got ${style.width}`);
    assert.ok(parseFloat(style.height) >= 24, `${sel} height >= 24, got ${style.height}`);
  }
});

// jsdom never matches :focus-visible (and reports the initial
// `outline-style: none` for every element), so the ring is read from
// the stylesheet rules.
test('model pill and model search show a focus ring instead of outline:none', () => {
  for (const id of ['model-btn', 'model-search']) {
    assert.ok(
      !/outline:\s*none/.test(cssRule(css, `#${id}`)),
      `#${id} must not suppress its outline`,
    );
    assert.ok(
      /outline:\s*1px solid var\(--accent\)/.test(
        cssRule(css, `#${id}:focus-visible`),
      ),
      `#${id}:focus-visible draws the accent ring`,
    );
  }
});

test('disabled menu items keep their label readable', () => {
  const item = doc.getElementById('settings-btn');
  item.disabled = true;
  const opacity = Number(win.getComputedStyle(item).opacity);
  item.disabled = false;
  assert.ok(opacity >= 0.5, `opacity >= 0.5, got ${opacity}`);
});

// Close the page so no timer main.js armed keeps node alive.
win.close();
report('ui_antipattern_a11y_labels');
