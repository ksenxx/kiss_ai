// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM cascade) parity tests for the sibling side panels
// of media/main.css: the three dimming overlays (history sidebar,
// settings, tricks) and their `.open` state, and the three panel close
// buttons and their hover state.  main.css used to spell each of them out as its own copy
// of the same declarations; they are grouped now, and this test pins
// that every sibling still computes to the same style -- so a future
// edit to one panel's look must be a deliberate un-grouping, not an
// accidental drift.  The rules are applied through jsdom's real
// stylesheet cascade on the real chat.html markup, not by reading the
// source text.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {inlineDesignTokens} = require('./designTokens');
const {newDom, MEDIA} = require('./simplify2_harness');

// No page script runs: only main.css (its design tokens inlined, as
// jsdom resolves no var()) is applied to the real chat.html markup.
function makeDom() {
  const win = newDom('outside-only');
  const css = inlineDesignTokens(
    fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8'),
  );
  const styleEl = win.document.createElement('style');
  styleEl.textContent = css;
  win.document.head.appendChild(styleEl);
  return win;
}

function styleOf(win, el, props) {
  const cs = win.getComputedStyle(el);
  const out = {};
  for (const p of props) out[p] = cs.getPropertyValue(p);
  return out;
}

function assertSiblingsAgree(win, ids, props, label) {
  const els = ids.map(id => {
    const el = win.document.getElementById(id);
    assert.ok(el, `#${id} must exist in chat.html`);
    return el;
  });
  const first = styleOf(win, els[0], props);
  for (const p of props) {
    assert.notStrictEqual(
      first[p],
      '',
      `${label}: #${ids[0]} must declare ${p} (the rule must match)`,
    );
  }
  for (let i = 1; i < els.length; i++) {
    assert.deepStrictEqual(
      styleOf(win, els[i], props),
      first,
      `${label}: #${ids[i]} must compute the same style as #${ids[0]}`,
    );
  }
}

const OVERLAYS = ['sidebar-overlay', 'settings-overlay', 'tricks-overlay'];
const CLOSE_BTNS = [
  'sidebar-close',
  'settings-panel-close',
  'tricks-panel-close',
];

function testOverlaysAgree() {
  const win = makeDom();
  const props = [
    'position',
    'inset',
    'background',
    'z-index',
    'opacity',
    'pointer-events',
    'transition',
  ];
  assertSiblingsAgree(win, OVERLAYS, props, 'closed overlays');
  assert.strictEqual(
    styleOf(win, win.document.getElementById('sidebar-overlay'), [
      'pointer-events',
    ])['pointer-events'],
    'none',
    'a closed overlay lets clicks through',
  );
  for (const id of OVERLAYS)
    win.document.getElementById(id).classList.add('open');
  assertSiblingsAgree(win, OVERLAYS, props, 'open overlays');
  assert.strictEqual(
    styleOf(win, win.document.getElementById('tricks-overlay'), ['opacity'])
      .opacity,
    '1',
    'an open overlay is visible',
  );
  win.close();
  console.log('  ok - the three overlays compute the same style');
}

function testCloseButtonsAgree() {
  const win = makeDom();
  const props = [
    'position',
    'top',
    'right',
    'background',
    'border',
    'color',
    'font-size',
    'cursor',
    'padding',
    'border-radius',
  ];
  assertSiblingsAgree(win, CLOSE_BTNS, props, 'panel close buttons');
  // The bottom sheet's close button sits above its own list pane.
  assert.strictEqual(
    styleOf(win, win.document.getElementById('tricks-panel-close'), [
      'z-index',
    ])['z-index'],
    '1',
    '#tricks-panel-close stacks above its list',
  );
  win.close();
  console.log('  ok - the three panel close buttons compute the same style');
}

function main() {
  testOverlaysAgree();
  testCloseButtonsAgree();
  console.log('all audit0902 css-parity tests passed');
}

main();
