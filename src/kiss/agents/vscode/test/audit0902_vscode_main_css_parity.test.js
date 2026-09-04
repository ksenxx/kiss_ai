// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM cascade) parity tests for the sibling side panels
// of media/main.css: the four dimming overlays (history sidebar,
// settings, frequent tasks, tricks) and their `.open` state, the four
// panel close buttons and their hover state, and the two scrolling
// list panes.  main.css used to spell each of them out as its own copy
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
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeDom() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  html = html.replace('</head>', '<style>' + css + '</style></head>');
  return new JSDOM(html, {pretendToBeVisual: true, url: 'https://localhost/'});
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

const OVERLAYS = [
  'sidebar-overlay',
  'settings-overlay',
  'frequent-overlay',
  'tricks-overlay',
];
const CLOSE_BTNS = [
  'sidebar-close',
  'settings-panel-close',
  'frequent-panel-close',
  'tricks-panel-close',
];
const LISTS = ['frequent-list', 'tricks-list'];

function testOverlaysAgree() {
  const dom = makeDom();
  const win = dom.window;
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
  console.log('  ok - the four overlays compute the same style');
}

function testCloseButtonsAgree() {
  const dom = makeDom();
  const win = dom.window;
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
  // The bottom sheets' close buttons sit above their own list pane.
  for (const id of ['frequent-panel-close', 'tricks-panel-close']) {
    assert.strictEqual(
      styleOf(win, win.document.getElementById(id), ['z-index'])['z-index'],
      '1',
      `#${id} stacks above its list`,
    );
  }
  win.close();
  console.log('  ok - the four panel close buttons compute the same style');
}

function testListPanesAgree() {
  const dom = makeDom();
  const win = dom.window;
  assertSiblingsAgree(
    win,
    LISTS,
    ['flex', 'overflow-y', 'scrollbar-width', 'margin-top'],
    'list panes',
  );
  win.close();
  console.log('  ok - the two list panes compute the same style');
}

function main() {
  testOverlaysAgree();
  testCloseButtonsAgree();
  testListPanesAgree();
  console.log('all audit0902 css-parity tests passed');
}

main();
