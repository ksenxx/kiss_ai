// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E regression for audit 2026-11-01 W6 F5: the content context menu's
// Select All selects the visible text as one Range per stretch between
// hidden sources, and since Chromium keeps only the first Range of a
// Selection, Copy used a cached string of the joined text while the
// live selection was still exactly the Range Select All installed.  A
// DOM Range is live: text inserted into the selected nodes by a
// preview's own script, or typed into a contenteditable, moved its
// boundaries along and the identity check still held, so Copy put the
// stale string on the clipboard while the page showed the new text.
// The live Ranges Select All installed (every stretch, not only the
// one the selection keeps) are now remembered instead and their text
// read afresh when Copy runs: what the selection covers at that moment,
// no more.  Round 2 (F5): re-reading the whole scope instead had Copy
// take text appended after the selection's end, which the selection's
// own highlight does not cover (a text node appended to the scope, or
// data appended to the end of a selected node -- Chromium's selection
// stays put for both, as a live Range does).
//
// jsdom's Ranges are live and its Selection keeps one Range, like
// Chromium's, so the real module shows the defect here; the stale-text
// tests failed against the pre-fix module.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

const MEDIA = path.join(__dirname, '..', 'media');

function makeWindow(html) {
  const win = new JSDOM('<body>' + html + '</body>', {
    runScripts: 'outside-only',
    url: 'https://localhost/',
  }).window;
  const copies = [];
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: text => {
        copies.push(text);
        return Promise.resolve();
      },
    },
  });
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.ContentContextMenu.installContentContextMenu(win.document);
  return {win, copies};
}

function menu(win, target, action) {
  target.dispatchEvent(
    new win.MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      clientX: 10,
      clientY: 10,
    }),
  );
  const item = win.document.querySelector(
    '#sorcar-content-context-menu .sorcar-ctx-item[data-action="' +
      action +
      '"]',
  );
  assert.ok(item && !item.classList.contains('disabled'), action + ' offered');
  item.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true, detail: 1}),
  );
}

async function main() {
  await test(
    'W6 F5: text inserted into the selected node after Select All is ' +
      'what Copy puts on the clipboard, not the text of the moment',
    async () => {
      const {win, copies} = makeWindow(
        '<div id="edit" contenteditable="true">Editable words</div>',
      );
      const edit = win.document.getElementById('edit');
      menu(win, edit, 'select-all');
      edit.firstChild.insertData(4, ' streamed');
      assert.strictEqual(win.getSelection().toString(), 'Edit streamedable words');
      menu(win, edit, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(copies, ['Edit streamedable words']);
    },
  );

  await test(
    'W6 F5: a multi-stretch Select All (text around a <script>) copies ' +
      'every stretch as it reads now, the hidden source still left out',
    async () => {
      const html =
        '<p id="one">One</p><script>var hidden = 1;</script><p id="two">Two</p>';
      const untouched = makeWindow(html);
      menu(untouched.win, untouched.win.document.body, 'select-all');
      menu(untouched.win, untouched.win.document.body, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(untouched.copies, ['One\nTwo'], 'untouched');
      // (jsdom, unlike Chromium, collapses the selection when the menu
      // focuses its first item, so each Copy here follows one Select
      // All on a fresh page.)
      const {win, copies} = makeWindow(html);
      const doc = win.document;
      menu(win, doc.body, 'select-all');
      // The second stretch is not part of the one Range the selection
      // keeps; the page's script inserts into it, and into the first.
      doc.getElementById('two').firstChild.insertData(1, 'w');
      doc.getElementById('one').firstChild.insertData(0, 'Stretch ');
      menu(win, doc.body, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(copies, ['Stretch One\nTwwo']);
      assert.ok(
        !copies[0].includes('hidden'),
        'the script text stays out: ' + copies[0],
      );
    },
  );

  await test(
    'W6 F5 (round 2): text appended after the selection -- a new node, ' +
      'or data past the end of a selected node -- is not copied',
    async () => {
      const {win, copies} = makeWindow(
        '<div id="edit" contenteditable="true">SELECTED</div>',
      );
      const doc = win.document;
      const edit = doc.getElementById('edit');
      menu(win, edit, 'select-all');
      edit.appendChild(doc.createTextNode(' UNSELECTED'));
      assert.strictEqual(win.getSelection().toString(), 'SELECTED');
      menu(win, edit, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(
        copies,
        ['SELECTED'],
        'Copy must put what the selection covers on the clipboard',
      );
      // Data appended to the end of a selected node is past the live
      // Range's end too (DOM "replace data"; Chromium's selection agrees),
      // while an insertion inside it is covered.
      const two = makeWindow(
        '<p id="one">One</p><script>var hidden = 1;</script><p id="two">Two</p>',
      );
      menu(two.win, two.win.document.body, 'select-all');
      two.win.document.getElementById('one').firstChild.appendData(' more');
      two.win.document.getElementById('two').firstChild.appendData(' more');
      two.win.document.getElementById('two').firstChild.insertData(0, 'In ');
      assert.strictEqual(two.win.getSelection().toString(), 'One');
      menu(two.win, two.win.document.body, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(two.copies, ['One\nIn Two']);
    },
  );

  await test(
    'W6 F5 (round 2): a hidden source inserted into a selected stretch ' +
      'after Select All stays out of the copy',
    async () => {
      const {win, copies} = makeWindow('<p id="one">One</p><p id="two">Two</p>');
      const doc = win.document;
      menu(win, doc.body, 'select-all');
      const script = doc.createElement('script');
      script.textContent = 'var hidden = 2;';
      doc.body.insertBefore(script, doc.getElementById('two'));
      menu(win, doc.body, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      // One stretch at Select All time: its text nodes read as one.
      assert.deepStrictEqual(copies, ['OneTwo']);
    },
  );

  await test(
    'W6 F5 guard: a selection the user makes afterwards drops the ' +
      'cache, and one re-installed with equal boundaries keeps it',
    async () => {
      const {win, copies} = makeWindow(
        '<p id="one">One</p><script>var hidden = 1;</script><p id="two">Two</p>',
      );
      const doc = win.document;
      const sel = win.getSelection();
      menu(win, doc.body, 'select-all');
      // What the clipboard fallback does on its way out: the same
      // boundaries as a new Range object.
      const clone = sel.getRangeAt(0).cloneRange();
      sel.removeAllRanges();
      sel.addRange(clone);
      menu(win, doc.body, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(copies, ['One\nTwo']);
      // The user selects "Two" by hand: Copy reads that selection.
      const range = doc.createRange();
      range.selectNodeContents(doc.getElementById('two'));
      sel.removeAllRanges();
      sel.addRange(range);
      menu(win, doc.getElementById('two'), 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(copies.slice(1), ['Two']);
    },
  );

  report('audit1101_W6_select_all_cache');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
