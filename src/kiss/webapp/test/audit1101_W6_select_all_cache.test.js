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
      // One stretch at Select All time, read as rendered: two paragraphs.
      assert.deepStrictEqual(copies, ['One\n\nTwo']);
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

  await test(
    'W8 F1: Select All leaves text the page hides (`hidden`, display:none, ' +
      'visibility:hidden) out of the copy, as it does script text',
    async () => {
      const {win, copies} = makeWindow(
        '<p id="one">One</p><span hidden>H</span><p style="display:none">D</p>' +
          '<p style="visibility:hidden">V</p><p id="two">Two</p>',
      );
      menu(win, win.document.body, 'select-all');
      menu(win, win.document.body, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(copies, ['One\nTwo']);
      // visibility is inherited: a visible child of an invisible parent shows.
      const seen = makeWindow(
        '<p style="visibility:hidden">V<b style="visibility:visible">Seen</b>W</p>',
      );
      menu(seen.win, seen.win.document.body, 'select-all');
      menu(seen.win, seen.win.document.body, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(seen.copies, ['Seen']);
      // Text the page hides after Select All stays out of the copy too:
      // a stretch of one text node (whose parent is hidden) and one of
      // several (whose common ancestor, or an ancestor above it, is).
      const html =
        '<p id="one">One</p><script>var hidden = 1;</script>' +
        '<div id="outer"><div id="wrap"><p>Two</p><p>Three</p></div></div>';
      for (const [target, style, expected] of [
        ['one', 'display:none', ['Two\n\nThree']],
        ['wrap', 'display:none', ['One']],
        ['outer', 'visibility:hidden', ['One']],
      ]) {
        const later = makeWindow(html);
        menu(later.win, later.win.document.body, 'select-all');
        later.win.document.getElementById(target).setAttribute('style', style);
        menu(later.win, later.win.document.body, 'copy');
        await new Promise(resolve => setImmediate(resolve));
        assert.deepStrictEqual(
          later.copies,
          expected,
          target + ' hidden after Select All',
        );
      }
    },
  );

  // The live Ranges are read as rendered, like the native Copy of the
  // selection: a block boundary is a line break (two for a paragraph),
  // a <br> one, cells are tab-separated, source whitespace collapses
  // outside <pre> -- instead of the text nodes' data run together.
  await test(
    'Select All then Copy keeps the paragraph and line breaks the ' +
      'page renders',
    async () => {
      const {win, copies} = makeWindow(
        '<p>Alpha</p>\n<p>Beta<br>Gamma</p>\n' +
          '<ul><li>one</li><li>two</li></ul>' +
          '<table><tr><td>a</td><td>b</td></tr></table>' +
          '<pre>  kept\n  indent</pre><div><span>in</span>line</div>',
      );
      menu(win, win.document.body, 'select-all');
      menu(win, win.document.body, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(copies, [
        'Alpha\n\nBeta\nGamma\n\none\ntwo\na\tb\n  kept\n  indent\ninline',
      ]);
    },
  );

  // Whitespace is preserved by the CSS `white-space` the text renders
  // under, not by the <pre> tag alone: `pre`, `pre-wrap` and
  // `break-spaces` (on any element, inherited by its children) keep
  // every space and line break, `pre-line` keeps the line breaks and
  // collapses the spaces around them, and a <pre> styled back to
  // `normal` collapses like any other block.  Collapsing crosses text
  // nodes as it does on screen: a collapsible space before a line
  // break (a preserved one, a <br>, a cell's end) goes, one after a
  // preserved space stays.
  await test('Select All then Copy keeps the whitespace CSS white-space preserves', async () => {
    const {win, copies} = makeWindow(
      '<style>.log{white-space:pre-wrap}</style>' +
        '<div style="white-space:pre-wrap">first line\n' +
        '    indented line\nlast line</div>' +
        '<div class="log"><span>  nested <b>bold</b>\n</span>tail</div>' +
        '<p style="white-space:pre-line">one  \n   two\n\nthree</p>' +
        '<div style="white-space:break-spaces">a  b</div>' +
        '<pre style="white-space:normal">x\n   y</pre>' +
        '<div style="white-space:pre-line">alpha <span>\n</span> beta</div>' +
        '<div><span style="white-space:pre-wrap">x  </span> y</div>' +
        '<p>Beta <br>Gamma</p>' +
        '<table><tr><td>a </td><td> b</td></tr></table>',
    );
    menu(win, win.document.body, 'select-all');
    menu(win, win.document.body, 'copy');
    await new Promise(resolve => setImmediate(resolve));
    assert.deepStrictEqual(copies, [
      'first line\n    indented line\nlast line\n' +
        '  nested bold\ntail\n\n' +
        'one\ntwo\n\nthree\n\n' +
        'a  b\nx y\n' +
        'alpha\nbeta\n' +
        'x   y\n\n' +
        'Beta\nGamma\n\n' +
        'a\tb',
    ]);
  });

  // A document starting with a newline before an inline <script>: the
  // whitespace must not become the one Range Chromium keeps (its
  // rendered text is empty, so Copy would be disabled).
  await test(
    'Select All starts its selection at visible text, not at the ' +
      'whitespace before a hidden source',
    async () => {
      const {win, copies} = makeWindow(
        '\n<script>void 0</script><p id="target">Visible report</p>',
      );
      menu(win, win.document.body, 'select-all');
      assert.strictEqual(win.getSelection().toString(), 'Visible report');
      menu(win, win.document.body, 'copy');
      await new Promise(resolve => setImmediate(resolve));
      assert.deepStrictEqual(copies, ['Visible report']);
    },
  );

  // Right-clicking a formatting child of an editing host: the menu acts
  // on the host (Select All selects all of it, Paste replaces a
  // selection crossing the child), not on the <b> alone.
  await test(
    'a menu opened on an inherited-contenteditable child acts on the ' +
      'editing host',
    async () => {
      const {win} = makeWindow(
        '<div id="host" contenteditable="true"><b id="b">One</b> Two</div>',
      );
      const doc = win.document;
      // jsdom has no isContentEditable: supply the inheritance the
      // specification defines (the attribute of the nearest ancestor).
      Object.defineProperty(win.HTMLElement.prototype, 'isContentEditable', {
        configurable: true,
        get() {
          for (let el = this; el; el = el.parentElement) {
            const v = el.getAttribute('contenteditable');
            if (v === '' || v === 'true') return true;
            if (v === 'false') return false;
          }
          return false;
        },
      });
      Object.defineProperty(win.navigator, 'clipboard', {
        configurable: true,
        value: {
          readText: () => Promise.resolve('X'),
          writeText: () => Promise.resolve(),
        },
      });
      const host = doc.getElementById('host');
      const b = doc.getElementById('b');
      menu(win, b, 'select-all');
      assert.strictEqual(win.getSelection().toString(), 'One Two');
      // Select "ne T" across the <b> and its sibling text, then Paste.
      // (jsdom drops the selection when the menu focuses its first item,
      // which Chromium does not, so it is installed once the menu is up.)
      b.dispatchEvent(
        new win.MouseEvent('contextmenu', {
          bubbles: true,
          cancelable: true,
          clientX: 10,
          clientY: 10,
        }),
      );
      const range = doc.createRange();
      range.setStart(b.firstChild, 1);
      range.setEnd(host.lastChild, 2);
      win.getSelection().removeAllRanges();
      win.getSelection().addRange(range);
      const paste = doc.querySelector(
        '#sorcar-content-context-menu .sorcar-ctx-item[data-action="paste"]',
      );
      assert.ok(paste && !paste.classList.contains('disabled'), 'paste offered');
      paste.dispatchEvent(
        new win.MouseEvent('click', {bubbles: true, cancelable: true, detail: 1}),
      );
      await new Promise(resolve => setImmediate(resolve));
      assert.strictEqual(host.textContent, 'OXwo');
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
