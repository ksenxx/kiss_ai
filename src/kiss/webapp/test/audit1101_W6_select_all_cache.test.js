// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E regression for audit 2026-11-01 W6 F5: the content context menu's
// Select All selects the visible text, and Copy while the live
// selection is still exactly the Range Select All installed used a
// cached string of its text.  A DOM Range is live: text inserted into
// the selected nodes by a preview's own script, or typed into a
// contenteditable, moved its boundaries along and the identity check
// still held, so Copy put the stale string on the clipboard while the
// page showed the new text.  Copy reads the live selection's text when
// it runs: what the selection covers at that moment, no more (round 2,
// F5: text appended after the selection's end, which the selection's
// own highlight does not cover, stays out -- a text node appended to
// the scope, or data appended to the end of the last selected node;
// Chromium's selection stays put for both, as a live Range does).
// Audit 2026-11-07 (A23): Select All used to install one Range per
// stretch between hidden sources, of which Chromium keeps the first,
// so the keyboard Copy got the first stretch alone; it is one Range
// now.  Audit 2026-11-08 (W1 F2): the menu Copy then read that Range
// through its own HTML/CSS text renderer, which copied what the native
// selection leaves out (a closed <details>, user-select:none text, a
// textarea's text, unchosen <option>s) and broke lines the page does
// not; it reads the document Selection's text, as rendered by the
// browser, like the keyboard Copy -- pinned, with the rendering (hidden
// sources and elements left out, paragraph breaks, whitespace), by the
// Chromium test src/kiss/tests/webapp/test_content_context_menu_paste_sweep1026.py.
//
// jsdom's Ranges are live and its Selection keeps one Range, like
// Chromium's, so the live-selection semantics show here; jsdom lays
// nothing out, so its Selection's text is the selected nodes' data.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {makeRunner, rightClick} = require('./ui_antipattern_harness');
const {MEDIA, flushMicrotasks} = require('./simplify2_harness');

const {test, report} = makeRunner();

function makeWindow(html, opts) {
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
  win.ContentContextMenu.installContentContextMenu(win.document, opts);
  return {win, copies};
}

function menu(win, target, action) {
  rightClick(win, target);
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

/** Menu-Copy on *target* and return what went on the clipboard. */
async function copied(win, copies, target) {
  const before = copies.length;
  menu(win, target, 'copy');
  await flushMicrotasks();
  assert.strictEqual(copies.length, before + 1, 'one copy');
  return copies[before];
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
      assert.strictEqual(
        win.getSelection().toString(),
        'Edit streamedable words',
      );
      assert.strictEqual(
        await copied(win, copies, edit),
        'Edit streamedable words',
      );
    },
  );

  await test(
    'W6 F5: Copy puts the text the live selection covers when it runs ' +
      'on the clipboard, insertions on both sides of a hidden source included',
    async () => {
      const html =
        '<p id="one">One</p><script>var hidden = 1;</script><p id="two">Two</p>';
      const untouched = makeWindow(html);
      menu(untouched.win, untouched.win.document.body, 'select-all');
      const sel = untouched.win.getSelection();
      assert.strictEqual(sel.rangeCount, 1);
      // (jsdom, unlike Chromium, collapses the selection when the menu
      // focuses its first item, so its text is read before each Copy,
      // and each Copy follows one Select All on a fresh page.)
      const shown = sel.toString();
      assert.strictEqual(
        await copied(
          untouched.win,
          untouched.copies,
          untouched.win.document.body,
        ),
        shown,
        'untouched',
      );
      const {win, copies} = makeWindow(html);
      const doc = win.document;
      menu(win, doc.body, 'select-all');
      // The page's script inserts into both paragraphs.
      doc.getElementById('two').firstChild.insertData(1, 'w');
      doc.getElementById('one').firstChild.insertData(0, 'Stretch ');
      const expected = win.getSelection().toString();
      const text = await copied(win, copies, doc.body);
      assert.strictEqual(text, expected);
      assert.ok(text.startsWith('Stretch One'), text);
      assert.ok(text.endsWith('Twwo'), text);
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
      assert.strictEqual(
        await copied(win, copies, edit),
        'SELECTED',
        'Copy must put what the selection covers on the clipboard',
      );
      // Data appended to the end of the last selected node is past the
      // live Range's end too (DOM "replace data"; Chromium's selection
      // agrees), while an insertion inside it, and data appended to a
      // node before the last, are covered.
      const two = makeWindow(
        '<p id="one">One</p><script>var hidden = 1;</script><p id="two">Two</p>',
      );
      menu(two.win, two.win.document.body, 'select-all');
      two.win.document.getElementById('one').firstChild.appendData(' more');
      two.win.document.getElementById('two').firstChild.appendData(' more');
      two.win.document.getElementById('two').firstChild.insertData(0, 'In ');
      const expected = two.win.getSelection().toString();
      const text = await copied(two.win, two.copies, two.win.document.body);
      assert.strictEqual(text, expected);
      assert.ok(text.startsWith('One more'), text);
      assert.ok(text.endsWith('In Two'), text);
    },
  );

  await test('a selection the user makes after Select All is what Copy reads', async () => {
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
    const expected = sel.toString();
    assert.strictEqual(await copied(win, copies, doc.body), expected);
    // The user selects "Two" by hand: Copy reads that selection.
    const range = doc.createRange();
    range.selectNodeContents(doc.getElementById('two'));
    sel.removeAllRanges();
    sel.addRange(range);
    assert.strictEqual(
      await copied(win, copies, doc.getElementById('two')),
      'Two',
    );
  });

  // Select All starts and ends at text the page shows: the hidden
  // elements (`hidden`, display:none, visibility:hidden) at the edges of
  // the scope are outside the selection, like script text.  (What the
  // selection covers between its ends is rendered by the browser; jsdom
  // reads every node's data.)
  await test(
    'W8 F1: Select All starts and ends at visible text, leaving the ' +
      'hidden elements at the edges out',
    async () => {
      const {win, copies} = makeWindow(
        '<span hidden>H</span><p style="display:none">D</p><p id="one">One</p>' +
          '<p id="two">Two</p><p style="visibility:hidden">V</p>',
      );
      const doc = win.document;
      menu(win, doc.body, 'select-all');
      const range = win.getSelection().getRangeAt(0);
      assert.strictEqual(
        range.startContainer,
        doc.getElementById('one').firstChild,
      );
      assert.strictEqual(
        range.endContainer,
        doc.getElementById('two').firstChild,
      );
      assert.strictEqual(await copied(win, copies, doc.body), 'OneTwo');
      // visibility is inherited: a visible child of an invisible parent shows.
      const seen = makeWindow(
        '<p style="visibility:hidden">V<b style="visibility:visible">Seen</b>W</p>',
      );
      menu(seen.win, seen.win.document.body, 'select-all');
      assert.strictEqual(seen.win.getSelection().toString(), 'Seen');
      assert.strictEqual(
        await copied(seen.win, seen.copies, seen.win.document.body),
        'Seen',
      );
    },
  );

  // The chat page installs the menu on its whole document and names the
  // content area as the scope of a right-click in it: Select All there
  // selects the opened file's text, not the chat beside it in the split
  // layout, which the body would take in (A23).
  await test(
    'Select All selects the scope the installer names for the clicked ' +
      'element, the body where it names none',
    async () => {
      const {win, copies} = makeWindow(
        '<div id="chat">PRIVATE CHAT</div>' +
          '<div id="area"><p id="file">FILE</p><p>CONTENT</p></div>',
        {scope: target => target.closest('#area')},
      );
      const doc = win.document;
      menu(win, doc.getElementById('file'), 'select-all');
      assert.strictEqual(win.getSelection().toString(), 'FILECONTENT');
      assert.strictEqual(
        await copied(win, copies, doc.getElementById('file')),
        'FILECONTENT',
      );
      menu(win, doc.getElementById('chat'), 'select-all');
      assert.strictEqual(
        win.getSelection().toString(),
        'PRIVATE CHATFILECONTENT',
      );
    },
  );

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
      assert.strictEqual(
        await copied(win, copies, win.document.body),
        'Visible report',
      );
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
      assert.ok(
        paste && !paste.classList.contains('disabled'),
        'paste offered',
      );
      paste.dispatchEvent(
        new win.MouseEvent('click', {
          bubbles: true,
          cancelable: true,
          detail: 1,
        }),
      );
      await flushMicrotasks();
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
