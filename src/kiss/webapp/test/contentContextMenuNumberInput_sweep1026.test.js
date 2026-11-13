// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Paste / Select All from the content context menu on an
// `<input type="number">` (e.g. `#cfg-max-budget`) or `type="email"`.
// Those input types have no text-selection API: `selectionStart` is null
// and `setRangeText` / `setSelectionRange` throw InvalidStateError (in
// Chromium and in jsdom alike).  Paste used to throw inside the clipboard
// promise's fulfilment handler, so the text was lost as an unhandled
// rejection; Select All threw synchronously out of the click handler.
//
// In a browser Paste on those fields goes through
// `document.execCommand('insertText')`, which replaces the native
// selection (pinned by the Chromium test
// `src/kiss/tests/webapp/test_content_context_menu_paste_sweep1026.py`).
// jsdom has no `execCommand`, so these tests cover the fallback that
// appends to the value and fires one `input` event; the `execCommand`
// refused (returns false) branch takes the same append path and is not
// reproducible end-to-end here without a test double.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM, VirtualConsole} = require('jsdom');
const {makeRunner, rightClick} = require('./ui_antipattern_harness');
const {MEDIA, settle} = require('./simplify2_harness');
// The runner closes the pages a test built when it has ended.
const windows = [];
const runner = makeRunner({windows});

// Rejections nobody handled and errors thrown out of event listeners
// (jsdom reports those on its virtual console, like a browser's console).
const unhandled = [];
process.on('unhandledRejection', e => unhandled.push(e));
const listenerErrors = [];

function makeDom(clipboardText) {
  unhandled.length = 0;
  listenerErrors.length = 0;
  const virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', e => listenerErrors.push(e));
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<input id="num" type="number" value="12">' +
      '<input id="mail" type="email" value="a@b">' +
      '<input id="txt" type="text" value="ab">' +
      '</body></html>',
    {
      runScripts: 'dangerously',
      pretendToBeVisual: true,
      url: 'https://localhost/',
      virtualConsole,
    },
  );
  const win = dom.window;
  Object.defineProperty(win.navigator, 'clipboard', {
    value: {readText: () => Promise.resolve(clipboardText)},
    configurable: true,
  });
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.ContentContextMenu.installContentContextMenu(win.document);
  windows.push(win);
  return win;
}

/**
 * Make the clipboard read of *win* pend.  The returned function answers
 * the pending read with its argument, and fails when the page asked for
 * none since the last answer (a paste that never reached the clipboard
 * would otherwise pass every "nothing was pasted" check).
 */
function deferClipboard(win) {
  let answer = null;
  Object.defineProperty(win.navigator, 'clipboard', {
    value: {
      readText: () =>
        new Promise(resolve => {
          answer = resolve;
        }),
    },
    configurable: true,
  });
  return text => {
    assert.ok(answer, 'the clipboard was asked');
    answer(text);
    answer = null;
  };
}

/** Open the menu on *target* and return its row for *action*. */
function menuRow(win, target, action) {
  rightClick(win, target);
  const item = win.document.querySelector(
    '#sorcar-content-context-menu .sorcar-ctx-item[data-action="' +
      action +
      '"]',
  );
  assert.ok(item, action + ' item is offered');
  return item;
}

function activate(win, target, action) {
  const item = menuRow(win, target, action);
  assert.ok(!item.classList.contains('disabled'), action + ' item is enabled');
  item.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true, detail: 1}),
  );
}

const run = (async () => {
  await runner.test(
    'Paste falls back to appending the clipboard text to a number input',
    async () => {
      const win = makeDom('34');
      const num = win.document.getElementById('num');
      const inputs = [];
      num.addEventListener('input', () => inputs.push(num.value));
      activate(win, num, 'paste');
      await settle();
      assert.strictEqual(num.value, '1234');
      assert.deepStrictEqual(
        inputs,
        ['1234'],
        'one input event after the paste',
      );
      assert.strictEqual(unhandled.length, 0, 'no unhandled rejection');
    },
  );

  await runner.test(
    'Paste falls back to appending on an email input (no selection API either)',
    async () => {
      const win = makeDom('.c');
      const mail = win.document.getElementById('mail');
      activate(win, mail, 'paste');
      await settle();
      assert.strictEqual(mail.value, 'a@b.c');
      assert.strictEqual(unhandled.length, 0, 'no unhandled rejection');
    },
  );

  await runner.test(
    'Paste still inserts at the caret of a text input',
    async () => {
      const win = makeDom('X');
      const txt = win.document.getElementById('txt');
      txt.focus();
      txt.setSelectionRange(1, 1);
      activate(win, txt, 'paste');
      await settle();
      assert.strictEqual(txt.value, 'aXb');
      assert.strictEqual(
        txt.selectionStart,
        2,
        'caret sits after the pasted text',
      );
      assert.strictEqual(unhandled.length, 0);
    },
  );

  await runner.test(
    'Select All on a number input does not throw and keeps focus',
    () => {
      const win = makeDom('');
      const num = win.document.getElementById('num');
      activate(win, num, 'select-all');
      assert.deepStrictEqual(
        listenerErrors,
        [],
        'no error escaped the click handler',
      );
      assert.strictEqual(
        win.document.activeElement,
        num,
        'the field is focused',
      );
      assert.strictEqual(
        win.document.querySelectorAll(
          '#sorcar-content-context-menu .sorcar-ctx-item',
        ).length,
        0,
        'the menu closed',
      );
    },
  );

  // A number input has no selection offsets, so Copy reads the document
  // Selection for it, which is the field's only while it holds the focus.
  // (jsdom's focus() empties the Selection, the menu's own on opening
  // included, so it is made again after each.)
  await runner.test(
    'Copy on a number input offers the document Selection only while ' +
      'the field holds the focus',
    () => {
      const win = makeDom('');
      const doc = win.document;
      const num = doc.getElementById('num');
      const other = doc.body.appendChild(doc.createElement('span'));
      other.textContent = 'elsewhere';
      win.getSelection().selectAllChildren(other);
      assert.ok(
        menuRow(win, num, 'copy').classList.contains('disabled'),
        'text selected elsewhere is not the unfocused field\'s to copy',
      );
      num.focus();
      win.getSelection().selectAllChildren(other);
      assert.ok(
        !menuRow(win, num, 'copy').classList.contains('disabled'),
        'the focused field copies what the Selection reads',
      );
    },
  );

  await runner.test(
    'Select All on a text input selects its whole value',
    () => {
      const win = makeDom('');
      const txt = win.document.getElementById('txt');
      activate(win, txt, 'select-all');
      assert.strictEqual(txt.selectionStart, 0);
      assert.strictEqual(txt.selectionEnd, 2);
    },
  );

  // An input that holds no text (a checkbox, a button) is not a field:
  // Select All over it selects what it selects beside it.  It used to
  // select the control's own (absent) text: nothing (A/W1-F1).  One of
  // an unknown type is the text field the browser makes of it.
  await runner.test(
    'Select All over a non-text input selects the text around it',
    () => {
      const win = makeDom('');
      const doc = win.document;
      doc.body.insertAdjacentHTML(
        'beforeend',
        '<p id="para">before <input type="checkbox" id="cb">' +
          '<input type="button" id="bt" value="Press"> after' +
          '<input type="odd" id="odd" value="abc"></p>',
      );
      activate(win, doc.getElementById('para'), 'select-all');
      const beside = win.getSelection().toString();
      assert.ok(beside.includes('before') && beside.includes('after'), beside);
      ['cb', 'bt'].forEach(id => {
        win.getSelection().removeAllRanges();
        activate(win, doc.getElementById(id), 'select-all');
        assert.strictEqual(win.getSelection().toString(), beside, id);
      });
      const odd = doc.getElementById('odd');
      activate(win, odd, 'select-all');
      assert.deepStrictEqual([odd.selectionStart, odd.selectionEnd], [0, 3]);
    },
  );

  // The clipboard read is asynchronous (the browser may first ask the
  // user for permission): a field the page made read-only, disabled or
  // removed meanwhile is left alone, as the native paste would leave it.
  await runner.test(
    'a Paste whose clipboard read lands after the field became read-only ' +
      'does not write into it',
    async () => {
      const win = makeDom('');
      const deliver = deferClipboard(win);
      const txt = win.document.getElementById('txt');
      txt.setSelectionRange(1, 1);
      const inputs = [];
      txt.addEventListener('input', () => inputs.push(txt.value));
      activate(win, txt, 'paste');
      txt.readOnly = true;
      deliver('X');
      await settle();
      assert.strictEqual(txt.value, 'ab', 'a read-only field is not written');
      assert.deepStrictEqual(inputs, [], 'and gets no input event');
      assert.deepStrictEqual(unhandled, []);
      assert.deepStrictEqual(listenerErrors, []);
    },
  );

  // The paste was asked for at the selection of that moment.  A
  // selection the user made while the read was pending (the caret
  // moved, other text selected) is not where it was asked for, and the
  // old offsets are nowhere over text the field has changed meanwhile:
  // the clipboard text is dropped then, in a text field and in a
  // contenteditable alike, and lands when nothing moved.
  await runner.test(
    'a Paste whose clipboard read lands after the selection moved pastes nowhere',
    async () => {
      const win = makeDom('');
      const doc = win.document;
      const deliver = deferClipboard(win);
      const txt = doc.getElementById('txt');
      txt.value = 'abcdef';
      const inputs = [];
      txt.addEventListener('input', () => inputs.push(txt.value));
      txt.setSelectionRange(1, 2);
      activate(win, txt, 'paste');
      txt.setSelectionRange(4, 5);
      deliver('X');
      await settle();
      assert.strictEqual(txt.value, 'abcdef', 'the new selection is kept');
      assert.deepStrictEqual(inputs, []);
      activate(win, txt, 'paste');
      txt.value = 'abcdefgh';
      txt.setSelectionRange(1, 2);
      deliver('X');
      await settle();
      assert.strictEqual(txt.value, 'abcdefgh', 'and the changed text');
      assert.deepStrictEqual(inputs, []);
      activate(win, txt, 'paste');
      deliver('X');
      await settle();
      assert.strictEqual(txt.value, 'aXcdefgh', 'an untouched one is pasted');
      assert.strictEqual(txt.selectionStart, 2);
      assert.deepStrictEqual(inputs, ['aXcdefgh']);

      doc.body.insertAdjacentHTML(
        'beforeend',
        '<div id="ce" contenteditable="true">abcdef</div>',
      );
      Object.defineProperty(win.HTMLElement.prototype, 'isContentEditable', {
        configurable: true,
        get() {
          return this.getAttribute('contenteditable') === 'true';
        },
      });
      const ce = doc.getElementById('ce');
      const sel = win.getSelection();
      function select(start, end) {
        const range = doc.createRange();
        range.setStart(ce.firstChild, start);
        range.setEnd(ce.firstChild, end);
        sel.removeAllRanges();
        sel.addRange(range);
      }
      // (jsdom drops the selection when the menu focuses its first
      // item, which Chromium does not: it is installed once the menu
      // is up, before the item is clicked.)
      function pasteAt(start, end) {
        ce.dispatchEvent(
          new win.MouseEvent('contextmenu', {bubbles: true, cancelable: true}),
        );
        select(start, end);
        doc
          .querySelector('#sorcar-content-context-menu [data-action="paste"]')
          .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
      }
      pasteAt(1, 2);
      select(4, 5);
      deliver('X');
      await settle();
      assert.strictEqual(ce.textContent, 'abcdef', 'contenteditable: kept');
      pasteAt(1, 2);
      deliver('X');
      await settle();
      assert.strictEqual(ce.textContent, 'aXcdef', 'contenteditable: pasted');
      assert.deepStrictEqual(unhandled, []);
      assert.deepStrictEqual(listenerErrors, []);
    },
  );

  // A paste asked for in one content tab must not land in another (W1
  // F1).  While the clipboard read is pending the user may switch tabs:
  // the field's view is hidden (display:none, still mounted) and a chat
  // composer focused.  In Chromium focus() on the hidden field fails
  // silently and the insertText command then wrote into the composer
  // (pinned by test_content_context_menu_paste_sweep1026.py); jsdom
  // focuses hidden elements, so here the fallback wrote into the hidden
  // field.  The paste is dropped instead when the field is no longer
  // rendered or the focus has moved to another control, and the focus
  // is never taken back from that control.
  await runner.test(
    'a Paste whose clipboard read lands after the field was hidden or ' +
      'another control focused writes nowhere',
    async () => {
      const win = makeDom('');
      const doc = win.document;
      const deliver = deferClipboard(win);
      doc.body.insertAdjacentHTML(
        'beforeend',
        '<div id="view"></div><textarea id="composer">DRAFT</textarea>',
      );
      const view = doc.getElementById('view');
      const txt = doc.getElementById('txt');
      view.appendChild(txt);
      const composer = doc.getElementById('composer');
      const inputs = [];
      doc.addEventListener('input', e => {
        inputs.push(e.target.id + '=' + e.target.value);
      });
      txt.focus();
      txt.setSelectionRange(2, 2);
      activate(win, txt, 'paste');
      // The tab switch: the view hidden, the composer focused at its end.
      view.style.display = 'none';
      composer.focus();
      composer.setSelectionRange(5, 5);
      deliver('CLIP');
      await settle();
      assert.strictEqual(txt.value, 'ab', 'the hidden field is left alone');
      assert.strictEqual(composer.value, 'DRAFT', 'and so is the composer');
      assert.strictEqual(doc.activeElement, composer, 'the focus stays');
      // The field still shown but the focus moved: dropped, focus kept.
      view.style.display = '';
      txt.focus();
      activate(win, txt, 'paste');
      composer.focus();
      deliver('CLIP');
      await settle();
      assert.strictEqual(txt.value, 'ab', 'a field the focus left');
      assert.strictEqual(doc.activeElement, composer, 'the focus stays');
      // The field hidden, the focus nowhere in particular: dropped.
      txt.focus();
      activate(win, txt, 'paste');
      view.style.display = 'none';
      txt.blur();
      deliver('CLIP');
      await settle();
      assert.strictEqual(txt.value, 'ab', 'a hidden field');
      assert.deepStrictEqual(inputs, [], 'no input event for a dropped paste');
      // Nothing moved: pasted at the caret.
      view.style.display = '';
      txt.focus();
      activate(win, txt, 'paste');
      deliver('CLIP');
      await settle();
      assert.strictEqual(txt.value, 'abCLIP');
      assert.deepStrictEqual(inputs, ['txt=abCLIP']);
      assert.deepStrictEqual(unhandled, []);
      assert.deepStrictEqual(listenerErrors, []);
    },
  );

  // The sandboxed preview iframe (an opaque origin, no clipboard-read
  // delegation) cannot read the clipboard: its Paste is offered disabled
  // instead of doing nothing; the keyboard shortcut shown still works.
  // jsdom does not load a srcdoc iframe, so the preview document is
  // built from the bootstrap markup main.js appends to the srcdoc, at
  // the opaque origin the sandbox gives it.
  await runner.test(
    'in the sandboxed preview document Paste is offered disabled',
    async () => {
      const win = makeDom('');
      const preview = new JSDOM(
        '<!DOCTYPE html><body><textarea id="ta">ab</textarea>' +
          win.ContentContextMenu.contentContextMenuBootstrapHtml() +
          '</body>',
        {runScripts: 'dangerously', pretendToBeVisual: true, url: 'about:srcdoc'},
      );
      const fwin = preview.window;
      windows.push(fwin);
      const fdoc = fwin.document;
      assert.strictEqual(String(fdoc.location.origin), 'null', 'opaque origin');
      const ta = fdoc.getElementById('ta');
      ta.dispatchEvent(
        new fwin.MouseEvent('contextmenu', {
          bubbles: true,
          cancelable: true,
          clientX: 10,
          clientY: 10,
        }),
      );
      const paste = fdoc.querySelector(
        '#sorcar-content-context-menu .sorcar-ctx-item[data-action="paste"]',
      );
      assert.ok(paste, 'Paste is listed');
      assert.strictEqual(
        paste.getAttribute('aria-disabled'),
        'true',
        'Paste is disabled where the clipboard cannot be read',
      );
      assert.strictEqual(paste.textContent.includes('V'), true, 'shortcut shown');
    },
  );
})();

run.then(() => {
  runner.report('contentContextMenuNumberInput_sweep1026');
});
