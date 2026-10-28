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
const {makeRunner} = require('./ui_antipattern_harness');
const {settle} = require('./simplify2_harness');
const runner = makeRunner();

const MEDIA = path.join(__dirname, '..', 'media');
const windows = [];

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

function activate(win, target, action) {
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
  assert.ok(item, action + ' item is offered');
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
})();

run.then(() => {
  while (windows.length) windows.pop().close();
  runner.report('contentContextMenuNumberInput_sweep1026');
});
