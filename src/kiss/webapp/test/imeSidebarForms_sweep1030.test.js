// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for Enter / Escape inside an IME composition
// in the side panel's text boxes (media/main.js).  The key that picks or
// drops a Japanese / Chinese / Korean candidate arrives as a keydown with
// `isComposing` or with keyCode 229 (Safari sends only the latter); it
// must neither submit nor cancel:
// * the Explorer's inline Rename / New File box;
// * a prompt toast ("Find in Folder...": Enter submits, Escape cancels).
// The Schedule "+" form's case (the inline form the Schedule and Machines
// panels share) is in sidebarPanels.test.js, whose coverage gate needs it.

'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();
const IME_FORMS = [{isComposing: true}, {keyCode: 229}];

function keydown(win, el, key, extra) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key,
      bubbles: true,
      cancelable: true,
      ...extra,
    }),
  );
}

function explorerPage() {
  const page = h.makePage();
  h.openExplorer(page.win, page.posted, [
    {name: 'a.txt', path: h.WD + '/a.txt', isDir: false},
    {name: 'dir', path: h.WD + '/dir', isDir: true},
  ]);
  return page;
}

function fsActions(posted, action) {
  return h.ofType(posted, 'fsAction').filter(m => m.action === action);
}

async function main() {
  await test('an IME Enter or Escape neither commits nor drops a rename', () => {
    const {win, posted} = explorerPage();
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/a.txt'), 'Rename...');
    const input = win.document.querySelector('.explorer-row .explorer-input');
    assert.ok(input, 'the rename box is open');
    input.value = 'にほんご';
    for (const form of IME_FORMS) {
      keydown(win, input, 'Enter', form);
      keydown(win, input, 'Escape', form);
    }
    assert.strictEqual(fsActions(posted, 'rename').length, 0);
    assert.ok(input.isConnected, 'the box is still open');
    keydown(win, input, 'Enter');
    const renames = fsActions(posted, 'rename');
    assert.strictEqual(renames.length, 1, 'a plain Enter commits');
    assert.strictEqual(renames[0].dest, h.WD + '/にほんご');
    win.close();
  });

  await test('an IME Enter or Escape neither submits nor cancels a prompt', () => {
    const {win, posted} = explorerPage();
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/dir'), 'Find in Folder...');
    const toastEl = h.toast(win, 'find-in-folder');
    assert.ok(toastEl, 'the prompt is open');
    const input = h.toastInput(toastEl);
    input.value = 'けんさく';
    for (const ime of IME_FORMS) {
      keydown(win, input, 'Enter', ime);
      keydown(win, input, 'Escape', ime);
    }
    assert.strictEqual(fsActions(posted, 'findInFolder').length, 0);
    assert.ok(toastEl.isConnected, 'the prompt is still open');
    keydown(win, input, 'Enter');
    assert.strictEqual(fsActions(posted, 'findInFolder').length, 1);
    win.close();
  });

  report('imeSidebarForms_sweep1030');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
