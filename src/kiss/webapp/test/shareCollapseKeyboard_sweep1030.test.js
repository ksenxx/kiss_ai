// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests of media/share.js on the document a shared chat page holds.
// The export is the live DOM's outerHTML, so every panel header keeps the
// attributes main.js gave it (tabindex="0", role="button",
// aria-expanded) but none of its listeners.  Before the fix the shared
// page re-created only the mouse click: Enter / Space on a focused header
// did nothing, so the content of a folded panel was out of reach without
// a mouse, and aria-expanded kept its export-time value after every
// toggle.  Each test failed against the pre-fix share.js.

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {click, makeSharePage} = require('./share_helpers');

const {test, report} = makeRunner();

/** A panel header as main.js addCollapse renders it, expanded. */
function header(title, extra) {
  return (
    '<div class="tc-h collapse-header" tabindex="0" role="button" ' +
    'aria-expanded="true"><span class="collapse-chv">\u25BE</span>' +
    title +
    (extra || '') +
    '<span class="collapse-preview"></span></div>'
  );
}

function key(win, el, name) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: name,
      bubbles: true,
      cancelable: true,
    }),
  );
}

/** One expanded tool panel, as exported. */
function onePanelPage() {
  const win = makeSharePage(
    '<div id="output"><div class="ev tc collapsible">' +
      header('Bash', '<button class="panel-copy-btn">Copy</button>') +
      '<div class="tc-b">echo hi</div></div></div>',
  );
  const panel = win.document.querySelector('.collapsible');
  return {win, panel, head: panel.querySelector('.collapse-header')};
}

async function main() {
  await test('Enter and Space on a focused header fold and unfold it', () => {
    const {win, panel, head} = onePanelPage();
    key(win, head, 'Enter');
    assert.ok(panel.classList.contains('collapsed'), 'Enter folds the panel');
    assert.strictEqual(head.getAttribute('aria-expanded'), 'false');
    key(win, head, ' ');
    assert.ok(!panel.classList.contains('collapsed'), 'Space unfolds it');
    assert.strictEqual(head.getAttribute('aria-expanded'), 'true');
    win.close();
  });

  await test('a mouse toggle keeps aria-expanded true to the state', () => {
    const {win, panel, head} = onePanelPage();
    click(head);
    assert.ok(panel.classList.contains('collapsed'));
    assert.strictEqual(head.getAttribute('aria-expanded'), 'false');
    win.close();
  });

  await test('keys on a button inside the header or other keys never toggle', () => {
    const {win, panel, head} = onePanelPage();
    key(win, head.querySelector('.panel-copy-btn'), 'Enter');
    key(win, head, 'a');
    assert.ok(!panel.classList.contains('collapsed'));
    assert.strictEqual(head.getAttribute('aria-expanded'), 'true');
    win.close();
  });

  await test('a fan-out folded with its parent reports itself folded', () => {
    const win = makeSharePage(
      '<div id="output"><div class="ev tc collapsible">' +
        header('outer') +
        '<div class="tc-b"><div class="ev tc tc-run-parallel collapsible">' +
        header('fan-out') +
        '<div class="tc-b">parallel work</div></div></div></div></div>',
    );
    const outer = win.document.querySelector('#output > .collapsible');
    const nestedHead = win.document.querySelector(
      '.tc-run-parallel > .collapse-header',
    );
    key(win, outer.querySelector(':scope > .collapse-header'), 'Enter');
    assert.strictEqual(nestedHead.getAttribute('aria-expanded'), 'false');
    win.close();
  });

  await test("a nested panel's header ahead of the panel's own keeps its state", () => {
    // addCollapse takes the header as given, so a panel may hold a nested
    // panel before its own header; only the panel's own header follows it.
    const win = makeSharePage(
      '<div id="output"><div class="ev tc collapsible">' +
        '<div class="tc-b"><div class="ev tc collapsible">' +
        header('inner') +
        '<div class="tc-b">inner work</div></div></div>' +
        header('outer') +
        '</div></div>',
    );
    const outer = win.document.querySelector('#output > .collapsible');
    const innerHead = outer.querySelector('.tc-b .collapse-header');
    const outerHead = outer.querySelector(':scope > .collapse-header');
    key(win, outerHead, 'Enter');
    assert.ok(outer.classList.contains('collapsed'));
    assert.strictEqual(outerHead.getAttribute('aria-expanded'), 'false');
    assert.strictEqual(innerHead.getAttribute('aria-expanded'), 'true');
    win.close();
  });

  report('shareCollapseKeyboard_sweep1030');
}

main();
