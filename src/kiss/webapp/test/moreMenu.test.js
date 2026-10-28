// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the composer's "..." overflow menu
// (#more-btn / #more-menu in chat.html) against the real chat.html +
// api.js + main.js:
//  - only one composer popup is open at a time (the "..." menu and the
//    model dropdown close each other),
//  - outside clicks close the menu (Escape closing it and handing focus
//    back to #more-btn is pinned by escapeClosesTopmostPopup.test.js),
//  - every required action lives in the menu (mic, share, attach,
//    Git Commit, Settings, theme toggle) and clicking an item closes
//    the menu even when the item's own handler stops propagation
//    (Git Commit does).

'use strict';

const assert = require('assert');
const {makePage, click} = require('./simplify2_harness');

// Tests 1 and 2 read the static markup, so they run on the page of
// test 3 before its first click.
function testMenuHoldsEveryMovedAction(win) {
  const menu = win.document.getElementById('more-menu');
  assert.ok(menu, '#more-menu must exist');
  for (const id of [
    'voice-btn',
    'share-btn',
    'upload-btn',
    'autocommit-btn',
    'settings-btn',
    'theme-btn',
  ]) {
    const item = win.document.getElementById(id);
    assert.ok(item, '#' + id + ' must exist');
    assert.ok(
      menu.contains(item),
      '#' + id + ' must live inside the "..." menu',
    );
    assert.strictEqual(
      item.tagName,
      'BUTTON',
      '#' + id + ' must be a native <button>',
    );
  }
  console.log('  ok - the "..." menu holds every moved action');
}

function testSettingsIsTheLastMenuItem(win) {
  const menu = win.document.getElementById('more-menu');
  const items = Array.from(menu.querySelectorAll('.more-menu-item'));
  assert.ok(items.length >= 2, 'the menu must hold several items');
  assert.strictEqual(
    items[items.length - 1].id,
    'settings-btn',
    'Settings must be the LAST item of the "..." menu on every surface',
  );
  console.log('  ok - Settings is the last "..." menu item');
}

function testOnlyOneComposerPopupAtATime(win) {
  const doc = win.document;
  const moreBtn = doc.getElementById('more-btn');
  const moreMenu = doc.getElementById('more-menu');
  const modelBtn = doc.getElementById('model-btn');
  const modelDD = doc.getElementById('model-dropdown');

  // "..." open, then the model button: the menu must yield.
  click(win, moreBtn);
  assert.ok(moreMenu.classList.contains('open'), 'menu opens');
  click(win, modelBtn);
  assert.ok(modelDD.classList.contains('open'), 'model dropdown opens');
  assert.ok(
    !moreMenu.classList.contains('open'),
    'opening the model dropdown must close the "..." menu',
  );
  assert.strictEqual(moreBtn.getAttribute('aria-expanded'), 'false');

  // Model dropdown open, then "...": the dropdown must yield.
  click(win, moreBtn);
  assert.ok(moreMenu.classList.contains('open'), 'menu reopens');
  assert.ok(
    !modelDD.classList.contains('open'),
    'opening the "..." menu must close the model dropdown',
  );
  win.close();
  console.log('  ok - only one composer popup is open at a time');
}

function testOutsideClickAndItemClickCloseMenu() {
  const {win} = makePage();
  const doc = win.document;
  const moreBtn = doc.getElementById('more-btn');
  const moreMenu = doc.getElementById('more-menu');

  click(win, moreBtn);
  assert.ok(moreMenu.classList.contains('open'), 'menu opens');
  click(win, doc.getElementById('task-input'));
  assert.ok(
    !moreMenu.classList.contains('open'),
    'a click outside the menu must close it',
  );

  // Git Commit's own handler calls stopPropagation(); the capture-phase
  // closer must still shut the menu.
  click(win, moreBtn);
  assert.ok(moreMenu.classList.contains('open'), 'menu reopens');
  click(win, doc.getElementById('autocommit-btn'));
  assert.ok(
    !moreMenu.classList.contains('open'),
    'activating Git Commit must close the menu despite stopPropagation',
  );
  win.close();
  console.log('  ok - outside clicks and item clicks close the menu');
}

function main() {
  console.log('moreMenu.test.js');
  const {win} = makePage();
  testMenuHoldsEveryMovedAction(win);
  testSettingsIsTheLastMenuItem(win);
  testOnlyOneComposerPopupAtATime(win);
  testOutsideClickAndItemClickCloseMenu();
  console.log('all ok');
}

main();
process.exit(0);
