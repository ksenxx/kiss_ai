// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// One Escape closes one popup, the topmost: with the Chats drawer and
// the "..." menu both open, the first press closes only the menu (and
// hands focus back to its button) and the second closes the drawer;
// the server-reset confirm sits above the menu.  Before, every popup
// had its own document-level Escape handler and all of them closed on
// the same press.

const assert = require('assert');
const {makePage} = require('./simplify2_harness');

function escape(win) {
  win.document.body.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'Escape',
      bubbles: true,
      cancelable: true,
    }),
  );
}

function testMenuClosesBeforeTheDrawer() {
  const {win} = makePage();
  const doc = win.document;
  const sidebar = doc.getElementById('sidebar');
  const moreBtn = doc.getElementById('more-btn');
  const moreMenu = doc.getElementById('more-menu');
  sidebar.classList.add('open');
  moreBtn.click();
  assert.ok(moreMenu.classList.contains('open'), 'setup: menu open');
  moreMenu.querySelector('button').focus();

  escape(win);
  assert.ok(!moreMenu.classList.contains('open'), 'the menu closed');
  assert.strictEqual(doc.activeElement, moreBtn, 'focus went to the button');
  assert.ok(sidebar.classList.contains('open'), 'the drawer stayed open');

  escape(win);
  assert.ok(!sidebar.classList.contains('open'), 'the drawer closed');
  win.close();
  console.log('  ok - Escape closes the menu, then the drawer');
}

function testServerResetConfirmClosesBeforeTheMenu() {
  const {win} = makePage();
  const doc = win.document;
  const modal = doc.getElementById('server-reset-confirm-modal');
  const moreBtn = doc.getElementById('more-btn');
  const moreMenu = doc.getElementById('more-menu');
  moreBtn.click();
  modal.classList.add('open');

  escape(win);
  assert.ok(!modal.classList.contains('open'), 'the confirm closed');
  assert.ok(moreMenu.classList.contains('open'), 'the menu stayed open');
  escape(win);
  assert.ok(!moreMenu.classList.contains('open'), 'then the menu closed');
  win.close();
  console.log('  ok - the server-reset confirm closes before the menu');
}

testMenuClosesBeforeTheDrawer();
testServerResetConfirmClosesBeforeTheMenu();
console.log('escapeClosesTopmostPopup: all passed');
