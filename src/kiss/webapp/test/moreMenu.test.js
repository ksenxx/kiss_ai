// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the composer's "..." overflow menu
// (#more-btn / #more-menu in chat.html) against the real chat.html +
// api.js + main.js:
//  - only one composer popup is open at a time (the "..." menu and the
//    model dropdown close each other),
//  - outside clicks close the menu, a click on a button of the
//    task-info strip included (Escape closing it and handing focus
//    back to #more-btn is pinned by escapeClosesTopmostPopup.test.js),
//  - the menu holds exactly five actions (inject promptlet, voice
//    trigger, save chat, attach files, iPhone Home Screen); appearance
//    and Settings live on the strip under the Chats list, Terminal,
//    Browser, Git commit and Create pull request on the strip under the
//    task-info panel; clicking an item closes the menu,
//  - the "Inject promptlet" item opens the promptlets sheet,
//  - the "iPhone Home Screen" item navigates the page to
//    /webclip.mobileconfig (jsdom cannot navigate, so the test records
//    what the page hands to location.assign) and closes the menu.

'use strict';

const assert = require('assert');
const {makePage, click} = require('./simplify2_harness');

// Tests 1 and 2 read the static markup, so they run on the page of
// test 3 before its first click.
const MENU_ITEMS = [
  'tricks-btn',
  'voice-btn',
  'share-btn',
  'upload-btn',
  'webclip-btn',
];

function testMenuHoldsExactlyItsFiveActions(win) {
  const menu = win.document.getElementById('more-menu');
  assert.ok(menu, '#more-menu must exist');
  assert.deepStrictEqual(
    Array.from(menu.querySelectorAll('button'), b => b.id),
    MENU_ITEMS,
    'the "..." menu holds these five buttons and no other',
  );
  assert.strictEqual(
    menu.querySelector('#tricks-btn .more-item-label').textContent,
    'Inject promptlet',
  );
  for (const item of menu.querySelectorAll('button')) {
    assert.ok(
      item.classList.contains('more-menu-item'),
      '#' + item.id + ' is a menu item',
    );
    assert.strictEqual(item.getAttribute('role'), 'menuitem');
    assert.ok(
      item.querySelector('.more-item-label').textContent.trim(),
      '#' + item.id + ' has a visible label',
    );
  }
  console.log('  ok - the "..." menu holds exactly its five actions');
}

function testTheOtherActionsLiveOnTheStrips(win) {
  const doc = win.document;
  const menu = doc.getElementById('more-menu');
  const strips = {
    'sidebar-footer': [
      'theme-btn',
      'contrast-btn',
      'theme-auto-btn',
    ],
    'meta-footer': [
      'settings-btn',
      'terminal-btn',
      'browser-btn',
      'autocommit-btn',
      'create-pr-btn',
    ],
  };
  for (const [stripId, ids] of Object.entries(strips)) {
    const strip = doc.getElementById(stripId);
    assert.ok(strip, '#' + stripId + ' must exist');
    for (const id of ids) {
      const btn = doc.getElementById(id);
      assert.ok(btn, '#' + id + ' must exist');
      assert.strictEqual(btn.tagName, 'BUTTON');
      assert.ok(strip.contains(btn), `#${id} lives on #${stripId}`);
      assert.ok(!menu.contains(btn), `#${id} left the "..." menu`);
    }
  }
  console.log('  ok - the actions that left the menu live on the strips');
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

  // An item closes the menu it is in.
  click(win, moreBtn);
  assert.ok(moreMenu.classList.contains('open'), 'menu reopens');
  click(win, doc.getElementById('upload-btn'));
  assert.ok(
    !moreMenu.classList.contains('open'),
    'activating an item must close the menu',
  );

  // Git commit left the menu for the task-info strip: a click on it is
  // a click outside the menu like any other.
  click(win, moreBtn);
  assert.ok(moreMenu.classList.contains('open'), 'menu reopens');
  click(win, doc.getElementById('autocommit-btn'));
  assert.ok(
    !moreMenu.classList.contains('open'),
    'a click on Git commit must close the menu',
  );
  assert.strictEqual(moreBtn.getAttribute('aria-expanded'), 'false');
  win.close();
  console.log('  ok - outside clicks and item clicks close the menu');
}

// jsdom cannot navigate and its Location is unforgeable, so the test
// swaps assign() on the implementation object behind the wrapper (the
// wrapper's "impl" symbol property) to record each destination.
function recordNavigations(win) {
  const destinations = [];
  const loc = win.location;
  const impl = Object.getOwnPropertySymbols(loc).find(
    s => s.description === 'impl',
  );
  assert.ok(impl, 'jsdom must expose the Location implementation');
  loc[impl].assign = url => destinations.push(url);
  return destinations;
}

function testWebclipItemNavigatesAndClosesMenu() {
  const {win} = makePage();
  const navigations = recordNavigations(win);
  const doc = win.document;
  const menu = doc.getElementById('more-menu');
  const item = doc.getElementById('webclip-btn');
  assert.ok(item && menu.contains(item), '#webclip-btn must live in the menu');
  assert.strictEqual(item.tagName, 'BUTTON');
  assert.strictEqual(
    item.querySelector('.more-item-label').textContent,
    'iPhone Home Screen',
  );
  click(win, doc.getElementById('more-btn'));
  assert.ok(menu.classList.contains('open'), 'menu opens');
  assert.deepStrictEqual(navigations, [], 'no navigation before the click');
  click(win, item);
  assert.deepStrictEqual(
    navigations,
    ['/webclip.mobileconfig'],
    'the click must navigate the page to the profile route, once',
  );
  assert.ok(!menu.classList.contains('open'), 'the click closes the menu');
  win.close();
  console.log('  ok - the iPhone Home Screen item navigates and closes the menu');
}

function main() {
  console.log('moreMenu.test.js');
  const {win} = makePage();
  testMenuHoldsExactlyItsFiveActions(win);
  testTheOtherActionsLiveOnTheStrips(win);
  testOnlyOneComposerPopupAtATime(win);
  testOutsideClickAndItemClickCloseMenu();
  testWebclipItemNavigatesAndClosesMenu();
  console.log('all ok');
}

main();
