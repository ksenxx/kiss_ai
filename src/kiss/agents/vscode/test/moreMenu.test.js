// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the composer's "..." overflow menu
// (#more-btn / #more-menu in chat.html) against the real chat.html +
// api.js + main.js:
//  - only one composer popup is open at a time (the "..." menu and the
//    model dropdown close each other),
//  - Escape closes the menu and hands focus back to #more-btn instead
//    of stranding it on a hidden menu item,
//  - outside clicks close the menu,
//  - every required action lives in the menu (mic, share, attach,
//    Git Commit, Settings, theme toggle) and clicking an item closes
//    the menu even when the item's own handler stops propagation
//    (Git Commit does).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function click(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

function testMenuHoldsEveryMovedAction() {
  const {win} = makeWebview();
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
  win.close();
  console.log('  ok - the "..." menu holds every moved action');
}

function testOnlyOneComposerPopupAtATime() {
  const {win} = makeWebview();
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

function testEscapeClosesMenuAndRestoresFocus() {
  const {win} = makeWebview();
  const doc = win.document;
  const moreBtn = doc.getElementById('more-btn');
  const moreMenu = doc.getElementById('more-menu');

  click(win, moreBtn);
  assert.ok(moreMenu.classList.contains('open'), 'menu opens');
  const voiceBtn = doc.getElementById('voice-btn');
  voiceBtn.focus();
  assert.strictEqual(doc.activeElement, voiceBtn, 'item focused (setup)');

  doc.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
  );
  assert.ok(
    !moreMenu.classList.contains('open'),
    'Escape must close the "..." menu',
  );
  assert.strictEqual(moreBtn.getAttribute('aria-expanded'), 'false');
  assert.strictEqual(
    doc.activeElement,
    moreBtn,
    'Escape must hand focus back to the "..." button, not strand it ' +
      'on a hidden menu item',
  );
  win.close();
  console.log('  ok - Escape closes the menu and restores focus');
}

function testOutsideClickAndItemClickCloseMenu() {
  const {win} = makeWebview();
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
  testMenuHoldsEveryMovedAction();
  testOnlyOneComposerPopupAtATime();
  testEscapeClosesMenuAndRestoresFocus();
  testOutsideClickAndItemClickCloseMenu();
  console.log('all ok');
}

main();
process.exit(0);
