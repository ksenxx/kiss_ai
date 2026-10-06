// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM, stub Monaco) tests for two corners of the two-row
// tab bar (see tabGroupsTwoRows.test.js for the rows themselves):
//
//   * a file opened FROM a file (the Explorer names the tab on screen as
//     the owner) stays in the chat's group when that file is closed:
//     closeContentTab hands it to the closed file's own owner;
//   * a dialog dismissed over a top-level file (its owning chat is gone,
//     so the group strip is hidden) hands focus back to the file's entry
//     on the main row, the row the user can see.

'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

function stripIds(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab[data-tab-id]'),
  ).map(el => el.dataset.tabId);
}

function mainIds(win) {
  return Array.from(
    win.document.querySelectorAll('#main-tab-list .chat-tab[data-tab-id]'),
  ).map(el => el.dataset.tabId);
}

function contentTabIdNamed(win, name) {
  const el = Array.from(
    win.document.querySelectorAll('.chat-tab.content-tab[data-tab-id]'),
  ).find(e => e.textContent.indexOf(name) >= 0);
  return el ? el.dataset.tabId : null;
}

async function testFileOpenedFromAFileStaysInTheGroupWhenThatFileCloses() {
  const ctx = h.makeWebview();
  const {win} = ctx;
  h.send(win, {type: 'configData', config: {work_dir: '/ws/a'}});
  h.send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', workDir: '/ws/a', chatId: 'chat-1'}],
  });
  h.send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/a/x.txt',
    name: 'x.txt',
    content: 'x',
  });
  const x = contentTabIdNamed(win, 'x.txt');
  assert.ok(x, 'x.txt opened');
  assert.strictEqual(win._testApi.getActiveTabId(), x, 'x.txt is on screen');
  // Opened while x.txt is on screen: the Explorer sends the active tab
  // (x.txt) as the owner.
  h.send(win, {
    type: 'fileContent',
    tabId: x,
    path: '/ws/a/y.txt',
    name: 'y.txt',
    content: 'y',
  });
  const y = contentTabIdNamed(win, 'y.txt');
  assert.ok(y, 'y.txt opened');
  assert.deepStrictEqual(
    stripIds(win),
    ['a1', x, y],
    'both files in the chat group',
  );
  assert.deepStrictEqual(mainIds(win), ['a1']);

  h.click(
    win,
    win.document.querySelector(
      `#tab-list .chat-tab[data-tab-id="${x}"] .chat-tab-close`,
    ),
  );
  assert.deepStrictEqual(
    mainIds(win),
    ['a1'],
    'closing x.txt must not turn y.txt into a top-level tab',
  );
  assert.deepStrictEqual(
    stripIds(win),
    ['a1', y],
    'y.txt moved up to the chat that x.txt belonged to',
  );
  win.close();
  console.log(
    '  ok - a file opened from a file stays in the group when that file closes',
  );
}

async function testKeepEditingOverATopLevelFileFocusesItsMainRowEntry() {
  const ctx = h.makeWebview();
  const {win} = ctx;
  await h.openDirtyContentTab(ctx);
  const file = contentTabIdNamed(win, 'notes.txt');
  assert.ok(file, 'notes.txt opened');
  // Its owning chat goes: the file is now a top-level tab on its own,
  // so the group strip is hidden.
  h.click(
    win,
    win.document.querySelector(
      '#main-tab-list .chat-tab[data-tab-id="a1"] .chat-tab-close',
    ),
  );
  assert.deepStrictEqual(mainIds(win), ['b1', file]);
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    file,
    'the file stays on screen',
  );
  assert.strictEqual(
    win.document.getElementById('tab-bar').style.display,
    'none',
    'a lone top-level file shows no strip',
  );

  const fileMain = win.document.querySelector(
    `#main-tab-list .chat-tab[data-tab-id="${file}"]`,
  );
  // A keyboard user is on the close control when the question opens;
  // the rows re-render meanwhile, so that control is gone by the time
  // the dialog closes and focus has to fall back.
  const closeBtn = fileMain.querySelector('.chat-tab-close');
  closeBtn.focus();
  h.click(win, closeBtn);
  const toast = win.document.querySelector(
    '[data-notification-id^="close-dirty-"]',
  );
  assert.ok(toast, 'closing the dirty file asks first');
  h.click(win, h.toastButton(toast, 'Keep editing'));
  assert.strictEqual(
    win.document.activeElement,
    win.document.querySelector(
      `#main-tab-list .chat-tab[data-tab-id="${file}"]`,
    ),
    'focus returns to the file\u2019s entry on the visible (main) row',
  );
  win.close();
  console.log(
    '  ok - Keep editing over a top-level file focuses its main-row entry',
  );
}

async function main() {
  await testFileOpenedFromAFileStaysInTheGroupWhenThatFileCloses();
  await testKeepEditingOverATopLevelFileFocusesItsMainRowEntry();
  console.log('tabGroupsContentOwnership.test.js: all tests passed');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
