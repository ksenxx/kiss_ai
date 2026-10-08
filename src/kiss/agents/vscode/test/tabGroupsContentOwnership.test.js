// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM, stub Monaco) tests for two corners of content-tab
// ownership (see tabGroupsTwoRows.test.js for the rows themselves):
//
//   * a file opened FROM a file (the Explorer names the tab on screen as
//     the owner) stays in the chat's group when that file is closed:
//     closeContentTab hands it to the closed file's own owner.  Checked
//     on both layouts: the desktop remote's split layout, where files
//     sit on the content pane's row (#content-tab-list), and a stacked
//     surface, where they share the group strip (#tab-list);
//   * a dialog dismissed over a top-level file in the content pane (its
//     owning chat was retired, so the file owns itself) hands focus back
//     to the composer, never to BODY.

'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

/** The tab row a content tab is listed on: the content pane's row in the
 *  split layout, the group strip on a stacked surface. */
function contentRow(stacked) {
  return stacked ? '#tab-list' : '#content-tab-list';
}

function rowIds(win, row) {
  return Array.from(
    win.document.querySelectorAll(`${row} .chat-tab[data-tab-id]`),
  ).map(el => el.dataset.tabId);
}

function contentTabIdNamed(win, name) {
  const el = Array.from(
    win.document.querySelectorAll('.chat-tab.content-tab[data-tab-id]'),
  ).find(e => e.textContent.indexOf(name) >= 0);
  return el ? el.dataset.tabId : null;
}

/** openTabs() record of *id*, or undefined once the tab is closed. */
function tabRecord(win, id) {
  return win._testApi.openTabs().find(t => t.id === id);
}

async function testFileOpenedFromAFileStaysInTheGroupWhenThatFileCloses(
  stacked,
) {
  const label = stacked ? 'stacked' : 'split';
  const ctx = h.makeWebview({narrow: stacked});
  const {win} = ctx;
  const row = contentRow(stacked);
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
  assert.ok(x, `${label}: x.txt opened`);
  // A stacked surface shows the file in place of the chat; the split
  // layout keeps the chat on screen and shows the file beside it.
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    stacked ? x : 'a1',
    `${label}: the tab on screen after opening x.txt`,
  );
  assert.strictEqual(tabRecord(win, x).rootId, 'a1', `${label}: x.txt in a1`);
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
  assert.ok(y, `${label}: y.txt opened`);
  assert.strictEqual(
    tabRecord(win, y).rootId,
    'a1',
    `${label}: y.txt belongs to the chat group through x.txt`,
  );
  assert.deepStrictEqual(
    rowIds(win, row),
    stacked ? ['a1', x, y] : [x, y],
    `${label}: both files on the row`,
  );

  h.click(
    win,
    win.document.querySelector(
      `${row} .chat-tab[data-tab-id="${x}"] .chat-tab-close`,
    ),
  );
  assert.strictEqual(tabRecord(win, x), undefined, `${label}: x.txt closed`);
  assert.strictEqual(
    tabRecord(win, y).rootId,
    'a1',
    `${label}: closing x.txt must not turn y.txt into a top-level tab`,
  );
  assert.deepStrictEqual(
    rowIds(win, row),
    stacked ? ['a1', y] : [y],
    `${label}: y.txt moved up to the chat that x.txt belonged to`,
  );
  win.close();
  console.log(
    `  ok - a file opened from a file stays in the group when that file closes (${label})`,
  );
}

async function testKeepEditingOverATopLevelFileFocusesTheComposer() {
  const ctx = h.makeWebview();
  const {win, posted} = ctx;
  await h.openDirtyContentTab(ctx);
  const file = contentTabIdNamed(win, 'notes.txt');
  assert.ok(file, 'notes.txt opened');
  assert.strictEqual(tabRecord(win, file).rootId, 'a1');
  // Its owning chat goes: "+" opens a fresh chat and retires the idle
  // chat left behind (idle for sure: the daemon's replay said so), so
  // the file is now a top-level tab on its own, still shown in the
  // content pane.
  h.send(win, {type: 'status', running: false, tabId: 'a1'});
  posted.length = 0;
  h.click(win, win.document.getElementById('new-chat-btn'));
  const fresh = win._testApi.getActiveTabId();
  assert.notStrictEqual(fresh, 'a1', 'a fresh chat is on screen');
  assert.ok(
    posted.some(m => m.type === 'closeTab' && m.tabId === 'a1'),
    'the idle chat left behind is retired',
  );
  assert.strictEqual(tabRecord(win, 'a1'), undefined, 'a1 is gone');
  assert.strictEqual(
    tabRecord(win, file).rootId,
    file,
    'the file is a top-level tab on its own',
  );
  assert.deepStrictEqual(
    rowIds(win, '#content-tab-list'),
    [file],
    'the file stays on the content pane row',
  );

  const entry = win.document.querySelector(
    `#content-tab-list .chat-tab[data-tab-id="${file}"]`,
  );
  // The user is on the close control when the question opens; the
  // click's blur (a mouse click drops the focus ring) means the dialog
  // has no usable opener to go back to and focus has to fall back.
  const closeBtn = entry.querySelector('.chat-tab-close');
  closeBtn.focus();
  h.click(win, closeBtn);
  const toast = win.document.querySelector(
    '[data-notification-id^="close-dirty-"]',
  );
  assert.ok(toast, 'closing the dirty file asks first');
  h.click(win, h.toastButton(toast, 'Keep editing'));
  assert.ok(tabRecord(win, file), 'Keep editing leaves the file open');
  assert.strictEqual(
    win.document.activeElement,
    win.document.getElementById('task-input'),
    'focus returns to the composer (visible beside the content pane)',
  );
  win.close();
  console.log(
    '  ok - Keep editing over a top-level file in the content pane focuses the composer',
  );
}

async function main() {
  await testFileOpenedFromAFileStaysInTheGroupWhenThatFileCloses(false);
  await testFileOpenedFromAFileStaysInTheGroupWhenThatFileCloses(true);
  await testKeepEditingOverATopLevelFileFocusesTheComposer();
  console.log('tabGroupsContentOwnership.test.js: all tests passed');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
