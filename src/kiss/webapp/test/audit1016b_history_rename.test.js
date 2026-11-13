// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Renaming a chat from the Chats panel (audit 2026-10-16 mainjs-3 F4,
// regression-diff F1): a double-click on the title opens the rename box
// and opens the chat ONCE (the two clicks of the gesture are not two
// header clicks); a chat whose rows all live on another machine is
// renamed on that machine (setChatSummary stamped `machine`).
// Runs with: node test/audit1016b_history_rename.test.js

const assert = require('assert');
const {ofType, plain, recordTimers} = require('./simplify2_harness');
const {key} = require('./ui_antipattern_harness');
const {
  makeRow,
  sendHistory,
  openChatsPage,
  chatHeader,
  chatTabs,
  freezeClock,
} = require('./history_helpers');

function title(win, chatId) {
  return chatHeader(win, chatId).querySelector(':scope > .history-chat-title');
}

function pencil(win, chatId) {
  return chatHeader(win, chatId).querySelector('.history-chat-rename');
}

function pencilShown(win, chatId) {
  const p = pencil(win, chatId);
  return !!p && win.getComputedStyle(p).display !== 'none';
}

function mouse(win, el, type, detail) {
  el.dispatchEvent(
    new win.MouseEvent(type, {bubbles: true, cancelable: true, detail}),
  );
}

/**
 * A double-click is click (detail 1), click (detail 2), dblclick.  The
 * first click opens the chat as a single click does; the second must
 * not open it AGAIN (a second new tab, a second newChat+resumeSession
 * and a closeTab of the first), and the dblclick opens the rename box.
 */
function testDoubleClickOpensTheChatOnce() {
  const {win, posted} = openChatsPage();
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, chat_summary: 'Fix the login bug'}),
  ]);
  const welcomeTab = win._testApi.getActiveTabId();
  const newChatsBefore = ofType(posted, 'newChat').length;
  const t = title(win, 'A');
  mouse(win, t, 'click', 1);
  mouse(win, t, 'click', 2);
  mouse(win, t, 'dblclick', 2);
  assert.strictEqual(
    ofType(posted, 'newChat').length - newChatsBefore,
    1,
    'one newChat frame for the gesture',
  );
  assert.strictEqual(
    ofType(posted, 'resumeSession').length,
    1,
    'the chat is resumed once',
  );
  // Leaving the idle welcome chat retires it (by design); the tab the
  // first click opened must not be retired by a second opening.
  assert.deepStrictEqual(
    ofType(posted, 'closeTab').map(m => m.tabId),
    [welcomeTab],
    'only the idle welcome tab was closed',
  );
  const open = chatTabs(win);
  assert.strictEqual(open.length, 1, 'one chat tab is open');
  assert.notStrictEqual(open[0].id, welcomeTab);
  assert.ok(
    title(win, 'A').querySelector('input'),
    'the double-click opened the rename box',
  );
  win.close();
  console.log('  ok - a double-click on the title opens the chat once');
}

/**
 * Every chat shows the pencil.  A chat with rows only from a remote
 * machine sends its setChatSummary to that machine, whose history.db
 * holds the rows; a chat with a local row (alone or mixed with remote
 * rows) is renamed on this daemon.
 */
function testRemoteOnlyChatIsRenamedOnItsMachine() {
  const {win, posted} = openChatsPage();
  sendHistory(win, posted, [
    makeRow({id: 'R', task_id: 1, machine: 'boxb', chat_summary: 'Remote'}),
    makeRow({id: 'R', task_id: 2, machine: 'boxb', chat_summary: 'Remote'}),
    makeRow({id: 'M', task_id: 3, machine: 'boxb', chat_summary: 'Mixed'}),
    makeRow({id: 'M', task_id: 4, chat_summary: 'Mixed'}),
    makeRow({id: 'L', task_id: 5, chat_summary: 'Local'}),
  ]);
  assert.strictEqual(pencilShown(win, 'R'), true, 'remote-only chat has it');
  assert.strictEqual(pencilShown(win, 'M'), true, 'mixed chat has it');
  assert.strictEqual(pencilShown(win, 'L'), true, 'local chat has it');
  rename(win, 'R', 'Remote, renamed');
  // The mixed chat renames as a local one does.
  rename(win, 'M', 'Mixed, renamed');
  assert.deepStrictEqual(plain(ofType(posted, 'setChatSummary')), [
    {
      type: 'setChatSummary',
      chatId: 'R',
      summary: 'Remote, renamed',
      machine: 'boxb',
    },
    {type: 'setChatSummary', chatId: 'M', summary: 'Mixed, renamed'},
  ]);
  assert.strictEqual(title(win, 'R').textContent, 'Remote, renamed');
  win.close();
  console.log('  ok - a remote-only chat is renamed on its machine');
}

// Audit 2026-11-08 (W8 F2): a rename the daemon could not write (no
// broadcast follows) shows the retyped title for its ten-second bound
// only: the rename itself arms the expiry, which asks for the refresh
// that paints the stored title.
function testAFailedRenameExpiresItsRetypedTitle() {
  const {win, posted} = openChatsPage();
  const now = 1700000000000;
  freezeClock(win, now);
  // The expiry: HISTORY_TITLE_OVERLAY_MS + 100 after the rename.
  const timers = recordTimers(win, 10100);
  sendHistory(win, posted, [makeRow({id: 'L', task_id: 5, chat_summary: 'Local'})]);
  const asked = ofType(posted, 'getHistory').length;
  rename(win, 'L', 'Refused title');
  assert.strictEqual(title(win, 'L').textContent, 'Refused title', 'shown at once');
  assert.strictEqual(
    timers.length,
    1,
    'BUG - the rename armed no expiry for its retyped title',
  );
  // Nothing came back in ten seconds: the expiry asks for a refresh...
  win.Date.now = () => now + 10100;
  timers[0].fn();
  assert.strictEqual(ofType(posted, 'getHistory').length, asked + 1, 'a refresh is asked');
  // ...whose stored title replaces the refused one.
  sendHistory(win, posted, [makeRow({id: 'L', task_id: 5, chat_summary: 'Local'})]);
  assert.strictEqual(title(win, 'L').textContent, 'Local');
  win.close();
  console.log('  ok - a failed rename shows its title for ten seconds only');
}

/** Retype *chatId*'s title as *text* through a double-click and Enter. */
function rename(win, chatId, text) {
  mouse(win, title(win, chatId), 'dblclick', 2);
  const input = title(win, chatId).querySelector('input');
  assert.ok(input, chatId + ': the double-click opens the rename box');
  input.value = text;
  key(win, input, 'Enter');
}

testDoubleClickOpensTheChatOnce();
testRemoteOnlyChatIsRenamedOnItsMachine();
testAFailedRenameExpiresItsRetypedTitle();
console.log('audit1016b_history_rename.test.js: all tests passed');
