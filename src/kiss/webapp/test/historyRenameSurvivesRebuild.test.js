// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A chat rename box open in the Chats panel must survive a history
// refresh that changes the rows (a `tasks_updated` broadcast lands
// whenever ANY task on the daemon persists a result).  The rebuild is
// parked, as it is under a held mouse button, until the rename ends;
// the rename's Enter then saves the text and the parked page lands.
// Runs with: node test/historyRenameSurvivesRebuild.test.js

const assert = require('assert');
const {send, ofType, settle: tick} = require('./simplify2_harness');
const {
  sendHistory,
  makeRow,
  openChatsPage: makePage,
  chatHeader,
} = require('./history_helpers');

/** Chat `chatId`'s panel in the Chats list, or null. */
function group(win, chatId) {
  return win.document.querySelector(
    '#history-list .history-chat-group[data-chat-id="' + chatId + '"]',
  );
}

function title(win, chatId) {
  return chatHeader(win, chatId).querySelector(':scope > .history-chat-title');
}

function key(win, el, k) {
  el.dispatchEvent(new win.KeyboardEvent('keydown', {key: k, bubbles: true}));
}

async function testRenameBoxSurvivesChangedRows() {
  const {win, posted} = makePage();
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, chat_summary: 'Fix the login bug'}),
  ]);
  win.document.querySelector('#history-list .history-chat-rename').click();
  const input = title(win, 'A').querySelector('input');
  assert.ok(input, 'setup: the pencil opened the rename box');
  input.value = 'Login fix';

  // Another task finished elsewhere: the daemon broadcasts, the client
  // refetches, the reply carries a new row.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'B', task_id: 2, chat_summary: 'Other chat'}),
    makeRow({id: 'A', task_id: 1, chat_summary: 'Fix the login bug'}),
  ]);
  await tick();
  assert.strictEqual(
    title(win, 'A').querySelector('input'),
    input,
    'the rename box must survive the refresh',
  );
  assert.strictEqual(input.value, 'Login fix', 'the typed text is kept');
  assert.strictEqual(group(win, 'B'), null, 'the new row waits');

  key(win, input, 'Enter');
  assert.deepStrictEqual(
    JSON.parse(JSON.stringify(ofType(posted, 'setChatSummary'))),
    [{type: 'setChatSummary', chatId: 'A', summary: 'Login fix'}],
  );
  await tick();
  assert.ok(group(win, 'B'), 'the parked page lands once the rename ends');
  assert.strictEqual(title(win, 'A').textContent, 'Login fix');
  assert.strictEqual(
    title(win, 'A').querySelector('input'),
    null,
    'no rename box after Enter',
  );
  win.close();
  console.log('  ok - the rename box survives a refresh with changed rows');
}

/** A cancelled rename (Escape) lands the parked page as a save does. */
async function testEscapeLandsTheParkedPage() {
  const {win, posted} = makePage();
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, chat_summary: 'Fix the login bug'}),
  ]);
  win.document.querySelector('#history-list .history-chat-rename').click();
  const input = title(win, 'A').querySelector('input');
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [
    makeRow({id: 'B', task_id: 2, chat_summary: 'Other chat'}),
    makeRow({id: 'A', task_id: 1, chat_summary: 'Fix the login bug'}),
  ]);
  await tick();
  assert.strictEqual(group(win, 'B'), null, 'setup: the page is parked');
  key(win, input, 'Escape');
  await tick();
  assert.strictEqual(ofType(posted, 'setChatSummary').length, 0);
  assert.ok(group(win, 'B'), 'Escape lands the parked page');
  assert.strictEqual(title(win, 'A').textContent, 'Fix the login bug');
  win.close();
  console.log('  ok - Escape lands the parked page');
}

/**
 * The identical-page fast path never rebuilt, so the box always lived
 * through it; a refresh returning the same rows keeps the box too.
 */
async function testSameRowsKeepTheBox() {
  const {win, posted} = makePage();
  const rows = [makeRow({id: 'A', task_id: 1, chat_summary: 'Fix it'})];
  sendHistory(win, posted, rows);
  win.document.querySelector('#history-list .history-chat-rename').click();
  const input = title(win, 'A').querySelector('input');
  send(win, {type: 'tasks_updated'});
  sendHistory(
    win,
    posted,
    rows.map(r => Object.assign({}, r)),
  );
  await tick();
  assert.strictEqual(title(win, 'A').querySelector('input'), input);
  win.close();
  console.log('  ok - a refresh with the same rows keeps the box');
}

async function main() {
  await testRenameBoxSurvivesChangedRows();
  await testEscapeLandsTheParkedPage();
  await testSameRowsKeepTheBox();
  console.log('historyRenameSurvivesRebuild: all passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
