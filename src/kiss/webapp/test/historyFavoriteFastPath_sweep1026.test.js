// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for the Chats list after a favourite click
// (media/main.js renderHistory / mergePendingFavorites).  A star click
// flips the row optimistically; a refetch that lands before the daemon
// wrote the favourite is overlaid with the click's value (the daemon's
// own value is stashed on the row as `_kissFavDaemon`).  The page is
// then identical to what is shown, so the identical-page fast path must
// keep the row elements: it used to rebuild the whole list once because
// the stash key made the incoming row "differ" from the shown one.

'use strict';

const assert = require('assert');
const {send, stubRaf, ofType} = require('./simplify2_harness');
const {sendHistory, makeRow, openChatsPage} = require('./history_helpers');

function rows(favorite) {
  return [
    makeRow({
      id: 'A',
      task_id: 1,
      chat_summary: 'Old task',
      is_favorite: favorite,
    }),
    makeRow({
      id: 'B',
      task_id: 2,
      chat_summary: 'Other task',
      is_favorite: false,
    }),
  ];
}

function rowEl(win) {
  return win.document.querySelector(
    '#history-list .history-chat-group[data-chat-id="A"] .sidebar-item',
  );
}

function main() {
  const {win, posted} = openChatsPage(stubRaf);
  sendHistory(win, posted, rows(false));
  const row = rowEl(win);
  assert.ok(row, 'the row is rendered');
  const star = row.querySelector('.sidebar-item-favorite');
  assert.ok(star, 'the row has a star');
  star.click();
  assert.ok(star.classList.contains('favorited'), 'the star flipped at once');
  assert.strictEqual(
    ofType(posted, 'setFavorite').length,
    1,
    'the write was sent',
  );

  // A refetch from BEFORE the write landed: the daemon still says no.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, rows(false));
  assert.strictEqual(
    rowEl(win),
    row,
    'the identical page kept the row element',
  );
  assert.ok(
    rowEl(win)
      .querySelector('.sidebar-item-favorite')
      .classList.contains('favorited'),
    'the optimistic star is still shown',
  );

  // The write landed: the daemon agrees; still the same rows on screen.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, rows(true));
  assert.strictEqual(rowEl(win), row, 'the agreeing page kept the row element');
  assert.ok(
    rowEl(win)
      .querySelector('.sidebar-item-favorite')
      .classList.contains('favorited'),
  );
  win.close();
}

function remoteRowEl(win) {
  return win.document.querySelector(
    '#history-list .history-chat-group[data-chat-id="A"] ' +
      '.sidebar-item[data-machine="gpu-box"]',
  );
}

function testARemoteRowOfTheSameTaskIdKeepsItsOwnFlag() {
  // A favourite is a write to THIS daemon's database.  A row of another
  // machine (the MACHINE filter switched within the optimistic window)
  // that happens to carry the same task id neither shows the local
  // star nor acknowledges the local write.
  const {win, posted} = openChatsPage(stubRaf);
  sendHistory(win, posted, rows(false));
  rowEl(win).querySelector('.sidebar-item-favorite').click();
  assert.strictEqual(ofType(posted, 'setFavorite').length, 1);
  const remote = () => [
    makeRow({
      id: 'A',
      task_id: 1,
      chat_summary: 'Remote task',
      is_favorite: false,
      machine: 'gpu-box',
    }),
  ];
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, remote());
  const remoteRow = remoteRowEl(win);
  assert.ok(remoteRow, 'the remote row is rendered');
  assert.strictEqual(
    remoteRow.dataset.favorite,
    '0',
    'BUG: the local favourite was painted onto the remote row',
  );
  // The remote row agreeing with the local toggle is no acknowledgement
  // either: back on the local page, a refetch from before the write
  // still shows the optimistic star.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [Object.assign(remote()[0], {is_favorite: true})]);
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, rows(false));
  assert.ok(
    rowEl(win).querySelector('.sidebar-item-favorite').classList.contains(
      'favorited',
    ),
    'BUG: the remote row consumed the local overlay',
  );
  win.close();
}

main();
testARemoteRowOfTheSameTaskIdKeepsItsOwnFlag();
console.log('historyFavoriteFastPath_sweep1026.test.js: all tests passed');
