// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Three additions to the chat page, tested against the real
// chat.html + main.js + panelCopy.js in JSDOM:
//
//  * every task panel in the task-history list shows its classification
//    tags ("work · coding") right after its "... ago" age label;
//  * every collapsible chat panel carries a "... ago" line under its
//    header counting from the chat's newest instant, fed by the daemon's
//    `chat_last_launched` / `chat_last_finished` stamps (falling back to
//    the newest row loaded) and kept current by the same 30 s sweep as
//    the task labels; its tooltip says "Finished <date>" once the newest
//    task is done, "Started <date>" while it runs, and the bare date
//    when only a launch stamp is known.
// (The thoughts panel's llm_call cost is panelCost.test.js.)

'use strict';

/* global require, console, process */

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  loadMainCss,
  captureTimers,
} = require('./simplify2_harness');
const {
  historyRows: rows,
  loadHistory,
  makeSession: session,
} = require('./history_helpers');

const HOUR_MS = 3600 * 1000;

function makePage() {
  // Every setTimeout the page arms, so a test can fire a long timer
  // (the 30 s "launched ... ago" sweep) without waiting for it.
  let timers;
  const {win, posted} = makeChatPage({
    beforeScripts(win) {
      stubRaf(win);
      loadMainCss(win);
      timers = captureTimers(win);
    },
  });
  return {win, posted, timers};
}

function groups(win) {
  return Array.from(
    win.document.querySelectorAll('#history-list > .history-chat-group'),
  );
}

function testTagsAfterLaunched() {
  const {win} = makePage();
  const now = Date.now();
  loadHistory(win, [
    session({
      id: 'chat-a',
      task_id: 'task-a',
      startTs: now - 2 * HOUR_MS,
      tags: 'work,coding, testing ,',
    }),
    session({
      id: 'chat-b',
      task_id: 'task-b',
      startTs: now - HOUR_MS,
      tags: '',
    }),
    session({id: 'chat-c', task_id: 'task-c', startTs: now - HOUR_MS}),
    session({
      id: 'chat-d',
      task_id: 'task-d',
      startTs: now - HOUR_MS,
      tags: 42,
    }),
  ]);
  const list = rows(win);
  assert.strictEqual(list.length, 4, 'one row per session');
  const tagged = list[0].querySelector('.sidebar-item-tags');
  assert.ok(tagged, 'a tagged row renders the tags span');
  assert.strictEqual(tagged.textContent, 'work · coding · testing');
  assert.strictEqual(tagged.title, 'Tags: work, coding, testing');
  const launched = list[0].querySelector('.sidebar-item-launched');
  assert.strictEqual(
    launched.nextElementSibling,
    tagged,
    'the tags sit immediately after the "... ago" age label',
  );
  assert.strictEqual(
    list[0].querySelector('.sidebar-item-collapse').nextElementSibling,
    launched,
    'the age label follows the last button of the strip',
  );
  assert.strictEqual(launched.textContent, '2 hours ago');
  assert.ok(
    launched.title.indexOf('Launched ') === 0,
    'the tooltip still spells out the launch instant',
  );
  assert.strictEqual(
    tagged.parentElement,
    list[0].querySelector('.sidebar-item-actions'),
    'the tags live in the action strip like the age label',
  );
  for (let i = 1; i < 4; i++) {
    assert.strictEqual(
      list[i].querySelector('.sidebar-item-tags'),
      null,
      `row ${i}: no tags span without usable tags`,
    );
    assert.ok(list[i].querySelector('.sidebar-item-launched'));
  }
  win.close();
  console.log('  ok - task rows show their tags after the age label');
}

function testChatLastLaunchedLine() {
  const {win} = makePage();
  const now = Date.now();
  const stamped = now - 3 * HOUR_MS;
  loadHistory(win, [
    // Newest row first, as the daemon sends them; both rows of chat-a
    // carry the chat's stamp.
    session({
      id: 'chat-a',
      task_id: 'task-a2',
      startTs: now - 5 * HOUR_MS,
      timestamp: Math.floor((now - 5 * HOUR_MS) / 1000),
      chat_last_launched: stamped,
    }),
    session({
      id: 'chat-a',
      task_id: 'task-a1',
      startTs: now - 30 * HOUR_MS,
      timestamp: Math.floor((now - 30 * HOUR_MS) / 1000),
      chat_last_launched: stamped,
    }),
    // An older daemon: no stamp, so the newest row loaded decides.
    session({
      id: 'chat-b',
      task_id: 'task-b2',
      startTs: now - 26 * HOUR_MS,
      timestamp: Math.floor((now - 26 * HOUR_MS) / 1000),
    }),
    session({
      id: 'chat-b',
      task_id: 'task-b1',
      startTs: now - 50 * HOUR_MS,
      timestamp: Math.floor((now - 50 * HOUR_MS) / 1000),
    }),
    // Neither a stamp nor a usable row instant: the line stays empty.
    session({id: 'chat-c', task_id: 'task-c', startTs: 0, timestamp: null}),
    // Epoch zero is a real launch instant (imported databases carry it).
    session({id: 'chat-d', task_id: 'task-d', startTs: 0, timestamp: 0}),
  ]);
  const gs = groups(win);
  assert.strictEqual(gs.length, 4, 'one chat panel per chat');
  const byChat = {};
  gs.forEach(g => {
    byChat[g.dataset.chatId] = g;
  });
  // The label closes the header's action strip, after the "Show
  // details" chevron, the rename pencil and the delete control (the
  // trash button and its hidden Delete / Cancel strip), where a task
  // panel shows its age label.
  const lineA = byChat['chat-a'].querySelector(
    ':scope > .history-chat-header > .history-chat-actions > .history-chat-launched',
  );
  assert.ok(lineA, 'the chat panel has a launched label in its action strip');
  assert.strictEqual(
    lineA.nextElementSibling,
    null,
    'the label ends the action strip',
  );
  const strip = Array.from(lineA.parentNode.children).map(el => el.className);
  assert.deepStrictEqual(
    strip,
    [
      'sidebar-item-collapse history-chat-collapse',
      'sidebar-item-edit history-chat-rename',
      'sidebar-item-delete',
      'sidebar-item-confirm',
      'history-chat-launched',
    ],
    'chevron, rename pencil, delete control, then the label',
  );
  assert.strictEqual(
    byChat['chat-a'].querySelector(':scope > .history-chat-header')
      .nextElementSibling,
    byChat['chat-a'].querySelector(':scope > .history-chat-body'),
    'the header sits directly above the task rows',
  );
  const labelA = lineA.querySelector('.sidebar-item-launched');
  assert.strictEqual(labelA.textContent, '3 hours ago');
  // chat_last_launched is the newest task's LAUNCH; without a
  // chat_last_finished stamp (an older daemon) the tooltip must not
  // claim the task finished then: it is the bare date.
  assert.strictEqual(
    labelA.title,
    new Date(stamped).toLocaleString(),
    'a bare launch stamp gets a neutral tooltip',
  );
  assert.strictEqual(
    byChat['chat-b'].querySelector(
      '.history-chat-launched .sidebar-item-launched',
    ).textContent,
    '1 day ago',
    'without a stamp the newest row loaded decides',
  );
  assert.strictEqual(
    byChat['chat-c'].querySelector('.history-chat-launched').children.length,
    0,
    'no usable instant renders no label',
  );
  const labelD = byChat['chat-d'].querySelector(
    '.history-chat-launched .sidebar-item-launched',
  );
  assert.ok(labelD, 'an epoch-zero launch instant renders a label');
  assert.strictEqual(labelD.dataset.launchTs, '0');
  assert.ok(/^\d+ years ago$/.test(labelD.textContent), labelD.textContent);

  // The chat panel is folded: the line stays visible (it is not part
  // of the body the chevron folds).
  const chevron = byChat['chat-a'].querySelector('.history-chat-collapse');
  if (!byChat['chat-a'].classList.contains('collapsed')) chevron.click();
  assert.ok(byChat['chat-a'].classList.contains('collapsed'), 'chat folded');
  assert.strictEqual(
    win.getComputedStyle(byChat['chat-a'].querySelector('.history-chat-body'))
      .display,
    'none',
    'the folded body is hidden',
  );
  assert.notStrictEqual(
    win.getComputedStyle(lineA).display,
    'none',
    'the launched line stays visible when the chat is folded',
  );
  assert.strictEqual(
    win.getComputedStyle(
      byChat['chat-c'].querySelector('.history-chat-launched'),
    ).display,
    'none',
    'an empty line takes no room',
  );

  // A refresh with a newer stamp moves the line forward; a stale
  // page does not move it back.  The daemon now also stamps the
  // newest task's completion: the line counts from it and says so.
  loadHistory(win, [
    session({
      id: 'chat-a',
      task_id: 'task-a3',
      startTs: now - 2 * HOUR_MS,
      endTs: now - HOUR_MS,
      timestamp: Math.floor((now - 2 * HOUR_MS) / 1000),
      chat_last_launched: now - 2 * HOUR_MS,
      chat_last_finished: now - HOUR_MS,
    }),
  ]);
  const lineA2 = groups(win)
    .find(g => g.dataset.chatId === 'chat-a')
    .querySelector('.history-chat-launched .sidebar-item-launched');
  assert.strictEqual(lineA2.textContent, '1 hour ago');
  assert.strictEqual(
    lineA2.title,
    'Finished ' + new Date(now - HOUR_MS).toLocaleString(),
    'a finished newest task: tooltip says Finished',
  );
  win.close();
  console.log(
    '  ok - chat panels show a "... ago" line without "last launched"',
  );
}

function testChatLineFollowsTheNewestTaskState() {
  const {win} = makePage();
  const now = Date.now();
  loadHistory(win, [
    // chat-run: its newest task is still running.  The stamps are
    // from the previous task's finish (the row is recomputed when a
    // task finishes); the running row's launch is newer and wins with
    // a "Started" tooltip.
    session({
      id: 'chat-run',
      task_id: 'task-r2',
      is_running: true,
      startTs: now - 10 * 60 * 1000,
      timestamp: Math.floor((now - 10 * 60 * 1000) / 1000),
      chat_last_launched: now - 3 * HOUR_MS,
      chat_last_finished: now - 2 * HOUR_MS,
    }),
    session({
      id: 'chat-run',
      task_id: 'task-r1',
      startTs: now - 3 * HOUR_MS,
      endTs: now - 2 * HOUR_MS,
      timestamp: Math.floor((now - 3 * HOUR_MS) / 1000),
      chat_last_launched: now - 3 * HOUR_MS,
      chat_last_finished: now - 2 * HOUR_MS,
    }),
    // chat-off: only an old finished task is on the page; the stamps
    // say the (unloaded) newest task finished later.
    session({
      id: 'chat-off',
      task_id: 'task-o1',
      startTs: now - 30 * HOUR_MS,
      endTs: now - 29 * HOUR_MS,
      timestamp: Math.floor((now - 30 * HOUR_MS) / 1000),
      chat_last_launched: now - 6 * HOUR_MS,
      chat_last_finished: now - 5 * HOUR_MS,
    }),
    // chat-old: a finished row with a completion time and no stamps
    // (an older daemon): the row's completion, "Finished".
    session({
      id: 'chat-old',
      task_id: 'task-x1',
      startTs: now - 9 * HOUR_MS,
      endTs: now - 8 * HOUR_MS,
      timestamp: Math.floor((now - 9 * HOUR_MS) / 1000),
    }),
  ]);
  const byChat = {};
  groups(win).forEach(g => {
    byChat[g.dataset.chatId] = g;
  });
  const label = id =>
    byChat[id].querySelector('.history-chat-launched .sidebar-item-launched');
  assert.strictEqual(label('chat-run').textContent, '10 minutes ago');
  assert.strictEqual(
    label('chat-run').title,
    'Started ' + new Date(now - 10 * 60 * 1000).toLocaleString(),
    'a running newest task: tooltip says Started',
  );
  assert.strictEqual(label('chat-off').textContent, '5 hours ago');
  assert.strictEqual(
    label('chat-off').title,
    'Finished ' + new Date(now - 5 * HOUR_MS).toLocaleString(),
    'the stamped completion beats an older row on the page',
  );
  assert.strictEqual(label('chat-old').textContent, '8 hours ago');
  assert.strictEqual(
    label('chat-old').title,
    'Finished ' + new Date(now - 8 * HOUR_MS).toLocaleString(),
    'without stamps a finished row counts from its completion',
  );
  win.close();
  console.log(
    '  ok - the chat line says Started while running and Finished once done',
  );
}

function testRefreshSweepRepaintsBothLabels() {
  const {win, timers} = makePage();
  const now = Date.now();
  loadHistory(win, [
    session({
      id: 'chat-a',
      task_id: 'task-a',
      startTs: now - 2 * HOUR_MS,
      tags: 'work',
      chat_last_launched: now - 2 * HOUR_MS,
    }),
  ]);
  const labels = Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item-launched'),
  );
  assert.strictEqual(labels.length, 2, 'the chat line and the task row');
  // Age both labels by five hours, then fire the 30 s sweep timer.
  labels.forEach(el => {
    el.dataset.launchTs = String(now - 7 * HOUR_MS);
  });
  const sweeps = timers.filter(t => t.ms === 30000);
  assert.ok(sweeps.length >= 1, 'the sweep timer is armed');
  sweeps[sweeps.length - 1].cb();
  const after = Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item-launched'),
  ).map(el => el.textContent);
  assert.deepStrictEqual(
    after.sort(),
    ['7 hours ago', '7 hours ago'],
    'the sweep repaints the chat line and the task row alike',
  );
  win.close();
  console.log('  ok - the 30 s sweep repaints the chat line and the task row');
}

try {
  testTagsAfterLaunched();
  testChatLastLaunchedLine();
  testChatLineFollowsTheNewestTaskState();
  testRefreshSweepRepaintsBothLabels();
  console.log('All historyTagsChatLaunchedPanelCost tests passed.');
} catch (e) {
  console.error(e);
  process.exit(1);
}
