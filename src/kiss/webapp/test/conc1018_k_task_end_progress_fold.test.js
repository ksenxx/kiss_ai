// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console */

// A task-end event takes down a `worktree_progress` line that outlived
// its operation (actionProgressTerminalPaths.test.js) and folds the
// transcript's event panels into one Trajectory panel.  The line must
// go BEFORE the fold: folded first, it is counted into the Trajectory
// preview ("2 events") and then removed from under it, so the collapsed
// header overstates what it hides; and when the line is the only
// foldable event an empty Trajectory panel is left behind.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

const TAB = 't1';

function makeTab() {
  const page = makePage();
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {
    type: 'tabs_state',
    tabs: [{tabId: TAB, chatId: 'chat-1', title: 't1', workDir: '/w'}],
  });
  page.win._testApi.endLaunch();
  assert.strictEqual(page.win._testApi.getActiveTabId(), TAB);
  return page;
}

function trajectory(win) {
  return win.document.querySelector('#output .ev.trajectory');
}

function previewText(panel) {
  const prev = panel.querySelector('.collapse-preview');
  return prev ? prev.textContent : '';
}

for (const endType of ['task_stopped', 'task_error', 'task_interrupted']) {
  const {win} = makeTab();
  send(win, {type: 'prompt', tabId: TAB, text: 'normal event before the end'});
  send(win, {type: 'text_end', tabId: TAB});
  send(win, {type: 'worktree_progress', tabId: TAB, message: 'Committing...'});
  assert.strictEqual(
    win.document.querySelectorAll('#output .ev.wt-progress').length,
    1,
    'the progress line is on screen before the task ends',
  );
  send(win, {type: endType, tabId: TAB});

  const panel = trajectory(win);
  assert.ok(panel, endType + ': the events fold into a Trajectory panel');
  const sub = panel.querySelector(':scope > .trajectory-sub');
  assert.strictEqual(
    sub.children.length,
    1,
    endType + ': only the prompt is under the Trajectory',
  );
  assert.strictEqual(
    win.document.querySelectorAll('#output .ev.wt-progress').length,
    0,
    endType + ': the progress line is gone',
  );
  assert.strictEqual(
    previewText(panel),
    '1 event',
    endType + ': the preview counts what the panel actually hides',
  );
  win.close();
  console.log(
    '  ok - ' + endType + ' removes the progress line before folding',
  );
}

// task_done takes the same tail: a progress line still up when the
// daemon reports the task done must not be counted either.
{
  const {win} = makeTab();
  send(win, {type: 'prompt', tabId: TAB, text: 'done with progress up'});
  send(win, {type: 'text_end', tabId: TAB});
  send(win, {type: 'worktree_progress', tabId: TAB, message: 'Merging...'});
  send(win, {type: 'task_done', tabId: TAB, success: true});
  const panel = trajectory(win);
  assert.ok(panel, 'task_done: the events fold into a Trajectory panel');
  assert.strictEqual(previewText(panel), '1 event');
  assert.strictEqual(
    win.document.querySelectorAll('#output .ev.wt-progress').length,
    0,
  );
  win.close();
  console.log('  ok - task_done removes the progress line before folding');
}

// The line is the only foldable event: nothing is left to fold, so no
// Trajectory panel appears at all.
{
  const {win} = makeTab();
  send(win, {type: 'worktree_progress', tabId: TAB, message: 'Committing...'});
  send(win, {type: 'task_stopped', tabId: TAB});
  assert.strictEqual(
    trajectory(win),
    null,
    'a lone progress line leaves no empty Trajectory panel behind',
  );
  assert.strictEqual(
    win.document.querySelectorAll('#output .ev.wt-progress').length,
    0,
  );
  win.close();
  console.log('  ok - a lone progress line leaves no empty Trajectory panel');
}
