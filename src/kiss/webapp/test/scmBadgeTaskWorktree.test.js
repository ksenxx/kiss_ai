// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the SOURCE CONTROL header badge when the
// repository has several worktrees: the badge counts the modified
// files of the worktree the tab's task runs in alone (the gitStatus
// reply's ``taskWorktree``, recorded by the daemon from the task's
// worktree_created), or of the current worktree when the tab has no
// worktree task -- never the sum over every worktree.  The Changes
// list below still shows every worktree with its own count.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  fakeMatchMedia,
  fakeIntervals,
  lastMsg,
} = require('./simplify2_harness');

const WD = '/ws/repo';
const WT = '/ws/repo/.kiss-worktrees/kiss_wt-1';

function makePage() {
  let clock;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts(win) {
      clock = fakeIntervals(win);
      stubRaf(win);
      fakeMatchMedia(win, true);
    },
  });
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  clock.tick(5000);
  return {win, posted, clock};
}

function changes(root, n) {
  const rows = [];
  for (let i = 0; i < n; i += 1) {
    rows.push({
      path: 'f' + i + '.py',
      absPath: root + '/f' + i + '.py',
      status: 'M',
      group: 'changes',
    });
  }
  return rows;
}

function statusReply(token, mainCount, wtCount, taskWorktree) {
  return {
    type: 'gitStatus',
    token,
    workDir: WD,
    repo: WD,
    branch: 'main',
    taskWorktree,
    changes: changes(WD, mainCount),
    worktrees: [
      {
        path: WD,
        name: 'repo',
        head: 'abc',
        branch: 'main',
        detached: false,
        current: true,
        changes: changes(WD, mainCount),
      },
      {
        path: WT,
        name: 'kiss_wt-1',
        head: 'def',
        branch: 'kiss/wt-1',
        detached: false,
        current: false,
        changes: changes(WT, wtCount),
      },
    ],
  };
}

function badge(win) {
  return win.document.getElementById('scm-hdr-count').textContent;
}

function testBadgeCountsTheTaskWorktreeOnly() {
  const {win, posted} = makePage();
  const req = lastMsg(posted, 'gitStatus');
  assert.ok(req, 'the page asked for the git status');
  assert.ok(
    req.tabId,
    'the request names the tab, so the daemon can answer with its worktree',
  );

  // The tab's task runs in the worktree: its 5 changes, not 2 + 5.
  send(win, statusReply(req.token, 2, 5, WT));
  assert.strictEqual(
    badge(win),
    '5',
    'the badge counts the task worktree alone',
  );
  const headers = win.document.querySelectorAll(
    '#scm-changes .scm-worktree-hdr',
  );
  assert.strictEqual(headers.length, 2, 'the list still shows both worktrees');
  assert.strictEqual(headers[0].querySelector('.scm-count').textContent, '2');
  assert.strictEqual(headers[1].querySelector('.scm-count').textContent, '5');

  // A trailing slash in the recorded dir still matches.
  send(win, statusReply(req.token, 2, 4, WT + '/'));
  assert.strictEqual(badge(win), '4');

  // A task run in a sub-folder of the worktree belongs to that
  // worktree: the deepest worktree containing the folder wins (the
  // worktree itself sits below the main checkout, which must not win).
  send(win, statusReply(req.token, 2, 6, WT + '/src/pkg'));
  assert.strictEqual(badge(win), '6', 'a sub-folder of the worktree counts it');
  send(win, statusReply(req.token, 7, 6, WD + '/docs'));
  assert.strictEqual(badge(win), '7', 'a sub-folder of the main checkout');
  // A sibling folder whose name merely starts with the worktree path
  // is not inside it.
  send(win, statusReply(req.token, 2, 6, WT + '0'));
  assert.strictEqual(badge(win), '2', 'a prefix-only match is not containment');

  // No worktree task: the current worktree's own count.
  send(win, statusReply(req.token, 2, 5, ''));
  assert.strictEqual(
    badge(win),
    '2',
    'without a worktree task the current worktree counts',
  );
  send(win, statusReply(req.token, 0, 5, ''));
  assert.strictEqual(
    badge(win),
    '',
    'no change in the current worktree: no badge',
  );

  // A recorded dir that is no worktree of this repository (another
  // repository's) falls back to the current worktree too.
  send(win, statusReply(req.token, 3, 5, '/elsewhere/wt'));
  assert.strictEqual(badge(win), '3');
  win.close();
  console.log(
    '  ok - the badge counts the modified files of the task worktree only',
  );
}

function testSingleWorktreeReplyKeepsCounting() {
  const {win, posted} = makePage();
  const req = lastMsg(posted, 'gitStatus');
  // An older daemon: no worktrees, no taskWorktree.
  send(win, {
    type: 'gitStatus',
    token: req.token,
    workDir: WD,
    repo: WD,
    branch: 'main',
    changes: changes(WD, 3),
  });
  assert.strictEqual(badge(win), '3');
  win.close();
  console.log('  ok - a single-worktree reply counts as before');
}

testBadgeCountsTheTaskWorktreeOnly();
testSingleWorktreeReplyKeepsCounting();
console.log('scmBadgeTaskWorktree.test.js passed');
