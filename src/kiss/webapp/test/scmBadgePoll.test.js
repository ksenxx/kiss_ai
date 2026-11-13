// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the SOURCE CONTROL header badge poll:
// once the daemon is connected, main.js asks git for the workspace's
// status every 5 s (gitStatus only, under the current token) so the
// badge counts the files being modified even while the section is
// folded; the poll pauses while the page is hidden or the daemon is
// away, and a workspace the view has not loaded yet is loaded whole.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  fakeMatchMedia,
  fakeIntervals,
  ofType,
  lastMsg,
  byId,
} = require('./simplify2_harness');

const WD = '/ws/repo';

function makePage(narrow) {
  let clock;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts(win) {
      clock = fakeIntervals(win);
      stubRaf(win);
      fakeMatchMedia(win, !narrow);
    },
  });
  return {win, posted, clock};
}

function statusReply(token, n) {
  const changes = [];
  for (let i = 0; i < n; i += 1) {
    changes.push({
      path: 'f' + i + '.py',
      absPath: WD + '/f' + i + '.py',
      status: 'M',
      group: 'changes',
    });
  }
  return {
    type: 'gitStatus',
    token,
    workDir: WD,
    repo: WD,
    branch: 'main',
    changes,
  };
}

function setHidden(win, hidden) {
  Object.defineProperty(win.document, 'hidden', {
    configurable: true,
    get: () => hidden,
  });
}

/** The subjects of the commits in the graph, top to bottom. */
function subjects(win) {
  return Array.from(
    win.document.querySelectorAll('#scm-graph .scm-commit-subject'),
    el => el.textContent,
  );
}

function testBadgeFollowsTheWorkingTree() {
  const {win, posted, clock} = makePage();
  const section = byId(win, 'meta-scm');
  // Folded: the view itself is not loaded, only the badge matters.
  if (!section.classList.contains('collapsed'))
    section
      .querySelector('.meta-section-toggle')
      .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  const before = ofType(posted, 'gitStatus').length;
  const logsBefore = ofType(posted, 'gitLog').length;

  clock.tick(5000);
  const st = ofType(posted, 'gitStatus');
  assert.ok(st.length > before, 'the 5 s poll asks for the git status');
  assert.strictEqual(st[st.length - 1].workDir, WD);
  assert.strictEqual(
    ofType(posted, 'gitLog').length,
    logsBefore + 1,
    'the first tick loads the unloaded workspace whole (status + log)',
  );
  send(win, statusReply(st[st.length - 1].token, 3));
  assert.strictEqual(
    byId(win, 'scm-hdr-count').textContent,
    '3',
    'the badge shows the number of modified files',
  );

  // Files changed on disk: the next tick asks again (status only, the
  // log moves on a commit) and the badge follows.
  const logsAfterLoad = ofType(posted, 'gitLog').length;
  clock.tick(5000);
  const again = ofType(posted, 'gitStatus');
  assert.strictEqual(again.length, st.length + 1, 'one gitStatus per tick');
  assert.strictEqual(
    ofType(posted, 'gitLog').length,
    logsAfterLoad,
    'a same-workspace tick does not re-ask for the log',
  );
  send(win, statusReply(again[again.length - 1].token, 1));
  assert.strictEqual(byId(win, 'scm-hdr-count').textContent, '1');
  send(win, statusReply(again[again.length - 1].token, 0));
  assert.strictEqual(byId(win, 'scm-hdr-count').textContent, '');
  win.close();
  console.log('  ok - the badge follows the working tree every 5 s');
}

function testPollPausesWhenHiddenOrDisconnected() {
  const {win, posted, clock} = makePage();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  clock.tick(5000);
  const n = ofType(posted, 'gitStatus').length;
  assert.ok(n > 0, 'the first tick asks for the git status');
  // Answer the outstanding request: an unanswered one would suppress
  // every later tick on its own (one request at a time) and hide a
  // missing hidden-page guard.
  send(win, statusReply(lastToken(posted), 0));

  setHidden(win, true);
  clock.tick(15000);
  assert.strictEqual(
    ofType(posted, 'gitStatus').length,
    n,
    'a hidden page asks git for nothing',
  );
  setHidden(win, false);

  send(win, {type: 'daemonStatus', connected: false});
  clock.tick(15000);
  assert.strictEqual(
    ofType(posted, 'gitStatus').length,
    n,
    'no poll while the daemon is away',
  );
  send(win, {type: 'daemonStatus', connected: true});
  clock.tick(5000);
  assert.ok(
    ofType(posted, 'gitStatus').length > n,
    'the poll resumes with the daemon',
  );
  win.close();
  console.log('  ok - the poll pauses while hidden or disconnected');
}

function testNoWorkspaceNoPoll() {
  const {win, posted, clock} = makePage();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: ''}});
  const n = ofType(posted, 'gitStatus').length;
  clock.tick(20000);
  assert.strictEqual(
    ofType(posted, 'gitStatus').length,
    n,
    'without a workspace there is nothing to count',
  );
  win.close();
  console.log('  ok - no workspace, no poll');
}

function logReply(token) {
  return {
    type: 'gitLog',
    token,
    workDir: WD,
    repo: WD,
    head: 'a1b2c3d',
    commits: [
      {
        sha: 'a1b2c3d',
        shortSha: 'a1b2c3d',
        parents: [],
        author: 'a',
        date: '2026-10-16T00:00:00Z',
        refs: [],
        subject: 'first',
        files: [],
      },
    ],
  };
}

function lastToken(posted) {
  return lastMsg(posted, 'gitStatus').token;
}

// Audit 2026-10-16 (mainjs-2 F2): a poll reply that repeats the status
// on screen must not rebuild the Changes list and the commit graph --
// that detached the focused row every 5 s, breaking keyboard
// navigation and the context menu's focus hand-back.  A changed reply
// still updates the list.
function testUnchangedPollReplyKeepsFocus() {
  const {win, posted, clock} = makePage();
  const doc = win.document;
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  const tok = lastToken(posted);
  send(win, statusReply(tok, 3));
  send(win, logReply(tok));
  const rows = doc.querySelectorAll('#scm-changes .scm-row');
  assert.strictEqual(rows.length, 3, 'the Changes list shows the 3 files');
  const commit = doc.querySelector('#scm-graph .scm-commit');
  assert.ok(commit, 'the graph shows the commit');
  rows[1].focus();
  assert.strictEqual(doc.activeElement, rows[1]);

  clock.tick(5000);
  send(win, statusReply(lastToken(posted), 3));
  assert.strictEqual(
    doc.activeElement,
    rows[1],
    'an unchanged poll reply leaves the focused row in place',
  );
  assert.ok(rows[1].isConnected, 'the row was not rebuilt');
  assert.ok(commit.isConnected, 'the graph was not rebuilt');
  assert.strictEqual(byId(win, 'scm-hdr-count').textContent, '3');

  // The graph row keeps its focus too.
  commit.focus();
  clock.tick(5000);
  send(win, statusReply(lastToken(posted), 3));
  assert.strictEqual(doc.activeElement, commit, 'the commit row kept focus');

  // The tree changed: the list follows (and the badge).
  clock.tick(5000);
  send(win, statusReply(lastToken(posted), 1));
  assert.strictEqual(
    doc.querySelectorAll('#scm-changes .scm-row').length,
    1,
    'a changed reply rebuilds the list',
  );
  assert.strictEqual(byId(win, 'scm-hdr-count').textContent, '1');
  win.close();
  console.log(
    '  ok - an unchanged poll reply keeps focus; a changed one updates',
  );
}

function testNoPollWhileThePanelIsOffScreen() {
  // Phone: the task-info panel is a drawer that starts closed.
  {
    const {win, posted, clock} = makePage(true);
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'configData', config: {work_dir: WD}});
    const n = ofType(posted, 'gitStatus').length;
    clock.tick(15000);
    assert.strictEqual(
      ofType(posted, 'gitStatus').length,
      n,
      'the closed drawer shows no badge: nothing to poll for',
    );
    byId(win, 'meta-drawer-btn').dispatchEvent(
      new win.MouseEvent('click', {bubbles: true}),
    );
    clock.tick(5000);
    assert.ok(
      ofType(posted, 'gitStatus').length > n,
      'the open drawer polls again',
    );
    win.close();
  }
  // Desktop: a content tab slides the docked panel off the pane; the
  // drawer tab brings it back.
  {
    const {win, posted, clock} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'configData', config: {work_dir: WD}});
    clock.tick(5000);
    // The daemon answers (an unanswered status is not asked again).
    send(win, statusReply(lastToken(posted), 0));
    send(win, {
      type: 'fileContent',
      path: WD + '/a.txt',
      name: 'a.txt',
      content: 'x',
      version: 'v1',
    });
    assert.ok(
      win.document.body.classList.contains('meta-hidden'),
      'the panel is hidden behind the content pane',
    );
    const n = ofType(posted, 'gitStatus').length;
    clock.tick(15000);
    assert.strictEqual(
      ofType(posted, 'gitStatus').length,
      n,
      'the hidden panel shows no badge: nothing to poll for',
    );
    byId(win, 'meta-drawer').dispatchEvent(
      new win.MouseEvent('click', {bubbles: true}),
    );
    clock.tick(5000);
    assert.ok(
      ofType(posted, 'gitStatus').length > n,
      'the panel shown again polls again',
    );
    win.close();
  }
  console.log('  ok - no poll while the task-info panel is off screen');
}

// A26_fable-1: the tick must not queue a new gitStatus behind one the
// daemon has not answered yet.  The daemon serves a connection's
// commands one after another, so with a slow git (a large repository,
// a slow disk) every unanswered tick would add a status request to
// the backlog a later Stop or submit waits behind.
function testOneOutstandingStatusRequest() {
  const {win, posted, clock} = makePage();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  clock.tick(5000);
  const sent = ofType(posted, 'gitStatus');
  assert.ok(sent.length > 0, 'the workspace is asked for its status');
  const n = sent.length;

  // Twenty ticks with no reply: the request on its way is the one
  // outstanding, nothing is queued behind it.
  clock.tick(20 * 5000);
  assert.strictEqual(
    ofType(posted, 'gitStatus').length,
    n,
    'BUG - the ticks queued gitStatus requests behind the unanswered one',
  );

  // The reply lands: the next tick asks again.
  send(win, statusReply(lastToken(posted), 2));
  assert.strictEqual(byId(win, 'scm-hdr-count').textContent, '2');
  clock.tick(5000);
  assert.strictEqual(
    ofType(posted, 'gitStatus').length,
    n + 1,
    'the tick after the reply polls again',
  );

  // A reply of a stale token (the request of a workspace left behind)
  // is not the answer to the outstanding one: still one in flight.
  send(win, statusReply('0', 9));
  clock.tick(5000);
  assert.strictEqual(
    ofType(posted, 'gitStatus').length,
    n + 1,
    'a stale reply does not release the poll',
  );

  // The daemon drops: its answer is not coming; the reconnected poll
  // asks afresh.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  clock.tick(5000);
  assert.ok(
    ofType(posted, 'gitStatus').length > n + 1,
    'the poll resumes after a reconnect',
  );
  win.close();
  console.log('  ok - one status request outstanding at a time');
}

// A26_claude-2: the badge poll asks for the status alone (the log moves
// on a commit), but a commit, checkout or reset made outside a task (a
// terminal) moves HEAD under the log on hand: a status reply whose
// heads differ from the log's re-asks the log under the same token, and
// the graph is rebuilt from the fresh pair, not from the old log.
function testAMovedHeadReloadsTheLog() {
  const {win, posted, clock} = makePage();
  const worktree = (reply, head) =>
    Object.assign(reply, {
      worktrees: [
        {
          path: WD,
          name: 'repo',
          head,
          branch: 'main',
          detached: false,
          current: true,
          changes: reply.changes || [],
        },
      ],
    });
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  const tok = lastToken(posted);
  send(win, worktree(statusReply(tok, 1), 'a1b2c3d'));
  send(win, worktree(logReply(tok), 'a1b2c3d'));
  assert.deepStrictEqual(subjects(win), ['Uncommitted changes', 'first']);
  const logs = ofType(posted, 'gitLog').length;

  // The tree alone changed: the log stands, nothing is re-asked.
  clock.tick(5000);
  send(win, worktree(statusReply(lastToken(posted), 2), 'a1b2c3d'));
  assert.strictEqual(ofType(posted, 'gitLog').length, logs, 'same head: no log');

  // `git commit` in a terminal: the poll sees the new HEAD and a clean
  // tree.
  clock.tick(5000);
  send(win, worktree(statusReply(lastToken(posted), 0), 'e5f6a7b'));
  assert.strictEqual(
    ofType(posted, 'gitLog').length,
    logs + 1,
    'BUG - the moved HEAD did not re-ask the log',
  );
  const req = lastMsg(posted, 'gitLog');
  assert.strictEqual(req.token, lastToken(posted), 'same generation');
  assert.strictEqual(req.workDir, WD);
  assert.deepStrictEqual(
    subjects(win),
    ['Uncommitted changes', 'first'],
    'the graph waits for the fresh log instead of showing the old one',
  );
  const fresh = worktree(logReply(req.token), 'e5f6a7b');
  fresh.head = 'e5f6a7b';
  fresh.commits.unshift({
    sha: 'e5f6a7b',
    shortSha: 'e5f6a7b',
    parents: ['a1b2c3d'],
    author: 'a',
    date: '2026-10-16T00:01:00Z',
    refs: ['HEAD -> main'],
    subject: 'second',
    files: [],
  });
  send(win, fresh);
  assert.deepStrictEqual(subjects(win), ['second', 'first'], 'the new commit shows');

  // The next poll agrees with the fresh log: nothing more is asked.
  clock.tick(5000);
  send(win, worktree(statusReply(lastToken(posted), 1), 'e5f6a7b'));
  assert.strictEqual(ofType(posted, 'gitLog').length, logs + 1);
  assert.deepStrictEqual(subjects(win), ['Uncommitted changes', 'second', 'first']);
  win.close();
  console.log('  ok - a HEAD moved outside a task reloads the log');
}

function worktreeEntry(path, name, head, current, branch) {
  return {
    path,
    name,
    head,
    branch: branch || name,
    detached: false,
    current: !!current,
    changes: [],
  };
}

function commitEntry(sha, parents, subject) {
  return {
    sha,
    shortSha: sha.slice(0, 7),
    parents,
    author: 'a',
    date: '2026-10-16T00:00:00Z',
    refs: [],
    subject,
    files: [],
  };
}

// Audit 2026-11-11 (W1): the graph's HEAD and branch labels come from
// the log's refs, so the log is also behind when a worktree's BRANCH
// moved under an unchanged commit (`git switch -c feature` in a
// terminal): the header named the new branch while the graph kept
// `HEAD -> main` until the next full refresh.
function testACheckoutAtTheSameCommitReloadsTheLog() {
  const {win, posted, clock} = makePage();
  const doc = win.document;
  const A = 'a1b2c3d';
  const head = () => doc.querySelector('#scm-graph .scm-ref.is-head');
  const onBranch = (reply, branch) => {
    reply.branch = branch;
    reply.worktrees = [worktreeEntry(WD, 'repo', A, true, branch)];
    return reply;
  };
  const logOn = (token, branch) => {
    const log = onBranch(logReply(token), branch);
    log.commits[0].refs = ['HEAD -> ' + branch];
    return log;
  };
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  const tok = lastToken(posted);
  send(win, onBranch(statusReply(tok, 0), 'main'));
  send(win, logOn(tok, 'main'));
  assert.strictEqual(head().textContent, 'main', 'setup: HEAD -> main');
  const logs = ofType(posted, 'gitLog').length;

  clock.tick(5000);
  send(win, onBranch(statusReply(lastToken(posted), 0), 'feature'));
  assert.strictEqual(byId(win, 'scm-branch').textContent, 'feature');
  assert.strictEqual(
    ofType(posted, 'gitLog').length,
    logs + 1,
    'BUG - the checkout at the same commit did not re-ask the log',
  );
  send(win, logOn(lastMsg(posted, 'gitLog').token, 'feature'));
  assert.strictEqual(
    head().textContent,
    'feature',
    'the graph names the branch checked out now',
  );

  // The next poll agrees with the fresh log: nothing more is asked.
  clock.tick(5000);
  send(win, onBranch(statusReply(lastToken(posted), 1), 'feature'));
  assert.strictEqual(ofType(posted, 'gitLog').length, logs + 1);
  win.close();
  console.log('  ok - a checkout at the same commit reloads the log');
}

// Audit 2026-11-11 (W1): a log that FAILED (a transient git error, a
// forwarded machine that did not answer) is asked again by every status
// that succeeds, at the poll's pace: an unchanged tree repeats its
// status for ever, and the error stayed in the graph until a manual
// refresh.
function testAFailedLogIsAskedAgain() {
  const {win, posted, clock} = makePage();
  const doc = win.document;
  const graph = () => byId(win, 'scm-graph').textContent;
  const withTree = reply =>
    Object.assign(reply, {
      worktrees: [worktreeEntry(WD, 'repo', 'a1b2c3d', true, 'main')],
    });
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  const tok = lastToken(posted);
  send(win, withTree(statusReply(tok, 0)));
  send(win, {type: 'gitLog', token: tok, error: 'git log failed'});
  assert.ok(graph().includes('git log failed'), 'setup: the error shows');
  const logs = ofType(posted, 'gitLog').length;

  // The same status again: the failed log is retried, once.
  clock.tick(5000);
  send(win, withTree(statusReply(lastToken(posted), 0)));
  assert.strictEqual(
    ofType(posted, 'gitLog').length,
    logs + 1,
    'BUG - a failed log was never asked again',
  );
  clock.tick(5000);
  send(win, withTree(statusReply(lastToken(posted), 0)));
  assert.strictEqual(
    ofType(posted, 'gitLog').length,
    logs + 1,
    'one retry outstanding at a time',
  );
  send(win, withTree(logReply(lastMsg(posted, 'gitLog').token)));
  assert.ok(!graph().includes('git log failed'), 'the error left the graph');
  assert.strictEqual(
    doc.querySelector('#scm-graph .scm-commit-subject').textContent,
    'first',
  );

  // A healthy log is not asked for by a repeated status.
  clock.tick(5000);
  send(win, withTree(statusReply(lastToken(posted), 0)));
  assert.strictEqual(ofType(posted, 'gitLog').length, logs + 1);
  win.close();
  console.log('  ok - a failed log is asked again by the status poll');
}

// Audit 2026-11-08 (W5 F2): the graph's "worktree" badges follow the
// status replies, which the badge poll keeps fresh, not the worktree
// list of the log on hand: a worktree moved (`git worktree move`) or
// two worktrees that swapped branches keep the same HEADs and branches,
// so the log is rightly not re-asked -- and the badges must still move.
function testStatusOnlyWorktreeChangesRepaintTheBadges() {
  const {win, posted, clock} = makePage();
  const doc = win.document;
  const A = 'a'.repeat(40);
  const B = 'b'.repeat(40);
  const C = 'c'.repeat(40);
  const badge = sha => {
    const ref = doc.querySelector(
      '.scm-commit[data-scm-sha="' + sha + '"] .scm-ref.is-worktree',
    );
    return ref ? ref.textContent + ' ' + ref.title : null;
  };
  const withTrees = (reply, trees) => Object.assign(reply, {worktrees: trees});
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  const tok = lastToken(posted);
  const trees = [
    worktreeEntry(WD, 'repo', A, true),
    worktreeEntry(WD + '-left', 'left', B),
    worktreeEntry(WD + '-right', 'right', C),
  ];
  send(win, withTrees(statusReply(tok, 0), trees));
  const log = withTrees(logReply(tok), trees);
  log.head = A;
  log.commits = [
    commitEntry(A, [B], 'A'),
    commitEntry(B, [C], 'B'),
    commitEntry(C, [], 'C'),
  ];
  send(win, log);
  assert.strictEqual(badge(B), 'left Worktree ' + WD + '-left');
  assert.strictEqual(badge(C), 'right Worktree ' + WD + '-right');
  const logs = ofType(posted, 'gitLog').length;

  // The two worktrees swapped branches (a checkout in each): every
  // branch still points at its commit, so the log's refs stand.
  clock.tick(5000);
  send(
    win,
    withTrees(statusReply(lastToken(posted), 0), [
      trees[0],
      worktreeEntry(WD + '-left', 'left', C, false, 'right'),
      worktreeEntry(WD + '-right', 'right', B, false, 'left'),
    ]),
  );
  assert.strictEqual(ofType(posted, 'gitLog').length, logs, 'same HEADs: no log');
  assert.strictEqual(
    badge(B),
    'right Worktree ' + WD + '-right',
    'BUG - the badge stayed on the worktree list of the old log',
  );
  assert.strictEqual(badge(C), 'left Worktree ' + WD + '-left');

  // `git worktree move repo-left repo-moved`: same HEAD, new folder.
  clock.tick(5000);
  send(
    win,
    withTrees(statusReply(lastToken(posted), 0), [
      trees[0],
      worktreeEntry(WD + '-moved', 'moved', C, false, 'right'),
      worktreeEntry(WD + '-right', 'right', B, false, 'left'),
    ]),
  );
  assert.strictEqual(ofType(posted, 'gitLog').length, logs, 'same HEADs: no log');
  assert.strictEqual(badge(C), 'moved Worktree ' + WD + '-moved');
  assert.strictEqual(badge(B), 'right Worktree ' + WD + '-right');
  win.close();
  console.log('  ok - a status-only worktree move or HEAD swap moves the badges');
}

// Audit 2026-11-08 (W5 F3): a full refresh asks for the status and the
// log together; its status, answered first, must not ask for the log
// a second time because the log still on hand (the previous
// generation's) walks from the old HEAD.  The moved-HEAD re-ask stays
// for the status-only poll (testAMovedHeadReloadsTheLog).
function testAFullRefreshAsksTheLogOnce() {
  const {win, posted} = makePage();
  const withTree = (reply, head) => {
    const tree = worktreeEntry(WD, 'repo', head, true);
    tree.changes = reply.changes || [];
    return Object.assign(reply, {worktrees: [tree]});
  };
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: WD}});
  const tok = lastToken(posted);
  send(win, withTree(statusReply(tok, 1), 'a1b2c3d'));
  send(win, withTree(logReply(tok), 'a1b2c3d'));
  assert.deepStrictEqual(subjects(win), ['Uncommitted changes', 'first']);

  // The user committed and pressed Refresh: one status + one log.
  byId(win, 'scm-refresh').click();
  const tok2 = lastToken(posted);
  assert.notStrictEqual(tok2, tok, 'a refresh is a new generation');
  const logsOf = t => ofType(posted, 'gitLog').filter(m => m.token === t);
  assert.strictEqual(logsOf(tok2).length, 1);
  send(win, withTree(statusReply(tok2, 0), 'e5f6a7b'));
  assert.strictEqual(
    logsOf(tok2).length,
    1,
    'BUG - the refresh status re-asked the log its refresh already asked',
  );
  const fresh = withTree(logReply(tok2), 'e5f6a7b');
  fresh.head = 'e5f6a7b';
  fresh.commits.unshift(commitEntry('e5f6a7b', ['a1b2c3d'], 'second'));
  send(win, fresh);
  assert.deepStrictEqual(subjects(win), ['second', 'first']);
  win.close();
  console.log('  ok - a full refresh asks the log once');
}

function main() {
  testBadgeFollowsTheWorkingTree();
  testPollPausesWhenHiddenOrDisconnected();
  testNoWorkspaceNoPoll();
  testUnchangedPollReplyKeepsFocus();
  testNoPollWhileThePanelIsOffScreen();
  testOneOutstandingStatusRequest();
  testAMovedHeadReloadsTheLog();
  testACheckoutAtTheSameCommitReloadsTheLog();
  testAFailedLogIsAskedAgain();
  testStatusOnlyWorktreeChangesRepaintTheBadges();
  testAFullRefreshAsksTheLogOnce();
  console.log('scmBadgePoll.test.js: all tests passed');
}

main();
