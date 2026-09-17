// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for message-ordering races and redundancies
// in the remote webapp's Explorer / Source Control sidebar
// (media/main.js).  Each test asserts the CORRECT behavior:
//
// * CAND-1: a successful `move` fsResult arriving after the user Cut a
//   DIFFERENT entry must not clear the Explorer clipboard — the newer
//   Cut survives and Paste still works,
// * CAND-3: clicking to expand a commit row whose sha is no longer in
//   the current log (a fresh gitLog landed while its paired gitStatus
//   is still in flight) must not mark the row expanded with an empty
//   files container, and must not remember that sha as expanded,
// * CAND-4: a forced Explorer refresh must not send a second listDir
//   for a folder whose first listing is still in flight,
// * CAND-5: changing the Source Control workspace clears the set of
//   expanded commits — a sha expanded in repo A is not auto-expanded
//   when the same sha appears in repo B.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body class="remote-chat"');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.matchMedia = function (query) {
    return {
      matches: query === '(min-width: 900px)',
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  Object.defineProperty(win.navigator, 'clipboard', {
    value: {writeText: () => Promise.resolve()},
    configurable: true,
  });
  win.prompt = () => null;
  win.confirm = () => true;
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'treeContextMenu.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=conc2026-races-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

const WD = '/ws/repo';
const WD2 = '/ws/other';

function pinWorkspace(win, wd) {
  send(win, {type: 'configData', config: {work_dir: wd || WD}});
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function rightClick(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      clientX: 10,
      clientY: 10,
    }),
  );
}

function byId(win, id) {
  return win.document.getElementById(id);
}

function ofType(posted, type) {
  return posted.filter(m => m.type === type);
}

function rows(win, sel) {
  return Array.from(win.document.querySelectorAll(sel));
}

function menuItem(win, label) {
  return rows(win, '#sidebar-context-menu .tree-ctx-item').find(
    el => el.querySelector('.tree-ctx-label').textContent === label,
  );
}

/** Open the Explorer on WD and answer its root listing with *entries*. */
function openExplorer(win, posted, entries) {
  pinWorkspace(win);
  click(win, byId(win, 'activity-explorer'));
  const list = ofType(posted, 'listDir');
  const req = list[list.length - 1];
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: req.path,
    root: req.path,
    entries,
  });
  return req;
}

function explorerRow(win, p) {
  return win.document.querySelector(
    '.explorer-row[data-explorer-path="' + p + '"]',
  );
}

const SHA_A = 'a'.repeat(40);
const SHA_B = 'b'.repeat(40);

function commit(sha, parents, subject) {
  return {
    sha,
    shortSha: sha.slice(0, 7),
    parents,
    author: 'A',
    date: new Date().toISOString(),
    refs: [],
    subject,
    message: subject,
    files: [],
  };
}

function worktree(wd, head) {
  return {
    path: wd,
    name: pathTail(wd),
    head,
    branch: 'main',
    detached: false,
    current: true,
    changes: [],
  };
}

function pathTail(p) {
  return p.slice(p.lastIndexOf('/') + 1);
}

/** Answer the LAST outstanding gitStatus + gitLog pair with *commits*. */
function answerScm(win, posted, wd, commits) {
  const st = ofType(posted, 'gitStatus');
  const lg = ofType(posted, 'gitLog');
  const head = commits.length ? commits[0].sha : '';
  send(win, {
    type: 'gitStatus',
    token: st[st.length - 1].token,
    workDir: wd,
    repo: wd,
    branch: 'main',
    changes: [],
    worktrees: [worktree(wd, head)],
  });
  send(win, {
    type: 'gitLog',
    token: lg[lg.length - 1].token,
    workDir: wd,
    repo: wd,
    head,
    commits,
    worktrees: [worktree(wd, head)],
  });
}

function commitRow(win, sha) {
  return rows(win, '#scm-graph .scm-commit').find(
    el => el.dataset.scmSha === sha,
  );
}

/** Open Source Control on *wd* showing history A -> B. */
function openScm(win, posted, wd) {
  pinWorkspace(win, wd);
  click(win, byId(win, 'activity-scm'));
  answerScm(win, posted, wd || WD, [
    commit(SHA_A, [SHA_B], 'Title line'),
    commit(SHA_B, [], 'root'),
  ]);
  return commitRow(win, SHA_A);
}

async function main() {
  await test('CAND-1: a late move fsResult does not clear a newer Cut', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
      {name: 'b.txt', path: WD + '/b.txt', isDir: false},
      {name: 'dir', path: WD + '/dir', isDir: true},
    ]);
    const dirRow = explorerRow(win, WD + '/dir');
    // Cut a.txt, then Paste it into dir: a `move` fsAction goes out.
    rightClick(win, explorerRow(win, WD + '/a.txt'));
    click(win, menuItem(win, 'Cut'));
    rightClick(win, dirRow);
    click(win, menuItem(win, 'Paste'));
    let moves = ofType(posted, 'fsAction').filter(m => m.action === 'move');
    assert.strictEqual(moves.length, 1);
    assert.strictEqual(moves[0].path, WD + '/a.txt');
    assert.strictEqual(moves[0].dest, WD + '/dir');
    // While the move reply is in flight the user Cuts b.txt.
    rightClick(win, explorerRow(win, WD + '/b.txt'));
    click(win, menuItem(win, 'Cut'));
    // The successful move reply for a.txt lands now.
    send(win, {
      type: 'fsResult',
      token: moves[0].token,
      path: WD + '/dir/a.txt',
    });
    // The clipboard still holds b.txt: Paste is enabled and pastes it.
    rightClick(win, dirRow);
    const paste = menuItem(win, 'Paste');
    assert.ok(paste, 'the Paste item exists');
    assert.ok(
      !paste.classList.contains('disabled') &&
        paste.getAttribute('aria-disabled') !== 'true',
      'BUG: the move reply for a.txt cleared the newer Cut of b.txt ' +
        '(Paste is disabled)',
    );
    click(win, paste);
    moves = ofType(posted, 'fsAction').filter(m => m.action === 'move');
    assert.strictEqual(
      moves.length,
      2,
      'BUG: Paste after the late move reply sent nothing',
    );
    assert.strictEqual(moves[1].path, WD + '/b.txt');
    assert.strictEqual(moves[1].dest, WD + '/dir');
  });

  await test('CAND-3: expanding a commit that fell out of the fresh log is a no-op', async () => {
    const {win, posted} = makeWebview();
    const rowA = openScm(win, posted);
    assert.ok(rowA, 'the commit row for A renders');
    // A manual refresh: a fresh gitStatus + gitLog pair goes out.
    click(win, byId(win, 'scm-refresh'));
    const lg = ofType(posted, 'gitLog');
    assert.ok(lg.length >= 2, 'the refresh asked for a new log');
    // The new log lands FIRST (history rewritten: A is gone); its
    // paired status is still in flight, so the old rows are on screen.
    send(win, {
      type: 'gitLog',
      token: lg[lg.length - 1].token,
      workDir: WD,
      repo: WD,
      head: SHA_B,
      commits: [commit(SHA_B, [], 'root')],
      worktrees: [worktree(WD, SHA_B)],
    });
    assert.ok(commitRow(win, SHA_A), 'the old row for A is still shown');
    // Clicking the stale row must not mark it expanded over an empty
    // files container.
    click(win, commitRow(win, SHA_A));
    const staleRow = commitRow(win, SHA_A);
    const files = staleRow.parentNode.querySelector('.scm-commit-files');
    assert.strictEqual(
      staleRow.getAttribute('aria-expanded'),
      'false',
      'BUG: a commit no longer in the log was marked expanded',
    );
    assert.ok(
      files.hidden,
      'BUG: an empty, unfilled files container was revealed',
    );
    assert.strictEqual(files.childNodes.length, 0);
    // Nor may the sha be remembered as expanded: when A comes back in
    // a later log it renders collapsed.
    const st = ofType(posted, 'gitStatus');
    send(win, {
      type: 'gitStatus',
      token: st[st.length - 1].token,
      workDir: WD,
      repo: WD,
      branch: 'main',
      changes: [],
      worktrees: [worktree(WD, SHA_B)],
    });
    click(win, byId(win, 'scm-refresh'));
    answerScm(win, posted, WD, [
      commit(SHA_A, [SHA_B], 'Title line'),
      commit(SHA_B, [], 'root'),
    ]);
    const backRow = commitRow(win, SHA_A);
    assert.ok(backRow, 'A is back in the graph');
    assert.strictEqual(
      backRow.getAttribute('aria-expanded'),
      'false',
      'BUG: the stale click was remembered and auto-expanded A',
    );
    // A commit that IS in the current log still expands normally.
    click(win, commitRow(win, SHA_B));
    const rowB = commitRow(win, SHA_B);
    assert.strictEqual(rowB.getAttribute('aria-expanded'), 'true');
    assert.ok(!rowB.parentNode.querySelector('.scm-commit-files').hidden);
  });

  await test('CAND-4: a forced refresh sends no second listDir for an in-flight folder', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, [
      {name: 'dir', path: WD + '/dir', isDir: true},
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
    ]);
    const listFor = p =>
      ofType(posted, 'listDir').filter(m => m.path === p).length;
    // Expand dir: its first listing goes out and stays in flight.
    click(win, explorerRow(win, WD + '/dir'));
    assert.strictEqual(listFor(WD + '/dir'), 1);
    const rootBefore = listFor(WD);
    // A forced refresh re-lists loaded folders (the root)...
    click(win, byId(win, 'explorer-refresh'));
    assert.strictEqual(
      listFor(WD),
      rootBefore + 1,
      'the loaded root is re-listed',
    );
    // ...but must not double up the listing already in flight.
    assert.strictEqual(
      listFor(WD + '/dir'),
      1,
      'BUG: a second listDir was sent for a folder whose first ' +
        'listing is still in flight',
    );
    // A second forced refresh while still in flight coalesces too.
    click(win, byId(win, 'explorer-refresh'));
    assert.strictEqual(
      listFor(WD + '/dir'),
      1,
      'repeated forced refreshes coalesce while the listing is in flight',
    );
    // The in-flight reply may predate whatever change forced the
    // refresh (e.g. a task wrote dir/generated.txt after the daemon
    // read the disk): when the stale reply lands, exactly ONE
    // follow-up listing must go out, or the forced refresh is lost and
    // the folder stays stale until some unrelated later refresh.
    send(win, {
      type: 'dirListing',
      token: ofType(posted, 'listDir').filter(m => m.path === WD + '/dir')[0]
        .token,
      path: WD + '/dir',
      root: WD,
      entries: [{name: 'c.txt', path: WD + '/dir/c.txt', isDir: false}],
    });
    assert.strictEqual(
      listFor(WD + '/dir'),
      2,
      'BUG: the forced refresh received while the first listing was in ' +
        'flight was dropped instead of queuing one follow-up listing',
    );
    assert.ok(
      !explorerRow(win, WD + '/dir/generated.txt'),
      'the stale reply (pre-change snapshot) rendered without the file',
    );
    // The follow-up reply carries the post-change snapshot.
    send(win, {
      type: 'dirListing',
      token: ofType(posted, 'listDir').filter(m => m.path === WD + '/dir')[1]
        .token,
      path: WD + '/dir',
      root: WD,
      entries: [
        {name: 'c.txt', path: WD + '/dir/c.txt', isDir: false},
        {name: 'generated.txt', path: WD + '/dir/generated.txt', isDir: false},
      ],
    });
    assert.strictEqual(
      listFor(WD + '/dir'),
      2,
      'the follow-up reply does not trigger yet another listing',
    );
    assert.ok(
      explorerRow(win, WD + '/dir/generated.txt'),
      'the task-created file is visible once the follow-up reply lands',
    );
    // Once settled, the folder is loaded and refreshes normally again.
    click(win, byId(win, 'explorer-refresh'));
    assert.strictEqual(
      listFor(WD + '/dir'),
      3,
      'a loaded folder is re-listed on the next forced refresh',
    );
  });

  await test('CAND-4b: task news between request and stale reply refreshes the folder', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, [{name: 'dir', path: WD + '/dir', isDir: true}]);
    const listFor = p =>
      ofType(posted, 'listDir').filter(m => m.path === p).length;
    // Expand dir: its FIRST listing goes out and stays in flight.
    click(win, explorerRow(win, WD + '/dir'));
    assert.strictEqual(listFor(WD + '/dir'), 1);
    // A task writes dir/generated.txt AFTER the daemon read the disk;
    // its status broadcast lands while the listing is still in flight
    // (task news forces refreshSidebarDataViews(true) independently of
    // this connection's serial command loop).
    const readyMsg = posted.find(m => m.type === 'ready');
    send(win, {type: 'status', tabId: readyMsg.tabId, running: false});
    assert.strictEqual(
      listFor(WD + '/dir'),
      1,
      'task news does not double up the in-flight listing',
    );
    // The stale pre-change snapshot lands: one follow-up must go out.
    send(win, {
      type: 'dirListing',
      token: ofType(posted, 'listDir').filter(m => m.path === WD + '/dir')[0]
        .token,
      path: WD + '/dir',
      root: WD,
      entries: [],
    });
    assert.strictEqual(
      listFor(WD + '/dir'),
      2,
      'BUG: the task-news forced refresh was dropped for the loading folder',
    );
    send(win, {
      type: 'dirListing',
      token: ofType(posted, 'listDir').filter(m => m.path === WD + '/dir')[1]
        .token,
      path: WD + '/dir',
      root: WD,
      entries: [
        {name: 'generated.txt', path: WD + '/dir/generated.txt', isDir: false},
      ],
    });
    assert.ok(
      explorerRow(win, WD + '/dir/generated.txt'),
      'the task-created file becomes visible without any manual refresh',
    );
  });

  await test('CAND-5: changing the SCM workspace clears the expanded commits', async () => {
    const {win, posted} = makeWebview();
    const rowA = openScm(win, posted);
    assert.ok(rowA, 'the commit row for A renders in repo A');
    click(win, rowA);
    assert.strictEqual(
      commitRow(win, SHA_A).getAttribute('aria-expanded'),
      'true',
      'A expands in repo A',
    );
    // The workspace changes to another repository.
    pinWorkspace(win, WD2);
    click(win, byId(win, 'activity-explorer'));
    click(win, byId(win, 'activity-scm'));
    const st = ofType(posted, 'gitStatus');
    assert.strictEqual(
      st[st.length - 1].workDir,
      WD2,
      'Source Control asks about the new workspace',
    );
    // Repo B happens to contain the same sha.
    answerScm(win, posted, WD2, [
      commit(SHA_A, [SHA_B], 'Same sha, other repo'),
      commit(SHA_B, [], 'root'),
    ]);
    const rowB = commitRow(win, SHA_A);
    assert.ok(rowB, 'the commit row for A renders in repo B');
    assert.strictEqual(
      rowB.getAttribute('aria-expanded'),
      'false',
      'BUG: a sha expanded in repo A was auto-expanded in repo B',
    );
    assert.ok(
      rowB.parentNode.querySelector('.scm-commit-files').hidden,
      'the files container starts hidden in the new repo',
    );
  });

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length) process.exit(1);
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
