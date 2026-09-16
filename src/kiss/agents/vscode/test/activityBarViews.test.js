// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the remote webapp's activity bar views
// (media/chat.html + media/main.js): the Explorer (listDir /
// dirListing) and Source Control (gitStatus / gitLog) protocol
// handling that a browser test cannot steer as precisely —
//
// * a dirListing / gitStatus / gitLog reply carrying a stale token (an
//   older tree generation or refresh) is ignored,
// * task news (tasks_updated -> refreshHistory) re-lists the folders on
//   screen and re-reads git, debounced, but only for the VISIBLE view,
// * the commit graph lays a diamond history out on two lanes and
//   truncates listings and errors are shown in place,
// * the VS Code webview (no body.remote-chat) never posts any of the
//   three commands even with a remembered non-Tasks view.

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

function makeWebview(opts) {
  const {remote = true, savedView = null} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) html = html.replace('<body', '<body class="remote-chat"');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  if (savedView) win.localStorage.setItem('kiss-sidebar-view', savedView);
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=activitybar-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const WD = '/ws/repo';

/** Deliver the config reply that pins the workspace. */
function pinWorkspace(win) {
  send(win, {type: 'configData', config: {work_dir: WD}});
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function byId(win, id) {
  return win.document.getElementById(id);
}

function ofType(posted, type) {
  return posted.filter(m => m.type === type);
}

function rowsIn(win, sel) {
  return Array.from(win.document.querySelectorAll(sel));
}

async function main() {
  await test('Explorer: root listing, stale-token replies ignored, errors in place', async () => {
    const {win, posted} = makeWebview();
    pinWorkspace(win);
    click(win, byId(win, 'activity-explorer'));
    const list = ofType(posted, 'listDir');
    assert.strictEqual(list.length, 1);
    assert.strictEqual(list[0].path, WD);
    assert.strictEqual(list[0].workDir, WD);
    const token = list[0].token;
    assert.ok(/^\d+:\/ws\/repo$/.test(token), 'token = generation:path');
    // A reply for an older tree generation is dropped.
    send(win, {
      type: 'dirListing',
      token: '0:' + WD,
      path: WD,
      root: WD,
      entries: [{name: 'ghost', path: WD + '/ghost', isDir: false}],
    });
    assert.strictEqual(rowsIn(win, '.explorer-row.is-file').length, 0);
    // A reply for a folder the tree never asked about is dropped.
    send(win, {
      type: 'dirListing',
      token: token.split(':')[0] + ':' + WD + '/nowhere',
      path: WD + '/nowhere',
      root: WD,
      entries: [{name: 'ghost', path: WD + '/nowhere/ghost', isDir: false}],
    });
    assert.strictEqual(rowsIn(win, '.explorer-row.is-file').length, 0);
    // The real reply fills the root, folders first as sent, and a
    // truncated listing says so.
    send(win, {
      type: 'dirListing',
      token,
      path: WD,
      root: WD,
      truncated: true,
      entries: [
        {name: 'src', path: WD + '/src', isDir: true},
        {name: 'a.py', path: WD + '/a.py', isDir: false},
      ],
    });
    const names = rowsIn(
      win,
      '.explorer-row[aria-level="2"] .explorer-name',
    ).map(e => e.textContent);
    assert.deepStrictEqual(
      JSON.stringify(names),
      JSON.stringify(['src', 'a.py']),
    );
    const notes = rowsIn(win, '.explorer-note').map(e => e.textContent);
    assert.deepStrictEqual(
      JSON.stringify(notes),
      JSON.stringify(['(more entries not shown)']),
    );
    // Expanding a folder asks once; an error reply lands in place.
    const srcRow = win.document.querySelector(
      '.explorer-row[data-explorer-path="' + WD + '/src"]',
    );
    click(win, srcRow);
    const list2 = ofType(posted, 'listDir');
    assert.strictEqual(list2.length, 2);
    assert.strictEqual(list2[1].path, WD + '/src');
    assert.ok(srcRow.classList.contains('loading'));
    send(win, {
      type: 'dirListing',
      token: list2[1].token,
      path: WD + '/src',
      root: WD,
      error: 'Failed to list src: permission denied',
    });
    assert.ok(!srcRow.classList.contains('loading'));
    assert.strictEqual(
      srcRow.nextSibling.querySelector('.explorer-note').textContent,
      'Failed to list src: permission denied',
    );
    // A file click posts openFile with the tree's workspace.
    const fileRow = win.document.querySelector(
      '.explorer-row[data-explorer-path="' + WD + '/a.py"]',
    );
    click(win, fileRow);
    const opened = ofType(posted, 'openFile');
    assert.strictEqual(opened.length, 1);
    assert.strictEqual(opened[0].path, WD + '/a.py');
    assert.strictEqual(opened[0].workDir, WD);
    win.close();
  });

  await test('Explorer: task news re-lists loaded folders only; empty folder note', async () => {
    const {win, posted} = makeWebview();
    pinWorkspace(win);
    click(win, byId(win, 'activity-explorer'));
    const first = ofType(posted, 'listDir')[0];
    send(win, {
      type: 'dirListing',
      token: first.token,
      path: WD,
      root: WD,
      entries: [
        {name: 'src', path: WD + '/src', isDir: true},
        {name: 'docs', path: WD + '/docs', isDir: true},
      ],
    });
    const before = ofType(posted, 'listDir').length;
    send(win, {type: 'tasks_updated'});
    const after = ofType(posted, 'listDir');
    // Only the root was listed so far: exactly one re-list, not one
    // per folder row.
    assert.strictEqual(after.length, before + 1);
    assert.strictEqual(after[after.length - 1].path, WD);
    // The re-listing keeps rows and drops vanished ones.
    send(win, {
      type: 'dirListing',
      token: after[after.length - 1].token,
      path: WD,
      root: WD,
      entries: [{name: 'src', path: WD + '/src', isDir: true}],
    });
    const names = rowsIn(
      win,
      '.explorer-row[aria-level="2"] .explorer-name',
    ).map(e => e.textContent);
    assert.deepStrictEqual(JSON.stringify(names), JSON.stringify(['src']));
    // An empty folder says so.
    const srcRow = win.document.querySelector(
      '.explorer-row[data-explorer-path="' + WD + '/src"]',
    );
    click(win, srcRow);
    const req = ofType(posted, 'listDir').pop();
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD + '/src',
      root: WD,
      entries: [],
    });
    assert.strictEqual(
      srcRow.nextSibling.querySelector('.explorer-note').textContent,
      '(empty)',
    );
    // Keyboard: Enter on a folder row toggles it (collapses here).
    srcRow.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
    );
    assert.strictEqual(srcRow.getAttribute('aria-expanded'), 'false');
    assert.ok(srcRow.nextSibling.hidden);
    win.close();
  });

  await test('Source Control: stale tokens ignored; diamond history on two lanes', async () => {
    const {win, posted} = makeWebview();
    pinWorkspace(win);
    click(win, byId(win, 'activity-scm'));
    const st = ofType(posted, 'gitStatus');
    const lg = ofType(posted, 'gitLog');
    assert.strictEqual(st.length, 1);
    assert.strictEqual(lg.length, 1);
    assert.strictEqual(st[0].token, lg[0].token);
    assert.strictEqual(st[0].workDir, WD);
    assert.strictEqual(lg[0].limit, 50);
    assert.strictEqual(
      byId(win, 'scm-changes').textContent.trim(),
      'Loading...',
    );
    // Stale replies leave "Loading..." up.
    send(win, {type: 'gitStatus', token: 'stale', workDir: WD, changes: []});
    send(win, {type: 'gitLog', token: 'stale', workDir: WD, commits: []});
    assert.strictEqual(
      byId(win, 'scm-changes').textContent.trim(),
      'Loading...',
    );
    assert.strictEqual(byId(win, 'scm-graph').textContent.trim(), 'Loading...');
    send(win, {
      type: 'gitStatus',
      token: st[0].token,
      workDir: WD,
      repo: WD,
      branch: 'main',
      changes: [
        {path: 'a.py', absPath: WD + '/a.py', status: 'M', group: 'changes'},
      ],
    });
    assert.strictEqual(byId(win, 'scm-branch').textContent, 'main');
    assert.strictEqual(byId(win, 'scm-changes-count').textContent, '1');
    // One group only: no group header.
    assert.strictEqual(rowsIn(win, '#scm-changes .scm-group-hdr').length, 0);
    // Diamond: M merges B (first parent) and C; both descend from R.
    const M = 'm'.repeat(40);
    const B = 'b'.repeat(40);
    const C = 'c'.repeat(40);
    const R = 'r'.repeat(40);
    const commit = (sha, parents, subject, files) => ({
      sha,
      shortSha: sha.slice(0, 7),
      parents,
      author: 'A',
      date: new Date().toISOString(),
      refs: [],
      subject,
      files: files || [],
    });
    send(win, {
      type: 'gitLog',
      token: lg[0].token,
      workDir: WD,
      repo: WD,
      head: M,
      commits: [
        commit(M, [B, C], 'merge'),
        commit(C, [R], 'side', [{path: 'c.py', status: 'A'}]),
        commit(B, [R], 'main work', [{path: 'b.py', status: 'M'}]),
        commit(R, [], 'root', [{path: 'r.py', status: 'A'}]),
      ],
    });
    const subjects = rowsIn(win, '#scm-graph .scm-commit-subject').map(
      e => e.textContent,
    );
    assert.deepStrictEqual(
      JSON.stringify(subjects),
      JSON.stringify([
        'Uncommitted changes',
        'merge',
        'side',
        'main work',
        'root',
      ]),
    );
    assert.strictEqual(byId(win, 'scm-graph-count').textContent, '4');
    const widths = rowsIn(win, '#scm-graph .scm-graph-cell').map(e =>
      Number(e.getAttribute('width')),
    );
    assert.deepStrictEqual(
      JSON.stringify(widths),
      JSON.stringify([28, 28, 28, 28, 28]),
    );
    const paths = rowsIn(win, '#scm-graph .scm-commit-wrap').map(
      e => e.querySelectorAll('path').length,
    );
    // uncommitted: 1 out.  merge: 1 in + 2 out.  side: 1 in + 1 out +
    // lane 0 passing through.  main work: 1 in + 1 out + the side
    // lane passing through — both already heading to R, so they
    // converge on one lane and root gets a single 'in'.
    assert.deepStrictEqual(
      JSON.stringify(paths),
      JSON.stringify([1, 3, 3, 3, 1]),
    );
    // Expand the side commit: its file opens with the repo root.
    const side = win.document.querySelector(
      '#scm-graph .scm-commit[data-scm-sha="' + C + '"]',
    );
    click(win, side);
    assert.strictEqual(side.getAttribute('aria-expanded'), 'true');
    const fileRow = side.parentNode.querySelector('.scm-commit-files .scm-row');
    assert.strictEqual(fileRow.dataset.scmPath, WD + '/c.py');
    click(win, fileRow);
    const opened = ofType(posted, 'openFile');
    assert.strictEqual(opened.length, 1);
    assert.strictEqual(opened[0].path, WD + '/c.py');
    // Section headers collapse their lists.
    click(win, byId(win, 'scm-graph-toggle'));
    assert.ok(byId(win, 'scm-graph').hidden);
    assert.strictEqual(
      byId(win, 'scm-graph-toggle').getAttribute('aria-expanded'),
      'false',
    );
    click(win, byId(win, 'scm-graph-toggle'));
    assert.ok(!byId(win, 'scm-graph').hidden);
    // Task news re-reads git once (debounced) for the visible view.
    const nSt = ofType(posted, 'gitStatus').length;
    send(win, {type: 'tasks_updated'});
    send(win, {type: 'tasks_updated'});
    send(win, {type: 'tasks_updated'});
    assert.strictEqual(ofType(posted, 'gitStatus').length, nSt);
    await sleep(600);
    assert.strictEqual(ofType(posted, 'gitStatus').length, nSt + 1);
    assert.strictEqual(
      ofType(posted, 'listDir').length,
      0,
      'Explorer hidden: no listDir',
    );
    // Expansion state survives the re-render.
    send(win, {
      type: 'gitLog',
      token: ofType(posted, 'gitLog').pop().token,
      workDir: WD,
      repo: WD,
      head: M,
      commits: [commit(C, [], 'side', [{path: 'c.py', status: 'A'}])],
    });
    const sideAgain = win.document.querySelector(
      '#scm-graph .scm-commit[data-scm-sha="' + C + '"]',
    );
    assert.strictEqual(sideAgain.getAttribute('aria-expanded'), 'true');
    win.close();
  });

  await test('Source Control: error replies and empty repo', async () => {
    const {win, posted} = makeWebview();
    pinWorkspace(win);
    click(win, byId(win, 'activity-scm'));
    const tok = ofType(posted, 'gitStatus')[0].token;
    send(win, {
      type: 'gitStatus',
      token: tok,
      workDir: WD,
      error: 'Not a git repository: /ws/repo',
    });
    send(win, {
      type: 'gitLog',
      token: tok,
      workDir: WD,
      error: 'Not a git repository: /ws/repo',
    });
    assert.strictEqual(
      byId(win, 'scm-changes').textContent.trim(),
      'Not a git repository: /ws/repo',
    );
    assert.strictEqual(
      byId(win, 'scm-graph').textContent.trim(),
      'Not a git repository: /ws/repo',
    );
    assert.strictEqual(byId(win, 'scm-branch').textContent, '');
    assert.strictEqual(byId(win, 'scm-changes-count').textContent, '');
    // The refresh button asks again with a fresh token.
    click(win, byId(win, 'scm-refresh'));
    const st = ofType(posted, 'gitStatus');
    assert.strictEqual(st.length, 2);
    assert.notStrictEqual(st[1].token, tok);
    send(win, {
      type: 'gitStatus',
      token: st[1].token,
      workDir: WD,
      repo: WD,
      branch: 'main',
      changes: [],
    });
    send(win, {
      type: 'gitLog',
      token: st[1].token,
      workDir: WD,
      repo: WD,
      head: '',
      commits: [],
    });
    assert.strictEqual(
      byId(win, 'scm-changes').textContent.trim(),
      'No changes',
    );
    assert.strictEqual(byId(win, 'scm-graph').textContent.trim(), 'No commits');
    win.close();
  });

  await test('view choice is remembered and restored (remote only)', async () => {
    const {win, posted} = makeWebview({savedView: 'explorer'});
    assert.strictEqual(
      byId(win, 'activity-explorer').getAttribute('aria-selected'),
      'true',
    );
    assert.ok(byId(win, 'sidebar-tab-history-panel').hidden);
    assert.ok(!byId(win, 'sidebar-explorer-panel').hidden);
    // No workspace yet: nothing was asked, the tree says so.
    assert.strictEqual(ofType(posted, 'listDir').length, 0);
    assert.strictEqual(
      byId(win, 'explorer-tree').textContent.trim(),
      'No workspace folder',
    );
    // The config reply pins the workspace and the tree loads.
    pinWorkspace(win);
    assert.strictEqual(ofType(posted, 'listDir').length, 1);
    click(win, byId(win, 'activity-tasks'));
    assert.strictEqual(win.localStorage.getItem('kiss-sidebar-view'), 'tasks');
    assert.ok(!byId(win, 'sidebar-tab-history-panel').hidden);
    win.close();
  });

  await test('VS Code webview: bar inert, Tasks view stays, no commands', async () => {
    const {win, posted} = makeWebview({remote: false, savedView: 'scm'});
    pinWorkspace(win);
    assert.strictEqual(
      byId(win, 'activity-tasks').getAttribute('aria-selected'),
      'true',
    );
    assert.ok(!byId(win, 'sidebar-tab-history-panel').hidden);
    assert.ok(byId(win, 'sidebar-scm-panel').hidden);
    send(win, {type: 'tasks_updated'});
    await sleep(500);
    assert.strictEqual(ofType(posted, 'listDir').length, 0);
    assert.strictEqual(ofType(posted, 'gitStatus').length, 0);
    assert.strictEqual(ofType(posted, 'gitLog').length, 0);
    // Stray replies are harmless.
    send(win, {
      type: 'dirListing',
      token: '1:/x',
      path: '/x',
      root: '/x',
      entries: [],
    });
    send(win, {type: 'gitStatus', token: '1', workDir: '/x', changes: []});
    send(win, {type: 'gitLog', token: '1', workDir: '/x', commits: []});
    win.close();
  });

  await test('Source Control: graph waits for the matching log during a refresh', async () => {
    const {win, posted} = makeWebview();
    pinWorkspace(win);
    click(win, byId(win, 'activity-scm'));
    const tok = ofType(posted, 'gitStatus')[0].token;
    const H1 = '1'.repeat(40);
    const H2 = '2'.repeat(40);
    const commit = (sha, parents, subject) => ({
      sha,
      shortSha: sha.slice(0, 7),
      parents,
      author: 'A',
      date: new Date().toISOString(),
      refs: [],
      subject,
      files: [],
    });
    send(win, {
      type: 'gitStatus',
      token: tok,
      workDir: WD,
      repo: WD,
      branch: 'main',
      changes: [],
    });
    // Status alone does not paint the graph: it still says Loading.
    assert.strictEqual(byId(win, 'scm-graph').textContent.trim(), 'Loading...');
    send(win, {
      type: 'gitLog',
      token: tok,
      workDir: WD,
      repo: WD,
      head: H1,
      commits: [commit(H1, [], 'one')],
    });
    let subjects = rowsIn(win, '#scm-graph .scm-commit-subject').map(
      e => e.textContent,
    );
    assert.deepStrictEqual(JSON.stringify(subjects), JSON.stringify(['one']));
    // A manual refresh: the NEW status (one change) arrives first.
    // The graph keeps showing the old pair rather than hanging a new
    // uncommitted row off the OLD head; the changes list repaints.
    click(win, byId(win, 'scm-refresh'));
    const tok2 = ofType(posted, 'gitStatus').pop().token;
    assert.notStrictEqual(tok2, tok);
    send(win, {
      type: 'gitStatus',
      token: tok2,
      workDir: WD,
      repo: WD,
      branch: 'main',
      changes: [
        {path: 'n.py', absPath: WD + '/n.py', status: 'U', group: 'changes'},
      ],
    });
    assert.strictEqual(byId(win, 'scm-changes-count').textContent, '1');
    subjects = rowsIn(win, '#scm-graph .scm-commit-subject').map(
      e => e.textContent,
    );
    assert.deepStrictEqual(JSON.stringify(subjects), JSON.stringify(['one']));
    send(win, {
      type: 'gitLog',
      token: tok2,
      workDir: WD,
      repo: WD,
      head: H2,
      commits: [commit(H2, [H1], 'two'), commit(H1, [], 'one')],
    });
    subjects = rowsIn(win, '#scm-graph .scm-commit-subject').map(
      e => e.textContent,
    );
    assert.deepStrictEqual(
      JSON.stringify(subjects),
      JSON.stringify(['Uncommitted changes', 'two', 'one']),
    );
    win.close();
  });

  await test('Explorer: an entry that turned from file into folder is a live folder', async () => {
    const {win, posted} = makeWebview();
    pinWorkspace(win);
    click(win, byId(win, 'activity-explorer'));
    const first = ofType(posted, 'listDir')[0];
    send(win, {
      type: 'dirListing',
      token: first.token,
      path: WD,
      root: WD,
      entries: [{name: 'thing', path: WD + '/thing', isDir: false}],
    });
    assert.ok(
      win.document.querySelector(
        '.explorer-row.is-file[data-explorer-path="' + WD + '/thing"]',
      ),
    );
    click(win, byId(win, 'explorer-refresh'));
    const again = ofType(posted, 'listDir').pop();
    send(win, {
      type: 'dirListing',
      token: again.token,
      path: WD,
      root: WD,
      entries: [{name: 'thing', path: WD + '/thing', isDir: true}],
    });
    const row = win.document.querySelector(
      '.explorer-row[data-explorer-path="' + WD + '/thing"]',
    );
    assert.ok(row.classList.contains('is-dir'));
    assert.strictEqual(
      rowsIn(win, '.explorer-row[data-explorer-path="' + WD + '/thing"]')
        .length,
      1,
    );
    click(win, row);
    assert.strictEqual(row.getAttribute('aria-expanded'), 'true');
    const req = ofType(posted, 'listDir').pop();
    assert.strictEqual(req.path, WD + '/thing');
    win.close();
  });

  await test('Explorer: two folders resolving to one target stay separate nodes', async () => {
    const {win, posted} = makeWebview();
    pinWorkspace(win);
    click(win, byId(win, 'activity-explorer'));
    const first = ofType(posted, 'listDir')[0];
    // The daemon canonicalizes symlinks: both aliases report the SAME
    // path for their children; the tree keys rows by the path it
    // asked for, so each alias keeps its own state.
    send(win, {
      type: 'dirListing',
      token: first.token,
      path: WD,
      root: WD,
      entries: [
        {name: 'alias-one', path: WD + '/target', isDir: true},
        {name: 'alias-two', path: WD + '/target', isDir: true},
      ],
    });
    const one = win.document.querySelector(
      '.explorer-row[data-explorer-path="' + WD + '/alias-one"]',
    );
    const two = win.document.querySelector(
      '.explorer-row[data-explorer-path="' + WD + '/alias-two"]',
    );
    assert.ok(one && two);
    click(win, one);
    const req = ofType(posted, 'listDir').pop();
    assert.strictEqual(req.path, WD + '/alias-one');
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD + '/target',
      root: WD,
      entries: [{name: 'sub', path: WD + '/target/sub', isDir: true}],
    });
    assert.strictEqual(one.getAttribute('aria-expanded'), 'true');
    assert.ok(!one.nextSibling.hidden, 'alias-one shows its children');
    assert.strictEqual(two.getAttribute('aria-expanded'), 'false');
    assert.ok(two.nextSibling.hidden, 'alias-two stays collapsed');
    assert.ok(
      win.document.querySelector(
        '.explorer-row[data-explorer-path="' + WD + '/alias-one/sub"]',
      ),
      'children are keyed under the alias that listed them',
    );
    win.close();
  });

  await test('active tab changes retarget the views to the tab workspace', async () => {
    // No pinned workspace: every tab is visible, and each browses the
    // folder it runs in.
    const {win, posted} = makeWebview();
    send(win, {type: 'configData', config: {work_dir: ''}});
    click(win, byId(win, 'activity-explorer'));
    assert.strictEqual(ofType(posted, 'listDir').length, 0);
    assert.strictEqual(
      byId(win, 'explorer-tree').textContent.trim(),
      'No workspace folder',
    );
    const mine = posted.find(m => m.type === 'ready').tabId;
    send(win, {
      type: 'tabs_state',
      tabs: [
        {tabId: mine, chatId: 'c1', title: 'mine', workDir: ''},
        {tabId: 'other-1', chatId: 'c2', title: 'other', workDir: '/ws/other'},
      ],
    });
    const otherStrip = win.document.querySelector(
      '.chat-tab[data-tab-id="other-1"]',
    );
    assert.ok(otherStrip, 'the adopted tab has a strip');
    click(win, otherStrip);
    const last = ofType(posted, 'listDir').pop();
    assert.ok(last, 'switching tabs re-roots the tree');
    assert.strictEqual(last.path, '/ws/other');
    assert.strictEqual(last.workDir, '/ws/other');
    // Back on the first tab the tree follows again.
    click(
      win,
      win.document.querySelector('.chat-tab[data-tab-id="' + mine + '"]'),
    );
    assert.strictEqual(
      byId(win, 'explorer-tree').textContent.trim(),
      'No workspace folder',
    );
    win.close();
  });

  console.log(`\n${passed} passed, ${failures.length} failed`);
  // main.js starts timers (metainfo polling, debounces) in every window
  // opened above; exit explicitly instead of waiting for them.
  process.exit(failures.length ? 1 : 0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
