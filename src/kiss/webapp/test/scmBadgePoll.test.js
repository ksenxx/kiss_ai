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
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const WD = '/ws/repo';

/**
 * Replace the window's setInterval/clearInterval with a fake clock so
 * the poll can be advanced deterministically instead of waited out.
 */
function installFakeIntervals(win) {
  let now = 0;
  let nextId = 1;
  const intervals = [];
  win.setInterval = function (cb, ms) {
    const iv = {id: nextId++, cb, ms: Number(ms) || 0, next: now + (Number(ms) || 0)};
    intervals.push(iv);
    return iv.id;
  };
  win.clearInterval = function (id) {
    const i = intervals.findIndex(iv => iv.id === id);
    if (i >= 0) intervals.splice(i, 1);
  };
  function tick(ms) {
    const target = now + ms;
    for (;;) {
      const due = intervals.filter(iv => iv.next <= target);
      if (due.length === 0) break;
      due.sort((a, b) => a.next - b.next);
      const iv = due[0];
      now = iv.next;
      iv.next += iv.ms;
      iv.cb();
    }
    now = target;
  }
  return {tick, intervals};
}

function makePage(narrow) {
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
  const clock = installFakeIntervals(win);
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};
  const posted = [];
  win.acquireTransport = function () {
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
      matches: !narrow && query === '(min-width: 900px)',
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
      '\n//# sourceURL=scm-badge-main.js',
  );
  return {win, posted, clock};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byId(win, id) {
  return win.document.getElementById(id);
}

function ofType(posted, type) {
  return posted.filter(m => m.type === type);
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
  return {type: 'gitStatus', token, workDir: WD, repo: WD, branch: 'main', changes};
}

function setHidden(win, hidden) {
  Object.defineProperty(win.document, 'hidden', {
    configurable: true,
    get: () => hidden,
  });
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
  assert.ok(logsAfterLoad >= logsBefore);
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
  const st = ofType(posted, 'gitStatus');
  return st[st.length - 1].token;
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
  console.log('  ok - an unchanged poll reply keeps focus; a changed one updates');
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

function main() {
  testBadgeFollowsTheWorkingTree();
  testPollPausesWhenHiddenOrDisconnected();
  testNoWorkspaceNoPoll();
  testUnchangedPollReplyKeepsFocus();
  testNoPollWhileThePanelIsOffScreen();
  console.log('scmBadgePoll.test.js: all tests passed');
  process.exit(0);
}

main();
