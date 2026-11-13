// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Shared jsdom harness for the ui_antipattern_*.test.js files (and a few
// others that drive the same context menus and dirty-tab dialogs): a full
// chat.html page with a stub acquireTransport, a stub Monaco (for content
// tabs), and helpers that open the Explorer / Source Control views and
// drive their context menus.
'use strict';

const fs = require('fs');
const harness = require('./simplify2_harness');

// UI_ANTIPATTERN_MAIN_JS points the harness at another main.js (an older
// revision, say) to show that a regression test fails without its fix.
const MAIN_JS = process.env.UI_ANTIPATTERN_MAIN_JS;
const WD = '/ws/repo';
const SHA_A = 'a'.repeat(40);
const SHA_B = 'b'.repeat(40);

// Every native dialog a page guarded by forbidNativeDialogs called, as
// `window.prompt` / `window.confirm`.  The guards throw, but a throw
// inside a DOM listener never reaches the test (jsdom reports it on its
// console and the dispatch returns normally), so makeRunner fails the
// test from this ledger as well.
const nativeDialogCalls = [];

/**
 * Make `w.prompt` / `w.confirm` record the call for makeRunner and throw:
 * the page ships its own dialogs and a native one would block the whole
 * page, so any call is a defect.
 */
function forbidNativeDialogs(w) {
  w.prompt = function () {
    nativeDialogCalls.push('window.prompt');
    throw new Error('window.prompt must not be called');
  };
  w.confirm = function () {
    nativeDialogCalls.push('window.confirm');
    throw new Error('window.confirm must not be called');
  };
}

/**
 * Build a page.  `win.prompt` / `win.confirm` throw: nothing may call
 * them.  The page is the desktop one (wide viewport: split chat/content
 * layout) unless `opts.narrow` is set, which builds the mobile page where
 * a content tab replaces the chat.
 */
function makePage(opts) {
  const wide = !(opts && opts.narrow);
  let created;
  const {win, posted} = harness.makePage({
    bodyClass: 'remote-chat',
    extraScripts: [
      'marked.min.js',
      'contentContextMenu.js',
      'treeContextMenu.js',
    ],
    sourceUrl: 'ui-antipattern-main.js',
    mainSource: src => (MAIN_JS ? fs.readFileSync(MAIN_JS, 'utf8') : src),
    beforeScripts(w) {
      harness.stubRaf(w);
      harness.fakeMatchMedia(w, wide);
      Object.defineProperty(w.navigator, 'clipboard', {
        value: {writeText: () => Promise.resolve()},
        configurable: true,
      });
      forbidNativeDialogs(w);
      created = harness.stubMonaco(w);
    },
  });
  return {win, posted, created};
}

const {send, sleep, ofType, byId} = harness;

/** A pointer click (detail 1), as a mouse produces. */
function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true, detail: 1}));
}

/** A keyboard-originated click: Enter / Space on a focused button (detail 0). */
function keyboardClick(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true, detail: 0}));
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

function key(win, el, name, extra) {
  return el.dispatchEvent(
    new win.KeyboardEvent(
      'keydown',
      Object.assign({key: name, bubbles: true, cancelable: true}, extra || {}),
    ),
  );
}

// The two forms a keydown inside an IME composition takes (`extra` for
// `key`): the standard flag, and the legacy keyCode 229 some browsers
// send instead.  Neither is a submit.
const IME_FORMS = [{isComposing: true}, {keyCode: 229}];

/**
 * Bring a workspace section of the task-info panel (meta-explorer /
 * meta-scm) on screen: expand it if it is collapsed.  An expanded
 * section is already live (the workspace pin loads it).
 */
function showSection(win, id) {
  const section = byId(win, id);
  if (section.classList.contains('collapsed'))
    click(win, section.querySelector('.meta-section-toggle'));
}

function all(win, sel) {
  return Array.from(win.document.querySelectorAll(sel));
}

function menuItem(win, label) {
  return all(win, '#sidebar-context-menu .tree-ctx-item').find(
    el => el.querySelector('.tree-ctx-label').textContent === label,
  );
}

function toast(win, id) {
  // Compared, not put in a selector: an id may carry quotes or a newline.
  return (
    all(win, '[data-notification-id]').find(
      el => el.dataset.notificationId === id,
    ) || null
  );
}

function toastButton(el, label) {
  return (
    Array.from(el.querySelectorAll('.kiss-notification-action')).find(
      b => b.textContent.trim() === label,
    ) || null
  );
}

function toastInput(el) {
  return el.querySelector('.kiss-notification-input');
}

/**
 * The notification id of the "already exists" question for *name* in the
 * folder *dest* on *machine* ('' is this machine), as main.js keys it.
 */
function overwriteId(machine, dest, name) {
  return 'fs-overwrite:' + JSON.stringify([machine, dest, name]);
}

/** Deliver the config reply that pins the workspace (*wd*, default WD). */
function pinWorkspace(win, wd) {
  send(win, {type: 'configData', config: {work_dir: wd || WD}});
}

/** Open the Explorer on WD and answer its root listing with *entries*. */
function openExplorer(win, posted, entries) {
  pinWorkspace(win);
  showSection(win, 'meta-explorer');
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

function commit(sha, parents, subject, message) {
  return {
    sha,
    shortSha: sha.slice(0, 7),
    parents,
    author: 'A',
    date: new Date().toISOString(),
    subject,
    message,
    refs: [],
  };
}

/** Open Source Control on WD with one commit and return its graph row. */
function openScm(win, posted) {
  pinWorkspace(win);
  showSection(win, 'meta-scm');
  const st = ofType(posted, 'gitStatus');
  const lg = ofType(posted, 'gitLog');
  const worktrees = [
    {
      path: WD,
      name: 'repo',
      head: SHA_A,
      branch: 'main',
      detached: false,
      current: true,
      changes: [],
    },
  ];
  send(win, {
    type: 'gitStatus',
    token: st[st.length - 1].token,
    workDir: WD,
    repo: WD,
    branch: 'main',
    changes: [],
    worktrees,
  });
  send(win, {
    type: 'gitLog',
    token: lg[lg.length - 1].token,
    workDir: WD,
    repo: WD,
    head: SHA_A,
    commits: [
      commit(SHA_A, [SHA_B], 'Title line', 'Title line\n\nBody paragraph.'),
      commit(SHA_B, [], 'root', 'root'),
    ],
    worktrees,
  });
  return all(win, '#scm-graph .scm-commit').find(
    el => el.querySelector('.scm-commit-subject').textContent === 'Title line',
  );
}

/** Right-click *row* and run the context-menu item labelled *label*. */
function runMenuItem(win, row, label) {
  rightClick(win, row);
  const item = menuItem(win, label);
  if (!item) throw new Error("no context-menu item '" + label + "'");
  click(win, item);
}

/** Right-click tab *tabNode* of the strip and run its menu item *label*. */
function pickTabMenuItem(win, tabNode, label) {
  rightClick(win, tabNode);
  const item = all(win, '#tab-context-menu .tab-ctx-item').find(
    el => el.textContent === label,
  );
  if (!item) throw new Error("no tab-menu item '" + label + "'");
  click(win, item);
}

/** The scoped transcript fetches (`resumeSession`) posted for tabs *ids*. */
function scopedFetches(posted, ids) {
  return ofType(posted, 'resumeSession').filter(
    m => m.scoped === true && ids.includes(m.tabId),
  );
}

/**
 * Open an editable content tab for /shared/notes.txt owned by chat tab
 * a1 and type into it, so the tab is dirty.  Returns the editor stub.
 */
async function openDirtyContentTab(ctx) {
  const {win, posted, created} = ctx;
  send(win, {type: 'configData', config: {work_dir: '/ws/a'}});
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'a1', workDir: '/ws/a', chatId: 'chat-1'},
      {tabId: 'b1', workDir: '/ws/b', chatId: 'chat-2'},
    ],
  });
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/shared/notes.txt',
    name: 'notes.txt',
    content: 'v1 on disk',
    version: 'ver-1',
  });
  // The stubbed editor is created on the next microtask.
  await sleep(0);
  if (created.length < 1) throw new Error('the content tab created no editor');
  created[0].editor._type('v1 edited');
  return {editor: created[0].editor, posted};
}

function contentTabStrips(win) {
  return win.document.querySelectorAll('.chat-tab.content-tab');
}

/**
 * Minimal test runner: `test(name, fn)` runs and records one test,
 * `report(title)` prints the totals and exits 1 on any failure.
 *
 * A synchronous `fn` is recorded before `test` returns, so a suite that
 * calls `test(...)` back to back and then `report(...)` without awaiting
 * counts every test.  An async `fn` returns a promise the suite must
 * await before `report`, which exits 1 while one is still running: its
 * failure would arrive after the totals and leave the exit status 0.
 * A test still running when the process ends fails it as well: a promise
 * that never settles keeps nothing on the event loop, so the suite
 * awaiting it would end with status 0 and no totals.
 *
 * `opts.windows` is an array the suite pushes every page it builds onto:
 * the runner closes and removes them when a test has ended, pass or fail,
 * and no other test is still running (an async one runs until its promise
 * has settled), so a page is never closed under a test that uses it and
 * nothing a page armed keeps node alive after the last test.
 */
function makeRunner(opts) {
  const windows = (opts && opts.windows) || [];
  let passed = 0;
  // Names of the async tests whose promise has not settled yet.
  const running = [];
  const failures = [];
  function closeWindows() {
    if (running.length) return;
    while (windows.length) windows.pop().close();
  }
  function pass(name) {
    // A forbidden native dialog called from a listener while the test
    // ran is the test's failure, whatever its own assertions said.
    if (nativeDialogCalls.length) {
      const calls = nativeDialogCalls.splice(0).join(', ');
      fail(name, new Error(`${calls} must not be called (native dialog)`));
      return;
    }
    closeWindows();
    passed++;
    console.log(`  \u2713 ${name}`);
  }
  function fail(name, e) {
    nativeDialogCalls.length = 0;
    closeWindows();
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
  function settle(name) {
    running.splice(running.indexOf(name), 1);
  }
  function test(name, fn) {
    let result;
    try {
      result = fn();
    } catch (e) {
      fail(name, e);
      return undefined;
    }
    if (result && typeof result.then === 'function') {
      running.push(name);
      return result.then(
        () => {
          settle(name);
          pass(name);
        },
        e => {
          settle(name);
          fail(name, e);
        },
      );
    }
    pass(name);
    return undefined;
  }
  function report(title) {
    if (running.length) {
      console.log(`\n${title}: ${running.length} async test(s) still running`);
      process.exit(1);
    }
    console.log(`\n${title}: ${passed} passed, ${failures.length} failed`);
    if (failures.length) process.exit(1);
  }
  // A status that already says failure (`report`'s exit, a crash) stands.
  function failRunningAtExit(code) {
    if (code !== 0 || !running.length) return;
    console.log(
      `\n${running.length} async test(s) still running at exit: ` +
        running.join(', '),
    );
    process.exitCode = 1;
  }
  process.on('exit', failRunningAtExit);
  /** How many tests failed so far (a probe for a runner under test). */
  function failed() {
    return failures.length;
  }
  return {test, report, failed};
}

module.exports = {
  WD,
  SHA_A,
  SHA_B,
  commit,
  makePage,
  forbidNativeDialogs,
  send,
  sleep,
  click,
  keyboardClick,
  rightClick,
  key,
  IME_FORMS,
  byId,
  showSection,
  ofType,
  all,
  menuItem,
  toast,
  toastButton,
  toastInput,
  overwriteId,
  openExplorer,
  explorerRow,
  openScm,
  runMenuItem,
  pickTabMenuItem,
  scopedFetches,
  openDirtyContentTab,
  contentTabStrips,
  makeRunner,
  pinWorkspace,
};
