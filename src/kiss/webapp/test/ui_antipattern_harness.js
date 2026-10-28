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
      // The page ships its own dialogs; a native prompt/confirm would block
      // the whole page.  Any call is a defect.
      w.prompt = function () {
        throw new Error('window.prompt must not be called');
      };
      w.confirm = function () {
        throw new Error('window.confirm must not be called');
      };
      created = harness.stubMonaco(w);
    },
  });
  return {win, posted, created};
}

const {send, sleep, ofType} = harness;

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

function byId(win, id) {
  return win.document.getElementById(id);
}

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
  return win.document.querySelector('[data-notification-id="' + id + '"]');
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
  // The editor is created asynchronously (Monaco loads on demand).
  for (let i = 0; i < 100 && created.length < 1; i++) await sleep(20);
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
 * await before `report`.
 */
function makeRunner() {
  let passed = 0;
  const failures = [];
  function pass(name) {
    passed++;
    console.log(`  \u2713 ${name}`);
  }
  function fail(name, e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
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
      return result.then(
        () => pass(name),
        e => fail(name, e),
      );
    }
    pass(name);
    return undefined;
  }
  function report(title) {
    console.log(`\n${title}: ${passed} passed, ${failures.length} failed`);
    if (failures.length) process.exit(1);
  }
  return {test, report};
}

module.exports = {
  WD,
  SHA_A,
  makePage,
  send,
  sleep,
  click,
  keyboardClick,
  rightClick,
  key,
  byId,
  showSection,
  ofType,
  all,
  menuItem,
  toast,
  toastButton,
  toastInput,
  openExplorer,
  explorerRow,
  openScm,
  runMenuItem,
  openDirtyContentTab,
  contentTabStrips,
  makeRunner,
  pinWorkspace,
};
