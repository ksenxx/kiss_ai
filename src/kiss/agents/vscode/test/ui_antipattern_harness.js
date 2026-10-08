// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Shared jsdom harness for the ui_antipattern_*.test.js files: a full
// chat.html webview with a stub VS Code API, a stub Monaco (for content
// tabs), and helpers that open the Explorer / Source Control views and
// drive their context menus.
'use strict';

const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
// UI_ANTIPATTERN_MAIN_JS points the harness at another main.js (an older
// revision, say) to show that a regression test fails without its fix.
const MAIN_JS =
  process.env.UI_ANTIPATTERN_MAIN_JS || path.join(MEDIA, 'main.js');
const WD = '/ws/repo';
const SHA_A = 'a'.repeat(40);
const SHA_B = 'b'.repeat(40);

/**
 * Build a webview.  `win.prompt` / `win.confirm` throw: nothing may call
 * them.  `opts.bodyClass` replaces the default `remote-chat` body class
 * (`editor-tab-mode` builds the VS Code editor-tab flavour).  The remote
 * page is the desktop one (wide viewport: split chat/content layout)
 * unless `opts.narrow` is set, which builds the mobile remote where a
 * content tab replaces the chat.
 */
function makeWebview(opts) {
  const bodyClass = (opts && opts.bodyClass) || 'remote-chat';
  const wide = !(opts && opts.narrow);
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body class="' + bodyClass + '"');
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
      matches: wide && query === '(min-width: 900px)',
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
  // The VS Code webview sandbox has no allow-modals: a native dialog
  // would silently return false / null there.  Any call is a defect.
  win.prompt = function () {
    throw new Error('window.prompt must not be called');
  };
  win.confirm = function () {
    throw new Error('window.confirm must not be called');
  };
  const created = [];
  win.monaco = {
    editor: {
      defineTheme: () => {},
      setTheme: () => {},
      create: (holder, opts) => {
        let value = String(opts.value === undefined ? '' : opts.value);
        let version = 1;
        const listeners = [];
        const model = {
          getAlternativeVersionId: () => version,
          getValue: () => value,
          getLineCount: () => value.split('\n').length,
        };
        const editor = {
          getModel: () => model,
          onDidChangeModelContent: cb => listeners.push(cb),
          setPosition: () => {},
          revealLineInCenter: () => {},
          getDomNode: () => holder,
          dispose: () => {},
          layout: () => {},
          focus: () => {},
          _type: text => {
            value = String(text);
            version += 1;
            listeners.slice().forEach(cb => cb());
          },
        };
        created.push({holder, editor, value: opts.value, opts});
        return editor;
      },
    },
  };
  for (const file of [
    'marked.min.js',
    'panelCopy.js',
    'api.js',
    'contentContextMenu.js',
    'treeContextMenu.js',
  ]) {
    win.eval(fs.readFileSync(path.join(MEDIA, file), 'utf8'));
  }
  win.eval(
    fs.readFileSync(MAIN_JS, 'utf8') + '\n//# sourceURL=ui-antipattern-main.js',
  );
  return {win, posted, created};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

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

function ofType(posted, type) {
  return posted.filter(m => m.type === type);
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

function pinWorkspace(win) {
  send(win, {type: 'configData', config: {work_dir: WD}});
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
  click(win, byId(win, 'activity-scm'));
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

/** Minimal test runner shared by the three files. */
function makeRunner() {
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
  function report(title) {
    console.log(`\n${title}: ${passed} passed, ${failures.length} failed`);
    if (failures.length) process.exit(1);
  }
  return {test, report};
}

module.exports = {
  WD,
  SHA_A,
  SHA_B,
  makeWebview,
  send,
  sleep,
  click,
  keyboardClick,
  rightClick,
  key,
  byId,
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
};
