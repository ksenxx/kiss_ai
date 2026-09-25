// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) regression tests for defects a read-only review
// found in the remote webapp's sidebar menus and folder picker
// (media/main.js + media/treeContextMenu.js):
//
// * a folder listing landing while a row is being renamed used to add
//   a second row for the same entry (and an error listing wiped the box),
// * "Select Folder" saved whatever was typed, even a path the daemon
//   had just failed to list,
// * a picked folder was not browsed when the active chat had pinned
//   another folder (sidebarWorkDir preferred the tab's own work dir),
// * a failed git action (a cherry-pick stopping on conflicts) left the
//   Source Control view stale,
// * "Copy Commit Message" copied only the subject line,
// * dismissing the optional tag message aborted "Create Tag..." instead
//   of creating a lightweight tag,
// * a successful Create Branch / Create Tag notification lost the name.

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
  const copied = [];
  Object.defineProperty(win.navigator, 'clipboard', {
    value: {
      writeText: text => {
        copied.push(text);
        return Promise.resolve();
      },
    },
    configurable: true,
  });
  const prompts = [];
  win.prompt = () => prompts.shift();
  win.confirm = () => true;
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'treeContextMenu.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=sidebar-fixes-main.js',
  );
  return {win, posted, copied, prompts};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const WD = '/ws/repo';

function pinWorkspace(win) {
  send(win, {type: 'configData', config: {work_dir: WD}});
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

function key(win, el, name, extra) {
  el.dispatchEvent(
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

function commit(sha, parents, subject, message) {
  return {
    sha,
    shortSha: sha.slice(0, 7),
    parents,
    author: 'A',
    date: new Date().toISOString(),
    refs: [],
    subject,
    message,
    files: [],
  };
}

/** Open Source Control on WD with a two-commit history. */
function openScm(win, posted) {
  pinWorkspace(win);
  click(win, byId(win, 'activity-scm'));
  const st = ofType(posted, 'gitStatus');
  const lg = ofType(posted, 'gitLog');
  const token = st[st.length - 1].token;
  send(win, {
    type: 'gitStatus',
    token,
    workDir: WD,
    repo: WD,
    branch: 'main',
    changes: [],
    worktrees: [
      {
        path: WD,
        name: 'repo',
        head: SHA_A,
        branch: 'main',
        detached: false,
        current: true,
        changes: [],
      },
    ],
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
    worktrees: [
      {
        path: WD,
        name: 'repo',
        head: SHA_A,
        branch: 'main',
        detached: false,
        current: true,
      },
    ],
  });
  return rows(win, '#scm-graph .scm-commit').find(
    el => el.querySelector('.scm-commit-subject').textContent === 'Title line',
  );
}

async function main() {
  await test('Explorer: a listing during Rename... adds no duplicate row and keeps the box', async () => {
    const {win, posted} = makeWebview();
    const req = openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
      {name: 'b.txt', path: WD + '/b.txt', isDir: false},
    ]);
    const row = explorerRow(win, WD + '/a.txt');
    rightClick(win, row);
    click(win, menuItem(win, 'Rename...'));
    await sleep(5);
    const input = row.querySelector('input.explorer-input');
    assert.ok(input, 'the rename box replaces the name');
    assert.strictEqual(win.document.activeElement, input, 'the box has focus');
    input.value = 'renamed.txt';
    input.setSelectionRange(2, 5);
    // A refresh of the same folder lands mid-edit (a task touched it).
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: req.path,
      root: req.path,
      entries: [
        {name: 'a.txt', path: WD + '/a.txt', isDir: false},
        {name: 'b.txt', path: WD + '/b.txt', isDir: false},
        {name: 'c.txt', path: WD + '/c.txt', isDir: false},
      ],
    });
    assert.strictEqual(
      rows(win, '.explorer-row[data-explorer-path="' + WD + '/a.txt"]').length,
      1,
      'BUG: the entry being renamed was listed twice',
    );
    assert.ok(input.isConnected, 'the rename box survives the refresh');
    assert.strictEqual(
      win.document.activeElement,
      input,
      'focus is given back',
    );
    assert.strictEqual(input.selectionStart, 2);
    assert.strictEqual(input.selectionEnd, 5);
    assert.strictEqual(rows(win, '.explorer-row.is-file').length, 3);
    await sleep(5);
    assert.strictEqual(
      ofType(posted, 'fsAction').length,
      0,
      'no premature commit',
    );
    // An error listing keeps the box too.
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: req.path,
      root: req.path,
      error: 'permission denied',
    });
    assert.ok(input.isConnected, 'an error listing keeps the rename box');
    assert.strictEqual(input.value, 'renamed.txt');
    // Escape cancels: one row, the old name, nothing sent.
    key(win, input, 'Escape');
    assert.strictEqual(
      rows(win, '.explorer-row[data-explorer-path="' + WD + '/a.txt"]').length,
      1,
    );
    assert.strictEqual(
      row.querySelector('.explorer-name').textContent,
      'a.txt',
    );
    assert.strictEqual(ofType(posted, 'fsAction').length, 0);
    // Enter commits a rename to a sibling path.
    rightClick(win, row);
    click(win, menuItem(win, 'Rename...'));
    await sleep(5);
    const input2 = row.querySelector('input.explorer-input');
    input2.value = 'z.txt';
    key(win, input2, 'Enter');
    const fsActions = ofType(posted, 'fsAction');
    assert.strictEqual(fsActions.length, 1);
    assert.strictEqual(fsActions[0].action, 'rename');
    assert.strictEqual(fsActions[0].path, WD + '/a.txt');
    assert.strictEqual(fsActions[0].dest, WD + '/z.txt');
    assert.strictEqual(fsActions[0].workDir, WD);
  });

  await test('Folder picker: Select only picks a folder the daemon listed', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    assert.ok(picker && !picker.hidden, 'the picker opens');
    let list = ofType(posted, 'listDir');
    let req = list[list.length - 1];
    assert.ok(String(req.token).startsWith('picker:'));
    assert.strictEqual(req.path, WD);
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD,
      root: WD,
      entries: [{name: 'sub', path: WD + '/sub', isDir: true}],
    });
    const input = picker.querySelector('.folder-picker-input');
    // (configData already pinned WD with one setWorkDir.)
    const pins = ofType(posted, 'setWorkDir').length;
    // Typed, never listed: Select lists it first...
    input.value = '/definitely/missing';
    click(win, picker.querySelector('.folder-picker-select'));
    assert.strictEqual(ofType(posted, 'saveConfig').length, 0, 'not saved yet');
    assert.strictEqual(ofType(posted, 'setWorkDir').length, pins);
    list = ofType(posted, 'listDir');
    req = list[list.length - 1];
    assert.strictEqual(req.path, '/definitely/missing');
    // ...and an error keeps the dialog open with nothing saved.
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: '/definitely/missing',
      root: '/definitely/missing',
      error: 'Directory not found: /definitely/missing',
    });
    assert.ok(
      !picker.hidden,
      'BUG: the picker closed on a folder that does not exist',
    );
    assert.strictEqual(
      ofType(posted, 'saveConfig').length,
      0,
      'BUG: a missing path was saved',
    );
    assert.strictEqual(ofType(posted, 'setWorkDir').length, pins);
    assert.ok(
      picker
        .querySelector('.folder-picker-note')
        .textContent.includes('not found'),
    );
    // A typed folder that does exist is picked once listed (canonical spelling).
    input.value = '/data/proj/';
    click(win, picker.querySelector('.folder-picker-select'));
    list = ofType(posted, 'listDir');
    req = list[list.length - 1];
    assert.strictEqual(req.path, '/data/proj/');
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: '/data/proj',
      root: '/data/proj',
      entries: [],
    });
    assert.ok(picker.hidden, 'the picker closes once the folder is listed');
    const saved = ofType(posted, 'saveConfig');
    assert.strictEqual(saved.length, 1);
    assert.strictEqual(saved[0].config.work_dir, '/data/proj');
    assert.strictEqual(
      ofType(posted, 'setWorkDir')[pins].workDir,
      '/data/proj',
    );
    // The Explorer re-roots at the picked folder.
    list = ofType(posted, 'listDir').filter(
      m => !String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(list[list.length - 1].path, '/data/proj');
    // Reopening: the listed folder itself and a listed subfolder pick at once.
    click(win, byId(win, 'explorer-pick-folder'));
    list = ofType(posted, 'listDir');
    req = list[list.length - 1];
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: '/data/proj',
      root: '/data/proj',
      entries: [{name: 'child', path: '/data/proj/child', isDir: true}],
    });
    click(win, picker.querySelector('.folder-picker-item'));
    assert.strictEqual(input.value, '/data/proj/child');
    click(win, picker.querySelector('.folder-picker-select'));
    assert.ok(picker.hidden);
    assert.strictEqual(
      ofType(posted, 'saveConfig')[1].config.work_dir,
      '/data/proj/child',
    );
  });

  await test('Folder picker: the picked folder is browsed even when the chat pinned another', async () => {
    const {win, posted} = makeWebview();
    pinWorkspace(win);
    // The active chat replays a task that ran in a sub-folder: its tab
    // pins that folder for browsing.
    send(win, {
      type: 'task_events',
      task: 'earlier task',
      task_id: 3,
      events: [],
      extra: JSON.stringify({work_dir: WD + '/sub'}),
    });
    click(win, byId(win, 'activity-explorer'));
    let list = ofType(posted, 'listDir');
    assert.strictEqual(
      list[list.length - 1].path,
      WD + '/sub',
      'the tab folder is browsed',
    );
    send(win, {
      type: 'dirListing',
      token: list[list.length - 1].token,
      path: WD + '/sub',
      root: WD + '/sub',
      entries: [],
    });
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    list = ofType(posted, 'listDir');
    let req = list[list.length - 1];
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD + '/sub',
      root: WD + '/sub',
      entries: [],
    });
    // Up to the parent, then Select it.
    click(win, picker.querySelector('.folder-picker-up'));
    list = ofType(posted, 'listDir');
    req = list[list.length - 1];
    assert.strictEqual(req.path, WD);
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD,
      root: WD,
      entries: [],
    });
    click(win, picker.querySelector('.folder-picker-select'));
    assert.ok(picker.hidden);
    list = ofType(posted, 'listDir').filter(
      m => !String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(
      list[list.length - 1].path,
      WD,
      'BUG: the Explorer kept browsing the tab folder after picking its parent',
    );
    assert.strictEqual(list[list.length - 1].workDir, WD);
    // The Source Control view follows too.
    click(win, byId(win, 'activity-scm'));
    const st = ofType(posted, 'gitStatus');
    assert.strictEqual(st[st.length - 1].workDir, WD);
  });

  await test('Folder picker: Up stays at a root ("/" and "C:\\")', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    const input = picker.querySelector('.folder-picker-input');
    input.value = '/';
    key(win, input, 'Enter');
    let list = ofType(posted, 'listDir');
    send(win, {
      type: 'dirListing',
      token: list[list.length - 1].token,
      path: '/',
      root: '/',
      entries: [],
    });
    click(win, picker.querySelector('.folder-picker-up'));
    list = ofType(posted, 'listDir');
    assert.strictEqual(list[list.length - 1].path, '/');
    input.value = 'C:\\';
    key(win, input, 'Enter');
    list = ofType(posted, 'listDir');
    send(win, {
      type: 'dirListing',
      token: list[list.length - 1].token,
      path: 'C:\\',
      root: 'C:\\',
      entries: [],
    });
    click(win, picker.querySelector('.folder-picker-up'));
    list = ofType(posted, 'listDir');
    assert.strictEqual(
      list[list.length - 1].path,
      'C:\\',
      'a drive root has no parent',
    );
    input.value = 'C:\\Users';
    key(win, input, 'Enter');
    list = ofType(posted, 'listDir');
    send(win, {
      type: 'dirListing',
      token: list[list.length - 1].token,
      path: 'C:\\Users',
      root: 'C:\\Users',
      entries: [],
    });
    click(win, picker.querySelector('.folder-picker-up'));
    list = ofType(posted, 'listDir');
    assert.strictEqual(list[list.length - 1].path, 'C:\\');
  });

  await test('Commit menu: a failed action still refreshes Source Control', async () => {
    const {win, posted} = makeWebview();
    const row = openScm(win, posted);
    assert.ok(row, 'the commit row renders');
    const before = ofType(posted, 'gitStatus').length;
    rightClick(win, row);
    click(win, menuItem(win, 'Cherry Pick'));
    const actions = ofType(posted, 'gitAction');
    assert.strictEqual(actions.length, 1);
    assert.strictEqual(actions[0].action, 'cherryPick');
    assert.strictEqual(actions[0].sha, SHA_A);
    send(win, {
      type: 'gitActionResult',
      token: actions[0].token,
      action: 'cherryPick',
      sha: SHA_A,
      error:
        'error: could not apply aaaaaaa... Title line\nhint: after resolving the conflicts...',
    });
    assert.ok(
      ofType(posted, 'gitStatus').length > before,
      'BUG: Source Control was not re-read after the cherry-pick stopped on conflicts',
    );
    const note = win.document.body.textContent;
    assert.ok(note.includes('could not apply'), 'the error is shown');
  });

  await test('Commit menu: Copy Commit Message copies the whole message', async () => {
    const {win, posted, copied} = makeWebview();
    const row = openScm(win, posted);
    rightClick(win, row);
    click(win, menuItem(win, 'Copy Commit Message'));
    assert.deepStrictEqual(copied, ['Title line\n\nBody paragraph.']);
    rightClick(win, row);
    click(win, menuItem(win, 'Copy Commit Hash'));
    assert.strictEqual(copied[1], SHA_A);
  });

  await test('Commit menu: Create Tag... without a message makes a lightweight tag; names survive', async () => {
    const {win, posted, prompts} = makeWebview();
    const row = openScm(win, posted);
    prompts.push('v9', null); // name, then the message box dismissed
    rightClick(win, row);
    click(win, menuItem(win, 'Create Tag...'));
    let actions = ofType(posted, 'gitAction');
    assert.strictEqual(
      actions.length,
      1,
      'BUG: dismissing the message aborted the tag',
    );
    assert.strictEqual(actions[0].action, 'createTag');
    assert.strictEqual(actions[0].name, 'v9');
    assert.strictEqual(actions[0].message, '');
    send(win, {
      type: 'gitActionResult',
      token: actions[0].token,
      action: 'createTag',
      sha: SHA_A,
      ok: true,
      output: '',
    });
    assert.ok(
      win.document.body.textContent.includes('Created tag v9'),
      'the notification names the tag',
    );
    // Cancelling the NAME box creates nothing.
    prompts.push(null);
    rightClick(win, row);
    click(win, menuItem(win, 'Create Tag...'));
    assert.strictEqual(ofType(posted, 'gitAction').length, 1);
    // Create Branch... names the branch in its notification.
    prompts.push('topic');
    rightClick(win, row);
    click(win, menuItem(win, 'Create Branch...'));
    actions = ofType(posted, 'gitAction');
    assert.strictEqual(actions.length, 2);
    assert.strictEqual(actions[1].action, 'createBranch');
    assert.strictEqual(actions[1].name, 'topic');
    send(win, {
      type: 'gitActionResult',
      token: actions[1].token,
      action: 'createBranch',
      sha: SHA_A,
      ok: true,
      output: '',
    });
    assert.ok(win.document.body.textContent.includes('Created branch topic'));
  });

  await test('Explorer shortcuts: Ctrl+Alt+C copies the path on Linux', async () => {
    const {win, posted, copied} = makeWebview();
    openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
    ]);
    const row = explorerRow(win, WD + '/a.txt');
    row.focus();
    key(win, row, 'c', {ctrlKey: true, altKey: true});
    assert.deepStrictEqual(copied, [WD + '/a.txt']);
    key(win, row, 'c', {ctrlKey: true, altKey: true, shiftKey: true});
    assert.strictEqual(copied[1], 'a.txt');
  });

  await test('Folder picker: a navigation while a Select waits does not pick the new folder', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    let list = ofType(posted, 'listDir');
    send(win, {
      type: 'dirListing',
      token: list[list.length - 1].token,
      path: WD,
      root: WD,
      entries: [],
    });
    const input = picker.querySelector('.folder-picker-input');
    input.value = '/a';
    click(win, picker.querySelector('.folder-picker-select'));
    list = ofType(posted, 'listDir');
    const selectReq = list[list.length - 1];
    assert.strictEqual(selectReq.path, '/a');
    // Before /a is listed the user types /b and presses Enter.
    input.value = '/b';
    key(win, input, 'Enter');
    list = ofType(posted, 'listDir');
    const navReq = list[list.length - 1];
    assert.strictEqual(navReq.path, '/b');
    send(win, {
      type: 'dirListing',
      token: navReq.token,
      path: '/b',
      root: '/b',
      entries: [],
    });
    assert.ok(
      !picker.hidden,
      'BUG: a plain navigation was taken as the pending Select',
    );
    assert.strictEqual(ofType(posted, 'saveConfig').length, 0);
    // The late /a reply is stale and picks nothing either.
    send(win, {
      type: 'dirListing',
      token: selectReq.token,
      path: '/a',
      root: '/a',
      entries: [],
    });
    assert.ok(!picker.hidden);
    assert.strictEqual(ofType(posted, 'saveConfig').length, 0);
    // A file-system root is refused with a note.
    input.value = '/';
    click(win, picker.querySelector('.folder-picker-select'));
    assert.ok(!picker.hidden);
    assert.ok(
      picker.querySelector('.folder-picker-note').textContent.includes('root'),
    );
    assert.strictEqual(ofType(posted, 'saveConfig').length, 0);
  });

  await test('Working directory panel: a folder opened there re-roots the views past a content tab', async () => {
    const {win, posted} = makeWebview();
    pinWorkspace(win);
    send(win, {
      type: 'task_events',
      task: 'earlier task',
      task_id: 4,
      events: [],
      extra: JSON.stringify({work_dir: WD + '/sub'}),
    });
    // A content tab opened from that chat is on screen.
    send(win, {
      type: 'fileContent',
      name: 'a.txt',
      path: WD + '/sub/a.txt',
      content: 'x',
    });
    click(win, byId(win, 'activity-explorer'));
    let list = ofType(posted, 'listDir');
    assert.strictEqual(list[list.length - 1].path, WD + '/sub');
    // The "Working directory" panel: type a new work dir and open it;
    // the daemon's listing confirms the folder and the client saves it.
    click(win, byId(win, 'more-btn'));
    click(win, byId(win, 'workdir-btn'));
    const wd = byId(win, 'workdir-input');
    wd.value = '/parent';
    wd.dispatchEvent(new win.Event('input', {bubbles: true}));
    click(win, byId(win, 'workdir-open-btn'));
    const checks = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('workdir:'),
    );
    send(win, {
      type: 'dirListing',
      token: checks[checks.length - 1].token,
      path: '/parent',
      root: '/parent',
      entries: [],
    });
    const saved = ofType(posted, 'saveConfig');
    assert.ok(
      saved.length >= 1 &&
        saved[saved.length - 1].config.work_dir === '/parent',
    );
    list = ofType(posted, 'listDir').filter(
      m => !String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(
      list[list.length - 1].path,
      '/parent',
      'BUG: the Explorer kept the content tab owner folder after a work dir change',
    );
  });

  await test('Explorer: an error refresh while renaming an expanded folder keeps its contents', async () => {
    const {win, posted} = makeWebview();
    const req = openExplorer(win, posted, [
      {name: 'dir', path: WD + '/dir', isDir: true},
    ]);
    const dirRow = explorerRow(win, WD + '/dir');
    click(win, dirRow);
    let list = ofType(posted, 'listDir');
    send(win, {
      type: 'dirListing',
      token: list[list.length - 1].token,
      path: WD + '/dir',
      root: WD,
      entries: [{name: 'child.txt', path: WD + '/dir/child.txt', isDir: false}],
    });
    assert.ok(explorerRow(win, WD + '/dir/child.txt'), 'the child is listed');
    rightClick(win, dirRow);
    click(win, menuItem(win, 'Rename...'));
    await sleep(5);
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD,
      root: WD,
      error: 'boom',
    });
    assert.ok(
      dirRow.isConnected && dirRow.querySelector('input.explorer-input'),
    );
    assert.ok(
      explorerRow(win, WD + '/dir/child.txt'),
      'BUG: the folder being renamed lost its listed children',
    );
    key(win, dirRow.querySelector('input.explorer-input'), 'Escape');
    assert.ok(explorerRow(win, WD + '/dir/child.txt'));
    assert.strictEqual(ofType(posted, 'fsAction').length, 0);
  });

  await test('Explorer: a refresh keeps the focused row focused and the single tab stop', async () => {
    const {win, posted} = makeWebview();
    const req = openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
      {name: 'b.txt', path: WD + '/b.txt', isDir: false},
    ]);
    const row = explorerRow(win, WD + '/a.txt');
    row.focus();
    assert.strictEqual(win.document.activeElement, row);
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD,
      root: WD,
      entries: [
        {name: 'a.txt', path: WD + '/a.txt', isDir: false},
        {name: 'b.txt', path: WD + '/b.txt', isDir: false},
      ],
    });
    assert.strictEqual(
      win.document.activeElement,
      row,
      'BUG: the focused row lost focus',
    );
    const stops = rows(win, '.explorer-row').filter(r => r.tabIndex === 0);
    assert.strictEqual(stops.length, 1, 'exactly one tab stop in the tree');
    assert.strictEqual(stops[0], row);
  });

  await test('Menus: opening a tree menu closes an open content context menu', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
    ]);
    send(win, {
      type: 'fileContent',
      name: 'n.txt',
      path: '/tmp/n.txt',
      content: 'hello',
    });
    const area = byId(win, 'content-tab-area');
    const view = area && area.querySelector('.content-tab-view');
    assert.ok(view, 'the content tab renders');
    rightClick(win, view);
    const contentMenu = byId(win, 'sorcar-content-context-menu');
    assert.ok(contentMenu && !contentMenu.hidden, 'the content menu opened');
    rightClick(win, explorerRow(win, WD + '/a.txt'));
    assert.ok(win.TreeContextMenu.isOpen(), 'the tree menu opened');
    const stillOpen = byId(win, 'sorcar-content-context-menu');
    assert.ok(
      !stillOpen || stillOpen.hidden || stillOpen.style.display === 'none',
      'BUG: the content context menu stayed open under the tree menu',
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
