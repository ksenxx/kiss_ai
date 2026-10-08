// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the remote webapp's multi-root Explorer
// and the chat-grouped task history (media/main.js):
//
// * "Add Folder to Explorer..." (header button / root context menu)
//   opens the folder picker in `add` mode and shows the pick as another
//   top-level folder, remembered in localStorage,
// * every top-level folder row carries "Set as Working Directory" /
//   "Remove Folder from Explorer" buttons (a check mark on the working
//   directory), also reachable from its context menu and the Delete key,
// * requests for entries under an added folder are confined to THAT
//   folder (listDir / openFile / fsAction workDir), nested top-level
//   folders keep separate nodes, and a paste across two folders uses
//   their deepest common folder (refused when that is "/"),
// * the history panel groups tasks by chat, newest chat (by its latest
//   task) first, with "Today" / "Yesterday" / date separators, across
//   pages and under the status filters.

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

function makeWebview(beforeMain) {
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
  // Native dialogs are gone from main.js (the VS Code webview sandbox
  // never showed them); any call is a regression.
  win.prompt = () => {
    throw new Error('window.prompt must not be called');
  };
  win.confirm = () => {
    throw new Error('window.confirm must not be called');
  };
  if (beforeMain) beforeMain(win);
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'treeContextMenu.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=explorer-roots-main.js',
  );
  return {win, posted, copied};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const WD = '/ws/repo';
const OTHER = '/data/other';

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

function all(win, sel) {
  return Array.from(win.document.querySelectorAll(sel));
}

/** Answer the in-webview Delete question (it replaced window.confirm). */
function confirmDelete(win) {
  const toast = win.document.querySelector(
    '[data-notification-id="fs-delete"]',
  );
  assert.ok(toast, 'Delete asks in-webview first');
  const btn = Array.from(
    toast.querySelectorAll('.kiss-notification-action'),
  ).find(b => b.textContent.trim() === 'Delete');
  click(win, btn);
}

function menuItem(win, label) {
  return all(win, '#sidebar-context-menu .tree-ctx-item').find(
    el => el.querySelector('.tree-ctx-label').textContent === label,
  );
}

function menuLabels(win) {
  return all(win, '#sidebar-context-menu .tree-ctx-item').map(
    el => el.querySelector('.tree-ctx-label').textContent,
  );
}

/** Answer the last listDir whose path is *p* with *entries*. */
function answer(win, posted, p, entries, workDir) {
  const reqs = ofType(posted, 'listDir').filter(
    m => m.path === p && (workDir === undefined || m.workDir === workDir),
  );
  const req = reqs[reqs.length - 1];
  assert.ok(req, 'a listDir for ' + p + ' was sent');
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: p,
    root: req.workDir,
    entries,
  });
  return req;
}

/** Open the Explorer on WD with *entries* at its root. */
function openExplorer(win, posted, entries) {
  send(win, {type: 'configData', config: {work_dir: WD}});
  click(win, byId(win, 'activity-explorer'));
  return answer(win, posted, WD, entries || []);
}

function rootRows(win) {
  return all(win, '#explorer-tree > .explorer-row.is-root');
}

/**
 * The daemon's answer to the working-directory check "Set as Working
 * Directory" makes (the same `listDir` with a 'workdir:' token that
 * "..." > Working directory sends): the check is asserted to exist and
 * be scoped to *dir* itself, then answered (with *error* when given).
 */
function answerWorkDirCheck(win, posted, dir, error) {
  const checks = ofType(posted, 'listDir').filter(
    m => String(m.token).startsWith('workdir:') && m.path === dir,
  );
  const req = checks[checks.length - 1];
  assert.ok(req, 'a workdir: listDir check for ' + dir + ' was sent');
  assert.strictEqual(req.workDir, dir, 'the check is scoped to the folder');
  if (error) {
    send(win, {type: 'dirListing', token: req.token, path: dir, error});
    return req;
  }
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: dir,
    root: dir,
    entries: [],
  });
  return req;
}

function rowFor(win, p, root) {
  // Matched in JS rather than a CSS attribute selector: a backslash in
  // the path would be a CSS escape there.
  return all(win, '.explorer-row[data-explorer-path]').find(
    r =>
      r.dataset.explorerPath === p &&
      (!root || r.dataset.explorerRoot === root),
  );
}

/** Add OTHER through the header button and the picker; answer its listing. */
function addOtherFolder(win, posted, entries) {
  click(win, byId(win, 'explorer-add-folder'));
  const picker = byId(win, 'folder-picker');
  assert.ok(picker && !picker.hidden, 'picker open');
  assert.strictEqual(
    picker.querySelector('#folder-picker-title').textContent,
    'Add Folder to Explorer',
  );
  assert.strictEqual(
    picker.querySelector('.folder-picker-select').textContent,
    'Add Folder',
  );
  // Browse to /data, highlight "other", Add Folder.
  const input = picker.querySelector('.folder-picker-input');
  input.value = '/data';
  key(win, input, 'Enter');
  const pick = ofType(posted, 'listDir').filter(m =>
    String(m.token).startsWith('picker:'),
  );
  const req = pick[pick.length - 1];
  assert.strictEqual(req.path, '/data');
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: '/data',
    root: '/data',
    entries: [{name: 'other', path: OTHER, isDir: true}],
  });
  click(win, picker.querySelector('.folder-picker-item'));
  click(win, picker.querySelector('.folder-picker-select'));
  assert.ok(picker.hidden, 'picker closed after Add Folder');
  answer(win, posted, OTHER, entries || [], OTHER);
}

async function main() {
  await test('Add Folder: the pick becomes a second top-level folder, confined to itself, and is remembered', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
    ]);
    const before = posted.length;
    addOtherFolder(win, posted, [
      {name: 'notes.md', path: OTHER + '/notes.md', isDir: false},
      {name: 'sub', path: OTHER + '/sub', isDir: true},
    ]);
    // Adding a folder never touches the working directory.
    assert.strictEqual(
      posted
        .slice(before)
        .filter(m => m.type === 'saveConfig' || m.type === 'setWorkDir').length,
      0,
      'no saveConfig / setWorkDir for Add Folder',
    );
    const roots = rootRows(win);
    assert.deepStrictEqual(
      roots.map(r => r.dataset.explorerPath),
      [WD, OTHER],
      'working directory first, then the added folder',
    );
    assert.ok(roots[0].classList.contains('is-workdir'));
    assert.ok(
      roots[0].querySelector('.explorer-root-mark'),
      'check mark on the work dir',
    );
    assert.ok(
      !roots[0].querySelector('.explorer-root-btn'),
      'no buttons on the work dir',
    );
    assert.ok(!roots[1].classList.contains('is-workdir'));
    assert.ok(
      roots[1].querySelector('.explorer-root-set'),
      'set button on the added folder',
    );
    assert.ok(
      roots[1].querySelector('.explorer-root-remove'),
      'remove button on the added folder',
    );
    assert.ok(
      roots[1].classList.contains('expanded'),
      'the added folder opens',
    );
    // Its listing was requested with the folder itself as workDir.
    const listed = ofType(posted, 'listDir').filter(m => m.path === OTHER);
    assert.strictEqual(listed.length, 1);
    assert.strictEqual(listed[0].workDir, OTHER);
    assert.ok(rowFor(win, OTHER + '/notes.md'), 'the added folder is listed');
    // The working directory's tree is untouched: no re-listing (the
    // picker's own browse of it is not a tree listing), rows kept.
    assert.strictEqual(
      ofType(posted, 'listDir').filter(
        m => m.path === WD && !String(m.token).startsWith('picker:'),
      ).length,
      1,
    );
    assert.ok(rowFor(win, WD + '/a.txt'), 'work dir rows kept');
    // Remembered for the next load.
    assert.deepStrictEqual(
      JSON.parse(win.localStorage.getItem('kiss-explorer-roots')),
      [OTHER],
    );
    // Opening a file under the added folder is confined to that folder.
    click(win, rowFor(win, OTHER + '/notes.md'));
    const opened = ofType(posted, 'openFile');
    assert.strictEqual(opened[opened.length - 1].path, OTHER + '/notes.md');
    assert.strictEqual(opened[opened.length - 1].workDir, OTHER);
    // Expanding a sub-folder of the added folder lists it in that folder.
    click(win, rowFor(win, OTHER + '/sub'));
    const sub = ofType(posted, 'listDir').filter(
      m => m.path === OTHER + '/sub',
    );
    assert.strictEqual(sub.length, 1);
    assert.strictEqual(sub[0].workDir, OTHER);
    // Adding the same folder again neither duplicates it nor re-saves.
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    const input = picker.querySelector('.folder-picker-input');
    input.value = OTHER;
    click(win, picker.querySelector('.folder-picker-select'));
    const pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    const req = pick[pick.length - 1];
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: OTHER,
      root: OTHER,
      entries: [],
    });
    assert.strictEqual(rootRows(win).length, 2, 'still two top-level folders');
    assert.deepStrictEqual(
      JSON.parse(win.localStorage.getItem('kiss-explorer-roots')),
      [OTHER],
    );
    win.close();
  });

  await test('Added folders come back on the next load; a file-system root is never added', async () => {
    const {win, posted} = makeWebview(w => {
      w.localStorage.setItem(
        'kiss-explorer-roots',
        JSON.stringify([OTHER, '/', 42, WD]),
      );
    });
    openExplorer(win, posted, []);
    // "/" (a root) and 42 (not a path) are dropped; WD is the working
    // directory and shows once, as the working directory.
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD, OTHER],
    );
    // Only the working directory starts expanded on a plain load.
    assert.ok(rootRows(win)[0].classList.contains('expanded'));
    assert.ok(!rootRows(win)[1].classList.contains('expanded'));
    assert.strictEqual(
      ofType(posted, 'listDir').filter(m => m.path === OTHER).length,
      0,
    );
    // Picking a file-system root in add mode is refused with a note.
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    picker.querySelector('.folder-picker-input').value = '/';
    click(win, picker.querySelector('.folder-picker-select'));
    assert.ok(!picker.hidden, 'picker stays open');
    assert.ok(
      /root/i.test(picker.querySelector('.folder-picker-note').textContent),
      'root refused',
    );
    assert.strictEqual(rootRows(win).length, 2);
    win.close();
  });

  await test('Set as Working Directory: the folder becomes the workspace and the old one stays listed', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    addOtherFolder(win, posted, []);
    const before = posted.length;
    click(win, rootRows(win)[1].querySelector('.explorer-root-set'));
    // The check mark does what "..." > Working directory does: the
    // daemon lists the folder first; nothing changes until it answers.
    assert.ok(
      !posted
        .slice(before)
        .some(m => m.type === 'saveConfig' || m.type === 'setWorkDir'),
      'no saveConfig / setWorkDir before the daemon confirms the folder',
    );
    assert.strictEqual(rootRows(win)[0].dataset.explorerPath, WD);
    answerWorkDirCheck(win, posted, OTHER);
    const after = posted.slice(before);
    // The daemon owns the one global working directory: the folder is
    // handed to it as setWorkDir (it persists and broadcasts it), never
    // as a settings save.
    assert.strictEqual(
      after.filter(m => m.type === 'saveConfig').length,
      0,
      'no saveConfig: the working directory is not a settings field',
    );
    const adopted = after.filter(m => m.type === 'setWorkDir');
    assert.strictEqual(adopted.length, 1, 'exactly one setWorkDir');
    assert.strictEqual(adopted[0].workDir, OTHER, 'setWorkDir OTHER');
    // The tree: OTHER first (working directory), the old work dir kept.
    const roots = rootRows(win);
    assert.deepStrictEqual(
      roots.map(r => r.dataset.explorerPath),
      [OTHER, WD],
    );
    assert.ok(roots[0].classList.contains('is-workdir'));
    assert.ok(roots[0].querySelector('.explorer-root-mark'));
    assert.ok(
      roots[1].querySelector('.explorer-root-set'),
      'the old work dir can be switched back to',
    );
    assert.deepStrictEqual(
      JSON.parse(win.localStorage.getItem('kiss-explorer-roots')).sort(),
      [OTHER, WD].sort(),
    );
    // The button click did not toggle the row it sits on.
    assert.ok(roots[0].classList.contains('expanded'), 'new work dir expanded');
    // Clicking the check mark on the working directory does nothing.
    const n = posted.length;
    click(win, roots[0].querySelector('.explorer-root-mark'));
    assert.strictEqual(posted.length, n);
    assert.ok(
      roots[0].classList.contains('expanded'),
      'row not collapsed by the mark click',
    );
    win.close();
  });

  await test('Remove Folder from Explorer: button, context menu and Delete key; never the working directory', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    addOtherFolder(win, posted, []);
    // Context menu of the working directory: switch / remove disabled,
    // no Rename / Delete (VS Code has none on a root either).
    rightClick(win, rootRows(win)[0]);
    const labels = menuLabels(win);
    assert.ok(labels.indexOf('Add Folder to Explorer...') >= 0);
    assert.ok(labels.indexOf('Rename...') < 0 && labels.indexOf('Delete') < 0);
    assert.ok(
      menuItem(win, 'Set as Working Directory').classList.contains('disabled'),
      'work dir: Set as Working Directory disabled',
    );
    assert.ok(
      menuItem(win, 'Remove Folder from Explorer').classList.contains(
        'disabled',
      ),
      'work dir: Remove disabled',
    );
    key(win, win.document, 'Escape');
    // Delete on the working directory row is a no-op.
    key(win, rootRows(win)[0], 'Delete');
    assert.strictEqual(rootRows(win).length, 2);
    // Context menu of the added folder: both enabled; Remove removes.
    rightClick(win, rootRows(win)[1]);
    assert.ok(
      !menuItem(win, 'Set as Working Directory').classList.contains('disabled'),
    );
    const remove = menuItem(win, 'Remove Folder from Explorer');
    assert.ok(remove && !remove.classList.contains('disabled'));
    click(win, remove);
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD],
    );
    assert.deepStrictEqual(
      JSON.parse(win.localStorage.getItem('kiss-explorer-roots')),
      [],
    );
    // Add again, remove with the hover button.
    addOtherFolder(win, posted, []);
    assert.strictEqual(rootRows(win).length, 2);
    const before = posted.length;
    click(win, rootRows(win)[1].querySelector('.explorer-root-remove'));
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD],
    );
    assert.strictEqual(
      posted.slice(before).filter(m => m.type === 'fsAction').length,
      0,
      'removing a folder from the Explorer deletes nothing on disk',
    );
    // Add again, remove with the Delete key on the focused row.
    addOtherFolder(win, posted, []);
    key(win, rootRows(win)[1], 'Delete');
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD],
    );
    // "Add Folder to Explorer..." from the root menu opens the picker in add mode.
    rightClick(win, rootRows(win)[0]);
    click(win, menuItem(win, 'Add Folder to Explorer...'));
    const picker = byId(win, 'folder-picker');
    assert.ok(!picker.hidden);
    assert.strictEqual(
      picker.querySelector('#folder-picker-title').textContent,
      'Add Folder to Explorer',
    );
    // ... while the Open Folder button keeps its own title.
    key(win, win.document, 'Escape');
    click(win, byId(win, 'explorer-pick-folder'));
    assert.strictEqual(
      picker.querySelector('#folder-picker-title').textContent,
      'Open Folder as Working Directory',
    );
    assert.strictEqual(
      picker.querySelector('.folder-picker-select').textContent,
      'Select Folder',
    );
    win.close();
  });

  await test('Nested top-level folders keep separate nodes; actions carry the folder they belong to', async () => {
    const {win, posted} = makeWebview();
    const SUB = WD + '/sub';
    openExplorer(win, posted, [
      {name: 'sub', path: SUB, isDir: true},
      {name: 'root.txt', path: WD + '/root.txt', isDir: false},
    ]);
    // Add WD/sub (a folder inside the working directory) as a top-level folder.
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    picker.querySelector('.folder-picker-input').value = SUB;
    click(win, picker.querySelector('.folder-picker-select'));
    const pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    send(win, {
      type: 'dirListing',
      token: pick[pick.length - 1].token,
      path: SUB,
      root: SUB,
      entries: [],
    });
    assert.ok(picker.hidden);
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD, SUB],
    );
    // The top-level SUB listing is confined to SUB ...
    const topReq = answer(
      win,
      posted,
      SUB,
      [{name: 'x.txt', path: SUB + '/x.txt', isDir: false}],
      SUB,
    );
    assert.strictEqual(topReq.workDir, SUB);
    // ... and does not fill the "sub" row under the working directory.
    const nested = rowFor(win, SUB, WD);
    const top = rowFor(win, SUB, SUB);
    assert.ok(nested && top && nested !== top, 'two rows for the same folder');
    assert.ok(top.classList.contains('expanded'));
    assert.ok(!nested.classList.contains('expanded'));
    assert.strictEqual(
      all(win, '.explorer-row[data-explorer-path="' + SUB + '/x.txt"]').length,
      1,
    );
    // Expanding the nested one asks with the working directory as workDir.
    click(win, nested);
    const nestedReq = answer(
      win,
      posted,
      SUB,
      [{name: 'x.txt', path: SUB + '/x.txt', isDir: false}],
      WD,
    );
    assert.strictEqual(nestedReq.workDir, WD);
    assert.notStrictEqual(
      nestedReq.token,
      topReq.token,
      'distinct node tokens',
    );
    assert.strictEqual(
      all(win, '.explorer-row[data-explorer-path="' + SUB + '/x.txt"]').length,
      2,
    );
    // An action on the top-level SUB's file is confined to SUB, on the
    // nested copy to the working directory.
    rightClick(win, rowFor(win, SUB + '/x.txt', SUB));
    click(win, menuItem(win, 'Delete'));
    confirmDelete(win);
    let fsReq = ofType(posted, 'fsAction').pop();
    assert.strictEqual(fsReq.action, 'delete');
    assert.strictEqual(fsReq.workDir, SUB);
    rightClick(win, rowFor(win, SUB + '/x.txt', WD));
    click(win, menuItem(win, 'Delete'));
    confirmDelete(win);
    fsReq = ofType(posted, 'fsAction').pop();
    assert.strictEqual(fsReq.workDir, WD);
    // Copy Relative Path is relative to the row's own top-level folder.
    win.close();
  });

  await test('Paste across two top-level folders uses their deepest common folder, or is refused', async () => {
    const {win, posted, copied} = makeWebview();
    openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
    ]);
    addOtherFolder(win, posted, [
      {name: 'notes.md', path: OTHER + '/notes.md', isDir: false},
    ]);
    // Copy Relative Path under the added folder is relative to it.
    rightClick(win, rowFor(win, OTHER + '/notes.md'));
    click(win, menuItem(win, 'Copy Relative Path'));
    await sleep(0);
    assert.strictEqual(copied[copied.length - 1], 'notes.md');
    // Copy /ws/repo/a.txt, paste into /data/other: only "/" is shared.
    rightClick(win, rowFor(win, WD + '/a.txt'));
    click(win, menuItem(win, 'Copy'));
    const before = posted.length;
    rightClick(win, rootRows(win)[1]);
    click(win, menuItem(win, 'Paste'));
    assert.strictEqual(
      posted.slice(before).filter(m => m.type === 'fsAction').length,
      0,
      'no fsAction can span "/"',
    );
    assert.ok(
      win.document.body.textContent.includes('share only the file-system root'),
      'the refusal is explained in a notification',
    );
    win.close();
    // Two folders under one parent: the paste is confined to the parent.
    const second = makeWebview();
    const SIB = '/ws/sibling';
    openExplorer(second.win, second.posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
    ]);
    click(second.win, byId(second.win, 'explorer-add-folder'));
    const picker = byId(second.win, 'folder-picker');
    picker.querySelector('.folder-picker-input').value = SIB;
    click(second.win, picker.querySelector('.folder-picker-select'));
    const pick = ofType(second.posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    send(second.win, {
      type: 'dirListing',
      token: pick[pick.length - 1].token,
      path: SIB,
      root: SIB,
      entries: [],
    });
    answer(second.win, second.posted, SIB, [], SIB);
    rightClick(second.win, rowFor(second.win, WD + '/a.txt'));
    click(second.win, menuItem(second.win, 'Copy'));
    rightClick(second.win, rootRows(second.win)[1]);
    click(second.win, menuItem(second.win, 'Paste'));
    const fsReq = ofType(second.posted, 'fsAction').pop();
    assert.ok(fsReq, 'paste sent');
    assert.strictEqual(fsReq.action, 'copy');
    assert.strictEqual(fsReq.path, WD + '/a.txt');
    assert.strictEqual(fsReq.dest, SIB);
    assert.strictEqual(fsReq.workDir, '/ws', 'deepest common folder');
    // A paste within one folder stays confined to that folder.
    rightClick(second.win, rootRows(second.win)[0]);
    click(second.win, menuItem(second.win, 'Paste'));
    assert.strictEqual(ofType(second.posted, 'fsAction').pop().workDir, WD);
    second.win.close();
  });

  await test('Refresh keeps the added folders; a task-news refresh re-lists them in place', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    addOtherFolder(win, posted, [
      {name: 'n.md', path: OTHER + '/n.md', isDir: false},
    ]);
    const before = posted.length;
    click(win, byId(win, 'explorer-refresh'));
    const relisted = posted.slice(before).filter(m => m.type === 'listDir');
    assert.deepStrictEqual(
      relisted.map(m => m.path + '|' + m.workDir).sort(),
      [OTHER + '|' + OTHER, WD + '|' + WD],
      'both top-level folders re-listed in place',
    );
    assert.strictEqual(rootRows(win).length, 2, 'tree not rebuilt');
    assert.ok(
      rowFor(win, OTHER + '/n.md'),
      'rows kept while the listing is in flight',
    );
    win.close();
  });

  // ---- History grouping ----

  const DAY = 86400;
  // Noon today, so "yesterday" and "three days ago" stay on their days
  // whatever the local time zone offset does to the test's clock.
  const todayNoon = (() => {
    const d = new Date();
    d.setHours(12, 0, 0, 0);
    return Math.floor(d.getTime() / 1000);
  })();

  function session(chat, task, ts, extra) {
    return Object.assign(
      {
        id: chat,
        task_id: task,
        title: 'task ' + task,
        preview: 'task ' + task,
        has_events: true,
        timestamp: ts,
        tokens: 1,
        cost: 0.1,
        steps: 1,
      },
      extra || {},
    );
  }

  function groups(win) {
    return all(win, '#history-list > .history-chat-group');
  }

  /**
   * A history row click opens its task in a fresh chat tab and asks
   * the daemon to resume it there.  The chat that was on screen (an
   * idle new chat) is retired at the same time, so the tab count does
   * not grow: the new active tab and the resumeSession it posts are the
   * evidence that the click was honoured.
   */
  function assertRowOpenedTask(
    win,
    posted,
    activeBefore,
    resumesBefore,
    taskId,
  ) {
    const active = win._testApi.getActiveTabId();
    assert.notStrictEqual(
      active,
      activeBefore,
      'the click opens a new chat tab',
    );
    assert.ok(
      win._testApi.openTabs().some(t => t.id === active),
      'the new chat tab is open',
    );
    const resumes = ofType(posted, 'resumeSession');
    assert.strictEqual(
      resumes.length,
      resumesBefore + 1,
      'one resume is asked',
    );
    assert.strictEqual(resumes[resumes.length - 1].taskId, taskId);
    assert.strictEqual(resumes[resumes.length - 1].tabId, active);
  }

  /** Deliver a history page stamped with the generation the panel last asked for. */
  function sendHistory(win, posted, offset, sessions) {
    const asked = ofType(posted, 'getHistory');
    send(win, {
      type: 'history',
      offset,
      generation: asked.length ? asked[asked.length - 1].generation : 0,
      sessions,
    });
  }

  function listShape(win) {
    return Array.from(byId(win, 'history-list').children)
      .filter(el => el.style.display !== 'none')
      .map(el =>
        el.classList.contains('history-day-sep')
          ? 'sep:' + el.textContent
          : el.classList.contains('history-chat-group')
            ? 'chat:' +
              el.dataset.chatId +
              '[' +
              Array.from(el.querySelectorAll('.sidebar-item'))
                .filter(r => r.style.display !== 'none')
                .map(r => r.querySelector('.sidebar-item-text').textContent)
                .join(',') +
              ']'
            : el.className,
      );
  }

  await test('History: tasks grouped by chat, chats by latest task, day separators', async () => {
    const {win, posted} = makeWebview();
    const threeDaysAgo = todayNoon - 3 * DAY;
    const label = new Date(threeDaysAgo * 1000).toLocaleDateString(undefined, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
    sendHistory(win, posted, 0, [
      session('A', 'a2', todayNoon),
      session('B', 'b2', todayNoon - 600),
      session('A', 'a1', todayNoon - 1200), // older task of A: joins A's block
      session('C', 'c1', todayNoon - DAY), // yesterday
      session('B', 'b1', todayNoon - DAY - 60),
      session('D', 'd1', threeDaysAgo),
      session('', 'e1', threeDaysAgo - 60), // no chat id: its own block
    ]);
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:A[task a2,task a1]',
      'chat:B[task b2,task b1]',
      'sep:Yesterday',
      'chat:C[task c1]',
      'sep:' + label,
      'chat:D[task d1]',
      'chat:[task e1]',
    ]);
    assert.strictEqual(all(win, '#history-list .sidebar-item').length, 7);
    const sep = all(win, '.history-day-sep')[0];
    assert.strictEqual(sep.getAttribute('role'), 'separator');
    // Task panels carry NO per-chat colour any more.
    assert.strictEqual(
      groups(win)[0].style.getPropertyValue('--task-color'),
      '',
      'no per-chat colour on the block',
    );
    assert.strictEqual(
      all(win, '#history-list .sidebar-item')[0].style.getPropertyValue(
        '--task-color',
      ),
      '',
      'no per-chat colour on the rows',
    );
    // Every chat block is a collapsible panel: a header (the chat's
    // first task) above a body holding the task rows.
    const groupA = groups(win)[0];
    const headerA = groupA.querySelector(':scope > .history-chat-header');
    assert.ok(headerA, 'chat block has a clickable header');
    assert.strictEqual(
      headerA.querySelector('.history-chat-title').textContent,
      'task a1',
      "the header follows the chat's oldest loaded task when the daemon " +
        'sends no chat_first_task',
    );
    assert.ok(
      groupA.querySelector(':scope > .history-chat-body .sidebar-item'),
      'task rows live in the body container',
    );
    win.close();
  });

  await test('History: chat panels collapse by default, stay open while running, remember the user toggle', async () => {
    const {win, posted} = makeWebview();
    const collapsed = g => g.classList.contains('collapsed');
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
      }),
      Object.assign(session('B', 'b1', todayNoon - 600), {is_running: true}),
      Object.assign(session('A', 'a1', todayNoon - 1200), {
        chat_first_task: 'first task of A',
      }),
    ]);
    let [gB, gA] = groups(win); // The running chat precedes newer idle chats.
    // The daemon names the chat's FIRST task (which may be beyond the
    // loaded page); the header shows it, clamped by CSS to 3 lines.
    assert.strictEqual(
      gA.querySelector('.history-chat-title').textContent,
      'first task of A',
    );
    assert.ok(collapsed(gA), 'an idle chat starts collapsed');
    assert.ok(!collapsed(gB), 'a chat with a running task starts open');
    assert.strictEqual(
      gA.querySelector('.history-chat-header').getAttribute('aria-expanded'),
      'false',
    );
    // The user opens A: the choice survives a re-render.
    gA.querySelector('.history-chat-header').click();
    assert.ok(!collapsed(gA), 'header click expands the panel');
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
      }),
      session('B', 'b1', todayNoon - 600),
    ]);
    [gA, gB] = groups(win);
    assert.ok(!collapsed(gA), 'the explicit expand survives the rebuild');
    assert.ok(
      collapsed(gB),
      "B's task finished and the user never toggled it: collapsed again",
    );
    // Collapsing hides the body, not the block.
    gA.querySelector('.history-chat-header').click();
    assert.ok(collapsed(gA), 'header click collapses the panel again');
    assert.ok(
      gA.querySelector(':scope > .history-chat-body .sidebar-item'),
      'the rows stay in the DOM under the collapsed header',
    );
    win.close();
  });

  await test("History: the chat header shows the daemon's chat_summary, falling back to the first task until one exists", async () => {
    const {win, posted} = makeWebview();
    const title = g => g.querySelector('.history-chat-title').textContent;
    // A is summarised; B has only its first task; C has neither (an
    // older daemon) and follows its oldest loaded row.
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
        chat_summary: 'Fix login and add tests',
      }),
      Object.assign(session('B', 'b1', todayNoon - 600), {
        chat_first_task: 'first task of B',
        chat_summary: '',
      }),
      session('C', 'c2', todayNoon - 1200),
      Object.assign(session('A', 'a1', todayNoon - 1800), {
        chat_first_task: 'first task of A',
        chat_summary: 'Fix login and add tests',
      }),
      session('C', 'c1', todayNoon - 2400),
    ]);
    let [gA, gB, gC] = groups(win);
    assert.strictEqual(title(gA), 'Fix login and add tests');
    assert.strictEqual(title(gB), 'first task of B');
    assert.strictEqual(title(gC), 'task c1');
    assert.strictEqual(
      gA.querySelector('.history-chat-header').textContent.trim(),
      'Fix login and add tests',
      "the summary is the header button's accessible name",
    );
    // B's task finishes and the daemon summarises the chat: the next
    // refresh renames the panel. A summary of only whitespace does not
    // count as one, so C keeps following its first task instead.
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
        chat_summary: 'Fix login and add tests',
      }),
      Object.assign(session('B', 'b1', todayNoon - 600), {
        chat_first_task: 'first task of B',
        chat_summary: 'Refactor the billing module',
      }),
      Object.assign(session('C', 'c2', todayNoon - 1200), {
        chat_first_task: 'first task of C',
        chat_summary: '   \n',
      }),
    ]);
    [gA, gB, gC] = groups(win);
    assert.strictEqual(title(gA), 'Fix login and add tests');
    assert.strictEqual(title(gB), 'Refactor the billing module');
    assert.strictEqual(title(gC), 'first task of C');
    // Only the first line of a multi-line summary is shown, and a row
    // that arrives without the chat-level fields (mixed daemon
    // payloads) never overwrites a name the daemon already gave.
    sendHistory(win, posted, 0, [
      Object.assign(session('A', 'a2', todayNoon), {
        chat_first_task: 'first task of A',
        chat_summary: '  Fix login\nand add tests',
      }),
      session('A', 'a1', todayNoon - 1800),
    ]);
    [gA] = groups(win);
    assert.strictEqual(title(gA), 'Fix login');
    win.close();
  });

  await test('History: a search expands every chat — past saved collapses and through the identical-refresh fast path — and clearing it restores the defaults', async () => {
    const {win, posted} = makeWebview();
    const collapsed = g => g.classList.contains('collapsed');
    const searchBox = byId(win, 'history-search');
    const setSearch = value => {
      searchBox.value = value;
      searchBox.dispatchEvent(new win.Event('input', {bubbles: true}));
    };
    const page = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page());
    let [gA, gB] = groups(win);
    // The user explicitly collapses A (expand + collapse stores the
    // choice), which must NOT hide A's matches inside a later search.
    gA.querySelector('.history-chat-header').click();
    gA.querySelector('.history-chat-header').click();
    assert.ok(collapsed(gA));
    setSearch('task');
    // The search's results equal the loaded page, so the reply lands
    // in the identical-refresh fast path: no rebuild, same nodes.
    sendHistory(win, posted, 0, page());
    [gA, gB] = groups(win);
    assert.ok(
      !collapsed(gA),
      'a search expands A even though the user collapsed it before',
    );
    assert.ok(!collapsed(gB), 'B expands for the search too');
    // A collapse DURING the search applies to the search view only.
    gA.querySelector('.history-chat-header').click();
    assert.ok(collapsed(gA), 'the panel can still be folded mid-search');
    // A changed-data rebuild while the SAME query stands (any task on
    // the daemon persisted a result) keeps the in-search fold.
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a2', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    [gA, gB] = groups(win);
    assert.ok(
      collapsed(gA),
      'the in-search fold survives a changed-data rebuild',
    );
    assert.ok(!collapsed(gB), 'unfolded chats stay expanded for the search');
    setSearch('');
    sendHistory(win, posted, 0, page());
    [gA, gB] = groups(win);
    assert.ok(collapsed(gA), "clearing the search restores A's saved collapse");
    assert.ok(collapsed(gB), 'clearing the search restores the default');
    // A NEW search starts from expanded matches again: the fold made
    // inside the previous search does not carry over.
    setSearch('task');
    sendHistory(win, posted, 0, page());
    [gA, gB] = groups(win);
    assert.ok(!collapsed(gA), 'a fresh search drops the old in-search fold');
    assert.ok(!collapsed(gB));
    win.close();
  });

  await test('History: keyboard focus on a chat header survives a changed-data rebuild', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const header = groups(win)[0].querySelector('.history-chat-header');
    header.focus();
    assert.strictEqual(win.document.activeElement, header);
    // Changed data (a new task appears) wipes and rebuilds the list;
    // the fresh header of the SAME chat must take the focus over.
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a2', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const headerAfter = groups(win)[0].querySelector('.history-chat-header');
    assert.notStrictEqual(headerAfter, header, 'the list was rebuilt');
    assert.strictEqual(
      groups(win)[0].dataset.chatId,
      'A',
      'A still leads (its newest task is the youngest)',
    );
    assert.strictEqual(
      win.document.activeElement,
      headerAfter,
      'focus lands on the rebuilt header of the same chat',
    );
    win.close();
  });

  function firstRow(win) {
    return win.document.querySelector('#history-list .sidebar-item');
  }

  // The touch grace timer (300 ms) and the macrotask hop behind it fire
  // late on a loaded machine, so wait for the rebuild instead of sleeping
  // a fixed margin past it.
  async function firstRowReplaced(win, row) {
    const deadline = Date.now() + 3000;
    while (firstRow(win) === row && Date.now() < deadline) await sleep(20);
    return firstRow(win) !== row;
  }

  await test('History: an identical refresh keeps the same DOM rows (no lost click, no lost focus)', async () => {
    const {win, posted} = makeWebview();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    rowBefore.focus();
    // A `tasks_updated` broadcast (ANY task on the daemon persisted a
    // result, or the post-ready nudge) refetches the first page.  When
    // it comes back identical, the rendered rows must be KEPT: wiping
    // and rebuilding them swallows a click in flight (mousedown on the
    // old row, mouseup on its replacement) and drops keyboard focus.
    const asksBefore = ofType(posted, 'getHistory').length;
    send(win, {type: 'tasks_updated'});
    const asks = ofType(posted, 'getHistory');
    assert.ok(
      asks.length > asksBefore,
      'tasks_updated refetches the open history panel',
    );
    sendHistory(win, posted, 0, page1());
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'identical refresh must keep the existing row node',
    );
    assert.strictEqual(
      win.document.activeElement,
      rowBefore,
      'keyboard focus survives an identical refresh',
    );
    // The KEPT row's handler is alive: clicking it opens its task in a
    // tab (this is the click the old wipe-and-rebuild used to swallow).
    const activeBefore = win._testApi.getActiveTabId();
    const resumesBefore = ofType(posted, 'resumeSession').length;
    click(win, rowBefore);
    assertRowOpenedTask(win, posted, activeBefore, resumesBefore, 'a1');
    win.close();
  });

  await test('History: changed data still rebuilds the rows', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('B', 'b2', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'changed data must rebuild the rows',
    );
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:B[task b2,task b1]',
      'chat:A[task a1]',
    ]);
    win.close();
  });

  await test('History: an identical refresh retires every stale pagination loader', async () => {
    const {win, posted} = makeWebview();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    // Loaders can pile up: the scroll handler appends one, then a
    // broadcast-driven refresh resets historyLoading while it is still
    // on screen, so a second scroll appends another (duplicate ids).
    // The old offset-0 wipe cleared them as a side effect; the reply
    // itself must now retire ALL of them even on the fast path.
    const list = byId(win, 'history-list');
    for (let i = 0; i < 2; i++) {
      const loader = win.document.createElement('div');
      loader.className = 'sidebar-loading';
      loader.id = 'history-loader';
      loader.textContent = 'Loading...';
      list.appendChild(loader);
    }
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, page1());
    assert.strictEqual(
      all(win, '#history-loader').length,
      0,
      'every loader row is removed by the identical refresh',
    );
    assert.strictEqual(firstRow(win), rowBefore, 'rows still kept');
    win.close();
  });

  await test('History: a huge identical page is still kept (field comparison, no serialization)', async () => {
    const {win, posted} = makeWebview();
    // Rows are compared field by field (huge prompt strings directly),
    // so even multi-megabyte pages take the fast path when unchanged.
    const big = 'x'.repeat(600 * 1024);
    const page1 = () => [
      session('A', 'a1', todayNoon, {title: big, preview: big}),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, page1());
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'a huge identical page must still keep its row nodes',
    );
    win.close();
  });

  await test('History: a kept row with a ticking duration gets its metrics refreshed in place', async () => {
    const {win, posted} = makeWebview();
    // A running task's duration is `Date.now() - startTs`; the fast
    // path must keep the node (so clicks/focus survive) but refresh the
    // metrics text so the duration keeps ticking.  This is the shape of
    // every running row and of completed rows lacking a recorded end.
    const startTs = Date.now() - 59000;
    const page1 = () => [
      session('A', 'a1', todayNoon, {is_running: true, startTs}),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    const metricsEl = rowBefore.querySelector('.running-item-metrics');
    const textBefore = metricsEl.textContent;
    assert.ok(/00:00:59|00:01:0\d/.test(textBefore), textBefore);
    const realNow = win.Date.now;
    win.Date.now = () => realNow() + 60000;
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, page1());
      assert.strictEqual(
        firstRow(win),
        rowBefore,
        'the ticking row node is kept',
      );
      const textAfter = rowBefore.querySelector(
        '.running-item-metrics',
      ).textContent;
      assert.notStrictEqual(
        textAfter,
        textBefore,
        'the duration must be recomputed in place',
      );
      assert.ok(/00:01:59|00:02:0\d/.test(textAfter), textAfter);
    } finally {
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: a rebuild with CHANGED data is deferred while the mouse is pressed on a row', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    // Press the row, and let a refresh with DIFFERENT data land while
    // the button is still down: rebuilding now would swallow the click.
    rowBefore.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the rebuild is deferred while the press is held',
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
    );
    // The rebuild must not run synchronously inside the mouseup
    // dispatch either: the browser synthesizes the click right after
    // mouseup, so a rebuild there would still swallow it.
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'no rebuild during the mouseup dispatch',
    );
    const activeBefore = win._testApi.getActiveTabId();
    const resumesBefore = ofType(posted, 'resumeSession').length;
    click(win, rowBefore);
    assertRowOpenedTask(win, posted, activeBefore, resumesBefore, 'a1');
    await sleep(20);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'the deferred rebuild lands right after the click',
    );
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:C[task c1]',
      'chat:A[task a1]',
      'chat:B[task b1]',
    ]);
    win.close();
  });

  await test('History: a rebuild is deferred during a TOUCH press (pointer events)', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    // Touch: only pointer events fire while the finger is down; the
    // compatibility mouse events arrive after the finger lifts.
    rowBefore.dispatchEvent(
      new win.MouseEvent('pointerdown', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the rebuild is deferred while the touch is held',
    );
    // On a tap, Chromium dispatches pointerup FIRST and the
    // compatibility mousedown/mouseup/click AFTER it: the parked page
    // must not land in that gap, or the click would hit whatever row
    // the rebuild put under the finger.
    const upEvent = new win.MouseEvent('pointerup', {
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperty(upEvent, 'pointerType', {value: 'touch'});
    rowBefore.dispatchEvent(upEvent);
    await sleep(50);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the parked page must wait out the compatibility-event gap',
    );
    const activeBefore = win._testApi.getActiveTabId();
    const resumesBefore = ofType(posted, 'resumeSession').length;
    click(win, rowBefore);
    assertRowOpenedTask(win, posted, activeBefore, resumesBefore, 'a1');
    assert.ok(
      await firstRowReplaced(win, rowBefore),
      'the deferred rebuild lands after the grace period',
    );
    win.close();
  });

  await test('History: a reused touch pointer ID does not release a newer touch', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    const pev = (type, id) => {
      const e = new win.MouseEvent(type, {bubbles: true, cancelable: true});
      Object.defineProperty(e, 'pointerId', {value: id});
      Object.defineProperty(e, 'pointerType', {value: 'touch'});
      return e;
    };
    // Touch 1 presses, a changed page lands and is parked, the finger
    // lifts (arming the 300ms compatibility-event grace timer) ...
    rowBefore.dispatchEvent(pev('pointerdown', 7));
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    rowBefore.dispatchEvent(pev('pointerup', 7));
    // ... and touch 2 begins INSIDE the grace period, with the browser
    // reusing pointer ID 7 for the new contact.  Touch 1's grace timer
    // must not release touch 2's press.
    rowBefore.dispatchEvent(pev('pointerdown', 7));
    await sleep(400);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'a reused pointer ID must not release a newer touch',
    );
    // Touch 2 lifts; its own grace period ends; the parked page lands.
    rowBefore.dispatchEvent(pev('pointerup', 7));
    assert.ok(
      await firstRowReplaced(win, rowBefore),
      'the parked page lands after the second touch releases',
    );
    win.close();
  });

  await test('History: a parked page stays parked when a second press begins before the deferred apply', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    rowBefore.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    // Release press 1 (queues the zero-delay apply), then start press 2
    // synchronously, BEFORE that macrotask can run: the apply must find
    // the new press held and re-park the page instead of rebuilding.
    rowBefore.dispatchEvent(
      new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    await sleep(30);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the deferred apply must re-park while a second press is held',
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
    );
    await sleep(30);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'the re-parked page lands after the second release',
    );
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:C[task c1]',
      'chat:A[task a1]',
      'chat:B[task b1]',
    ]);
    win.close();
  });

  await test('History: a rebuild is deferred while a keyboard activation (Space) is held', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    const fav = rowBefore.querySelector('.sidebar-item-favorite');
    fav.focus();
    // A native button activates Space on KEYUP: a rebuild landing
    // between keydown and keyup would detach the button and swallow
    // the activation, exactly like a swallowed mouse click.
    fav.dispatchEvent(
      new win.KeyboardEvent('keydown', {
        key: ' ',
        bubbles: true,
        cancelable: true,
      }),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'the rebuild is deferred while Space is held on a list control',
    );
    // Overlapping activation keys: pressing and RELEASING Enter while
    // Space is still held must not release the guard.
    fav.dispatchEvent(
      new win.KeyboardEvent('keydown', {
        key: 'Enter',
        bubbles: true,
        cancelable: true,
      }),
    );
    fav.dispatchEvent(
      new win.KeyboardEvent('keyup', {
        key: 'Enter',
        bubbles: true,
        cancelable: true,
      }),
    );
    await sleep(20);
    assert.strictEqual(
      firstRow(win),
      rowBefore,
      'releasing Enter must not release the still-held Space',
    );
    // Chromium dispatches the Space keyup FIRST and synthesizes the
    // button's click right after it (before any queued macrotask): the
    // guard's release must not rebuild synchronously inside the keyup
    // dispatch, or the click would hit a detached button.
    fav.dispatchEvent(
      new win.KeyboardEvent('keyup', {
        key: ' ',
        bubbles: true,
        cancelable: true,
      }),
    );
    assert.ok(
      fav.isConnected,
      'no rebuild during the keyup dispatch (the activation click is next)',
    );
    click(win, fav);
    await sleep(20);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'the parked page lands after the keyboard activation',
    );
    const rebuiltA = all(win, '#history-list .sidebar-item')[1];
    assert.ok(
      rebuiltA
        .querySelector('.sidebar-item-favorite')
        .classList.contains('favorited'),
      'the activation took effect (optimistic star on the rebuilt row)',
    );
    win.close();
  });

  await test('History: a press whose release never arrives cannot park the panel forever', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    const timers = [];
    const origSetTimeout = win.setTimeout;
    const realNow = win.Date.now;
    const T = realNow();
    win.setTimeout = (fn, ms) => {
      if (ms >= 250 && ms <= 5500) {
        timers.push({fn, ms});
        return 1e9 + timers.length;
      }
      return origSetTimeout(fn, ms);
    };
    try {
      win.Date.now = () => T;
      // A press with no matching release (the release event was eaten:
      // e.g. a drag released outside the webview's iframe).
      rowBefore.dispatchEvent(
        new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
      );
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('C', 'c1', todayNoon + 60),
        session('A', 'a1', todayNoon),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.strictEqual(
        firstRow(win),
        rowBefore,
        'parked while the press appears held',
      );
      assert.strictEqual(
        timers.length,
        1,
        'a stale-press safety timer is armed with the parked page',
      );
      // A NEW press begins two seconds in: when the old timer fires at
      // the bound it must NOT clear the younger press's latches — it
      // re-arms for that press's own remainder instead.
      win.Date.now = () => T + 2000;
      firstRow(win).dispatchEvent(
        new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
      );
      win.Date.now = () => T + 5000;
      timers[0].fn();
      await sleep(20);
      assert.strictEqual(
        firstRow(win),
        rowBefore,
        'the safety timer must not release a press younger than the bound',
      );
      assert.strictEqual(timers.length, 2, 're-armed for the remainder');
      assert.ok(
        timers[1].ms <= 2000,
        `re-armed for the remainder, got ${timers[1].ms}ms`,
      );
      // The younger press too never releases: at ITS bound the latches
      // are dropped and the parked page finally lands.
      win.Date.now = () => T + 7100;
      timers[1].fn();
      await sleep(20);
      assert.notStrictEqual(
        firstRow(win),
        rowBefore,
        'the safety timer drops the stale latches and lands the page',
      );
    } finally {
      win.setTimeout = origSetTimeout;
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: a favourite click alone arms the reconciliation refetch', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const timers = [];
    const origSetTimeout = win.setTimeout;
    const realNow = win.Date.now;
    win.setTimeout = (fn, ms) => {
      if (ms >= 9500 && ms <= 10500) {
        timers.push(fn);
        return 1e9 + timers.length;
      }
      return origSetTimeout(fn, ms);
    };
    try {
      // The write has no acknowledgement and may FAIL: with no other
      // refresh ever arriving, the click itself must plan the refetch
      // that lets the daemon's value reappear at the bound.
      click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
      assert.strictEqual(
        timers.length,
        1,
        'the click armed the reconciliation refetch',
      );
      win.Date.now = () => realNow() + 10200;
      const asked = ofType(posted, 'getHistory').length;
      timers[0]();
      assert.ok(
        ofType(posted, 'getHistory').length > asked,
        'the refetch was posted with no page ever delivered in between',
      );
      // The (failed) write never happened on the daemon: past the
      // bound its contrary value rules again.
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.ok(
        !firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'the daemon value shows once the overlay expired',
      );
    } finally {
      win.setTimeout = origSetTimeout;
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: hiding the page mid-press releases the guard', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    const pev = (type, id) => {
      const e = new win.MouseEvent(type, {bubbles: true, cancelable: true});
      Object.defineProperty(e, 'pointerId', {value: id});
      Object.defineProperty(e, 'pointerType', {value: 'touch'});
      return e;
    };
    // A touch press, then the page is hidden before any release event.
    rowBefore.dispatchEvent(pev('pointerdown', 3));
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.strictEqual(firstRow(win), rowBefore, 'parked during the press');
    Object.defineProperty(win.document, 'hidden', {
      value: true,
      configurable: true,
    });
    win.document.dispatchEvent(new win.Event('visibilitychange'));
    await sleep(20);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'hiding the page drops the latches and lands the parked page',
    );
    win.close();
  });

  await test('History: a cancelled press (drag/scroll) releases the guard without a mouseup', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowBefore = firstRow(win);
    // Chromium can emit pointerdown -> mousedown -> dragstart ->
    // pointercancel with NO mouseup (dragging selected row text): the
    // cancel must release both flags or rendering stays parked forever.
    rowBefore.dispatchEvent(
      new win.MouseEvent('pointerdown', {bubbles: true, cancelable: true}),
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    rowBefore.dispatchEvent(
      new win.MouseEvent('pointercancel', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('C', 'c1', todayNoon + 60),
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.notStrictEqual(
      firstRow(win),
      rowBefore,
      'rendering is not latched after a cancelled press',
    );
    win.close();
  });

  await test('History: a duplicate later page is dropped instead of duplicating rows', async () => {
    const {win, posted} = makeWebview();
    const fullPage = (tag, base) =>
      Array.from({length: 50}, (_, i) =>
        session('C' + tag + i, 't' + tag + i, base - i * 60),
      );
    sendHistory(win, posted, 0, fullPage('x', todayNoon));
    sendHistory(win, posted, 50, fullPage('y', todayNoon - 5000));
    assert.strictEqual(all(win, '#history-list .sidebar-item').length, 100);
    // A second offset-50 reply (an overlapped same-generation request)
    // no longer extends at the cursor (now 100): it must be dropped.
    sendHistory(win, posted, 50, fullPage('z', todayNoon - 9000));
    assert.strictEqual(
      all(win, '#history-list .sidebar-item').length,
      100,
      'the stale duplicate page is dropped',
    );
    byId(win, 'history-list').dispatchEvent(new win.Event('scroll'));
    const asks = ofType(posted, 'getHistory');
    assert.strictEqual(
      asks[asks.length - 1].offset,
      100,
      'pagination continues at the rendered row count',
    );
    win.close();
  });

  await test('History: no second same-generation request can overlap a pending refresh', async () => {
    const {win, posted} = makeWebview();
    const fullPage = tag =>
      Array.from({length: 50}, (_, i) =>
        session('C' + tag + i, 't' + tag + i, todayNoon - i * 60),
      );
    sendHistory(win, posted, 0, fullPage('x'));
    // A broadcast resets pagination and refetches; until that reply
    // lands, a bottom-scroll must NOT fire an overlapping request of
    // the same generation (its late reply would re-order pagination).
    send(win, {type: 'tasks_updated'});
    const asked = ofType(posted, 'getHistory').length;
    byId(win, 'history-list').dispatchEvent(new win.Event('scroll'));
    assert.strictEqual(
      ofType(posted, 'getHistory').length,
      asked,
      'the scroll is parked while the refresh is in flight',
    );
    sendHistory(win, posted, 0, fullPage('x'));
    byId(win, 'history-list').dispatchEvent(new win.Event('scroll'));
    assert.strictEqual(
      ofType(posted, 'getHistory').length,
      asked + 1,
      'after the reply, the scroll paginates again',
    );
    win.close();
  });

  await test('History: a same-day clock change alone does not rebuild, a day change does', async () => {
    const {win, posted} = makeWebview();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    // A wall-clock correction across local midnight must relabel
    // "Today"/"Yesterday", so the fast path is off on a new local day.
    const origToDateString = win.Date.prototype.toDateString;
    win.Date.prototype.toDateString = function () {
      return 'Fri Jan 01 2100';
    };
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, page1());
      assert.notStrictEqual(
        firstRow(win),
        rowBefore,
        'a new local day must rebuild (separator labels changed)',
      );
    } finally {
      win.Date.prototype.toDateString = origToDateString;
    }
    win.close();
  });

  await test('History: the favourite overlay plans a refetch so its expiry is visible without broadcasts', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    // Capture the expiry timer main.js arms at the toggle (and keeps
    // serving while a page still disagrees inside the bound).
    const longTimers = [];
    const origSetTimeout = win.setTimeout;
    win.setTimeout = (fn, ms) => {
      if (ms > 5000) {
        longTimers.push(fn);
        return 1e9 + longTimers.length;
      }
      return origSetTimeout(fn, ms);
    };
    const realNow = win.Date.now;
    try {
      click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600),
      ]);
      // At least the expiry refetch is armed (the midnight-relabel
      // timer is captured by the same >5s filter).
      assert.ok(longTimers.length >= 1, 'an expiry refetch is armed');
      // At the deadline the timer refetches; the daemon's (contrary)
      // value now rules because the overlay has expired.
      win.Date.now = () => realNow() + 11000;
      const asked = ofType(posted, 'getHistory').length;
      longTimers.forEach(fn => fn());
      assert.ok(
        ofType(posted, 'getHistory').length > asked,
        'the expiry timer refetches the history',
      );
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.ok(
        !firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'the daemon value shows at the deadline without any broadcast',
      );
    } finally {
      win.Date.now = realNow;
      win.setTimeout = origSetTimeout;
    }
    win.close();
  });

  await test('History: the expiry refetch serves the EARLIEST pending favourite deadline', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const realNow = win.Date.now;
    const origSetTimeout = win.setTimeout;
    try {
      const T = realNow();
      const timers = [];
      win.setTimeout = (fn, ms) => {
        if (ms > 3000 && ms < 20000) {
          timers.push({fn, ms});
          // A truthy fake id: main.js keeps the armed timer's id, and 0
          // would read as "no timer armed".
          return 1e9 + timers.length;
        }
        return origSetTimeout(fn, ms);
      };
      const rows = all(win, '#history-list .sidebar-item');
      // A is starred at T, B five seconds later: A's overlay expires at
      // T+10s, B's at T+15s.  The first click arms the timer; the
      // second has a LATER deadline and must not move it.
      win.Date.now = () => T;
      click(win, rows[0].querySelector('.sidebar-item-favorite'));
      assert.strictEqual(timers.length, 1, "A's click armed the timer");
      win.Date.now = () => T + 5000;
      click(win, rows[1].querySelector('.sidebar-item-favorite'));
      assert.strictEqual(
        timers.length,
        1,
        "B's later deadline must not move the armed timer",
      );
      // The timer fires (both overlays still pending) and refetches; the
      // reply's PAGE ORDER puts the later deadline (B) first: the next
      // timer must be re-armed for A's earlier deadline (~4.1s away at
      // T+6s), not left at B's (~9.1s away).
      win.Date.now = () => T + 6000;
      timers[0].fn();
      sendHistory(win, posted, 0, [
        session('B', 'b1', todayNoon - 600, {steps: 3}),
        session('A', 'a1', todayNoon, {steps: 3}),
      ]);
      assert.strictEqual(timers.length, 3, 'armed for B, re-armed for A');
      assert.ok(
        timers[2].ms >= 4000 && timers[2].ms <= 4300,
        `re-armed for the earliest pending deadline, got ${timers[2].ms}ms`,
      );
      // At A's deadline the timer refetches; A's overlay has expired
      // (daemon value rules) while B's still holds.
      win.Date.now = () => T + 10200;
      const asked = ofType(posted, 'getHistory').length;
      timers[2].fn();
      assert.ok(
        ofType(posted, 'getHistory').length > asked,
        'the earliest-deadline timer refetches the history',
      );
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600, {steps: 3}),
      ]);
      const after = all(win, '#history-list .sidebar-item');
      assert.ok(
        !after[0]
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        "A's overlay expired at its own 10s bound",
      );
      assert.ok(
        after[1]
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        "B's younger overlay still holds",
      );
    } finally {
      win.Date.now = realNow;
      win.setTimeout = origSetTimeout;
    }
    win.close();
  });

  await test('History: an agreeing page retires the overlay and cancels its refetch', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const origSetTimeout = win.setTimeout;
    const origClearTimeout = win.clearTimeout;
    const armed = [];
    const cleared = [];
    win.setTimeout = (fn, ms) => {
      if (ms >= 9500 && ms <= 10500) {
        armed.push(fn);
        return 1e9 + armed.length;
      }
      return origSetTimeout(fn, ms);
    };
    win.clearTimeout = id => {
      if (id >= 1e9) {
        cleared.push(id);
        return undefined;
      }
      return origClearTimeout(id);
    };
    try {
      click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
      assert.strictEqual(armed.length, 1, 'the click armed the refetch');
      // The daemon's own data AGREES with the toggle: the overlay is
      // retired and the now-pointless reconciliation refetch cancelled.
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {is_favorite: true}),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.deepStrictEqual(
        cleared,
        [1e9 + 1],
        'the agreeing page cancelled the armed refetch',
      );
    } finally {
      win.setTimeout = origSetTimeout;
      win.clearTimeout = origClearTimeout;
    }
    win.close();
  });

  await test('History: a parked page cannot acknowledge its own favourite overlay', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const origSetTimeout = win.setTimeout;
    const origClearTimeout = win.clearTimeout;
    const realNow = win.Date.now;
    const armed = [];
    const cleared = [];
    win.setTimeout = (fn, ms) => {
      if (ms >= 9500 && ms <= 10500) {
        armed.push(fn);
        return 1e9 + armed.length;
      }
      return origSetTimeout(fn, ms);
    };
    win.clearTimeout = id => {
      if (id >= 1e9) {
        cleared.push(id);
        return undefined;
      }
      return origClearTimeout(id);
    };
    try {
      click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
      assert.strictEqual(armed.length, 1, 'the click armed the refetch');
      // A CONTRARY page (the daemon never saw the write) arrives while
      // the row is pressed: it is merged (overlay applied), parked, and
      // merged AGAIN when it lands after the release.  The second merge
      // must not mistake the overlay for daemon agreement.
      const rowBefore = firstRow(win);
      rowBefore.dispatchEvent(
        new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
      );
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600),
      ]);
      rowBefore.dispatchEvent(
        new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
      );
      await sleep(30);
      assert.notStrictEqual(firstRow(win), rowBefore, 'parked page landed');
      assert.ok(
        firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'the optimistic star survives the parked page landing',
      );
      assert.strictEqual(
        cleared.length,
        0,
        'landing the parked page must not cancel the reconciliation refetch',
      );
      // The write really failed: at the bound the refetch lets the
      // daemon's contrary value rule again.
      win.Date.now = () => realNow() + 10200;
      const asked = ofType(posted, 'getHistory').length;
      armed[0]();
      assert.ok(
        ofType(posted, 'getHistory').length > asked,
        'the reconciliation refetch still fires at the bound',
      );
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 3}),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.ok(
        !firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'the daemon value rules once the overlay expired',
      );
    } finally {
      win.setTimeout = origSetTimeout;
      win.clearTimeout = origClearTimeout;
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: the favourite overlay expires so the daemon becomes authoritative again', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    click(win, firstRow(win).querySelector('.sidebar-item-favorite'));
    // Within the bound, a contrary (pre-write) page keeps the star.
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 3}),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.ok(
      firstRow(win)
        .querySelector('.sidebar-item-favorite')
        .classList.contains('favorited'),
      'the optimistic star holds within the bound',
    );
    // Past the bound (another client may have toggled it back, or the
    // write may have failed), the daemon's value rules.
    const realNow = win.Date.now;
    win.Date.now = () => realNow() + 11000;
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, [
        session('A', 'a1', todayNoon, {steps: 4}),
        session('B', 'b1', todayNoon - 600),
      ]);
      assert.ok(
        !firstRow(win)
          .querySelector('.sidebar-item-favorite')
          .classList.contains('favorited'),
        'after the bound the server value rules',
      );
    } finally {
      win.Date.now = realNow;
    }
    win.close();
  });

  await test('History: an optimistic favourite survives a page deferred across its own click', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const favBtn = firstRow(win).querySelector('.sidebar-item-favorite');
    // Press the star; while the button is held, a refresh lands that
    // predates the toggle (star still off) with an unrelated change.
    favBtn.dispatchEvent(
      new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
    );
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 9}),
      session('B', 'b1', todayNoon - 600),
    ]);
    click(win, favBtn);
    favBtn.dispatchEvent(
      new win.MouseEvent('mouseup', {bubbles: true, cancelable: true}),
    );
    await sleep(20);
    const favAfter = firstRow(win).querySelector('.sidebar-item-favorite');
    assert.notStrictEqual(favAfter, favBtn, 'the row was rebuilt');
    assert.ok(
      favAfter.classList.contains('favorited'),
      'the optimistic star is preserved over the stale page',
    );
    // Once the daemon's own data agrees, the overlay retires and the
    // server value keeps ruling.
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 9, is_favorite: true}),
      session('B', 'b1', todayNoon - 600),
    ]);
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 9, is_favorite: false}),
      session('B', 'b1', todayNoon - 600),
    ]);
    assert.ok(
      !firstRow(win)
        .querySelector('.sidebar-item-favorite')
        .classList.contains('favorited'),
      'after the daemon agreed once, later server data rules again',
    );
    win.close();
  });

  await test('History: a duplicate offset-0 reply in one generation does not corrupt pagination', async () => {
    const {win, posted} = makeWebview();
    const fullPage = tag =>
      Array.from({length: 50}, (_, i) =>
        session('C' + tag + i, 't' + tag + i, todayNoon - i * 60),
      );
    sendHistory(win, posted, 0, fullPage('x'));
    // A second offset-0 reply in the SAME generation (a scroll's
    // duplicate racing a deferred rebuild) must assign the cursor, not
    // advance it past what is on screen.
    sendHistory(win, posted, 0, fullPage('y'));
    byId(win, 'history-list').dispatchEvent(new win.Event('scroll'));
    const asks = ofType(posted, 'getHistory');
    const last = asks[asks.length - 1];
    assert.strictEqual(
      last.offset,
      50,
      'the next page is requested at the rendered row count',
    );
    win.close();
  });

  await test('History: keyboard focus moves to the same task after a changed-data rebuild', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rows = all(win, '#history-list .sidebar-item');
    rows[1].focus();
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 7}),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowsAfter = all(win, '#history-list .sidebar-item');
    assert.notStrictEqual(rowsAfter[1], rows[1], 'rows were rebuilt');
    assert.strictEqual(
      win.document.activeElement,
      rowsAfter[1],
      "focus lands on task b1's fresh row",
    );
    // Focus on an inline control (the favourite star) survives as the
    // SAME control on the rebuilt row: the next Space/Enter must repeat
    // the control's action, not open the task.
    rowsAfter[0].querySelector('.sidebar-item-favorite').focus();
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {steps: 8}),
      session('B', 'b1', todayNoon - 600),
    ]);
    const rowsFinal = all(win, '#history-list .sidebar-item');
    assert.strictEqual(
      win.document.activeElement,
      rowsFinal[0].querySelector('.sidebar-item-favorite'),
      "focus lands on task a1's fresh favourite control",
    );
    win.close();
  });

  await test('History: a locale change makes the next refresh rebuild (row times are localized)', async () => {
    const {win, posted} = makeWebview();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    const origDtf = win.Intl.DateTimeFormat;
    win.Intl.DateTimeFormat = function () {
      return {resolvedOptions: () => ({locale: 'de-DE', timeZone: 'UTC'})};
    };
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, page1());
      assert.notStrictEqual(
        firstRow(win),
        rowBefore,
        'a locale change must rebuild the rows',
      );
    } finally {
      win.Intl.DateTimeFormat = origDtf;
    }
    win.close();
  });

  await test('History: a time-zone change makes the next refresh rebuild (row times are localized)', async () => {
    const {win, posted} = makeWebview();
    const page1 = () => [
      session('A', 'a1', todayNoon),
      session('B', 'b1', todayNoon - 600),
    ];
    sendHistory(win, posted, 0, page1());
    const rowBefore = firstRow(win);
    // Same data, different clock localization: the rows' timestamps
    // were rendered under the old zone, so the fast path must not keep
    // them.
    const origOffset = win.Date.prototype.getTimezoneOffset;
    win.Date.prototype.getTimezoneOffset = function () {
      return origOffset.call(this) + 60;
    };
    try {
      send(win, {type: 'tasks_updated'});
      sendHistory(win, posted, 0, page1());
      assert.notStrictEqual(
        firstRow(win),
        rowBefore,
        'a zone change must rebuild the rows',
      );
    } finally {
      win.Date.prototype.getTimezoneOffset = origOffset;
    }
    win.close();
  });

  await test('History: a later page adds older tasks to existing blocks and opens older ones after', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a2', todayNoon),
      session('B', 'b1', todayNoon - 60),
    ]);
    sendHistory(win, posted, 2, [
      session('A', 'a1', todayNoon - 120),
      session('C', 'c1', todayNoon - DAY),
    ]);
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:A[task a2,task a1]',
      'chat:B[task b1]',
      'sep:Yesterday',
      'chat:C[task c1]',
    ]);
    // A fresh first page replaces everything (no duplicate blocks).
    sendHistory(win, posted, 0, [
      session('B', 'b2', todayNoon + 1),
      session('A', 'a2', todayNoon),
    ]);
    assert.deepStrictEqual(listShape(win), [
      'sep:Today',
      'chat:B[task b2]',
      'chat:A[task a2]',
    ]);
    assert.strictEqual(all(win, '.history-day-sep').length, 1);
    // An empty first page shows the empty note, no separators.
    sendHistory(win, posted, 0, []);
    assert.strictEqual(all(win, '.history-day-sep').length, 0);
    assert.ok(
      /No conversations yet/.test(byId(win, 'history-list').textContent),
    );
    win.close();
  });

  await test('History: filters hide blocks and separators left without a visible task', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [
      session('A', 'a1', todayNoon, {failed: true}),
      session('B', 'b1', todayNoon - DAY),
      session('C', 'c1', todayNoon - 2 * DAY, {failed: true}),
    ]);
    assert.strictEqual(listShape(win).length, 6);
    // Hide failed tasks: chats A and C go, with the Today separator and
    // the two-days-ago one; Yesterday / B stay.
    const errors = byId(win, 'hf-errors');
    assert.ok(errors, 'errors filter chip');
    click(win, errors);
    const shape = listShape(win);
    assert.deepStrictEqual(shape, ['sep:Yesterday', 'chat:B[task b1]']);
    // Show them again: everything is back.
    click(win, errors);
    assert.strictEqual(listShape(win).length, 6);
    win.close();
  });

  await test('History: epoch 0 is a date, a missing timestamp is "Undated"; other years show the year', async () => {
    const {win, posted} = makeWebview();
    const lastYear = new Date();
    lastYear.setFullYear(lastYear.getFullYear() - 1);
    lastYear.setHours(12, 0, 0, 0);
    const lastYearTs = Math.floor(lastYear.getTime() / 1000);
    const missing = session('C', 'c1', 0);
    delete missing.timestamp;
    sendHistory(win, posted, 0, [
      session('A', 'a1', lastYearTs),
      session('B', 'b1', 0),
      missing,
      session('D', 'd1', 'not a number'),
    ]);
    const seps = all(win, '.history-day-sep').map(el => el.textContent);
    const fmt = d =>
      d.toLocaleDateString(undefined, {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    // Epoch zero is 1970 (the date filter treats it the same way);
    // the missing and the unparsable timestamps share one "Undated".
    assert.deepStrictEqual(seps, [fmt(lastYear), fmt(new Date(0)), 'Undated']);
    win.close();
  });

  await test('History: separators are rebuilt when the page becomes visible again (midnight, time zone)', async () => {
    const {win, posted} = makeWebview();
    sendHistory(win, posted, 0, [session('A', 'a1', todayNoon)]);
    let sep = all(win, '.history-day-sep')[0];
    assert.strictEqual(sep.textContent, 'Today');
    assert.strictEqual(sep.dataset.ts, String(todayNoon));
    // A label computed on another day is recomputed when the page wakes.
    sep.textContent = 'Yesterday';
    win.document.dispatchEvent(new win.Event('visibilitychange'));
    sep = all(win, '.history-day-sep')[0];
    assert.strictEqual(all(win, '.history-day-sep').length, 1);
    assert.strictEqual(sep.textContent, 'Today');
    // A time-zone change can move a chat to another local day: the
    // partition is rebuilt, not just the words.  2026-01-10T00:30Z and
    // 2026-01-09T23:30Z are two days in UTC, one day in Los Angeles.
    const savedTz = process.env.TZ;
    process.env.TZ = 'UTC';
    try {
      const t1 = Date.UTC(2026, 0, 10, 0, 30) / 1000;
      const t2 = Date.UTC(2026, 0, 9, 23, 30) / 1000;
      sendHistory(win, posted, 0, [
        session('B', 'b1', t1),
        session('C', 'c1', t2),
      ]);
      assert.deepStrictEqual(
        all(win, '.history-day-sep').map(el => el.dataset.day),
        ['2026-1-10', '2026-1-9'],
      );
      process.env.TZ = 'America/Los_Angeles';
      win.document.dispatchEvent(new win.Event('visibilitychange'));
      assert.deepStrictEqual(
        all(win, '.history-day-sep').map(el => el.dataset.day),
        ['2026-1-9'],
        'one local day after the zone change',
      );
      assert.deepStrictEqual(listShape(win).slice(0, 3), [
        'sep:' +
          new Date(t1 * 1000).toLocaleDateString(undefined, {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
          }),
        'chat:B[task b1]',
        'chat:C[task c1]',
      ]);
    } finally {
      if (savedTz === undefined) delete process.env.TZ;
      else process.env.TZ = savedTz;
    }
    // A timestamp beyond what Date can represent is "Undated" once,
    // together with a missing one (no duplicate Undated separators).
    const missing = session('E', 'e1', 0);
    delete missing.timestamp;
    sendHistory(win, posted, 0, [session('D', 'd1', 1e300), missing]);
    assert.deepStrictEqual(
      all(win, '.history-day-sep').map(el => el.textContent),
      ['Undated'],
    );
    win.close();
  });

  // ---- Review regressions (Explorer) ----

  await test('Duplicate stored folders show once and are removed together', async () => {
    const {win, posted} = makeWebview(w => {
      w.localStorage.setItem(
        'kiss-explorer-roots',
        JSON.stringify([OTHER, OTHER + '/', OTHER]),
      );
    });
    openExplorer(win, posted, []);
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD, OTHER],
    );
    click(win, rootRows(win)[1].querySelector('.explorer-root-remove'));
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD],
    );
    assert.deepStrictEqual(
      JSON.parse(win.localStorage.getItem('kiss-explorer-roots')),
      [],
    );
    win.close();
  });

  await test('POSIX folder names may end in a backslash or a space', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    const odd = '/data/odd\\';
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    const input = picker.querySelector('.folder-picker-input');
    input.value = odd;
    click(win, picker.querySelector('.folder-picker-select'));
    let pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    let req = pick[pick.length - 1];
    assert.strictEqual(req.path, odd, 'the backslash is part of the name');
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: odd,
      root: odd,
      entries: [],
    });
    assert.ok(picker.hidden);
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD, odd],
    );
    assert.strictEqual(
      ofType(posted, 'listDir').filter(
        m =>
          m.path === odd &&
          m.workDir === odd &&
          !String(m.token).startsWith('picker:'),
      ).length,
      1,
      'listed once by the tree (the picker browse is separate)',
    );
    // A name ending in a space is requested as typed (only a blank box
    // falls back to the folder on screen).
    const spaced = '/data/spaced ';
    click(win, byId(win, 'explorer-add-folder'));
    input.value = spaced;
    key(win, input, 'Enter');
    pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    req = pick[pick.length - 1];
    assert.strictEqual(req.path, spaced);
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: spaced,
      root: spaced,
      entries: [],
    });
    assert.strictEqual(input.value, spaced, 'the box keeps the exact spelling');
    click(win, picker.querySelector('.folder-picker-select'));
    assert.ok(picker.hidden);
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD, odd, spaced],
    );
    // A blank box means "the folder on screen".
    click(win, byId(win, 'explorer-add-folder'));
    input.value = '   ';
    key(win, input, 'Enter');
    pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(
      pick[pick.length - 1].path,
      WD,
      'blank box: no navigation away from the folder shown',
    );
    win.close();
  });

  await test('Paste between sibling folders whose names contain backslashes stays on POSIX separators', async () => {
    const {win, posted} = makeWebview();
    const A = '/tmp/lit\\name/one';
    const B = '/tmp/lit\\name/two';
    send(win, {type: 'configData', config: {work_dir: A}});
    click(win, byId(win, 'activity-explorer'));
    answer(
      win,
      posted,
      A,
      [{name: 'f.txt', path: A + '/f.txt', isDir: false}],
      A,
    );
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    picker.querySelector('.folder-picker-input').value = B;
    click(win, picker.querySelector('.folder-picker-select'));
    const pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    send(win, {
      type: 'dirListing',
      token: pick[pick.length - 1].token,
      path: B,
      root: B,
      entries: [],
    });
    answer(win, posted, B, [], B);
    rightClick(win, rowFor(win, A + '/f.txt'));
    click(win, menuItem(win, 'Copy'));
    rightClick(win, rootRows(win)[1]);
    click(win, menuItem(win, 'Paste'));
    const fsReq = ofType(posted, 'fsAction').pop();
    assert.ok(fsReq, 'paste sent');
    assert.strictEqual(fsReq.workDir, '/tmp/lit\\name');
    win.close();
  });

  await test('Set / Remove from the context menu keep the keyboard in the tree', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    addOtherFolder(win, posted, []);
    rootRows(win)[1].focus();
    rightClick(win, rootRows(win)[1]);
    click(win, menuItem(win, 'Remove Folder from Explorer'));
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD],
    );
    let active = win.document.activeElement;
    assert.ok(
      active && active.classList.contains('explorer-row'),
      'focus back on a row',
    );
    assert.strictEqual(active.tabIndex, 0, 'on the tab stop');
    addOtherFolder(win, posted, []);
    rootRows(win)[1].focus();
    rightClick(win, rootRows(win)[1]);
    click(win, menuItem(win, 'Set as Working Directory'));
    answerWorkDirCheck(win, posted, OTHER);
    assert.strictEqual(rootRows(win)[0].dataset.explorerPath, OTHER);
    active = win.document.activeElement;
    assert.ok(
      active && active.classList.contains('explorer-row'),
      'focus on the rebuilt tree',
    );
    assert.strictEqual(active.tabIndex, 0);
    win.close();
  });

  await test('Windows folders: one spelling per folder; renamed / deleted folders update tabs below them', async () => {
    const {win, posted} = makeWebview(w => {
      w.localStorage.setItem(
        'kiss-explorer-roots',
        JSON.stringify(['c:/repo', 'C:\\Repo\\', 'C:\\Other']),
      );
    });
    const WWD = 'C:\\Repo';
    send(win, {type: 'configData', config: {work_dir: WWD}});
    click(win, byId(win, 'activity-explorer'));
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WWD, 'C:\\Other'],
      'case / separator variants of the work dir collapse into it',
    );
    answer(
      win,
      posted,
      WWD,
      [{name: 'folder', path: WWD + '\\folder', isDir: true}],
      WWD,
    );
    // Open a file inside the folder, then delete the folder: its tab closes.
    send(win, {
      type: 'fileContent',
      path: WWD + '\\folder\\file.txt',
      name: 'file.txt',
      content: 'x',
      tabId: 'tab-1',
    });
    const tabsBefore = all(win, '.chat-tab').length;
    rightClick(win, rowFor(win, WWD + '\\folder'));
    click(win, menuItem(win, 'Delete'));
    confirmDelete(win);
    const del = ofType(posted, 'fsAction').pop();
    assert.strictEqual(del.action, 'delete');
    assert.strictEqual(del.workDir, WWD);
    send(win, {
      type: 'fsResult',
      token: del.token,
      action: 'delete',
      path: WWD + '\\folder',
    });
    assert.strictEqual(
      all(win, '.chat-tab').length,
      tabsBefore - 1,
      'the tab under the deleted folder closed',
    );
    win.close();
  });

  await test('POSIX names with backslashes: root label, picker Up and Copy Relative Path treat only "/" as a separator', async () => {
    const {win, posted, copied} = makeWebview();
    const A = '/tmp/lit\\name';
    send(win, {type: 'configData', config: {work_dir: A}});
    click(win, byId(win, 'activity-explorer'));
    answer(
      win,
      posted,
      A,
      [{name: 'f.txt', path: A + '/f.txt', isDir: false}],
      A,
    );
    assert.strictEqual(
      rootRows(win)[0].querySelector('.explorer-name').textContent,
      'lit\\name',
      'the whole last segment is the label',
    );
    rightClick(win, rowFor(win, A + '/f.txt'));
    click(win, menuItem(win, 'Copy Relative Path'));
    await sleep(0);
    assert.strictEqual(copied[copied.length - 1], 'f.txt');
    // Picker: Up from /tmp/lit\name goes to /tmp.
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    let pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(pick[pick.length - 1].path, A);
    click(win, picker.querySelector('.folder-picker-up'));
    pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(pick[pick.length - 1].path, '/tmp');
    key(win, win.document, 'Escape');
    win.close();
  });

  await test('Set as Working Directory keeps the keyboard in the tree past the composer focus retries', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    // The active chat is a task of the old workspace: switching the
    // workspace hides its tab and activates another one, which
    // schedules the composer's deferred focus (100 / 300 ms).
    send(win, {
      type: 'task_events',
      task: 'earlier task',
      task_id: 3,
      events: [],
      extra: JSON.stringify({work_dir: WD}),
    });
    answer(win, posted, WD, [], WD);
    addOtherFolder(win, posted, []);
    rootRows(win)[1].focus();
    rightClick(win, rootRows(win)[1]);
    click(win, menuItem(win, 'Set as Working Directory'));
    answerWorkDirCheck(win, posted, OTHER);
    assert.strictEqual(rootRows(win)[0].dataset.explorerPath, OTHER);
    await sleep(400);
    const active = win.document.activeElement;
    assert.ok(
      active && active.classList.contains('explorer-row'),
      'focus still on a tree row after the retries, got ' +
        (active ? active.id || active.className : 'none'),
    );
    win.close();
  });

  await test('Set as Working Directory on a folder the daemon refuses: an error notification, the tree unchanged', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    addOtherFolder(win, posted, []);
    const before = posted.length;
    click(win, rootRows(win)[1].querySelector('.explorer-root-set'));
    answerWorkDirCheck(win, posted, OTHER, 'Not a directory: ' + OTHER);
    assert.ok(
      !posted
        .slice(before)
        .some(m => m.type === 'saveConfig' || m.type === 'setWorkDir'),
      'a refused folder never reaches saveConfig / setWorkDir',
    );
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD, OTHER],
      'the working directory is unchanged',
    );
    assert.ok(rootRows(win)[0].classList.contains('is-workdir'));
    // The panel is closed, so the refusal is a notification, not a
    // line in the panel's error box.
    const toast = win.document.querySelector('.kiss-notification');
    assert.ok(toast, 'an error notification is shown');
    assert.ok(
      toast.textContent.indexOf('Not a directory: ' + OTHER) >= 0,
      "the notification carries the daemon's reason",
    );
    const panelError = byId(win, 'workdir-error');
    assert.ok(panelError.hidden, "the closed panel's error line stays hidden");
    win.close();
  });

  await test('Set as Working Directory: a late daemon reply leaves the keyboard where the user moved it', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    addOtherFolder(win, posted, []);
    rootRows(win)[1].focus();
    click(win, rootRows(win)[1].querySelector('.explorer-root-set'));
    // While the daemon checks the folder the user goes back to typing.
    const input = byId(win, 'task-input');
    input.focus();
    assert.strictEqual(win.document.activeElement, input);
    answerWorkDirCheck(win, posted, OTHER);
    assert.strictEqual(rootRows(win)[0].dataset.explorerPath, OTHER);
    assert.strictEqual(
      win.document.activeElement,
      input,
      'the switch does not pull the keyboard back into the tree',
    );
    win.close();
  });

  await test("The Working directory panel's own Open never focuses the Explorer, even after an abandoned check-mark switch", async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    addOtherFolder(win, posted, []);
    rootRows(win)[1].focus();
    click(win, rootRows(win)[1].querySelector('.explorer-root-set'));
    const tickCheck = ofType(posted, 'listDir')
      .filter(m => String(m.token).startsWith('workdir:'))
      .pop();
    // Before the daemon answers, the user opens "..." > Working
    // directory and opens a third folder from there.
    click(win, byId(win, 'more-btn'));
    click(win, byId(win, 'workdir-btn'));
    const box = byId(win, 'workdir-input');
    box.value = '/data/third';
    box.dispatchEvent(new win.Event('input', {bubbles: true}));
    click(win, byId(win, 'workdir-open-btn'));
    // The check mark's own reply is stale now and changes nothing.
    send(win, {
      type: 'dirListing',
      token: tickCheck.token,
      path: OTHER,
      root: OTHER,
      entries: [],
    });
    assert.strictEqual(rootRows(win)[0].dataset.explorerPath, WD);
    answerWorkDirCheck(win, posted, '/data/third');
    assert.strictEqual(rootRows(win)[0].dataset.explorerPath, '/data/third');
    const active = win.document.activeElement;
    assert.ok(
      !(active && active.classList.contains('explorer-row')),
      "the panel's switch leaves the Explorer alone, focus on " +
        (active ? active.id || active.className : 'none'),
    );
    win.close();
  });

  // ---- "Add Folder to Explorer..." offers the recently opened folders --

  const NOW_S = Math.floor(Date.now() / 1000);
  const RECENTS = [
    {path: '/data/old', ts: NOW_S - 86400 * 3},
    {path: WD, ts: NOW_S},
    {path: OTHER, ts: NOW_S - 3600},
    {path: '/data/newer', ts: NOW_S - 60},
  ];

  function recentRows(picker) {
    return Array.from(
      picker.querySelectorAll('.folder-picker-recent-list .workdir-item'),
    );
  }

  function lastPickerListing(posted) {
    return ofType(posted, 'listDir')
      .filter(m => String(m.token).startsWith('picker:'))
      .pop();
  }

  await test('Add Folder: the picker lists the recently opened folders (newest first, those already shown left out) and one click adds one', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    send(win, {
      type: 'configData',
      config: {work_dir: WD, recent_work_dirs: RECENTS},
    });
    const before = posted.length;
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    // The dialog asks for the daemon's latest list, as the panel does.
    assert.strictEqual(ofType(posted.slice(before), 'getConfig').length, 1);
    const section = picker.querySelector('.folder-picker-recent');
    assert.ok(!section.hidden, 'the recent section shows');
    assert.deepStrictEqual(
      recentRows(picker).map(r => r.dataset.path),
      ['/data/newer', OTHER, '/data/old'],
      'newest first; the working directory (already shown) left out',
    );
    const row = recentRows(picker)[1];
    assert.strictEqual(row.title, 'Add ' + OTHER);
    assert.ok(
      /opened .*ago/.test(row.querySelector('.workdir-item-ago').textContent),
      'the row says when the folder was last opened',
    );
    // One click: the daemon lists the folder, then it is added.
    click(win, row);
    const req = lastPickerListing(posted);
    assert.strictEqual(req.path, OTHER);
    assert.strictEqual(req.workDir, OTHER);
    assert.ok(!picker.hidden, 'the dialog waits for the listing');
    assert.strictEqual(rootRows(win).length, 1, 'nothing added yet');
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: OTHER,
      root: OTHER,
      entries: [],
    });
    assert.ok(picker.hidden, 'the dialog closes once the folder is listed');
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD, OTHER],
    );
    assert.deepStrictEqual(
      JSON.parse(win.localStorage.getItem('kiss-explorer-roots')),
      [OTHER],
    );
    assert.strictEqual(
      posted
        .slice(before)
        .filter(m => m.type === 'saveConfig' || m.type === 'setWorkDir').length,
      0,
      'adding never touches the working directory',
    );
    // Reopened, the list no longer offers the folder just added.
    click(win, byId(win, 'explorer-add-folder'));
    assert.deepStrictEqual(
      recentRows(picker).map(r => r.dataset.path),
      ['/data/newer', '/data/old'],
    );
    // A fresher configData repaints the open dialog's list.
    send(win, {
      type: 'configData',
      config: {
        work_dir: WD,
        recent_work_dirs: RECENTS.concat([
          {path: '/data/fresh', ts: NOW_S + 5},
        ]),
      },
    });
    assert.deepStrictEqual(
      recentRows(picker).map(r => r.dataset.path),
      ['/data/fresh', '/data/newer', '/data/old'],
    );
    win.close();
  });

  await test('Add Folder: a recent folder the daemon refuses stays unadded, with the reason in the dialog', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    send(win, {
      type: 'configData',
      config: {work_dir: WD, recent_work_dirs: RECENTS},
    });
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    // Enter on a focused row picks it like a click.
    const row = recentRows(picker).find(r => r.dataset.path === '/data/old');
    key(win, row, 'Enter');
    const req = lastPickerListing(posted);
    assert.strictEqual(req.path, '/data/old');
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: '/data/old',
      error: 'No such directory',
    });
    assert.ok(!picker.hidden, 'the dialog stays open');
    assert.strictEqual(
      picker.querySelector('.folder-picker-note').textContent,
      'No such directory',
    );
    assert.strictEqual(rootRows(win).length, 1, 'nothing added');
    assert.strictEqual(win.localStorage.getItem('kiss-explorer-roots'), null);
    // Other keys on a row do nothing; Space picks.
    key(win, row, 'ArrowDown');
    assert.strictEqual(
      lastPickerListing(posted),
      req,
      'ArrowDown asks nothing',
    );
    key(win, row, ' ');
    assert.notStrictEqual(lastPickerListing(posted), req, 'Space picks');
    assert.strictEqual(lastPickerListing(posted).path, '/data/old');
    win.close();
  });

  await test('The open dialog follows the Explorer: a repaint keeps the focused row, and a folder that becomes a top-level folder leaves the list', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    send(win, {
      type: 'configData',
      config: {work_dir: WD, recent_work_dirs: RECENTS},
    });
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    const row = recentRows(picker).find(r => r.dataset.path === OTHER);
    row.focus();
    // The daemon's reply to the dialog's getConfig rebuilds the rows;
    // the keyboard stays on the same folder.
    send(win, {
      type: 'configData',
      config: {work_dir: WD, recent_work_dirs: RECENTS},
    });
    assert.notStrictEqual(
      win.document.activeElement,
      row,
      'the rows were rebuilt',
    );
    assert.strictEqual(win.document.activeElement.dataset.path, OTHER);
    assert.ok(picker.contains(win.document.activeElement));
    // Meanwhile another client of the daemon made OTHER the working
    // directory (the next configData reports it): the Explorer rebuilds
    // around it and the dialog stops offering it.
    send(win, {
      type: 'configData',
      config: {work_dir: OTHER, recent_work_dirs: RECENTS},
    });
    answer(win, posted, OTHER, [], OTHER);
    assert.strictEqual(rootRows(win)[0].dataset.explorerPath, OTHER);
    assert.deepStrictEqual(
      recentRows(picker).map(r => r.dataset.path),
      [WD, '/data/newer', '/data/old'],
      'the new working directory left, the old one offered',
    );
    assert.ok(
      !picker.contains(win.document.activeElement) ||
        win.document.activeElement.dataset.path !== OTHER,
      'no row of the vanished folder keeps focus',
    );
    win.close();
  });

  await test('Add Folder: a recent folder the dialog already lists is added at once, without another listing', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    send(win, {
      type: 'configData',
      config: {
        work_dir: WD,
        recent_work_dirs: [{path: WD + '/sub', ts: NOW_S}],
      },
    });
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    const req = lastPickerListing(posted);
    assert.strictEqual(
      req.path,
      WD,
      'the dialog opens on the working directory',
    );
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD,
      root: WD,
      entries: [{name: 'sub', path: WD + '/sub', isDir: true}],
    });
    const listings = ofType(posted, 'listDir').length;
    click(win, recentRows(picker)[0]);
    assert.ok(picker.hidden, 'added at once');
    assert.deepStrictEqual(
      rootRows(win).map(r => r.dataset.explorerPath),
      [WD, WD + '/sub'],
    );
    // The only new listDir is the Explorer's own listing of the new folder.
    const added = posted
      .slice(0)
      .filter(m => m.type === 'listDir')
      .slice(listings);
    assert.strictEqual(added.length, 1);
    assert.strictEqual(added[0].path, WD + '/sub');
    assert.ok(!String(added[0].token).startsWith('picker:'));
    win.close();
  });

  await test('The recent section is hidden with nothing left to offer and in the "Open Folder" picker', async () => {
    const {win, posted} = makeWebview();
    openExplorer(win, posted, []);
    // No recents at all.
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    assert.ok(picker.querySelector('.folder-picker-recent').hidden);
    click(win, picker.querySelector('.folder-picker-cancel'));
    // Every recent folder is already a top-level folder.
    send(win, {
      type: 'configData',
      config: {work_dir: WD, recent_work_dirs: [{path: WD, ts: NOW_S}]},
    });
    click(win, byId(win, 'explorer-add-folder'));
    assert.ok(picker.querySelector('.folder-picker-recent').hidden);
    assert.strictEqual(recentRows(picker).length, 0);
    click(win, picker.querySelector('.folder-picker-cancel'));
    // "Open Folder as Working Directory" never shows the section (the
    // Working directory panel is where those are opened from) and
    // asks for no configData.
    send(win, {
      type: 'configData',
      config: {work_dir: WD, recent_work_dirs: RECENTS},
    });
    const before = posted.length;
    click(win, byId(win, 'explorer-pick-folder'));
    assert.strictEqual(
      picker.querySelector('#folder-picker-title').textContent,
      'Open Folder as Working Directory',
    );
    assert.ok(picker.querySelector('.folder-picker-recent').hidden);
    assert.strictEqual(recentRows(picker).length, 0);
    assert.strictEqual(ofType(posted.slice(before), 'getConfig').length, 0);
    // A configData arriving meanwhile keeps it hidden.
    send(win, {
      type: 'configData',
      config: {work_dir: WD, recent_work_dirs: RECENTS},
    });
    assert.ok(picker.querySelector('.folder-picker-recent').hidden);
    // The panel's own list still carries every folder, titled "Open".
    click(win, picker.querySelector('.folder-picker-cancel'));
    click(win, byId(win, 'more-btn'));
    click(win, byId(win, 'workdir-btn'));
    const panelRows = all(win, '#workdir-list .workdir-item');
    assert.strictEqual(panelRows.length, 4);
    assert.strictEqual(panelRows[0].title, 'Open ' + WD);
    win.close();
  });

  console.log(`\n${passed} passed, ${failures.length} failed`);
  process.exit(failures.length ? 1 : 0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
