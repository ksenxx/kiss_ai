// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the remote webapp's multi-root Explorer
// (media/main.js); the chat-grouped task history is historyGroups.test.js:
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
//   their deepest common folder (refused when that is "/").

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  byId,
  sleep,
  stubRaf,
  fakeMatchMedia,
  ofType,
} = require('./simplify2_harness');

const {test, report} = require('./ui_antipattern_harness').makeRunner();
const {
  all,
  showSection,
  menuItem,
  rightClick,
} = require('./ui_antipattern_harness');

function makePage(beforeMain) {
  const copied = [];
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: win => {
      stubRaf(win);
      fakeMatchMedia(win, true);
      Object.defineProperty(win.navigator, 'clipboard', {
        value: {
          writeText: text => {
            copied.push(text);
            return Promise.resolve();
          },
        },
        configurable: true,
      });
      // Native dialogs are gone from main.js; any call is a regression.
      win.prompt = () => {
        throw new Error('window.prompt must not be called');
      };
      win.confirm = () => {
        throw new Error('window.confirm must not be called');
      };
      if (beforeMain) beforeMain(win);
    },
    extraScripts: [
      'marked.min.js',
      'contentContextMenu.js',
      'treeContextMenu.js',
    ],
  });
  return {win, posted, copied};
}

const WD = '/ws/repo';
const OTHER = '/data/other';

function key(win, el, name, extra) {
  el.dispatchEvent(
    new win.KeyboardEvent(
      'keydown',
      Object.assign({key: name, bubbles: true, cancelable: true}, extra || {}),
    ),
  );
}

/** Answer the in-page Delete question (it replaced window.confirm). */
function confirmDelete(win) {
  const toast = win.document.querySelector(
    '[data-notification-id="fs-delete"]',
  );
  assert.ok(toast, 'Delete asks in-page first');
  const btn = Array.from(
    toast.querySelectorAll('.kiss-notification-action'),
  ).find(b => b.textContent.trim() === 'Delete');
  click(win, btn);
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
  showSection(win, 'meta-explorer');
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
    const {win, posted} = makePage();
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
    const {win, posted} = makePage(w => {
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
    const {win, posted} = makePage();
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
    const {win, posted} = makePage();
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
    const {win, posted} = makePage();
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
    const {win, posted, copied} = makePage();
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
    const second = makePage();
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

  await test('Open and Open to the Side send the same openFile, the latter in the background', async () => {
    const {win, posted} = makePage();
    const FILE = WD + '/root.txt';
    openExplorer(win, posted, [{name: 'root.txt', path: FILE, isDir: false}]);
    click(win, rowFor(win, FILE, WD));
    let req = ofType(posted, 'openFile').pop();
    assert.ok(req, 'a click opens the file');
    assert.strictEqual(req.workDir, WD);
    assert.ok(!('background' in req), 'a plain open is not a background one');
    rightClick(win, rowFor(win, FILE, WD));
    click(win, menuItem(win, 'Open to the Side'));
    req = ofType(posted, 'openFile').pop();
    assert.strictEqual(req.path, FILE);
    assert.strictEqual(req.workDir, WD);
    assert.strictEqual(req.background, true, 'opened without leaving the tab');
    win.close();
  });

  await test('Refresh keeps the added folders; a task-news refresh re-lists them in place', async () => {
    const {win, posted} = makePage();
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

  // ---- Review regressions (Explorer) ----

  await test('Duplicate stored folders show once and are removed together', async () => {
    const {win, posted} = makePage(w => {
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
    const {win, posted} = makePage();
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
    const {win, posted} = makePage();
    const A = '/tmp/lit\\name/one';
    const B = '/tmp/lit\\name/two';
    send(win, {type: 'configData', config: {work_dir: A}});
    showSection(win, 'meta-explorer');
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
    const {win, posted} = makePage();
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
    const {win, posted} = makePage(w => {
      w.localStorage.setItem(
        'kiss-explorer-roots',
        JSON.stringify(['c:/repo', 'C:\\Repo\\', 'C:\\Other']),
      );
    });
    const WWD = 'C:\\Repo';
    send(win, {type: 'configData', config: {work_dir: WWD}});
    showSection(win, 'meta-explorer');
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
    const {win, posted, copied} = makePage();
    const A = '/tmp/lit\\name';
    send(win, {type: 'configData', config: {work_dir: A}});
    showSection(win, 'meta-explorer');
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

  await test('a folder clicked in the picker is highlighted or entered, never opened as a file', async () => {
    // The picker's rows carry their folder in data-path, like the
    // chat's file links do; the file-link click handler must leave
    // them alone (a directory "opened" as a file would land a listing
    // tab in the content pane and slide the task-info panel off).
    const {win, posted} = makePage();
    openExplorer(win, posted, []);
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    let pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    send(win, {
      type: 'dirListing',
      token: pick[pick.length - 1].token,
      path: WD,
      root: WD,
      entries: [{name: 'dir', path: WD + '/dir', isDir: true}],
    });
    const item = picker.querySelector('.folder-picker-item');
    const opens = ofType(posted, 'openFile').length;
    click(win, item);
    assert.strictEqual(item.getAttribute('aria-selected'), 'true');
    click(win, item);
    item.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
    pick = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(pick[pick.length - 1].path, WD + '/dir', 'entered');
    assert.strictEqual(
      ofType(posted, 'openFile').length,
      opens,
      'no folder was opened as a file',
    );
    key(win, win.document, 'Escape');
    win.close();
  });

  await test('Set as Working Directory keeps the keyboard in the tree past the composer focus retries', async () => {
    const {win, posted} = makePage();
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
    const {win, posted} = makePage();
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
    // The refusal is an error notification.
    const toast = win.document.querySelector('.kiss-notification');
    assert.ok(toast, 'an error notification is shown');
    assert.ok(
      toast.textContent.indexOf('Not a directory: ' + OTHER) >= 0,
      "the notification carries the daemon's reason",
    );
    win.close();
  });

  await test('Set as Working Directory: a late daemon reply leaves the keyboard where the user moved it', async () => {
    const {win, posted} = makePage();
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

  await test("The chat header drop-down's own pick never focuses the Explorer, even after an abandoned check-mark switch", async () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, []);
    addOtherFolder(win, posted, []);
    rootRows(win)[1].focus();
    click(win, rootRows(win)[1].querySelector('.explorer-root-set'));
    const tickCheck = ofType(posted, 'listDir')
      .filter(m => String(m.token).startsWith('workdir:'))
      .pop();
    // Before the daemon answers, the user picks a third folder in the
    // chat header's working directory drop-down.
    send(win, {
      type: 'configData',
      config: {work_dir: WD, recent_work_dirs: [{path: '/data/third', ts: 1}]},
    });
    const select = byId(win, 'chat-workdir');
    select.value = '/data/third';
    assert.strictEqual(select.value, '/data/third');
    select.dispatchEvent(new win.Event('change', {bubbles: true}));
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
      "the drop-down's switch leaves the Explorer alone, focus on " +
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
    const {win, posted} = makePage();
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
    const {win, posted} = makePage();
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
    const {win, posted} = makePage();
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
    const {win, posted} = makePage();
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
    const {win, posted} = makePage();
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
    // The chat header's drop-down still lists every folder.
    click(win, picker.querySelector('.folder-picker-cancel'));
    const listed = Array.from(byId(win, 'chat-workdir').options, o => o.value);
    assert.deepStrictEqual(listed.slice(0, 4), [
      WD,
      '/data/newer',
      OTHER,
      '/data/old',
    ]);
    assert.strictEqual(listed.length, 5, 'plus "Choose ..."');
    win.close();
  });

  report('explorerRoots');
  process.exit(0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
