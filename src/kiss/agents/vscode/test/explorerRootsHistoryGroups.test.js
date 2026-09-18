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
  win.prompt = () => '';
  win.confirm = () => true;
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
    const after = posted.slice(before);
    const saved = after.find(m => m.type === 'saveConfig');
    assert.ok(
      saved && saved.config.work_dir === OTHER,
      'saveConfig {work_dir: OTHER}',
    );
    const pinned = after.find(m => m.type === 'setWorkDir');
    assert.ok(pinned && pinned.workDir === OTHER, 'setWorkDir OTHER');
    assert.strictEqual(byId(win, 'cfg-work-dir').value, OTHER);
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
    let fsReq = ofType(posted, 'fsAction').pop();
    assert.strictEqual(fsReq.action, 'delete');
    assert.strictEqual(fsReq.workDir, SUB);
    rightClick(win, rowFor(win, SUB + '/x.txt', WD));
    click(win, menuItem(win, 'Delete'));
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
    assert.ok(
      groups(win)[0].style.getPropertyValue('--task-color'),
      'block coloured by chat',
    );
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

  console.log(`\n${passed} passed, ${failures.length} failed`);
  process.exit(failures.length ? 1 : 0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
