// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the Explorer's file operations and the
// working-directory pickers of media/main.js:
//
// * an editor tab follows its file when Cut + Paste moves it;
// * the editor's "Open in Browser" button follows a rename to or from
//   .html;
// * Rename... of a folder renames it in its parent, not into itself;
// * Cut and Set as Working Directory act on the machine the menu was
//   opened on, even when the tree turned to another machine meanwhile;
// * a paste answered after the tree turned to another machine opens no
//   folder there;
// * the folder dialog refuses a typed path the daemon spells as a
//   file-system root ("/tmp/.." is "/");
// * repaints before a machine's directories arrive ask it only once.

'use strict';

const assert = require('assert');
const {
  makePage: buildPage,
  send,
  click,
  ofType,
  sleep,
  pickMachine,
  stubRaf,
  fakeMatchMedia,
} = require('./simplify2_harness');
const {
  WD,
  makePage,
  makeRunner,
  byId,
  key,
  menuItem,
  rightClick,
  runMenuItem,
  openExplorer,
  explorerRow,
  showSection,
  toast,
  toastButton,
} = require('./ui_antipattern_harness');
const {MACHINES} = require('./picked_machine_helpers');
const {answerListing} = require('./workdir_helpers');

const {test, report} = makeRunner();

// gpu-box's workspace has the same path as this machine's, so a path
// cut here names an entry there too.
const BOX_DIRS = {
  type: 'machineWorkDirs',
  machine: 'gpu-box',
  workDir: WD,
  recentWorkDirs: [{path: WD, ts: 1}],
  error: '',
};

/**
 * A page on this machine's WD (with /extra added to the Explorer
 * earlier) whose daemon also reaches gpu-box.
 */
function machinePage() {
  const page = buildPage({
    bodyClass: 'remote-chat',
    extraScripts: ['contentContextMenu.js', 'treeContextMenu.js'],
    beforeScripts(w) {
      stubRaf(w);
      fakeMatchMedia(w, true);
      w.localStorage.setItem('kiss-explorer-roots', '["/extra"]');
    },
  });
  const {win, posted} = page;
  send(win, {type: 'configData', config: {work_dir: WD}, machine: 'laptop'});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  showSection(win, 'meta-explorer');
  answerListing(win, posted, WD, '', [
    {name: 'foo.py', path: WD + '/foo.py', isDir: false},
    {name: 'sub', path: WD + '/sub', isDir: true},
  ]);
  return page;
}

/** Turn the active chat (and so the Explorer) to gpu-box. */
function bindToBox(win) {
  send(win, {
    type: 'tabMachine',
    tabId: win._testApi.getActiveTabId(),
    machine: 'gpu-box',
  });
  send(win, BOX_DIRS);
}

/**
 * Open *path* in an editor of chat *owner* showing *content*; with
 * *token*, as the answer to the request that carried it (a reload).
 */
async function openFile(win, owner, path, content, token) {
  send(win, {
    type: 'fileContent',
    tabId: owner,
    path,
    name: path.slice(path.lastIndexOf('/') + 1),
    content,
    version: 'v-' + path,
    token,
  });
  // The stubbed editor is created on the next microtask.
  await sleep(0);
}

/** The open editors' tabs, in tab order (an array of this realm). */
function editorTabs(win) {
  return Array.from(win._testApi.openTabs()).filter(t => t.isContentTab);
}

/** The paths of the open editors, in tab order. */
function editorPaths(win) {
  return editorTabs(win).map(t => t.contentPath);
}

/** The text in the one editor of *path*, brought on screen. */
function editorText(win, created, path) {
  const open = editorTabs(win).filter(t => t.contentPath === path);
  assert.strictEqual(open.length, 1, 'one editor of ' + path);
  win._testApi.switchToTab(open[0].id);
  // An open tab's view stays in the page, hidden unless it is the one shown.
  const shown = created.filter(c => {
    const view = c.holder.closest('.content-tab-view');
    return c.holder.isConnected && view.style.display !== 'none';
  });
  assert.strictEqual(shown.length, 1, 'one editor on screen');
  return shown[0].editor.getModel().getValue();
}

/** Rename... the entry at *path* to *name*; returns the fsAction sent. */
function renameEntry(win, posted, path, name) {
  const row = explorerRow(win, path);
  runMenuItem(win, row, 'Rename...');
  const input = row.querySelector('input.explorer-input');
  input.value = name;
  key(win, input, 'Enter');
  return ofType(posted, 'fsAction').pop();
}

/** Answer the newest fsAction "already exists", and press Replace. */
function pressReplace(win, posted) {
  const first = ofType(posted, 'fsAction').pop();
  send(win, {
    type: 'fsResult',
    token: first.token,
    error: 'exists',
    exists: true,
  });
  click(
    win,
    Array.from(win.document.querySelectorAll('.kiss-notification-action')).find(
      b => b.textContent.trim() === 'Replace',
    ),
  );
}

/** Answer the newest fsAction, the replacing retry: the entry is at *path* now. */
function answerReplaced(win, posted, path) {
  const retry = ofType(posted, 'fsAction').pop();
  assert.strictEqual(retry.overwrite, true);
  send(win, {type: 'fsResult', token: retry.token, ok: true, path});
}

/**
 * Answer the newest fsAction "already exists", press Replace and
 * answer the retry: the entry is at *path* now.
 */
function replaceAt(win, posted, path) {
  pressReplace(win, posted);
  answerReplaced(win, posted, path);
}

/**
 * The question about the unsaved edits that replacing the entry really
 * at *path* would discard (null when it is not asked).
 */
function unsavedQuestion(win, path) {
  return toast(win, 'fs-overwrite-unsaved:' + JSON.stringify(['', path]));
}

/**
 * Answer the newest listDir of *p* with *entries*; the folder is
 * really at *real* (*p* itself unless it is reached through a link).
 */
function listFolder(win, posted, p, entries, real) {
  const req = ofType(posted, 'listDir')
    .filter(m => m.path === p)
    .pop();
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: real || p,
    root: req.workDir,
    entries,
  });
}

async function main() {
  await test('an editor tab follows its file through Cut + Paste', async () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
      {name: 'dir', path: WD + '/dir', isDir: true},
    ]);
    send(win, {
      type: 'fileContent',
      tabId: win._testApi.getActiveTabId(),
      path: WD + '/a.txt',
      name: 'a.txt',
      content: 'text',
      version: 'v1',
    });
    await sleep(0);
    runMenuItem(win, explorerRow(win, WD + '/a.txt'), 'Cut');
    runMenuItem(win, explorerRow(win, WD + '/dir'), 'Paste');
    const move = ofType(posted, 'fsAction').pop();
    assert.strictEqual(move.action, 'move');
    send(win, {
      type: 'fsResult',
      token: move.token,
      ok: true,
      path: WD + '/dir/a.txt',
    });
    const tab = win._testApi.openTabs().find(t => t.isContentTab);
    assert.strictEqual(tab.contentPath, WD + '/dir/a.txt');
    assert.strictEqual(tab.title, 'a.txt');
    // Pasted into the folder it is in, the file stays, and so does its
    // editor: nothing was replaced.
    listFolder(win, posted, WD + '/dir', [{name: 'a.txt'}]);
    runMenuItem(win, explorerRow(win, WD + '/dir/a.txt'), 'Cut');
    runMenuItem(win, explorerRow(win, WD + '/dir'), 'Paste');
    const stay = ofType(posted, 'fsAction').pop();
    assert.deepStrictEqual(
      [stay.path, stay.dest],
      [WD + '/dir/a.txt', WD + '/dir'],
    );
    send(win, {type: 'fsResult', token: stay.token, ok: true, path: stay.path});
    assert.deepStrictEqual(editorPaths(win), [WD + '/dir/a.txt']);
    // Two files cut, and one of the two moves fails: that file is still
    // cut, and the next Paste moves it.
    listFolder(win, posted, WD, [
      {name: 'b.txt'},
      {name: 'c.txt'},
      {name: 'dir', isDir: true},
    ]);
    const ctrlClick = row =>
      row.dispatchEvent(
        new win.MouseEvent('click', {bubbles: true, ctrlKey: true}),
      );
    win.document
      .querySelectorAll('.explorer-row.is-selected')
      .forEach(ctrlClick);
    for (const name of ['b.txt', 'c.txt'])
      ctrlClick(explorerRow(win, WD + '/' + name));
    runMenuItem(win, explorerRow(win, WD + '/b.txt'), 'Cut');
    runMenuItem(win, explorerRow(win, WD + '/dir'), 'Paste');
    const [moved, failed] = ofType(posted, 'fsAction').slice(-2);
    assert.deepStrictEqual(
      [moved.path, failed.path],
      [WD + '/b.txt', WD + '/c.txt'],
    );
    send(win, {
      type: 'fsResult',
      token: moved.token,
      ok: true,
      path: WD + '/dir/b.txt',
    });
    send(win, {
      type: 'fsResult',
      token: failed.token,
      error: 'move failed: busy',
    });
    runMenuItem(win, explorerRow(win, WD + '/dir'), 'Paste');
    const again = ofType(posted, 'fsAction').pop();
    assert.notStrictEqual(again.token, failed.token);
    assert.deepStrictEqual([again.action, again.path], ['move', WD + '/c.txt']);
    win.close();
  });

  await test('Open in Browser follows a rename to or from .html', async () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
    ]);
    send(win, {
      type: 'fileContent',
      tabId: win._testApi.getActiveTabId(),
      path: WD + '/a.txt',
      name: 'a.txt',
      content: '<p>page</p>',
      version: 'v1',
    });
    await sleep(0);
    const bar = win.document.querySelector('.content-save-bar');
    const rename = (from, to) => {
      const row = explorerRow(win, WD + '/' + from);
      runMenuItem(win, row, 'Rename...');
      const input = row.querySelector('input.explorer-input');
      input.value = to;
      key(win, input, 'Enter');
      const token = ofType(posted, 'fsAction').pop().token;
      send(win, {type: 'fsResult', token, ok: true, path: WD + '/' + to});
      answerListing(win, posted, WD, '', [
        {name: to, path: WD + '/' + to, isDir: false},
      ]);
      return bar.querySelector('.content-browser-btn');
    };
    assert.strictEqual(bar.querySelector('.content-browser-btn'), null);
    const btn = rename('a.txt', 'a.html');
    assert.ok(btn, 'a file renamed to .html can be opened in the browser');
    assert.strictEqual(
      btn.nextElementSibling,
      bar.querySelector('.content-save-btn'),
      'right before Save',
    );
    assert.strictEqual(rename('a.html', 'a.HTM'), btn, 'still an HTML file');
    assert.strictEqual(rename('a.HTM', 'a.txt'), null, 'no longer one');
    assert.ok(
      bar.querySelector('.content-save-path').textContent.includes('a.txt'),
    );
    win.close();
  });

  await test('a rename or move over an open file leaves its path one editor, the moved file\u2019s', async () => {
    const {win, posted, created} = makePage();
    openExplorer(win, posted, [
      {name: 'a.txt'},
      {name: 'b.txt'},
      {name: 'dir', isDir: true},
      {name: 'src', isDir: true},
    ]);
    const owner = win._testApi.getActiveTabId();
    await openFile(win, owner, WD + '/a.txt', 'a text');
    await openFile(win, owner, WD + '/b.txt', 'b text');
    await openFile(win, owner, WD + '/dir/b.txt', 'dir b text');
    // Rename... a.txt to b.txt, Replace: the editor of the b.txt that is
    // gone closes, and a.txt's has the name.
    renameEntry(win, posted, WD + '/a.txt', 'b.txt');
    replaceAt(win, posted, WD + '/b.txt');
    assert.deepStrictEqual(editorPaths(win), [
      WD + '/b.txt',
      WD + '/dir/b.txt',
    ]);
    assert.strictEqual(editorText(win, created, WD + '/b.txt'), 'a text');
    // Cut + Paste into dir, Replace: likewise.
    listFolder(win, posted, WD, [
      {name: 'b.txt'},
      {name: 'dir', isDir: true},
      {name: 'src', isDir: true},
    ]);
    runMenuItem(win, explorerRow(win, WD + '/b.txt'), 'Cut');
    runMenuItem(win, explorerRow(win, WD + '/dir'), 'Paste');
    replaceAt(win, posted, WD + '/dir/b.txt');
    assert.deepStrictEqual(editorPaths(win), [WD + '/dir/b.txt']);
    assert.strictEqual(editorText(win, created, WD + '/dir/b.txt'), 'a text');
    // A folder replaced by another: the editors of its files close, and
    // one whose file the other folder brings is that file's.
    await openFile(win, owner, WD + '/dir/only.txt', 'only text');
    await openFile(win, owner, WD + '/src/b.txt', 'src b text');
    renameEntry(win, posted, WD + '/src', 'dir');
    replaceAt(win, posted, WD + '/dir');
    assert.deepStrictEqual(editorPaths(win), [WD + '/dir/b.txt']);
    assert.strictEqual(
      editorText(win, created, WD + '/dir/b.txt'),
      'src b text',
    );
    win.close();
  });

  await test('unsaved edits of a file to replace are asked about before it is replaced: Keep editing replaces nothing, Replace leaves its path one editor', async () => {
    const {win, posted, created} = makePage();
    const entries = [
      {name: 'a.txt'},
      {name: 'b.txt'},
      {name: 'dir', isDir: true},
      {name: 'src', isDir: true},
    ];
    openExplorer(win, posted, entries);
    const owner = win._testApi.getActiveTabId();
    const sent = () => ofType(posted, 'fsAction').length;
    const closeQuestions = () =>
      editorTabs(win).filter(t => toast(win, 'close-dirty-' + t.id));
    await openFile(win, owner, WD + '/a.txt', 'a text');
    await openFile(win, owner, WD + '/b.txt', 'b text');
    created[0].editor._type('a edited');
    created[1].editor._type('b edited');
    // Both with unsaved edits.  Replace: b.txt's are asked about first,
    // and nothing is sent meanwhile.
    renameEntry(win, posted, WD + '/a.txt', 'b.txt');
    pressReplace(win, posted);
    const before = sent();
    let asked = unsavedQuestion(win, WD + '/b.txt');
    assert.ok(
      asked.textContent.includes(
        "'b.txt' has unsaved changes. Replacing it discards them.",
      ),
    );
    // Keep editing: nothing is replaced, each editor is its own file's.
    click(win, toastButton(asked, 'Keep editing'));
    assert.strictEqual(sent(), before);
    assert.deepStrictEqual(editorPaths(win), [WD + '/a.txt', WD + '/b.txt']);
    assert.strictEqual(editorText(win, created, WD + '/b.txt'), 'b edited');
    // Replace: b.txt's editor goes with the file that is gone, unasked,
    // and a.txt's has the path, its edits with it.
    renameEntry(win, posted, WD + '/a.txt', 'b.txt');
    pressReplace(win, posted);
    click(win, toastButton(unsavedQuestion(win, WD + '/b.txt'), 'Replace'));
    answerReplaced(win, posted, WD + '/b.txt');
    assert.deepStrictEqual(editorPaths(win), [WD + '/b.txt']);
    assert.strictEqual(editorText(win, created, WD + '/b.txt'), 'a edited');
    assert.deepStrictEqual(closeQuestions(), []);
    // A folder to replace: the question counts the files with unsaved
    // edits in it.  The moved file's own editor is not one of them.
    listFolder(win, posted, WD, entries.slice(1));
    await openFile(win, owner, WD + '/dir/b.txt', 'dir b text');
    await openFile(win, owner, WD + '/dir/c.txt', 'dir c text');
    await openFile(win, owner, WD + '/src/b.txt', 'src b text');
    for (const editor of created.slice(2)) editor.editor._type('edited');
    renameEntry(win, posted, WD + '/src', 'dir');
    pressReplace(win, posted);
    asked = unsavedQuestion(win, WD + '/dir');
    assert.ok(
      asked.textContent.includes("'dir' has unsaved changes in 2 files."),
    );
    click(win, toastButton(asked, 'Replace'));
    answerReplaced(win, posted, WD + '/dir');
    assert.deepStrictEqual(editorPaths(win), [
      WD + '/b.txt',
      WD + '/dir/b.txt',
    ]);
    assert.deepStrictEqual(closeQuestions(), []);
    // A copy replaces a file under its editor, which stays with its
    // edits: nothing is asked about them.
    listFolder(win, posted, WD, entries.slice(1, 3));
    runMenuItem(win, explorerRow(win, WD + '/b.txt'), 'Copy');
    runMenuItem(win, explorerRow(win, WD + '/dir'), 'Paste');
    replaceAt(win, posted, WD + '/dir/b.txt');
    assert.deepStrictEqual(editorPaths(win), [
      WD + '/b.txt',
      WD + '/dir/b.txt',
    ]);
    assert.strictEqual(editorText(win, created, WD + '/dir/b.txt'), 'edited');
    win.close();
  });

  await test('edits typed into a file’s editor while the file is being replaced are kept and asked about, and no other editor takes their path', async () => {
    const {win, posted, created} = makePage();
    openExplorer(win, posted, [{name: 'a.txt'}, {name: 'b.txt'}]);
    const owner = win._testApi.getActiveTabId();
    const question = path =>
      toast(
        win,
        'close-dirty-' + editorTabs(win).find(t => t.contentPath === path).id,
      );
    await openFile(win, owner, WD + '/a.txt', 'a text');
    await openFile(win, owner, WD + '/b.txt', 'b text');
    // Nothing unsaved when Replace is pressed, so nothing is asked then.
    renameEntry(win, posted, WD + '/a.txt', 'b.txt');
    pressReplace(win, posted);
    created[1].editor._type('b edited');
    answerReplaced(win, posted, WD + '/b.txt');
    // b.txt's editor asks before it closes, as a deleted file's does;
    // a.txt's, with nothing unsaved, closes rather than share the path.
    assert.deepStrictEqual(editorPaths(win), [WD + '/b.txt']);
    assert.strictEqual(editorText(win, created, WD + '/b.txt'), 'b edited');
    click(win, toastButton(question(WD + '/b.txt'), "Don't save"));
    assert.deepStrictEqual(editorPaths(win), []);
    // Both with unsaved edits by then: neither is dropped unasked, and
    // the renamed file's editor stays what it was rather than be a
    // second editor of the path.
    listFolder(win, posted, WD, [{name: 'b.txt'}, {name: 'c.txt'}]);
    await openFile(win, owner, WD + '/b.txt', 'a text');
    await openFile(win, owner, WD + '/c.txt', 'c text');
    created[2].editor._type('a edited');
    renameEntry(win, posted, WD + '/b.txt', 'c.txt');
    pressReplace(win, posted);
    created[3].editor._type('c edited');
    answerReplaced(win, posted, WD + '/c.txt');
    assert.deepStrictEqual(editorPaths(win), [WD + '/b.txt', WD + '/c.txt']);
    click(win, toastButton(question(WD + '/c.txt'), 'Keep editing'));
    assert.deepStrictEqual(editorPaths(win), [WD + '/b.txt', WD + '/c.txt']);
    assert.strictEqual(editorText(win, created, WD + '/b.txt'), 'a edited');
    assert.strictEqual(editorText(win, created, WD + '/c.txt'), 'c edited');
    // The same file once more, its edits now given up with Replace
    // before it is replaced, and typed into again after that: Replace
    // gave up the text it asked about, so the editor asks all the same.
    const giveUpAndReplace = () => {
      renameEntry(win, posted, WD + '/b.txt', 'c.txt');
      pressReplace(win, posted);
      click(win, toastButton(unsavedQuestion(win, WD + '/c.txt'), 'Replace'));
    };
    const version = c => c.editor.getModel().getAlternativeVersionId();
    giveUpAndReplace();
    created[3].editor._type('c typed late');
    answerReplaced(win, posted, WD + '/c.txt');
    assert.deepStrictEqual(editorPaths(win), [WD + '/b.txt', WD + '/c.txt']);
    assert.strictEqual(editorText(win, created, WD + '/c.txt'), 'c typed late');
    click(win, toastButton(question(WD + '/c.txt'), 'Keep editing'));
    // Nor did it give up what is typed into the editor of the file
    // reloaded since, though that editor numbers its edits from one
    // again and gets to the number the replaced one had.
    giveUpAndReplace();
    const view = created[3].holder.closest('.content-tab-view');
    click(win, view.querySelector('.content-refresh-btn'));
    const reload = ofType(posted, 'openFile').pop().token;
    await openFile(win, owner, WD + '/c.txt', 'c on disk', reload);
    while (version(created[4]) < version(created[3]))
      created[4].editor._type('c reloaded');
    answerReplaced(win, posted, WD + '/c.txt');
    assert.strictEqual(editorText(win, created, WD + '/c.txt'), 'c reloaded');
    click(win, toastButton(question(WD + '/c.txt'), 'Keep editing'));
    // Nor what is typed after it and saved before the file is replaced:
    // the file it was saved to is gone, the editor holds its only copy.
    giveUpAndReplace();
    created[4].editor._type('c saved late');
    send(win, {type: 'daemonStatus', connected: true});
    const bar = created[4].holder.closest('.content-tab-view');
    click(win, bar.querySelector('.content-save-btn'));
    const save = ofType(posted, 'saveFile').pop();
    send(win, {type: 'fileSaved', token: save.token, ok: true, version: 'v2'});
    answerReplaced(win, posted, WD + '/c.txt');
    assert.strictEqual(editorText(win, created, WD + '/c.txt'), 'c saved late');
    assert.ok(question(WD + '/c.txt'));
    // No state of that editor is on disk, whatever is undone in it.
    created[4].editor._type('more');
    created[4].editor._undo();
    assert.ok(!bar.querySelector('.content-save-btn').disabled);
    // Nor did it give up the edits of a file its question did not
    // count, typed while it was open.
    listFolder(win, posted, WD, [{name: 'dir', isDir: true}]);
    await openFile(win, owner, WD + '/src/x.txt', 'x text');
    await openFile(win, owner, WD + '/src/y.txt', 'y text');
    const [x, y] = created.slice(-2);
    x.editor._type('x edited');
    renameEntry(win, posted, WD + '/dir', 'src');
    pressReplace(win, posted);
    y.editor._type('y edited');
    click(win, toastButton(unsavedQuestion(win, WD + '/src'), 'Replace'));
    answerReplaced(win, posted, WD + '/src');
    assert.ok(!editorPaths(win).includes(WD + '/src/x.txt'));
    assert.ok(question(WD + '/src/y.txt'));
    win.close();
  });

  await test('an editor follows a file renamed, moved or deleted through a symlinked folder; a link is not its target', async () => {
    const {win, posted} = makePage();
    // The daemon opens a file by its real path; the tree names it
    // through the link.
    const REAL = '/data/real';
    const link = name => ({name, isDir: true, real: REAL});
    openExplorer(win, posted, [link('alias'), {name: 'dir', isDir: true}]);
    const owner = win._testApi.getActiveTabId();
    click(win, explorerRow(win, WD + '/alias'));
    listFolder(win, posted, WD + '/alias', [{name: 'a.txt'}], REAL);
    await openFile(win, owner, REAL + '/a.txt', 'text');
    await openFile(win, owner, REAL + '/c.txt', 'text');
    const renamed = [REAL + '/b.txt', REAL + '/c.txt'];
    const answer = path => {
      const {token} = ofType(posted, 'fsAction').pop();
      send(win, {type: 'fsResult', token, ok: true, path});
    };
    const pressDelete = path => {
      runMenuItem(win, explorerRow(win, path), 'Delete');
      click(win, toastButton(toast(win, 'fs-delete'), 'Delete'));
      answer(path);
    };
    renameEntry(win, posted, WD + '/alias/a.txt', 'b.txt');
    answer(WD + '/alias/b.txt');
    assert.deepStrictEqual(editorPaths(win), renamed);
    // Cut + Paste out of the linked folder, and back into it.
    listFolder(win, posted, WD + '/alias', [{name: 'b.txt'}], REAL);
    runMenuItem(win, explorerRow(win, WD + '/alias/b.txt'), 'Cut');
    runMenuItem(win, explorerRow(win, WD + '/dir'), 'Paste');
    answer(WD + '/dir/b.txt');
    assert.strictEqual(editorPaths(win)[0], WD + '/dir/b.txt');
    listFolder(win, posted, WD + '/alias', [], REAL);
    listFolder(win, posted, WD + '/dir', [{name: 'b.txt'}]);
    runMenuItem(win, explorerRow(win, WD + '/dir/b.txt'), 'Cut');
    runMenuItem(win, explorerRow(win, WD + '/alias'), 'Paste');
    answer(WD + '/alias/b.txt');
    assert.strictEqual(editorPaths(win)[0], REAL + '/b.txt');
    // The link itself renamed: its target has not moved.
    renameEntry(win, posted, WD + '/alias', 'alias2');
    answer(WD + '/alias2');
    assert.deepStrictEqual(editorPaths(win), renamed);
    // A file deleted through the link: its editor closes.
    listFolder(win, posted, WD, [link('alias2'), {name: 'dir', isDir: true}]);
    click(win, explorerRow(win, WD + '/alias2'));
    listFolder(win, posted, WD + '/alias2', [{name: 'b.txt'}], REAL);
    pressDelete(WD + '/alias2/b.txt');
    assert.deepStrictEqual(editorPaths(win), [REAL + '/c.txt']);
    // The link itself deleted: its target's files are still there.
    pressDelete(WD + '/alias2');
    assert.deepStrictEqual(editorPaths(win), [REAL + '/c.txt']);
    win.close();
  });

  await test('a Delete confirmed, or a Cut pasted, after the tree turned to another folder still finds the editor of a file reached through a link', async () => {
    const {win, posted} = makePage();
    const REAL = '/data/real';
    openExplorer(win, posted, [{name: 'alias', isDir: true, real: REAL}]);
    const owner = win._testApi.getActiveTabId();
    click(win, explorerRow(win, WD + '/alias'));
    listFolder(
      win,
      posted,
      WD + '/alias',
      [{name: 'a.txt'}, {name: 'b.txt'}],
      REAL,
    );
    await openFile(win, owner, REAL + '/a.txt', 'a text');
    await openFile(win, owner, REAL + '/b.txt', 'b text');
    // Asked about a.txt, and b.txt cut, while the tree lists the link.
    runMenuItem(win, explorerRow(win, WD + '/alias/b.txt'), 'Cut');
    runMenuItem(win, explorerRow(win, WD + '/alias/a.txt'), 'Delete');
    // Another working directory: the tree no longer knows the link.
    send(win, {type: 'configData', config: {work_dir: '/ws/other'}});
    win._testApi.switchToTab(owner);
    click(win, byId(win, 'explorer-refresh'));
    listFolder(win, posted, '/ws/other', [{name: 'dir', isDir: true}]);
    assert.strictEqual(explorerRow(win, WD + '/alias/a.txt'), null);
    const answer = path => {
      const {token} = ofType(posted, 'fsAction').pop();
      send(win, {type: 'fsResult', token, ok: true, path});
    };
    click(win, toastButton(toast(win, 'fs-delete'), 'Delete'));
    answer(WD + '/alias/a.txt');
    assert.deepStrictEqual(editorPaths(win), [REAL + '/b.txt']);
    runMenuItem(win, explorerRow(win, '/ws/other/dir'), 'Paste');
    answer('/ws/other/dir/b.txt');
    assert.deepStrictEqual(editorPaths(win), ['/ws/other/dir/b.txt']);
    win.close();
  });

  await test('Rename... of a folder keeps it in its parent', async () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, [{name: 'src', path: WD + '/src', isDir: true}]);
    const row = explorerRow(win, WD + '/src');
    runMenuItem(win, row, 'Rename...');
    const input = row.querySelector('input.explorer-input');
    input.value = 'lib';
    key(win, input, 'Enter');
    const [rename] = ofType(posted, 'fsAction');
    assert.strictEqual(rename.action, 'rename');
    assert.strictEqual(rename.path, WD + '/src');
    assert.strictEqual(rename.dest, WD + '/lib');
    win.close();
  });

  await test('Cut keeps the machine its menu was opened on', () => {
    const {win, posted} = machinePage();
    rightClick(win, explorerRow(win, WD + '/foo.py'));
    bindToBox(win);
    click(win, menuItem(win, 'Cut'));
    answerListing(win, posted, WD, 'gpu-box', [
      {name: 'foo.py', path: WD + '/foo.py', isDir: false},
      {name: 'pkg', path: WD + '/pkg', isDir: true},
    ]);
    const before = ofType(posted, 'fsAction').length;
    runMenuItem(win, explorerRow(win, WD + '/pkg'), 'Paste');
    assert.strictEqual(
      ofType(posted, 'fsAction').length,
      before,
      "this machine's file is not moved on gpu-box",
    );
    win.close();
  });

  await test('a paste reply opens its folder only on the machine it ran on', () => {
    const {win, posted} = machinePage();
    const paste = () => {
      runMenuItem(win, explorerRow(win, WD + '/foo.py'), 'Copy');
      runMenuItem(win, explorerRow(win, WD + '/sub'), 'Paste');
      return ofType(posted, 'fsAction').pop();
    };
    const reply = copy =>
      send(win, {
        type: 'fsResult',
        token: copy.token,
        ok: true,
        path: WD + '/sub/foo.py',
      });
    const boxSubListings = () =>
      ofType(posted, 'listDir').filter(
        m => m.path === WD + '/sub' && m.machine === 'gpu-box',
      ).length;
    // The tree turns to gpu-box before this machine's paste is answered.
    const local = paste();
    bindToBox(win);
    answerListing(win, posted, WD, 'gpu-box', [
      {name: 'foo.py', path: WD + '/foo.py', isDir: false},
      {name: 'sub', path: WD + '/sub', isDir: true},
    ]);
    reply(local);
    const sub = explorerRow(win, WD + '/sub');
    assert.strictEqual(
      sub.getAttribute('aria-expanded'),
      'false',
      "BUG: this machine's paste opened gpu-box's folder",
    );
    assert.strictEqual(boxSubListings(), 0);
    // A paste on the machine the tree shows still opens its folder.
    reply(paste());
    assert.strictEqual(sub.getAttribute('aria-expanded'), 'true');
    assert.strictEqual(boxSubListings(), 1);
    win.close();
  });

  await test("Set as Working Directory keeps the menu's machine", () => {
    const {win, posted} = machinePage();
    rightClick(win, explorerRow(win, '/extra'));
    bindToBox(win);
    click(win, menuItem(win, 'Set as Working Directory'));
    const check = ofType(posted, 'listDir')
      .filter(m => String(m.token).startsWith('workdir:'))
      .pop();
    assert.strictEqual(check.path, '/extra');
    assert.strictEqual(check.machine, undefined, 'checked on this machine');
    win.close();
  });

  await test('the folder dialog refuses a path the daemon spells as a root', () => {
    const {win, posted} = makePage();
    openExplorer(win, posted, []);
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    answerListing(win, posted, WD, '', []);
    picker.querySelector('.folder-picker-input').value = '/tmp/..';
    click(win, picker.querySelector('.folder-picker-select'));
    const req = ofType(posted, 'listDir').pop();
    assert.strictEqual(req.path, '/tmp/..');
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: '/',
      root: '/',
      entries: [],
    });
    assert.strictEqual(ofType(posted, 'setWorkDir').length, 0);
    assert.ok(!picker.hidden, 'the dialog stays open');
    assert.ok(
      picker
        .querySelector('.folder-picker-note')
        .textContent.includes('file-system root'),
    );
    assert.ok(explorerRow(win, WD), 'the Explorer keeps the workspace');
    win.close();
  });

  await test('repaints before the reply ask a machine for its directories once', () => {
    const {win, posted} = makePage();
    send(win, {type: 'configData', config: {work_dir: WD}, machine: 'laptop'});
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
    const asks = () => ofType(posted, 'getMachineWorkDirs').length;
    pickMachine(win, 'gpu-box');
    assert.strictEqual(asks(), 1);
    for (let i = 0; i < 3; i++) send(win, {type: 'status', running: false});
    assert.strictEqual(asks(), 1, 'the repaints ask nothing more');
    send(win, BOX_DIRS);
    send(win, {type: 'status', running: false});
    assert.strictEqual(asks(), 1);
    // An outage swallows a reply: the reconnected page asks again.
    pickMachine(win, '');
    send(win, {type: 'daemonStatus', connected: false});
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 2});
    const sent = asks();
    pickMachine(win, 'gpu-box');
    assert.ok(asks() > sent, 'asked again after the outage');
    win.close();
  });

  report('explorerOps_sweep1030');
}

main();
