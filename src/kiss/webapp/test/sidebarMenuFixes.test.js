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
const {
  makePage: makeChatPage,
  stubRaf,
  fakeMatchMedia,
  stubClipboard,
  withTeardown,
} = require('./simplify2_harness');
const {
  send,
  sleep,
  click,
  rightClick,
  key,
  byId,
  showSection,
  ofType,
  all,
  makeRunner,
  forbidNativeDialogs,
  menuItem,
  toast,
  toastInput,
  WD,
  SHA_A,
  openExplorer,
  explorerRow,
  openScm,
  pinWorkspace,
} = require('./ui_antipattern_harness');
const {trackWindow, closeWindows} = require('./workdir_helpers');

const {test: runTest, report} = makeRunner();
// Each test's windows are closed once it has settled.
const test = withTeardown(runTest, closeWindows);

function makePage() {
  let copied;
  function beforeScripts(win) {
    stubRaf(win);
    fakeMatchMedia(win, true);
    copied = stubClipboard(win).writes;
    // Native dialogs are gone from main.js; a call fails the test.
    forbidNativeDialogs(win);
  }
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts,
    extraScripts: [
      'marked.min.js',
      'contentContextMenu.js',
      'treeContextMenu.js',
    ],
  });
  trackWindow(win);
  return {win, posted, copied};
}

/** The in-page prompt toast *id* and its text input. */
function promptToast(win, id) {
  const el = toast(win, id);
  assert.ok(el, 'the prompt ' + id + ' is on screen');
  return {toast: el, input: toastInput(el)};
}

/** Type *value* into the prompt toast *id* and press Enter. */
function answerPrompt(win, id, value) {
  const {input} = promptToast(win, id);
  input.value = value;
  key(win, input, 'Enter');
}

async function main() {
  await test('Explorer: a listing during Rename... adds no duplicate row and keeps the box', async () => {
    const {win, posted} = makePage();
    const req = openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
      {name: 'b.txt', path: WD + '/b.txt', isDir: false},
    ]);
    const row = explorerRow(win, WD + '/a.txt');
    rightClick(win, row);
    click(win, menuItem(win, 'Rename...'));
    await sleep(0);
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
      all(win, '.explorer-row[data-explorer-path="' + WD + '/a.txt"]').length,
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
    assert.strictEqual(all(win, '.explorer-row.is-file').length, 3);
    // jsdom fires no blur when the listing moves the focused box through
    // a fragment, so fire the one a browser may: the box has its focus
    // back when the blur's deferred check runs, and nothing commits.
    input.dispatchEvent(new win.FocusEvent('blur'));
    await sleep(0);
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
      all(win, '.explorer-row[data-explorer-path="' + WD + '/a.txt"]').length,
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
    await sleep(0);
    const input2 = row.querySelector('input.explorer-input');
    input2.value = 'z.txt';
    key(win, input2, 'Enter');
    const fsActions = ofType(posted, 'fsAction');
    assert.strictEqual(fsActions.length, 1);
    assert.strictEqual(fsActions[0].action, 'rename');
    assert.strictEqual(fsActions[0].path, WD + '/a.txt');
    assert.strictEqual(fsActions[0].dest, WD + '/z.txt');
    assert.strictEqual(fsActions[0].workDir, WD);
    // So does a blur that sticks (a click elsewhere).
    rightClick(win, row);
    click(win, menuItem(win, 'Rename...'));
    await sleep(0);
    const input3 = row.querySelector('input.explorer-input');
    input3.value = 'y.txt';
    input3.blur();
    await sleep(0);
    const blurred = ofType(posted, 'fsAction');
    assert.strictEqual(blurred.length, 2, 'a blur that sticks commits');
    assert.strictEqual(blurred[1].dest, WD + '/y.txt');
  });

  await test('Folder picker: Select only picks a folder the daemon listed', () => {
    const {win, posted} = makePage();
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
    // configData is adopted directly: no setWorkDir echo, and the
    // working directory is never a settings save.
    assert.strictEqual(ofType(posted, 'setWorkDir').length, 0);
    assert.strictEqual(ofType(posted, 'saveConfig').length, 0);
    // Typed, never listed: Select lists it first...
    input.value = '/definitely/missing';
    click(win, picker.querySelector('.folder-picker-select'));
    assert.strictEqual(
      ofType(posted, 'setWorkDir').length,
      0,
      'not adopted yet',
    );
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
      ofType(posted, 'setWorkDir').length,
      0,
      'BUG: a missing path was adopted',
    );
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
    // The daemon is handed the global working directory (it persists
    // and broadcasts it); nothing goes through saveConfig.
    const adopted = ofType(posted, 'setWorkDir');
    assert.strictEqual(adopted.length, 1);
    assert.strictEqual(adopted[0].workDir, '/data/proj');
    assert.strictEqual(ofType(posted, 'saveConfig').length, 0);
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
      ofType(posted, 'setWorkDir')[1].workDir,
      '/data/proj/child',
    );
    assert.strictEqual(ofType(posted, 'saveConfig').length, 0);
  });

  await test("Folder picker: the global working directory is browsed, not the chat's task folder, and a pick re-roots", () => {
    const {win, posted} = makePage();
    pinWorkspace(win);
    // The active chat replays a task that ran in a sub-folder.  There
    // is one global working directory now: the tab's task folder does
    // not change what the Explorer browses.
    send(win, {
      type: 'task_events',
      task: 'earlier task',
      task_id: 3,
      events: [],
      extra: JSON.stringify({work_dir: WD + '/sub'}),
    });
    showSection(win, 'meta-explorer');
    let list = ofType(posted, 'listDir');
    assert.strictEqual(
      list[list.length - 1].path,
      WD,
      'the global working directory is browsed, not the tab folder',
    );
    send(win, {
      type: 'dirListing',
      token: list[list.length - 1].token,
      path: WD,
      root: WD,
      entries: [{name: 'sub', path: WD + '/sub', isDir: true}],
    });
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    list = ofType(posted, 'listDir');
    const req = list[list.length - 1];
    assert.strictEqual(req.path, WD, 'the picker opens at the global folder');
    send(win, {
      type: 'dirListing',
      token: req.token,
      path: WD,
      root: WD,
      entries: [{name: 'sub', path: WD + '/sub', isDir: true}],
    });
    // Pick the listed sub-folder: it becomes the global working directory.
    click(win, picker.querySelector('.folder-picker-item'));
    click(win, picker.querySelector('.folder-picker-select'));
    assert.ok(picker.hidden);
    const adopted = ofType(posted, 'setWorkDir');
    assert.strictEqual(adopted.length, 1);
    assert.strictEqual(adopted[0].workDir, WD + '/sub');
    assert.strictEqual(ofType(posted, 'saveConfig').length, 0);
    list = ofType(posted, 'listDir').filter(
      m => !String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(
      list[list.length - 1].path,
      WD + '/sub',
      'BUG: the Explorer kept browsing the old folder after the pick',
    );
    assert.strictEqual(list[list.length - 1].workDir, WD + '/sub');
    // The Source Control view follows too.
    showSection(win, 'meta-scm');
    const st = ofType(posted, 'gitStatus');
    assert.strictEqual(st[st.length - 1].workDir, WD + '/sub');
    // Another surface's pick arrives as the daemon's broadcast: this
    // client re-scopes to it the same way.
    send(win, {type: 'workDirChanged', workDir: '/elsewhere'});
    const st2 = ofType(posted, 'gitStatus');
    assert.strictEqual(st2[st2.length - 1].workDir, '/elsewhere');
    showSection(win, 'meta-explorer');
    list = ofType(posted, 'listDir').filter(
      m => !String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(list[list.length - 1].path, '/elsewhere');
    assert.strictEqual(
      ofType(posted, 'setWorkDir').length,
      1,
      'a broadcast is adopted, never echoed back as setWorkDir',
    );
  });

  await test('Folder picker: Up stays at a root ("/" and "C:\\")', () => {
    const {win, posted} = makePage();
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

  await test('Commit menu: a failed action still refreshes Source Control', () => {
    const {win, posted} = makePage();
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

  await test('Commit menu: Copy Commit Message copies the whole message', () => {
    const {win, posted, copied} = makePage();
    const row = openScm(win, posted);
    rightClick(win, row);
    click(win, menuItem(win, 'Copy Commit Message'));
    assert.deepStrictEqual(copied, ['Title line\n\nBody paragraph.']);
    rightClick(win, row);
    click(win, menuItem(win, 'Copy Commit Hash'));
    assert.strictEqual(copied[1], SHA_A);
  });

  await test('Commit menu: Create Tag... without a message makes a lightweight tag; names survive', () => {
    const {win, posted} = makePage();
    const row = openScm(win, posted);
    rightClick(win, row);
    click(win, menuItem(win, 'Create Tag...'));
    // The name, then the message box submitted empty (in-page
    // prompts replaced window.prompt, which the page never showed).
    answerPrompt(win, 'git-create-tag', 'v9');
    answerPrompt(win, 'git-create-tag-message', '');
    let actions = ofType(posted, 'gitAction');
    assert.strictEqual(
      actions.length,
      1,
      'BUG: an empty message aborted the tag',
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
    rightClick(win, row);
    click(win, menuItem(win, 'Create Tag...'));
    key(win, promptToast(win, 'git-create-tag').input, 'Escape');
    assert.strictEqual(ofType(posted, 'gitAction').length, 1);
    // Create Branch... names the branch in its notification.
    rightClick(win, row);
    click(win, menuItem(win, 'Create Branch...'));
    answerPrompt(win, 'git-create-branch', 'topic');
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

  await test('Explorer shortcuts: Ctrl+Alt+C copies the path on Linux', () => {
    const {win, posted, copied} = makePage();
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

  await test('Folder picker: a navigation while a Select waits does not pick the new folder', () => {
    const {win, posted} = makePage();
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
    // A pick is adopted through setWorkDir: none went out.
    assert.strictEqual(ofType(posted, 'setWorkDir').length, 0);
    // The late /a reply is stale and picks nothing either.
    send(win, {
      type: 'dirListing',
      token: selectReq.token,
      path: '/a',
      root: '/a',
      entries: [],
    });
    assert.ok(!picker.hidden);
    assert.strictEqual(ofType(posted, 'setWorkDir').length, 0);
    // A file-system root is refused with a note.
    input.value = '/';
    click(win, picker.querySelector('.folder-picker-select'));
    assert.ok(!picker.hidden);
    assert.ok(
      picker.querySelector('.folder-picker-note').textContent.includes('root'),
    );
    assert.strictEqual(ofType(posted, 'setWorkDir').length, 0);
  });

  await test('Chat header drop-down: a folder picked there re-roots the views past a content tab', () => {
    const {win, posted} = makePage();
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
    showSection(win, 'meta-explorer');
    let list = ofType(posted, 'listDir');
    // The global working directory is browsed past the content tab and
    // its owner chat alike (the owner's task folder is not a scope).
    assert.strictEqual(list[list.length - 1].path, WD);
    // The chat header's working directory drop-down: pick another
    // directory; the daemon's listing confirms the folder and the
    // client hands it to the daemon as the global working directory.
    send(win, {
      type: 'configData',
      config: {work_dir: WD, recent_work_dirs: [{path: '/parent', ts: 1}]},
    });
    const wd = byId(win, 'chat-workdir');
    wd.value = '/parent';
    assert.strictEqual(wd.value, '/parent', 'the folder is listed');
    wd.dispatchEvent(new win.Event('change', {bubbles: true}));
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
    const adopted = ofType(posted, 'setWorkDir');
    assert.strictEqual(adopted.length, 1);
    assert.strictEqual(adopted[0].workDir, '/parent');
    assert.strictEqual(ofType(posted, 'saveConfig').length, 0);
    assert.strictEqual(wd.value, '/parent', 'the drop-down shows the folder');
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
    const {win, posted} = makePage();
    const req = openExplorer(win, posted, [
      {name: 'dir', path: WD + '/dir', isDir: true},
    ]);
    const dirRow = explorerRow(win, WD + '/dir');
    click(win, dirRow);
    const list = ofType(posted, 'listDir');
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
    await sleep(0);
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

  await test('Explorer: a refresh keeps the focused row focused and the single tab stop', () => {
    const {win, posted} = makePage();
    const req = openExplorer(win, posted, [
      {name: 'a.txt', path: WD + '/a.txt', isDir: false},
      {name: 'b.txt', path: WD + '/b.txt', isDir: false},
    ]);
    const row = explorerRow(win, WD + '/a.txt');
    row.focus();
    assert.strictEqual(win.document.activeElement, row);
    const listing = {
      type: 'dirListing',
      token: req.token,
      path: WD,
      root: WD,
      entries: [
        {name: 'a.txt', path: WD + '/a.txt', isDir: false},
        {name: 'b.txt', path: WD + '/b.txt', isDir: false},
      ],
    };
    send(win, listing);
    assert.strictEqual(
      win.document.activeElement,
      row,
      'BUG: the focused row lost focus',
    );
    const tabStops = () =>
      all(win, '.explorer-row').filter(r => r.tabIndex === 0);
    assert.deepStrictEqual(tabStops(), [row], 'exactly one tab stop');
    // An error refresh removes the rows: the focus and the tab stop a
    // removed row held go to the folder's own row.
    const failed = {...listing, entries: undefined, error: 'Permission denied'};
    const folder = explorerRow(win, WD);
    send(win, failed);
    assert.ok(!row.isConnected, 'the error replaced the rows');
    assert.strictEqual(
      win.document.activeElement,
      folder,
      'BUG: the focus fell out of the tree',
    );
    assert.deepStrictEqual(tabStops(), [folder], 'BUG: no tab stop is left');
    // A control outside the tree keeps its focus; the stop its removed
    // row held is still repaired.
    send(win, listing);
    explorerRow(win, WD + '/a.txt').focus();
    const input = byId(win, 'task-input');
    input.focus();
    send(win, failed);
    assert.strictEqual(win.document.activeElement, input);
    assert.deepStrictEqual(tabStops(), [folder]);
  });

  await test('Menus: opening a tree menu closes an open content context menu', () => {
    const {win, posted} = makePage();
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
    // Closing may remove the menu, so it is looked up without byId's
    // existence check.
    const contentMenuOpen = () => {
      const menu = win.document.getElementById('sorcar-content-context-menu');
      return !!menu && !menu.hidden && menu.style.display !== 'none';
    };
    rightClick(win, view);
    assert.ok(contentMenuOpen(), 'the content menu opened');
    rightClick(win, explorerRow(win, WD + '/a.txt'));
    const treeMenu = byId(win, 'sidebar-context-menu');
    assert.ok(treeMenu && !treeMenu.hidden, 'the tree menu opened');
    assert.ok(
      !contentMenuOpen(),
      'BUG: the content context menu stayed open under the tree menu',
    );
    // So does a right-click outside the content area, where the browser's
    // own menu opens (the composer).
    rightClick(win, view);
    assert.ok(contentMenuOpen(), 'the content menu reopened');
    rightClick(win, byId(win, 'task-input'));
    assert.ok(
      !contentMenuOpen(),
      'BUG: the content context menu stayed open under the native menu',
    );
  });

  // The menu is installed on the whole document and filters on the
  // content area, so a Select All there must select the opened file's
  // pane alone: the body would take in the chat beside it in the split
  // layout and the following Copy would carry the conversation (A23).
  await test('Menus: Select All in a content tab selects the file, not the chat beside it', async () => {
    const {win, copied} = makePage();
    send(win, {type: 'system_output', text: 'PRIVATE CHAT CONTENT'});
    assert.ok(
      win.document.body.textContent.includes('PRIVATE CHAT CONTENT'),
      'the chat pane shows the message',
    );
    send(win, {
      type: 'fileContent',
      name: 'n.txt',
      path: '/tmp/n.txt',
      content: 'FILE CONTENT',
    });
    // No Monaco in jsdom: the CDN loader fails and the <pre> fallback
    // renders the file's text.
    const loader = win.document.head.querySelector(
      'script[src*="vs/loader.js"]',
    );
    assert.ok(loader, 'the Monaco loader is requested');
    loader.dispatchEvent(new win.Event('error'));
    await sleep(0);
    const view = byId(win, 'content-tab-area').querySelector(
      '.content-tab-view',
    );
    const code = view.querySelector('.content-code-fallback');
    assert.ok(code, 'the file renders as text');
    rightClick(win, code);
    const item = win.document.querySelector(
      '#sorcar-content-context-menu .sorcar-ctx-item[data-action="select-all"]',
    );
    assert.ok(item, 'the content menu offers Select All');
    click(win, item);
    assert.strictEqual(
      win.getSelection().toString(),
      'FILE CONTENT',
      'BUG: Select All in the file pane selected the page, chat included',
    );
    rightClick(win, code);
    click(
      win,
      win.document.querySelector(
        '#sorcar-content-context-menu .sorcar-ctx-item[data-action="copy"]',
      ),
    );
    await sleep(0);
    assert.deepStrictEqual(copied, ['FILE CONTENT']);
  });

  report('sidebarMenuFixes');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
