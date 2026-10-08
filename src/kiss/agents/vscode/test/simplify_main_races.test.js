// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for stale-callback and dropped-reply races in
// the chat webview (media/main.js) found by the vscode-main simplification
// pass.  Each test drives the real page through host messages and asserts
// the CORRECT behaviour:
//
// * a daemon outage releases every in-flight Explorer listing, content
//   save and manual Git Commit instead of leaving them stuck,
// * a fileContent reply from another chat neither replaces a dirty
//   editor nor consumes the tab's own pending reload,
// * a failure's `result` withdraws the empty provisional Thoughts panel,
// * a Shift keyup lost to a window blur does not turn Enter into a
//   newline,
// * a text result (search, git show) opened after a tab switch belongs
//   to the chat that asked for it, and a file opened FROM a file view is
//   reported to the host under its root chat,
// * an Edit diff of thousands of lines renders without freezing.

'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

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

function connect(win) {
  h.send(win, {type: 'configData', config: {work_dir: h.WD}, apiKeys: {}});
  h.send(win, {type: 'daemonStatus', connected: true});
}

function tabsState(win, entries) {
  h.send(win, {
    type: 'tabs_state',
    tabs: entries.map(e => ({
      tabId: e.id,
      chatId: e.chat || '',
      title: e.id,
      workDir: e.workDir || '',
    })),
  });
}

function lastOfType(posted, type) {
  const all = h.ofType(posted, type);
  return all[all.length - 1];
}

async function waitFor(predicate, message) {
  for (let i = 0; i < 200; i++) {
    const v = predicate();
    if (v) return v;
    await h.sleep(10);
  }
  throw new Error(message);
}

const FILE_PATH = '/shared/notes.txt';

/** An editable content tab owned by chat a1 whose editor holds edits. */
async function openDirtyTab(ctx) {
  const {win, created} = ctx;
  tabsState(win, [
    {id: 'a1', workDir: '/ws/a', chat: 'chat-1'},
    {id: 'b1', workDir: '/ws/b', chat: 'chat-2'},
  ]);
  h.send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: FILE_PATH,
    name: 'notes.txt',
    content: 'v1 on disk',
    version: 'ver-1',
  });
  await waitFor(() => created.length >= 1, 'the content tab creates an editor');
  created[0].editor._type('v1 edited');
  return win.document.querySelector('.content-save-btn');
}

(async () => {
  await test('an outage releases a folder whose first listing never came', () => {
    const {win, posted} = h.makeWebview();
    connect(win);
    h.openExplorer(win, posted, [
      {name: 'foo.py', path: h.WD + '/foo.py', isDir: false},
      {name: 'src', path: h.WD + '/src', isDir: true},
    ]);
    const row = h.explorerRow(win, h.WD + '/src');
    h.click(win, row);
    const before = h.ofType(posted, 'listDir').length;
    assert.ok(row.classList.contains('loading'), 'precondition: in flight');
    h.send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    assert.ok(!row.classList.contains('loading'), 'no longer waiting');
    assert.ok(!row.classList.contains('expanded'), 'collapsed again');
    h.send(win, {type: 'daemonStatus', connected: true, reconnecting: true});
    // The reconnect re-lists the folders listed before (the root), not
    // the one whose listing was lost.
    const relisted = h.ofType(posted, 'listDir').slice(before);
    assert.ok(relisted.every(m => m.path === h.WD), 'only the root is re-listed');
    const afterReconnect = h.ofType(posted, 'listDir').length;
    h.click(win, row);
    assert.strictEqual(
      h.ofType(posted, 'listDir').length,
      afterReconnect + 1,
      'the next expansion asks for the listing afresh',
    );
    const kids = row.nextElementSibling;
    assert.strictEqual(
      kids.querySelectorAll('.explorer-note').length,
      1,
      'one Loading... note, not one per attempt',
    );
    const req = lastOfType(posted, 'listDir');
    h.send(win, {
      type: 'dirListing',
      token: req.token,
      path: req.path,
      root: h.WD,
      entries: [{name: 'a.py', path: h.WD + '/src/a.py', isDir: false}],
    });
    assert.ok(h.explorerRow(win, h.WD + '/src/a.py'), 'the listing lands');
    win.close();
  });

  await test('an outage fails a save in flight so the tab can save again', async () => {
    const ctx = h.makeWebview();
    const {win, posted} = ctx;
    connect(win);
    const saveBtn = await openDirtyTab(ctx);
    h.click(win, saveBtn);
    assert.strictEqual(h.ofType(posted, 'saveFile').length, 1);
    assert.strictEqual(saveBtn.disabled, true, 'precondition: saving');
    h.send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    const status = win.document.querySelector('.content-save-status');
    assert.strictEqual(
      status.textContent,
      'Save failed: the server connection dropped',
    );
    assert.ok(status.classList.contains('error'));
    assert.strictEqual(saveBtn.disabled, false, 'Save works again');
    h.send(win, {type: 'daemonStatus', connected: true, reconnecting: true});
    h.click(win, saveBtn);
    const saves = h.ofType(posted, 'saveFile');
    assert.strictEqual(saves.length, 2, 'the retry goes out');
    assert.notStrictEqual(saves[0].token, saves[1].token);
    // The straggling reply to the dead request is ignored.
    h.send(win, {type: 'fileSaved', token: saves[0].token, ok: true, version: 'v2'});
    assert.strictEqual(saveBtn.disabled, true, 'still waiting for the retry');
    h.send(win, {type: 'fileSaved', token: saves[1].token, ok: true, version: 'v2'});
    assert.strictEqual(status.textContent, 'Saved');
    win.close();
  });

  await test('an outage re-arms the manual Git Commit button', () => {
    const {win, posted} = h.makeWebview();
    connect(win);
    tabsState(win, [{id: 'a1', workDir: '/ws/a', chat: 'chat-1'}]);
    const btn = h.byId(win, 'autocommit-btn');
    h.send(win, {type: 'gitCommit'});
    assert.strictEqual(h.ofType(posted, 'autocommitAction').length, 1);
    assert.strictEqual(btn.disabled, true, 'precondition: committing');
    h.send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    assert.strictEqual(btn.disabled, false);
    assert.strictEqual(btn.querySelector('.more-item-label').textContent, 'Git Commit');
    h.send(win, {type: 'daemonStatus', connected: true, reconnecting: true});
    h.send(win, {type: 'gitCommit'});
    assert.strictEqual(h.ofType(posted, 'autocommitAction').length, 2, 'a new commit can start');
    win.close();
  });

  await test("another chat's open of the path neither replaces edits nor eats the reload", async () => {
    const ctx = h.makeWebview();
    const {win, posted, created} = ctx;
    connect(win);
    const saveBtn = await openDirtyTab(ctx);
    h.click(win, saveBtn);
    const save = lastOfType(posted, 'saveFile');
    h.send(win, {type: 'fileSaved', token: save.token, ok: false, conflict: true, error: 'changed on disk'});
    const toast = win.document.querySelector('[data-notification-id^="file-save-conflict-"]');
    h.click(win, h.toastButton(toast, 'Reload from disk'));
    assert.strictEqual(h.ofType(posted, 'openFile').length, 1, 'the reload is requested');
    // Chat b1's own open of the same path lands first.
    h.send(win, {type: 'fileContent', tabId: 'b1', path: FILE_PATH, name: 'notes.txt', content: 'v2 on disk', version: 'ver-2'});
    await h.sleep(30);
    assert.strictEqual(created.length, 1, "b1's reply must not replace the dirty editor");
    // The tab's own reload reply still replaces the text.
    h.send(win, {type: 'fileContent', tabId: 'a1', path: FILE_PATH, name: 'notes.txt', content: 'v3 on disk', version: 'ver-3'});
    await waitFor(() => created.length >= 2, 'the reload reply re-renders the editor');
    assert.strictEqual(created[created.length - 1].value, 'v3 on disk');
    // The reload is spent: a later plain open protects the new edits again.
    created[created.length - 1].editor._type('v3 edited');
    h.send(win, {type: 'fileContent', tabId: 'a1', path: FILE_PATH, name: 'notes.txt', content: 'v4 on disk', version: 'ver-4'});
    await h.sleep(30);
    assert.strictEqual(created.length, 2, 'the edits survive a plain open');
    win.close();
  });

  await test("a failure's result withdraws the empty provisional Thoughts panel", () => {
    const {win} = h.makeWebview();
    connect(win);
    tabsState(win, [{id: 'a1', workDir: '/ws/a', chat: 'chat-1'}]);
    h.send(win, {type: 'status', running: true, tabId: 'a1'});
    h.send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: 'a1'});
    h.send(win, {type: 'tool_result', name: 'Bash', output: 'x', tabId: 'a1'});
    const out = h.byId(win, 'output');
    assert.strictEqual(out.querySelectorAll('.llm-panel').length, 1, 'precondition: a panel is opened on spec');
    h.send(win, {type: 'result', success: false, text: 'boom', tabId: 'a1'});
    h.send(win, {type: 'task_stopped', tabId: 'a1'});
    assert.strictEqual(out.querySelectorAll('.llm-panel').length, 0, 'nothing was said into it: it is gone');
    win.close();
  });

  await test('a Shift keyup lost to a window blur does not stick', () => {
    const {win, posted} = h.makeWebview();
    connect(win);
    tabsState(win, [{id: 'a1', workDir: '/ws/a', chat: 'chat-1'}]);
    const inp = h.byId(win, 'task-input');
    inp.value = 'hello';
    win.document.dispatchEvent(new win.KeyboardEvent('keydown', {key: 'Shift', bubbles: true}));
    win.dispatchEvent(new win.Event('blur'));
    const before = h.ofType(posted, 'submit').length;
    inp.dispatchEvent(new win.InputEvent('beforeinput', {inputType: 'insertLineBreak', bubbles: true, cancelable: true}));
    assert.strictEqual(h.ofType(posted, 'submit').length, before + 1, 'Enter sends');
    win.close();
  });

  await test('a search result opened after a tab switch belongs to the chat that asked', () => {
    // The stacked (narrow) remote: a content tab replaces the chat on
    // screen, so the host is told which chat it belongs to.
    const {win, posted} = h.makeWebview({narrow: true});
    connect(win);
    tabsState(win, [
      {id: 'a1', workDir: h.WD, chat: 'chat-1'},
      {id: 'b1', workDir: h.WD, chat: 'chat-2'},
    ]);
    // The stacked remote lists the Explorer only while its drawer is open.
    h.click(win, h.byId(win, 'meta-drawer-btn'));
    h.openExplorer(win, posted, [{name: 'src', path: h.WD + '/src', isDir: true}]);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/src'), 'Find in Folder...');
    const box = win.document.activeElement;
    box.value = 'needle';
    h.key(win, box, 'Enter');
    const req = lastOfType(posted, 'fsAction');
    assert.strictEqual(req.action, 'findInFolder');
    // The user switches to b1 before the reply lands; the reply echoes
    // the tab the request was sent as.
    h.send(win, {type: 'openChatFromHistory', chatId: 'chat-2'});
    assert.strictEqual(lastOfType(posted, 'activeTabChanged').tabId, 'b1', 'precondition');
    h.send(win, {type: 'fsResult', token: req.token, tabId: req.tabId, action: 'findInFolder', text: 'src/a.py:1: needle', count: 1});
    assert.strictEqual(h.all(win, '.chat-tab.content-tab').length, 1, 'the result tab opened');
    // The result tab is on screen; the `ready` sent after an outage
    // names the chat it belongs to.
    h.send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    h.send(win, {type: 'daemonStatus', connected: true, reconnecting: true});
    assert.strictEqual(lastOfType(posted, 'ready').tabId, req.tabId, 'the result tab is owned by the chat that asked, not the one on screen');
    win.close();
  });

  await test('a file opened from a file view is reported under its root chat', () => {
    const {win, posted} = h.makeWebview({narrow: true});
    connect(win);
    tabsState(win, [
      {id: 'b1', workDir: '/ws/b', chat: 'chat-2'},
      {id: 'a1', workDir: '/ws/a', chat: 'chat-1'},
    ]);
    h.send(win, {type: 'openChatFromHistory', chatId: 'chat-1'});
    assert.strictEqual(lastOfType(posted, 'activeTabChanged').tabId, 'a1', 'precondition');
    h.send(win, {type: 'fileContent', tabId: 'a1', path: '/ws/a/x.md', name: 'x.md', content: '# x'});
    const strips = h.all(win, '.chat-tab.content-tab');
    assert.strictEqual(strips.length, 1);
    const viewId = strips[0].dataset.tabId;
    // A link clicked inside the x.md view asks as that view.
    h.send(win, {type: 'fileContent', tabId: viewId, path: '/ws/a/y.md', name: 'y.md', content: '# y'});
    assert.strictEqual(h.all(win, '.chat-tab.content-tab').length, 2);
    h.send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    h.send(win, {type: 'daemonStatus', connected: true, reconnecting: true});
    assert.strictEqual(lastOfType(posted, 'ready').tabId, 'a1', 'the chat at the root of the owner chain, not the first chat');
    win.close();
  });

  await test('an Edit diff of thousands of lines renders promptly', () => {
    const {win} = h.makeWebview();
    connect(win);
    tabsState(win, [{id: 'a1', workDir: '/ws/a', chat: 'chat-1'}]);
    const lines = n => Array.from({length: n}, (_, i) => 'line ' + i);
    const oldS = lines(3000).join('\n');
    const newS = lines(3000).map((l, i) => (i === 1500 ? 'changed' : l)).join('\n');
    const t0 = Date.now();
    h.send(win, {type: 'tool_call', name: 'Edit', path: '/ws/a/f.py', old_string: oldS, new_string: newS, tabId: 'a1'});
    const out = h.byId(win, 'output');
    assert.ok(Date.now() - t0 < 5000, 'no freeze');
    assert.strictEqual(out.querySelectorAll('.diff-old').length, 1, 'one line removed');
    assert.strictEqual(out.querySelectorAll('.diff-new').length, 1, 'one line added');
    assert.strictEqual(out.querySelectorAll('.diff-ctx').length, 2999, 'the rest is context');
    // Beyond the table's size cap the diff degrades to remove-all / add-all.
    const huge = lines(5000).join('\n');
    h.send(win, {type: 'tool_call', name: 'Edit', path: '/ws/a/g.py', old_string: huge, new_string: huge + '\nextra', tabId: 'a1'});
    assert.strictEqual(out.querySelectorAll('.diff-old').length, 1 + 5000);
    assert.strictEqual(out.querySelectorAll('.diff-new').length, 1 + 5001);
    win.close();
  });

  console.log(`\nsimplify_main_races: ${passed} passed, ${failures.length} failed`);
  if (failures.length) process.exit(1);
})();
