// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end (jsdom) tests for the two refresh buttons of media/main.js:
//
//   * the first row of the Chats sidebar, above the Projects row and
//     the search box, holds a button that reloads the whole app page
//     (the browser's reload, which the standalone app window and a
//     home-screen app have no address bar for);
//   * every content tab that shows something on disk (a code file, a
//     .md / .html file, a PDF or image, a report, a folder listing) has
//     a button at its top right that re-reads it from disk through
//     reloadContentTab: on a Save bar it is the bar's last item, else it
//     floats over the view's top right corner.  A virtual tab (a
//     commit's patch) has nothing on disk to re-read and gets none.
//
// The reloads are counted by the harness's reloadCountingConsole.

const assert = require('assert');
const {
  makePage: buildPage,
  send,
  stubMonaco,
  waitFor,
  click,
  flushMicrotasks,
  reloadCountingConsole,
} = require('./simplify2_harness');
const {
  writeReport,
  reportRequests,
  answerReport,
} = require('./subagent_tab_helpers');

function makePage() {
  const reloads = [];
  let created;
  const {win, posted} = buildPage({
    virtualConsole: reloadCountingConsole(reloads),
    extraScripts: ['marked.min.js'],
    beforeScripts: w => {
      created = stubMonaco(w);
      w.URL.createObjectURL = () => 'blob:fake';
      w.URL.revokeObjectURL = () => {};
    },
  });
  return {win, posted, created, reloads};
}

function openFiles(posted) {
  return posted.filter(m => m.type === 'openFile');
}

function refreshBtns(win) {
  return win.document.querySelectorAll(
    '#content-tab-area .content-refresh-btn',
  );
}

function testSidebarRefreshButtonReloadsTheApp() {
  const {win, reloads} = makePage();
  const row = win.document.getElementById(
    'sidebar-tab-history-panel',
  ).firstElementChild;
  assert.ok(
    row.classList.contains('sidebar-refresh-row'),
    "the refresh row is the panel's first",
  );
  const btn = row.querySelector('#chats-refresh-btn');
  assert.ok(btn, 'the row holds the refresh button');
  assert.strictEqual(
    row.lastElementChild,
    btn,
    "the button is at the row's right end",
  );
  // The row is the button alone (the panel has no "Chats" title row);
  // the Projects row comes under it and the search box follows.
  assert.strictEqual(row.textContent.trim(), '');
  assert.strictEqual(row.parentNode.querySelector('.sidebar-hdr'), null);
  const projects = row.nextElementSibling;
  assert.strictEqual(projects.id, 'projects-row');
  assert.ok(
    projects.nextElementSibling.querySelector('#history-search'),
    'the search box follows the Projects row',
  );
  assert.strictEqual(btn.getAttribute('aria-label'), 'Refresh the app');
  assert.ok(btn.querySelector('svg'), 'the button shows an icon');
  assert.strictEqual(reloads.length, 0, 'no reload before the click');
  click(win, btn);
  assert.strictEqual(reloads.length, 1, 'the click reloads the page');
  win.close();
  console.log(
    '  ok - the refresh button, in the row above Projects, reloads the app',
  );
}

async function testEditableFileButtonSitsInSaveBarAndReloadsFromDisk() {
  const {win, posted, created, reloads} = makePage();
  send(win, {type: 'configData', config: {work_dir: '/ws/a'}});
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', workDir: '/ws/a', chatId: 'chat-1'}],
  });
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/a/notes.txt',
    name: 'notes.txt',
    content: 'v1 on disk',
    version: 'ver-1',
  });
  const {editor} = await waitFor(
    () => created[0],
    'the file opens in an editor',
  );
  const bar = win.document.querySelector('#content-tab-area .content-save-bar');
  assert.ok(bar, 'an editable file has a Save bar');
  const btns = refreshBtns(win);
  assert.strictEqual(btns.length, 1, 'exactly one refresh button per tab');
  assert.strictEqual(btns[0].parentNode, bar, 'the button is in the Save bar');
  assert.strictEqual(
    bar.lastElementChild,
    btns[0],
    "it is the bar's last item (top right)",
  );
  assert.ok(
    !btns[0].classList.contains('content-refresh-overlay'),
    'a bar button does not float',
  );
  assert.strictEqual(btns[0].getAttribute('aria-label'), 'Reload from disk');

  // Edit, then refresh: the daemon is asked for the file and its reply
  // replaces the edited text (reloadContentTab's contract).
  editor._type('v1 edited');
  assert.strictEqual(openFiles(posted).length, 0, 'nothing asked yet');
  click(win, btns[0]);
  const asks = openFiles(posted);
  assert.strictEqual(asks.length, 1, 'the click asks the daemon for the file');
  assert.strictEqual(asks[0].path, '/ws/a/notes.txt');
  assert.strictEqual(asks[0].tabId, 'a1', 'asked as the owning chat');
  assert.strictEqual(asks[0].workDir, '/ws/a');
  assert.ok(asks[0].token, 'the reload carries its token');
  // The daemon echoes the request's token on its reply.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    token: asks[0].token,
    path: '/ws/a/notes.txt',
    name: 'notes.txt',
    content: 'v2 on disk',
    version: 'ver-2',
  });
  await waitFor(
    () => created.length >= 2 && created[1].value === 'v2 on disk',
    'the reply replaces the edited text with the file on disk',
  );
  assert.strictEqual(
    refreshBtns(win).length,
    1,
    'the re-rendered tab has one refresh button again',
  );
  assert.strictEqual(
    reloads.length,
    0,
    'a file refresh never reloads the page',
  );
  win.close();
  console.log(
    "  ok - an editable file's button sits in the Save bar and reloads from disk",
  );
}

async function testMarkdownButtonInBarAndBareHtmlButtonFloats() {
  const {win, created} = makePage();
  send(win, {
    type: 'fileContent',
    path: '/ws/README.md',
    name: 'README.md',
    content: '# hi',
    version: 'ver-1',
  });
  let btns = refreshBtns(win);
  assert.strictEqual(btns.length, 1, 'a previewable .md has one button');
  assert.ok(
    btns[0].closest('.content-save-bar'),
    'a .md with a version stamp puts it in its Save bar',
  );
  assert.strictEqual(
    btns[0].closest('.content-save-bar').lastElementChild,
    btns[0],
    'after the Edit source / Preview toggle and Save',
  );

  // A read-only .html (no version stamp): its bar holds the path and
  // "Open in Browser", nothing to save.
  send(win, {
    type: 'fileContent',
    path: '/ws/page.html',
    name: 'page.html',
    content: '<p>page</p>',
  });
  btns = refreshBtns(win);
  assert.strictEqual(btns.length, 2, 'the second tab has its own button');
  const bar = btns[1].closest('.content-save-bar');
  assert.ok(bar, 'the read-only html view has a bar too');
  assert.strictEqual(bar.querySelector('.content-save-btn'), null);
  assert.strictEqual(
    btns[1].previousElementSibling.className,
    'content-browser-btn',
    'after Open in Browser',
  );
  assert.ok(!btns[1].classList.contains('content-refresh-overlay'));
  // An editor is only created once the Monaco loader's promise resolves.
  await flushMicrotasks();
  assert.strictEqual(created.length, 0, 'no editor for the two previews');
  win.close();
  console.log('  ok - a .md and a read-only .html button sit in their bars');
}

function testBinaryAndDirectoryFloatReportInBarVirtualHasNone() {
  const {win, posted} = makePage();
  const chat = win._testApi.getActiveTabId();
  send(win, {
    type: 'fileContent',
    path: '/ws/pic.png',
    name: 'pic.png',
    binary: true,
    mime: 'image/png',
    base64: 'AAAA',
  });
  assert.strictEqual(refreshBtns(win).length, 1, 'an image tab has a button');
  assert.ok(refreshBtns(win)[0].classList.contains('content-refresh-overlay'));

  // A finished task's report is the file itself, as the daemon read it.
  writeReport(win, 'reports/report.md', '# report', {tabId: chat});
  send(win, {type: 'task_done', success: true, tabId: chat});
  answerReport(win, reportRequests(posted)[0], '# report');
  assert.strictEqual(refreshBtns(win).length, 2, 'a report tab has a button');
  assert.ok(
    refreshBtns(win)[1].closest('.content-save-bar'),
    "a report is its file's own view: the button is in the Save bar",
  );

  send(win, {
    type: 'fileContent',
    path: '/ws/docs',
    name: 'docs',
    isDirectory: true,
    content: '/ws/docs:\n\na.txt\n',
  });
  const btns = refreshBtns(win);
  assert.strictEqual(btns.length, 3, 'a folder listing has a button');
  const before = openFiles(posted).length;
  click(win, btns[2]);
  const asks = openFiles(posted);
  assert.strictEqual(asks.length, before + 1, 'the click re-lists the folder');
  assert.strictEqual(asks[asks.length - 1].path, '/ws/docs');

  send(win, {
    type: 'fileContent',
    path: 'git-show:///ws/abc1234',
    name: 'abc1234 - fix',
    content: 'diff --git a/x b/x\n',
    languageName: 'x.diff',
    isVirtual: true,
  });
  assert.strictEqual(
    refreshBtns(win).length,
    3,
    "a virtual tab (a commit's patch) gets no refresh button",
  );
  win.close();
  console.log(
    "  ok - image and folder buttons float, a report's is in its bar; a virtual tab has none",
  );
}

(async () => {
  try {
    testSidebarRefreshButtonReloadsTheApp();
    await testEditableFileButtonSitsInSaveBarAndReloadsFromDisk();
    await testMarkdownButtonInBarAndBareHtmlButtonFloats();
    testBinaryAndDirectoryFloatReportInBarVirtualHasNone();
    console.log('refreshButtons.test.js: all tests passed');
  } catch (err) {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exitCode = 1;
  }
})();
