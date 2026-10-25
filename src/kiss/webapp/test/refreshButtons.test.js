// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end (jsdom) tests for the two refresh buttons of media/main.js:
//
//   * the Chats row of the sidebar ends with a button that reloads the
//     whole app page (the browser's reload, which the standalone app
//     window and a home-screen app have no address bar for);
//   * every content tab that shows something on disk (a code file, a
//     .md / .html file, a PDF or image, a report, a folder listing) has
//     a button at its top right that re-reads it from disk through
//     reloadContentTab: on a Save bar it is the bar's last item, else it
//     floats over the view's top right corner.  A virtual tab (a
//     commit's patch) has nothing on disk to re-read and gets none.
//
// jsdom cannot navigate: every window.location.reload() surfaces as a
// "Not implemented: navigation" jsdomError, which is how the reloads
// are counted.

const assert = require('assert');
const {VirtualConsole} = require('jsdom');
const {
  makePage: buildPage,
  send,
  stubMonaco,
  waitFor,
  click,
} = require('./simplify2_harness');

function makePage() {
  const reloads = [];
  const vc = new VirtualConsole();
  vc.on('jsdomError', err => {
    if (/navigation/.test(String(err && err.message))) reloads.push(1);
  });
  let created;
  const {win, posted} = buildPage({
    virtualConsole: vc,
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

function testChatsRowRefreshButtonReloadsTheApp() {
  const {win, reloads} = makePage();
  const hdr = win.document.querySelector(
    '#sidebar-tab-history-panel .sidebar-hdr',
  );
  assert.ok(hdr, 'the Chats header row exists');
  assert.ok(
    hdr.classList.contains('chats-hdr'),
    'the row is the flex Chats row',
  );
  assert.strictEqual(hdr.firstElementChild.textContent.trim(), 'Chats');
  const btn = hdr.querySelector('#chats-refresh-btn');
  assert.ok(btn, 'the Chats row holds the refresh button');
  assert.strictEqual(
    hdr.lastElementChild,
    btn,
    "the button is at the row's right end",
  );
  assert.strictEqual(btn.getAttribute('aria-label'), 'Refresh the app');
  assert.ok(btn.querySelector('svg'), 'the button shows an icon');
  assert.strictEqual(reloads.length, 0, 'no reload before the click');
  click(win, btn);
  assert.strictEqual(reloads.length, 1, 'the click reloads the page');
  win.close();
  console.log('  ok - the Chats row refresh button reloads the app');
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
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
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

  // A read-only .html (no version stamp: a bare iframe, no Save bar).
  send(win, {
    type: 'fileContent',
    path: '/ws/page.html',
    name: 'page.html',
    content: '<p>page</p>',
  });
  btns = refreshBtns(win);
  assert.strictEqual(btns.length, 2, 'the second tab has its own button');
  const floating = Array.from(btns).find(b => !b.closest('.content-save-bar'));
  assert.ok(floating, 'the bare html view gets a button outside any bar');
  assert.ok(
    floating.classList.contains('content-refresh-overlay'),
    'it floats over the view',
  );
  assert.ok(
    floating.parentNode.classList.contains('content-tab-view'),
    'it is a direct child of the tab view',
  );
  assert.strictEqual(created.length, 0, 'no editor for the two previews');
  win.close();
  console.log(
    '  ok - a .md button sits in the bar, a bare .html button floats',
  );
}

async function testBinaryReportAndDirectoryFloatVirtualHasNone() {
  const {win, posted} = makePage();
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

  send(win, {
    type: 'fileContent',
    path: '/ws/report.md',
    name: 'report.md',
    content: '<h1>report</h1>',
    isReport: true,
  });
  assert.strictEqual(refreshBtns(win).length, 2, 'a report tab has a button');

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
    '  ok - image, report and folder buttons float; a virtual tab has none',
  );
}

(async () => {
  try {
    testChatsRowRefreshButtonReloadsTheApp();
    await testEditableFileButtonSitsInSaveBarAndReloadsFromDisk();
    await testMarkdownButtonInBarAndBareHtmlButtonFloats();
    await testBinaryReportAndDirectoryFloatVirtualHasNone();
    console.log('refreshButtons.test.js: all tests passed');
  } catch (err) {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exitCode = 1;
  }
})();
