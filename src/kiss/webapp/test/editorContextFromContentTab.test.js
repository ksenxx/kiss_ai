// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the in-page editor context: the file tab
// the user viewed last is "the file open in the editor" for a prompt
// (`submit.activeFile`) and for ghost-text completion
// (`complete.activeFile` + `activeFileContent`, the tab's buffer). This
// is what gives the webapp its editor context.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  clickTab,
  recordTimers,
  fireTimers,
} = require('./simplify2_harness');
const {writeReport: writeReportAs} = require('./subagent_tab_helpers');

const pages = [];

/**
 * A remote-chat page whose 300 ms timers are recorded (cancellation
 * aware), so the ghost debounce can be fired by hand instead of waited
 * for, and a debounce main.js cleared never fires.  *extraScripts* are
 * loaded after main.js.
 */
function makePage(extraScripts = []) {
  let timers = [];
  const page = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts,
    beforeScripts: win => {
      timers = recordTimers(win, 300);
    },
  });
  pages.push(page.win);
  return {win: page.win, posted: page.posted, timers};
}

function contentTabId(win) {
  const el = Array.from(win.document.querySelectorAll('.chat-tab')).find(
    e => e.dataset.tabId && e.dataset.tabId !== 'a1',
  );
  assert.ok(el, 'a content tab strip must exist');
  return el.dataset.tabId;
}

function submitPrompt(win, posted, text) {
  const before = posted.filter(m => m.type === 'submit').length;
  win.document.getElementById('task-input').value = text;
  win.document
    .getElementById('send-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const sent = posted.filter(m => m.type === 'submit');
  assert.strictEqual(sent.length, before + 1, 'one submit is posted');
  return sent[sent.length - 1];
}

/**
 * Type *text* into the composer and fire the 300 ms ghost debounce the
 * input armed (and did not clear), so the `complete` request is posted
 * at once.
 */
function requestGhost({win, posted, timers}, text) {
  const before = posted.filter(m => m.type === 'complete').length;
  const armed = timers.length;
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.selectionStart = text.length;
  inp.selectionEnd = text.length;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  const live = timers.slice(armed).filter(t => !t.cleared);
  assert.strictEqual(live.length, 1, 'the input armed one live ghost debounce');
  fireTimers(live);
  assert.strictEqual(
    posted.filter(m => m.type === 'complete').length,
    before + 1,
    'one ghost request is posted',
  );
}

function openChat(win) {
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws/a'}],
  });
}

function testViewedFileTabIsTheEditorContext() {
  const page = makePage();
  const {win, posted} = page;
  openChat(win);
  // No file tab yet: neither message names a file.
  assert.strictEqual(submitPrompt(win, posted, 'hello').activeFile, undefined);
  send(win, {type: 'status', running: false, tabId: 'a1'});
  requestGhost(page, 'hel');
  let completes = posted.filter(m => m.type === 'complete');
  assert.ok(completes.length >= 1, 'a ghost request is posted');
  assert.strictEqual(
    completes[completes.length - 1].activeFile,
    '',
    'completion names no file explicitly (the daemon clears its snapshot)',
  );

  // The user opens a markdown file: its tab is focused (viewed).
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/a/notes.md',
    name: 'notes.md',
    content: '# Notes\n\nunsaved buffer text',
    version: 'v1',
  });
  const fileTab = contentTabId(win);
  // Back on the chat: the viewed file is the context of what they type.
  clickTab(win, 'a1');
  const sub = submitPrompt(win, posted, 'summarize the notes');
  assert.strictEqual(sub.activeFile, '/ws/a/notes.md');
  assert.strictEqual(sub.activeFileContent, undefined, 'a run needs no buffer');

  send(win, {type: 'status', running: false, tabId: 'a1'});
  requestGhost(page, 'summ');
  completes = posted.filter(m => m.type === 'complete');
  const last = completes[completes.length - 1];
  assert.strictEqual(last.activeFile, '/ws/a/notes.md');
  assert.strictEqual(
    last.activeFileContent,
    '# Notes\n\nunsaved buffer text',
    'completion sees the tab buffer, not the file on disk',
  );

  // Closing the file tab ends the context.  The mobile chat screen lists
  // no content tabs, so the user goes to the file first and closes it
  // from its own strip entry; the chat is back on screen afterwards.
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws/a'}],
  });
  clickTab(win, fileTab);
  win.document
    .querySelector(`.chat-tab[data-tab-id="${fileTab}"] .chat-tab-close`)
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    'a1',
    'the chat is on screen again',
  );
  send(win, {type: 'status', running: false, tabId: 'a1'});
  assert.strictEqual(
    submitPrompt(win, posted, 'and now?').activeFile,
    undefined,
    'a closed file tab is no context',
  );
  // ...and completion says so explicitly, so the daemon forgets the
  // closed file's snapshot instead of keeping it (an absent activeFile
  // means "keep" there).
  send(win, {type: 'status', running: false, tabId: 'a1'});
  requestGhost(page, 'and n');
  completes = posted.filter(m => m.type === 'complete');
  assert.strictEqual(completes[completes.length - 1].activeFile, '');
  assert.strictEqual(
    completes[completes.length - 1].activeFileContent,
    undefined,
  );
}

function testEmptiedBufferIsSentAsEmptyContent() {
  // A file whose buffer is empty is "open with an empty buffer": the
  // completion carries activeFileContent "" (not nothing, which would
  // leave the daemon's previous snapshot of the file in place).
  const page = makePage();
  const {win, posted} = page;
  openChat(win);
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/a/empty.md',
    name: 'empty.md',
    content: '',
    version: 'v1',
  });
  clickTab(win, 'a1');
  send(win, {type: 'status', running: false, tabId: 'a1'});
  requestGhost(page, 'emp');
  const completes = posted.filter(m => m.type === 'complete');
  const last = completes[completes.length - 1];
  assert.strictEqual(last.activeFile, '/ws/a/empty.md');
  assert.strictEqual(
    last.activeFileContent,
    '',
    'an empty buffer is sent as ""',
  );
}

function testVirtualAndDirectoryTabsAreNoEditorContext() {
  // A commit's patch (git-show://...) and a folder listing are content
  // tabs but not files on disk: never the active file.
  const page = makePage();
  const {win, posted} = page;
  openChat(win);
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: 'git-show:///ws/a/abc1234',
    name: 'abc1234 - fix',
    content: 'diff --git a/x b/x\n',
    languageName: 'x.diff',
    isVirtual: true,
  });
  clickTab(win, 'a1');
  assert.strictEqual(
    submitPrompt(win, posted, 'explain this commit').activeFile,
    undefined,
    'a virtual git tab is not a file on disk',
  );
  send(win, {type: 'status', running: false, tabId: 'a1'});
  requestGhost(page, 'expl');
  let completes = posted.filter(m => m.type === 'complete');
  assert.strictEqual(completes[completes.length - 1].activeFile, '');

  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/a/src',
    name: 'src',
    content: 'a.py\nb.py\n',
    isDirectory: true,
  });
  clickTab(win, 'a1');
  assert.strictEqual(
    submitPrompt(win, posted, 'list them').activeFile,
    undefined,
    'a folder listing is not a file',
  );

  // A real file viewed afterwards is the context again.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/a/src/a.py',
    name: 'a.py',
    content: 'x = 1\n',
    version: 'v1',
  });
  clickTab(win, 'a1');
  assert.strictEqual(
    submitPrompt(win, posted, 'and a.py?').activeFile,
    '/ws/a/src/a.py',
  );
  send(win, {type: 'status', running: false, tabId: 'a1'});
  requestGhost(page, 'and a');
  completes = posted.filter(m => m.type === 'complete');
  assert.strictEqual(
    completes[completes.length - 1].activeFile,
    '/ws/a/src/a.py',
  );
  assert.strictEqual(
    completes[completes.length - 1].activeFileContent,
    'x = 1\n',
  );
}

function writeReport(win, filePath, content) {
  writeReportAs(win, filePath, content, {tabId: 'a1'});
}

function testReportTabReportsItsMarkdownSource() {
  // A finished task's report opens as converted HTML; completion must
  // see the report file's own text (never the HTML), and a regenerated
  // report must replace the earlier text rather than leaving it out
  // (an omitted buffer keeps the daemon's previous snapshot).
  const page = makePage();
  const {win, posted} = page;
  openChat(win);
  const reportPath = '/ws/a/reports/summary.md';
  writeReport(win, reportPath, '# Report\n\noldReportIdentifier\n');
  send(win, {type: 'task_done', success: true, tabId: 'a1'});
  contentTabId(win);
  clickTab(win, 'a1');
  send(win, {type: 'status', running: false, tabId: 'a1'});
  requestGhost(page, 'oldRe');
  let completes = posted.filter(m => m.type === 'complete');
  let last = completes[completes.length - 1];
  assert.strictEqual(last.activeFile, reportPath);
  assert.strictEqual(
    last.activeFileContent,
    '# Report\n\noldReportIdentifier\n',
  );

  send(win, {type: 'status', running: true, tabId: 'a1'});
  writeReport(win, reportPath, '# Report\n\nnewReportIdentifier\n');
  send(win, {type: 'task_done', success: true, tabId: 'a1'});
  clickTab(win, 'a1');
  send(win, {type: 'status', running: false, tabId: 'a1'});
  requestGhost(page, 'newRe');
  completes = posted.filter(m => m.type === 'complete');
  last = completes[completes.length - 1];
  assert.strictEqual(last.activeFile, reportPath);
  assert.strictEqual(
    last.activeFileContent,
    '# Report\n\nnewReportIdentifier\n',
    'the regenerated report replaces the earlier text',
  );
}

function testReadOnlyPreviewReportsNoBuffer() {
  // A read-only preview (no version stamp: an HTML page past the
  // editor cap, which can run to tens of MB) has nothing the user can
  // edit, so the completion names the file with a null buffer: the
  // daemon drops any snapshot it kept of the path and reads the head
  // of the file from disk, instead of the page travelling back whole
  // with every keystroke.  The pieces of a page the daemon sent in
  // chunks are joined before the tab opens.
  const page = makePage();
  const {win, posted} = page;
  openChat(win);
  const head = {type: 'fileContent', tabId: 'a1', path: '/ws/a/reports/chat-big.html'};
  send(win, {
    ...head,
    name: 'chat-big.html',
    content: '<!DOCTYPE html><html><body><p id="first">first piece</p>',
    more: true,
  });
  assert.throws(() => contentTabId(win), 'no tab until the last piece is in');
  send(win, {...head, content: '<p id="last">last piece</p></body></html>'});
  contentTabId(win);
  const frame = win.document.querySelector('.content-html-frame');
  assert.ok(frame, 'the joined page renders in the preview frame');
  assert.ok(frame.srcdoc.indexOf('first piece') >= 0, 'with its first piece');
  assert.ok(frame.srcdoc.indexOf('last piece') >= 0, 'and its last');
  assert.strictEqual(
    win.document.querySelector('.content-save-bar'),
    null,
    'read-only: no Save bar',
  );
  clickTab(win, 'a1');
  assert.strictEqual(
    submitPrompt(win, posted, 'what does the page say?').activeFile,
    '/ws/a/reports/chat-big.html',
  );
  send(win, {type: 'status', running: false, tabId: 'a1'});
  requestGhost(page, 'what');
  const completes = posted.filter(m => m.type === 'complete');
  const last = completes[completes.length - 1];
  assert.strictEqual(last.activeFile, '/ws/a/reports/chat-big.html');
  assert.ok(
    'activeFileContent' in last && last.activeFileContent === null,
    'a read-only preview reports a null buffer, not an absent one',
  );
}

function testBrowserTabIsNoEditorContext() {
  // A streamed browser tab is a content tab with a URL, not a file.
  // The streamed-browser view lives in its own script, browserTab.js.
  const {win, posted} = makePage(['browserTab.js']);
  openChat(win);
  send(win, {
    type: 'openBrowserTab',
    tab_id: 'browser-1',
    url: 'https://example.com',
    title: 'Example',
    focus: true,
  });
  assert.strictEqual(contentTabId(win), 'browser-1', 'the browser tab opened');
  const browserStrip = win.document.querySelector(
    '#tab-list .chat-tab[data-tab-id="browser-1"]',
  );
  assert.ok(browserStrip.classList.contains('active'), 'and was focused');
  clickTab(win, 'a1');
  assert.strictEqual(
    submitPrompt(win, posted, 'what is on that page?').activeFile,
    undefined,
    'a browser tab is never "the file open in the editor"',
  );
}

{
  const tests = [
    testViewedFileTabIsTheEditorContext,
    testEmptiedBufferIsSentAsEmptyContent,
    testVirtualAndDirectoryTabsAreNoEditorContext,
    testReportTabReportsItsMarkdownSource,
    testReadOnlyPreviewReportsNoBuffer,
    testBrowserTabIsNoEditorContext,
  ];
  let failures = 0;
  for (const t of tests) {
    try {
      t();
      console.log('PASS', t.name);
    } catch (err) {
      failures += 1;
      console.error('FAIL', t.name);
      console.error(err && err.stack ? err.stack : err);
    }
  }
  for (const win of pages) win.close();
  if (failures > 0) {
    console.error(failures + ' test(s) failed');
    process.exitCode = 1;
  } else {
    console.log(
      'All ' + tests.length + ' editorContextFromContentTab tests passed',
    );
  }
}
