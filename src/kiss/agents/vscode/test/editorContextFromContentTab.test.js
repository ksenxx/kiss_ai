// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the in-page editor context: the file tab
// the user viewed last is "the file open in the editor" for a prompt
// (`submit.activeFile`) and for ghost-text completion
// (`complete.activeFile` + `activeFileContent`, the tab's buffer). This
// is what gives the remote webapp the context the VS Code host takes
// from its native editors; the host's own visible editor still wins
// there (see SorcarSidebarView `submit` / `complete`).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// A tab on the group strip (#tab-list) is clicked there; a chat that is
// not on it is picked the way the Chats panel does it.
function clickTab(win, tabId) {
  const el = win.document.querySelector(
    '#tab-list ' + `.chat-tab[data-tab-id="${tabId}"]`,
  );
  if (el) el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  else win._testApi.switchToTab(tabId);
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

function requestGhost(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.selectionStart = text.length;
  inp.selectionEnd = text.length;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  return new Promise(r => setTimeout(r, 400));
}

function openChat(win) {
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws/a'}],
  });
}

async function testViewedFileTabIsTheEditorContext() {
  const {win, posted} = makeWebview();
  openChat(win);
  // No file tab yet: neither message names a file.
  assert.strictEqual(submitPrompt(win, posted, 'hello').activeFile, undefined);
  send(win, {type: 'status', running: false, tabId: 'a1'});
  await requestGhost(win, 'hel');
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
  await requestGhost(win, 'summ');
  completes = posted.filter(m => m.type === 'complete');
  const last = completes[completes.length - 1];
  assert.strictEqual(last.activeFile, '/ws/a/notes.md');
  assert.strictEqual(
    last.activeFileContent,
    '# Notes\n\nunsaved buffer text',
    'completion sees the tab buffer, like the unsaved VS Code document',
  );

  // Closing the file tab ends the context.  The mobile chat screen lists
  // no content tabs, so the user goes to the file first and closes it
  // from its own strip entry; the chat is back on screen afterwards.
  send(win, {type: 'tabs_state', tabs: [{tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws/a'}]});
  clickTab(win, fileTab);
  win.document
    .querySelector(`.chat-tab[data-tab-id="${fileTab}"] .chat-tab-close`)
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(win._testApi.getActiveTabId(), 'a1', 'the chat is on screen again');
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
  await requestGhost(win, 'and n');
  completes = posted.filter(m => m.type === 'complete');
  assert.strictEqual(completes[completes.length - 1].activeFile, '');
  assert.strictEqual(completes[completes.length - 1].activeFileContent, undefined);
}

async function testEmptiedBufferIsSentAsEmptyContent() {
  // A file whose buffer is empty is "open with an empty buffer": the
  // completion carries activeFileContent "" (not nothing, which would
  // leave the daemon's previous snapshot of the file in place).
  const {win, posted} = makeWebview();
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
  await requestGhost(win, 'emp');
  const completes = posted.filter(m => m.type === 'complete');
  const last = completes[completes.length - 1];
  assert.strictEqual(last.activeFile, '/ws/a/empty.md');
  assert.strictEqual(last.activeFileContent, '', 'an empty buffer is sent as ""');
}

async function testVirtualAndDirectoryTabsAreNoEditorContext() {
  // A commit's patch (git-show://...) and a folder listing are content
  // tabs but not files on disk: never the active file.
  const {win, posted} = makeWebview();
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
  await requestGhost(win, 'expl');
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
  assert.strictEqual(submitPrompt(win, posted, 'and a.py?').activeFile, '/ws/a/src/a.py');
  send(win, {type: 'status', running: false, tabId: 'a1'});
  await requestGhost(win, 'and a');
  completes = posted.filter(m => m.type === 'complete');
  assert.strictEqual(completes[completes.length - 1].activeFile, '/ws/a/src/a.py');
  assert.strictEqual(completes[completes.length - 1].activeFileContent, 'x = 1\n');
}

function writeReport(win, filePath, content) {
  const extra = {tabId: 'a1'};
  send(win, Object.assign({type: 'tool_call', name: 'Write', path: filePath, content}, extra));
  send(
    win,
    Object.assign(
      {
        type: 'tool_result',
        content: 'Successfully wrote ' + content.length + ' characters to ' + filePath,
        is_error: false,
        tool_name: 'Write',
        path: filePath,
      },
      extra,
    ),
  );
}

async function testReportTabReportsItsMarkdownSource() {
  // A finished task's report opens as converted HTML; completion must
  // see the report file's own text (never the HTML), and a regenerated
  // report must replace the earlier text rather than leaving it out
  // (an omitted buffer keeps the daemon's previous snapshot).
  const {win, posted} = makeWebview();
  openChat(win);
  const reportPath = '/ws/a/reports/summary.md';
  writeReport(win, reportPath, '# Report\n\noldReportIdentifier\n');
  send(win, {type: 'task_done', success: true, tabId: 'a1'});
  contentTabId(win);
  clickTab(win, 'a1');
  send(win, {type: 'status', running: false, tabId: 'a1'});
  await requestGhost(win, 'oldRe');
  let completes = posted.filter(m => m.type === 'complete');
  let last = completes[completes.length - 1];
  assert.strictEqual(last.activeFile, reportPath);
  assert.strictEqual(last.activeFileContent, '# Report\n\noldReportIdentifier\n');

  send(win, {type: 'status', running: true, tabId: 'a1'});
  writeReport(win, reportPath, '# Report\n\nnewReportIdentifier\n');
  send(win, {type: 'task_done', success: true, tabId: 'a1'});
  clickTab(win, 'a1');
  send(win, {type: 'status', running: false, tabId: 'a1'});
  await requestGhost(win, 'newRe');
  completes = posted.filter(m => m.type === 'complete');
  last = completes[completes.length - 1];
  assert.strictEqual(last.activeFile, reportPath);
  assert.strictEqual(
    last.activeFileContent,
    '# Report\n\nnewReportIdentifier\n',
    'the regenerated report replaces the earlier text',
  );
}

async function testBrowserTabIsNoEditorContext() {
  // A streamed browser tab is a content tab with a URL, not a file.
  const {win, posted} = makeWebview();
  openChat(win);
  // The streamed-browser view is a separate script (browserTab.js);
  // a minimal stand-in with the members main.js touches suffices.
  const noop = function () {};
  win.BrowserTabView = {
    create: function () {
      return {
        el: win.document.createElement('div'),
        setBadge: noop,
        setVisible: noop,
        state: noop,
        error: noop,
        frame: noop,
        resubscribe: noop,
        dispose: noop,
      };
    },
  };
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

(async () => {
  const tests = [
    testViewedFileTabIsTheEditorContext,
    testEmptiedBufferIsSentAsEmptyContent,
    testVirtualAndDirectoryTabsAreNoEditorContext,
    testReportTabReportsItsMarkdownSource,
    testBrowserTabIsNoEditorContext,
  ];
  let failures = 0;
  for (const t of tests) {
    try {
      await t();
      console.log('PASS', t.name);
    } catch (err) {
      failures += 1;
      console.error('FAIL', t.name);
      console.error(err && err.stack ? err.stack : err);
    }
  }
  if (failures > 0) {
    console.error(failures + ' test(s) failed');
    process.exit(1);
  }
  console.log('All ' + tests.length + ' editorContextFromContentTab tests passed');
  process.exit(0);
})();
