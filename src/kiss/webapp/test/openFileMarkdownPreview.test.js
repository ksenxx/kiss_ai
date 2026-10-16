// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A clicked .md/.markdown file link opens the file converted to HTML
// (marked) and rendered in a content TAB -- a markdown preview --
// instead of showing the raw markdown source as text.  The daemon
// answers the page's openFile with a fileContent message
// (web_server._handle_open_file); this test drives the page (jsdom,
// media/main.js) with those replies over the direct transport:
//  1. a .md reply opens a content tab whose iframe renders the HTML
//     conversion, with an "Edit source" toggle since the file can be
//     saved back;
//  2. .markdown works the same way;
//  3. a path-only prompt is sent as a submit (the daemon decides it is a
//     file and answers promptOpened + fileContent), which renders the
//     same preview;
//  4. a Python file still opens as text (the Monaco editor);
//  5. .html still renders as HTML, not through the markdown converter;
//  6. an existing .md path outside the work dir previews too;
//  7. a missing .md path (the daemon replies with an error) opens
//     nothing.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {makePage, send, waitFor} = require('./simplify2_harness.js');

const MEDIA = path.join(__dirname, '..', 'media');

function pageWithMarkedAndMonaco() {
  const created = [];
  // The page's transport, so a test can read the state it persists
  // (the unsent drafts a reload hands back).
  let transport = null;
  const page = makePage({
    beforeScripts: win => {
      const acquire = win.acquireTransport;
      win.acquireTransport = () => {
        transport = acquire();
        return transport;
      };
      win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
      win.monaco = {
        editor: {
          defineTheme: () => {},
          setTheme: () => {},
          create: (holder, opts) => {
            const value = String(opts.value === undefined ? '' : opts.value);
            const model = {
              getAlternativeVersionId: () => 1,
              getValue: () => value,
              getLineCount: () => value.split('\n').length,
            };
            created.push({holder, value: opts.value, language: opts.language});
            return {
              getModel: () => model,
              onDidChangeModelContent: () => {},
              setPosition: () => {},
              revealLineInCenter: () => {},
              getDomNode: () => holder,
              dispose: () => {},
              layout: () => {},
              focus: () => {},
            };
          },
        },
      };
    },
  });
  return Object.assign(page, {created, persistedState: () => transport.getState()});
}

function contentTabs(win) {
  return win._testApi.openTabs().filter(t => t.isContentTab);
}

// The html frames of content views on screen (a background content
// tab keeps its view in the DOM, hidden).
function frames(win) {
  return Array.from(
    win.document.querySelectorAll('.content-html-frame'),
  ).filter(f => f.closest('.content-tab-view').style.display !== 'none');
}

function activeFrameDoc(win) {
  const fr = frames(win);
  assert.strictEqual(fr.length, 1, 'exactly one html frame on screen');
  return fr[0].srcdoc;
}

// The user clicks a link in the chat, so the chat is on screen when
// the daemon's reply lands.
function openFile(win, tabId, filePath, content) {
  win._testApi.switchToTab(tabId);
  send(win, {
    type: 'fileContent',
    path: filePath,
    name: filePath.split('/').pop(),
    tabId,
    content,
    version: '1:1',
  });
}

function assertMarkdownPreview(win, heading, label) {
  const doc = activeFrameDoc(win);
  assert.ok(
    new RegExp('<h1[^>]*>' + heading + '</h1>').test(doc),
    label + ': the markdown heading is converted to <h1>, got ' +
      doc.slice(-200),
  );
  assert.ok(
    !doc.includes('# ' + heading),
    label + ': the raw markdown source is not shown',
  );
}

async function main() {
  const page = pageWithMarkedAndMonaco();
  const {win, created, posted} = page;
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'configData', config: {work_dir: '/ws/proj'}, apiKeys: {}});

  // 1. A .md reply opens the rendered markdown preview tab.
  openFile(win, tabId, '/ws/proj/docs/notes.md', '# Notes\n\nsome *text*\n');
  assert.strictEqual(contentTabs(win).length, 1, 'one content tab opened');
  assert.ok(contentTabs(win)[0].title.includes('notes.md'));
  assertMarkdownPreview(win, 'Notes', '.md');
  assert.ok(activeFrameDoc(win).includes('<em>text</em>'));
  assert.strictEqual(created.length, 0, 'no text editor for the preview');
  const toggle = Array.from(
    win.document.querySelectorAll('button'),
  ).find(
    b =>
      /edit source/i.test(b.textContent) &&
      b.closest('.content-tab-view') &&
      b.closest('.content-tab-view').style.display !== 'none',
  );
  assert.ok(toggle, 'a saveable .md preview offers an "Edit source" toggle');
  console.log('  ok - .md opens the rendered markdown preview tab');

  // 2. .markdown works the same way.
  openFile(win, tabId, '/ws/proj/docs/guide.markdown', '# Guide\n');
  assert.strictEqual(contentTabs(win).length, 2);
  assertMarkdownPreview(win, 'Guide', '.markdown');
  console.log('  ok - .markdown opens the rendered preview tab too');

  // 3. A path-only prompt goes out as a submit; the daemon recognises
  // the file, answers promptOpened and serves it like a click.
  win._testApi.switchToTab(tabId);
  const inp = win.document.getElementById('task-input');
  inp.value = 'docs/readme.md';
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
  const submit = await waitFor(
    () => posted.find(m => m.type === 'submit' && m.prompt === 'docs/readme.md'),
    'the path-only prompt is submitted for the daemon to resolve',
  );
  assert.strictEqual(submit.tabId, tabId);
  // Until the daemon answers, the submitted path is a prompt it has not
  // acknowledged: a reload would hand it back as an unsent draft.
  win.dispatchEvent(new win.Event('pagehide'));
  // (Object.entries: the page's objects come from jsdom's realm, whose
  // Object.prototype deepStrictEqual would not match.)
  assert.deepStrictEqual(
    Object.entries(page.persistedState().inputDrafts),
    [[tabId, 'docs/readme.md']],
    'an unacknowledged prompt is kept as the tab\'s draft',
  );
  send(win, {type: 'promptOpened', tabId, taskId: submit.taskId});
  openFile(win, tabId, '/ws/proj/docs/readme.md', '# Readme\n');
  assert.strictEqual(contentTabs(win).length, 3);
  assertMarkdownPreview(win, 'Readme', 'path-only submit');
  // promptOpened acknowledged the prompt without a task: the path must
  // not come back into the composer after a reload.
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(
    Object.entries(page.persistedState().inputDrafts),
    [],
    'the opened path is no longer an unsent draft',
  );
  console.log('  ok - path-only submit renders markdown like a click');

  // 4. A Python file opens as text in the editor.
  openFile(win, tabId, '/ws/proj/src/app.py', 'print(1)\n');
  assert.strictEqual(frames(win).length, 0, 'no html frame for a .py file');
  await waitFor(() => created.length === 1, 'the text editor was created');
  assert.strictEqual(created[0].language, 'python');
  console.log('  ok - non-markdown files still open as text');

  // 5. .html renders as HTML, untouched by the markdown converter.
  openFile(win, tabId, '/ws/proj/site/page.html', '<p># not a heading</p>');
  const htmlDoc = activeFrameDoc(win);
  assert.ok(htmlDoc.includes('<p># not a heading</p>'));
  assert.ok(!/<h1/.test(htmlDoc), 'html is not run through marked');
  console.log('  ok - .html still renders as HTML in its own tab');

  // 6. An .md path outside the work dir previews too.
  openFile(win, tabId, '/elsewhere/other.md', '# Outside\n');
  assertMarkdownPreview(win, 'Outside', 'outside work dir');
  console.log('  ok - outside-workspace md path previews like any other');

  // 7. A missing .md path opens nothing: the daemon answers an error.
  const count = contentTabs(win).length;
  win._testApi.switchToTab(tabId);
  send(win, {
    type: 'fileContent',
    path: 'no/such/notes.md',
    name: 'notes.md',
    tabId,
    error: 'File not found: no/such/notes.md',
  });
  assert.strictEqual(contentTabs(win).length, count, 'missing md: no tab');
  const toast = win.document.querySelector(
    '.kiss-notification[data-notification-id="file-open-error"]',
  );
  assert.ok(toast && toast.textContent.includes('File not found'));
  console.log('  ok - missing md path opens nothing');
  win.close();
}

main()
  .then(() => console.log('openFileMarkdownPreview.test.js: all tests passed'))
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
