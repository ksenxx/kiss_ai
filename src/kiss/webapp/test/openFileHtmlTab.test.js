// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A clicked .html/.htm file link renders the page in a content TAB (a
// sandboxed iframe) instead of showing its source as text.  The daemon
// answers the page's openFile with a fileContent message
// (web_server._handle_open_file); this test drives the page (jsdom,
// media/main.js) with those replies over the direct transport:
//  1. an .html reply opens a content tab whose iframe renders the page;
//  2. a second reply for the same path reuses that tab and refreshes it;
//  3. after the user closes the tab, the next reply creates a new one;
//  4. .htm renders the same way, and a document without <head> too;
//  5. a Python file still opens as text (the Monaco editor), not a frame;
//  6. an existing path outside the work dir opens like any other;
//  7. a missing path (the daemon replies with an error) opens nothing.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  waitFor,
  htmlFrames,
  activeFrameDoc,
  stubMonaco,
} = require('./simplify2_harness.js');
const {contentTabs} = require('./subagent_tab_helpers');

function pageWithMonaco() {
  let created;
  const page = makePage({
    beforeScripts: win => {
      created = stubMonaco(win);
    },
  });
  return Object.assign(page, {created});
}

function closeActiveContentTab(win) {
  const el = win.document.querySelector(
    '#content-tab-list .chat-tab.active .chat-tab-close, ' +
      '#tab-list .chat-tab.active .chat-tab-close',
  );
  assert.ok(el, 'the active content tab has a close button');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

// The user clicks a link in the chat, so the chat is on screen when
// the daemon's reply lands.
function openFile(win, tabId, path, content, extra) {
  const name = path.split('/').pop();
  win._testApi.switchToTab(tabId);
  send(
    win,
    Object.assign(
      {type: 'fileContent', path, name, tabId, content, version: '1:1'},
      extra || {},
    ),
  );
}

async function main() {
  const page = pageWithMonaco();
  const {win, created} = page;
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'configData', config: {work_dir: '/ws/proj'}, apiKeys: {}});

  // 1. An .html reply opens a rendered content tab.
  openFile(
    win,
    tabId,
    '/ws/proj/site/index.html',
    '<!DOCTYPE html><html><head><title>t</title></head>' +
      '<body><h1>Hello page</h1></body></html>',
  );
  assert.strictEqual(contentTabs(win).length, 1, 'one content tab opened');
  assert.ok(
    contentTabs(win)[0].title.includes('index.html'),
    'the tab is titled after the file',
  );
  assert.ok(activeFrameDoc(win).includes('<h1>Hello page</h1>'));
  assert.strictEqual(created.length, 0, 'no text editor for an html file');
  console.log('  ok - .html opens a rendered content tab');

  // 2. A second reply for the same path reuses the tab, refreshed.
  openFile(
    win,
    tabId,
    '/ws/proj/site/index.html',
    '<html><body><h1>Hello again</h1></body></html>',
  );
  assert.strictEqual(contentTabs(win).length, 1, 'same tab reused');
  assert.ok(activeFrameDoc(win).includes('Hello again'), 'content refreshed');
  console.log('  ok - second open reuses and refreshes the same tab');

  // 3. After the user closes the tab, the next reply creates a new one.
  const firstId = contentTabs(win)[0].id;
  closeActiveContentTab(win);
  assert.strictEqual(contentTabs(win).length, 0, 'tab closed');
  openFile(
    win,
    tabId,
    '/ws/proj/site/index.html',
    '<html><body><h1>Third</h1></body></html>',
  );
  assert.strictEqual(contentTabs(win).length, 1);
  assert.notStrictEqual(contentTabs(win)[0].id, firstId, 'a new tab');
  assert.ok(activeFrameDoc(win).includes('Third'));
  console.log('  ok - closed tab is recreated on next open');

  // 4. .htm renders too, and a document without <head> as well.
  openFile(win, tabId, '/ws/proj/site/old.htm', '<p>no head here</p>');
  assert.strictEqual(contentTabs(win).length, 2);
  assert.ok(activeFrameDoc(win).includes('no head here'));
  console.log('  ok - .htm renders in a tab, with or without <head>');

  // 5. A Python file opens as text in the editor, not in a frame.
  openFile(win, tabId, '/ws/proj/src/app.py', 'print(1)\n');
  assert.strictEqual(contentTabs(win).length, 3);
  assert.strictEqual(htmlFrames(win).length, 0, 'no html frame for a .py file');
  await waitFor(() => created.length === 1, 'the text editor was created');
  assert.strictEqual(created[0].value, 'print(1)\n');
  assert.strictEqual(created[0].language, 'python');
  console.log('  ok - non-html files still open as text');

  // 6. An .html path outside the work dir opens like any other.
  openFile(win, tabId, '/elsewhere/page.html', '<b>outside</b>');
  assert.strictEqual(contentTabs(win).length, 4);
  assert.ok(activeFrameDoc(win).includes('<b>outside</b>'));
  console.log('  ok - outside-workspace html path opens like any other');

  // 7. A missing path opens nothing: the daemon answers with an error,
  // which the page toasts to the user who clicked.
  win._testApi.switchToTab(tabId);
  send(win, {
    type: 'fileContent',
    path: 'no/such/page.html',
    name: 'page.html',
    tabId,
    error: 'File not found: no/such/page.html',
  });
  assert.strictEqual(contentTabs(win).length, 4, 'missing html: no new tab');
  const toast = win.document.querySelector(
    '.kiss-notification[data-notification-id="file-open-error"]',
  );
  assert.ok(toast, 'the error is toasted to the user');
  assert.ok(toast.textContent.includes('File not found: no/such/page.html'));
  console.log('  ok - missing html path opens nothing');
  win.close();
}

main()
  .then(() => console.log('openFileHtmlTab.test.js: all tests passed'))
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
