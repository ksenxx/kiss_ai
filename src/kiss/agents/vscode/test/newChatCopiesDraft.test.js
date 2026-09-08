// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests: opening a NEW CHAT carries the composer
// draft (the text sitting in the task textarea) into the new chat's
// textarea on every surface (media/main.js):
//  - editor-tabs mode: the + button / Cmd+T (host `clearChat`) post
//    `openChatPanel` WITH `pendingText`, and a panel booted with the
//    host-stamped `data-kiss-pending-text` seeds its textarea from it
//    (multi-line drafts included — attribute newlines survive);
//  - a panel booted WITHOUT the attribute starts with an empty
//    textarea, and an empty draft still posts `pendingText: ''`;
//  - sidebar / remote mode (no editor-tab-mode class): createNewTab
//    copies the draft into the new internal tab's textarea, and the
//    original tab keeps its own copy.
//
// Unreachable-branch note: init()'s `!inp.value` guard cannot be false
// at boot (chat.html ships an empty textarea and nothing runs before
// init), so that branch is not exercised here.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs);
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=new-chat-draft-main.js',
  );

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byType(posted, type) {
  return posted.filter(m => m.type === type);
}

const ROOT = 'root-tab-0001';

function editorAttrs(extra) {
  return (
    ' class="editor-tab-mode"' +
    ` data-kiss-tab-id="${ROOT}"` +
    ' data-kiss-tab-title="My chat"' +
    (extra || '')
  );
}

// The + button in editor-tabs mode posts openChatPanel carrying the
// composer draft; an untouched composer still sends pendingText ''.
function testPlusButtonCarriesDraft() {
  const {win, posted} = makeWebview(editorAttrs());
  const inp = win.document.getElementById('task-input');
  const addBtn = win.document.getElementById('new-chat-btn');

  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  let opens = byType(posted, 'openChatPanel');
  assert.strictEqual(opens.length, 1, 'the + posts openChatPanel');
  assert.strictEqual(
    opens[0].pendingText,
    '',
    'an empty composer travels as an empty draft',
  );

  inp.value = 'refactor the parser';
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  opens = byType(posted, 'openChatPanel');
  assert.strictEqual(opens.length, 2);
  assert.strictEqual(
    opens[1].pendingText,
    'refactor the parser',
    'the + must carry the composer draft to the host',
  );
  assert.strictEqual(
    inp.value,
    'refactor the parser',
    'the opening panel keeps its own draft',
  );
}

// Cmd+T reaches an editor-tab panel as the host's `clearChat`; a used
// tab takes the createNewTab path, which must carry the draft too.
function testClearChatCarriesDraft() {
  const {win, posted} = makeWebview(editorAttrs());
  // A used tab (welcome hidden) so clearChat takes the createNewTab path.
  win._testApi.hideWelcome();
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft typed before Cmd+T';

  send(win, {type: 'clearChat'});
  const opens = byType(posted, 'openChatPanel');
  assert.strictEqual(opens.length, 1, 'clearChat asks the host for a panel');
  assert.strictEqual(
    opens[0].pendingText,
    'draft typed before Cmd+T',
    'Cmd+T must carry the composer draft to the host',
  );
}

// A panel booted with data-kiss-pending-text (the host stamped the
// opener's draft) seeds its textarea from it — multi-line included.
function testBootSeedsTextareaFromPendingText() {
  const draft = 'line one\nline two';
  const {win, posted} = makeWebview(
    editorAttrs(' data-kiss-pending-text="line one\nline two"'),
  );
  const inp = win.document.getElementById('task-input');
  assert.strictEqual(
    inp.value,
    draft,
    'boot must seed the textarea from data-kiss-pending-text',
  );
  const readies = byType(posted, 'ready');
  assert.strictEqual(readies.length, 1, 'seeding must not break ready');
  assert.strictEqual(readies[0].tabId, ROOT);

  // The draft survives a hop to another tab and back (it was adopted
  // into the root tab's inputValue, not only painted into the DOM).
  send(win, {
    type: 'output',
    tabId: ROOT,
    data: {type: 'subagent_tab', tab_id: 'sub-1', parent_tab_id: ROOT},
  });
  win._testApi.endLaunch();
  const subStrip = win.document.querySelector(
    '#tab-list .chat-tab[data-tab-id="sub-1"]',
  );
  if (subStrip) {
    subStrip.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
    assert.notStrictEqual(win._testApi.getActiveTabId(), ROOT);
    const rootStrip = win.document.querySelector(
      `#tab-list .chat-tab[data-tab-id="${ROOT}"]`,
    );
    rootStrip.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
    assert.strictEqual(
      inp.value,
      draft,
      'the seeded draft must round-trip through a tab switch',
    );
  }
}

// Without the attribute the textarea starts empty (the pre-change
// behavior is preserved for plain opens).
function testBootWithoutPendingTextStaysEmpty() {
  const {win} = makeWebview(editorAttrs());
  assert.strictEqual(
    win.document.getElementById('task-input').value,
    '',
    'no data-kiss-pending-text: the textarea starts empty',
  );
}

// Sidebar / remote surface (no editor-tab-mode class): the in-webview
// createNewTab already copies the draft into the new internal tab.
// Guard that parity so "all surfaces" stays true.
function testSidebarCreateNewTabCopiesDraft() {
  const {win, posted} = makeWebview(' class=""');
  const inp = win.document.getElementById('task-input');
  const firstTab = win._testApi.getActiveTabId();
  inp.value = 'sidebar draft';

  win._testApi.createNewTab();
  const newTab = win._testApi.getActiveTabId();
  assert.notStrictEqual(newTab, firstTab, 'a new internal tab is active');
  assert.strictEqual(
    inp.value,
    'sidebar draft',
    'the new internal tab starts with the copied draft',
  );
  assert.strictEqual(
    byType(posted, 'openChatPanel').length,
    0,
    'no editor tab is requested outside editor-tabs mode',
  );

  // The originating tab kept its own copy.
  win._testApi.endLaunch();
  const firstStrip = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${firstTab}"]`,
  );
  assert.ok(firstStrip, 'the first tab is still in the tab bar');
  firstStrip.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    inp.value,
    'sidebar draft',
    'the originating tab keeps its draft',
  );
}

function main() {
  const tests = [
    testPlusButtonCarriesDraft,
    testClearChatCarriesDraft,
    testBootSeedsTextareaFromPendingText,
    testBootWithoutPendingTextStaysEmpty,
    testSidebarCreateNewTabCopiesDraft,
  ];
  for (const t of tests) {
    t();
    console.log(`  ok - ${t.name}`);
  }
  console.log('newChatCopiesDraft.test.js: all tests passed');
}

main();
