// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests: opening a NEW CHAT carries the composer
// draft (the text sitting in the task textarea) into the new chat's
// textarea (media/main.js): the + button copies the draft into the new
// internal tab's textarea.
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

function makePage() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
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
  win.acquireTransport = function () {
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

// + on a chat that is still new keeps it; on a used chat it opens a
// new internal tab that starts with the copied draft.
function testPlusCopiesDraft() {
  const {win} = makePage();
  const inp = win.document.getElementById('task-input');
  const addBtn = win.document.getElementById('new-chat-btn');
  const firstTab = win._testApi.getActiveTabId();

  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    firstTab,
    'the + on a new chat reuses it',
  );

  // The daemon echo makes the chat used; now + opens a tab.
  send(win, {type: 'setTaskText', text: 'first task'});
  inp.value = 'refactor the parser';
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const newTab = win._testApi.getActiveTabId();
  assert.notStrictEqual(newTab, firstTab, 'a new internal tab is active');
  assert.strictEqual(
    inp.value,
    'refactor the parser',
    'the new internal tab starts with the copied draft',
  );
  // The idle chat left behind is retired (+ moves on from it; the
  // Chats panel keeps it one click away).
  win._testApi.endLaunch();
  assert.ok(
    !win._testApi.openTabs().some(t => t.id === firstTab),
    'the idle chat left behind is retired',
  );
  win.close();
}

// The page starts with an empty textarea.
function testBootStartsEmpty() {
  const {win} = makePage();
  assert.strictEqual(
    win.document.getElementById('task-input').value,
    '',
    'the textarea starts empty',
  );
  win.close();
}

function main() {
  const tests = [testPlusCopiesDraft, testBootStartsEmpty];
  for (const t of tests) {
    t();
    console.log(`  ok - ${t.name}`);
  }
  console.log('newChatCopiesDraft.test.js: all tests passed');
}

main();
