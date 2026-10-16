// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end test: "+" from a BUSY chat (a task is running, so the chat
// stays open) MOVES the composer draft into the new chat.  The busy
// chat keeps no copy, neither on screen when the user switches back nor
// in the persisted tab state a reload would restore.
//
// Before the fix createNewTab() ran saveCurrentTab() before carrying
// the text, so the draft landed in both chats whenever the chat left
// behind could not be retired.

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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  let state;
  win.acquireTransport = function () {
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, getState: () => state};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickPlus(win) {
  win.document
    .getElementById('new-chat-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function chatIds(win) {
  return Array.from(win._testApi.openTabs(), t => t.id);
}

function composer(win) {
  return win.document.getElementById('task-input');
}

function testPlusFromBusyChatMovesTheDraft() {
  const {win, getState} = makePage();
  const boot = win._testApi.getActiveTabId();
  send(win, {type: 'setTaskText', text: 'long task'});
  send(win, {type: 'status', running: true, tabId: boot});
  composer(win).value = 'carry me';

  clickPlus(win);

  const fresh = win._testApi.getActiveTabId();
  assert.notStrictEqual(fresh, boot, 'a new chat opens');
  assert.strictEqual(composer(win).value, 'carry me', 'the draft moved');
  assert.ok(chatIds(win).includes(boot), 'the busy chat stays open');

  win._testApi.switchToTab(boot);
  assert.strictEqual(
    composer(win).value,
    '',
    'the busy chat keeps no copy of the draft',
  );
  const drafts = (getState() || {}).inputDrafts || {};
  assert.ok(!drafts[boot], 'no copy is persisted for the busy chat');

  win._testApi.switchToTab(fresh);
  assert.strictEqual(composer(win).value, 'carry me', 'the new chat has it');
  win.close();
  console.log('  ok - + from a busy chat moves the draft, leaving no copy');
}

function main() {
  testPlusFromBusyChatMovesTheDraft();
  console.log('\n1 passed, 0 failed');
}

main();
