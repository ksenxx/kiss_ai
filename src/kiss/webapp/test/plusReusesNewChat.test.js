// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the "+" button reusing a chat that is still new.
//
// A new chat is a root chat tab no task has touched: it shows the
// welcome, has no daemon chat id, nothing running, pending or unacked.
// "+" (openNewChat) keeps the chat on screen when it is still new, brings
// an existing new chat forward (one opened earlier here, or in another
// browser and mirrored by the registry) when the chat on screen has a
// conversation, and only opens a tab when no new chat exists.  The chat
// left behind is retired when idle, as before; its composer draft moves
// to the new chat unless that chat has a draft of its own.

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

  const posted = [];
  win.acquireTransport = function () {
    let state;
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
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

function activeId(win) {
  return win._testApi.getActiveTabId();
}

function composer(win) {
  return win.document.getElementById('task-input');
}

/** A daemon echo that gives the tab on screen a task. */
function useActiveChat(win, text) {
  send(win, {type: 'setTaskText', text: text || 'first task'});
}

// The registry snapshot every browser mirrors: the tabs here plus
// *extra* entries (a new chat opened in another browser has no chat id).
function snapshot(win, extra) {
  const tabs = win._testApi
    .openTabs()
    .filter(t => !t.isContentTab && !t.isSubagentTab)
    .map(t => ({tabId: t.id, title: t.title}));
  send(win, {type: 'tabs_state', tabs: tabs.concat(extra || [])});
}

function testPlusOnNewChatKeepsIt() {
  const {win, posted} = makePage();
  const boot = activeId(win);
  posted.length = 0;

  clickPlus(win);

  assert.strictEqual(activeId(win), boot, 'the new chat on screen stays');
  assert.deepStrictEqual(chatIds(win), [boot], 'no second tab opens');
  assert.ok(
    !posted.some(m => m.type === 'newChat'),
    'nothing is registered with the daemon',
  );
  assert.strictEqual(
    win.document.getElementById('welcome').style.display,
    '',
    'the welcome stays up',
  );
  win.close();
  console.log('  ok - + on a chat that is still new keeps it');
}

function testPlusOnNewChatKeepsItsDraft() {
  const {win} = makePage();
  const boot = activeId(win);
  composer(win).value = 'half-typed prompt';

  clickPlus(win);

  assert.strictEqual(activeId(win), boot);
  assert.strictEqual(composer(win).value, 'half-typed prompt');
  win.close();
  console.log('  ok - + on a new chat keeps the draft in place');
}

function testPlusOnUsedChatOpensATabWhenNoneIsNew() {
  const {win, posted} = makePage();
  const boot = activeId(win);
  useActiveChat(win);
  posted.length = 0;

  clickPlus(win);

  const fresh = activeId(win);
  assert.notStrictEqual(fresh, boot, 'a tab opens');
  assert.ok(
    posted.some(m => m.type === 'newChat' && m.tabId === fresh),
    'the tab is registered with the daemon',
  );
  assert.deepStrictEqual(
    chatIds(win),
    [fresh],
    'the idle chat left behind is retired',
  );
  win.close();
  console.log('  ok - + on a used chat opens a tab when no chat is new');
}

function testPlusOnUsedChatBringsAnExistingNewChatForward() {
  const {win, posted} = makePage();
  const boot = activeId(win);
  useActiveChat(win, 'boot task');
  // Register the boot tab so a later snapshot can retire it, and
  // mirror a new chat opened in another browser.
  snapshot(win, [{tabId: 'other-browser-new', title: 'new chat'}]);
  assert.deepStrictEqual(chatIds(win), [boot, 'other-browser-new']);
  assert.strictEqual(activeId(win), boot, 'the mirrored tab stays behind');
  posted.length = 0;

  clickPlus(win);

  assert.strictEqual(
    activeId(win),
    'other-browser-new',
    'the existing new chat comes forward',
  );
  assert.ok(
    !posted.some(m => m.type === 'newChat'),
    'no second new chat is registered',
  );
  assert.deepStrictEqual(
    chatIds(win),
    ['other-browser-new'],
    'the idle chat left behind is retired',
  );
  assert.strictEqual(
    win.document.getElementById('welcome').style.display,
    '',
    'the new chat shows the welcome',
  );
  win.close();
  console.log('  ok - + on a used chat brings an existing new chat forward');
}

function testPlusMovesTheDraftToTheNewChatBroughtForward() {
  const {win} = makePage();
  const boot = activeId(win);
  useActiveChat(win);
  snapshot(win, [{tabId: 'spare', title: 'new chat'}]);
  composer(win).value = 'carry me';

  clickPlus(win);

  assert.strictEqual(activeId(win), 'spare');
  assert.strictEqual(composer(win).value, 'carry me', 'the draft moved');
  assert.deepStrictEqual(
    chatIds(win),
    ['spare'],
    'the draft moved, so the chat left behind had nothing to keep',
  );
  assert.ok(!chatIds(win).includes(boot));
  win.close();
  console.log('  ok - + moves the draft into the new chat it brings forward');
}

function testPlusKeepsTheNewChatsOwnDraftAndTheChatLeftBehind() {
  const {win} = makePage();
  const boot = activeId(win);
  // Draft something in the boot chat, open a new chat with "+" (the
  // draft moves along), then come back and give the boot chat a task
  // and a second draft.
  composer(win).value = 'spare draft';
  useActiveChat(win);
  clickPlus(win);
  const spare = activeId(win);
  assert.notStrictEqual(spare, boot);
  assert.strictEqual(composer(win).value, 'spare draft');
  // The boot chat was idle and its draft moved, so it was retired;
  // open it again from the Chats panel path with a task of its own.
  send(win, {type: 'tabs_state', tabs: [{tabId: spare, title: 'new chat'}]});
  win._testApi.createNewTab();
  const second = activeId(win);
  useActiveChat(win, 'second task');
  composer(win).value = 'second draft';

  clickPlus(win);

  assert.strictEqual(activeId(win), spare, 'the new chat comes forward');
  assert.strictEqual(
    composer(win).value,
    'spare draft',
    'the new chat keeps its own draft',
  );
  assert.deepStrictEqual(
    chatIds(win).sort(),
    [second, spare].sort(),
    'the chat left behind keeps its draft, so it stays open',
  );
  win.close();
  console.log(
    '  ok - + leaves a drafted chat open when the new chat has a draft',
  );
}

function testPlusSkipsAChatWhoseStatusIsUnknown() {
  const {win} = makePage();
  const boot = activeId(win);
  useActiveChat(win);
  // A chat mirrored WITH a chat id ran a task somewhere: never a new chat.
  snapshot(win, [{tabId: 'ran-elsewhere', title: 'deploy', chatId: 'c-9'}]);

  clickPlus(win);

  const fresh = activeId(win);
  assert.notStrictEqual(fresh, 'ran-elsewhere');
  assert.notStrictEqual(fresh, boot);
  assert.ok(chatIds(win).includes('ran-elsewhere'), 'the used chat stays');
  win.close();
  console.log('  ok - + never reuses a chat that ran a task elsewhere');
}

function testPlusSkipsAChatWithAPendingPrompt() {
  const {win} = makePage();
  const boot = activeId(win);
  // Submit from the boot chat: the prompt is unacked until the daemon
  // echoes it, so the chat is busy and not new.
  composer(win).value = 'run this';
  win.document
    .getElementById('send-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  clickPlus(win);

  const fresh = activeId(win);
  assert.notStrictEqual(fresh, boot, 'a submitted chat is not reused');
  assert.ok(chatIds(win).includes(boot), 'a submitted chat is not retired');
  win.close();
  console.log('  ok - + never reuses a chat whose prompt is in flight');
}

function main() {
  testPlusOnNewChatKeepsIt();
  testPlusOnNewChatKeepsItsDraft();
  testPlusOnUsedChatOpensATabWhenNoneIsNew();
  testPlusOnUsedChatBringsAnExistingNewChatForward();
  testPlusMovesTheDraftToTheNewChatBroughtForward();
  testPlusKeepsTheNewChatsOwnDraftAndTheChatLeftBehind();
  testPlusSkipsAChatWhoseStatusIsUnknown();
  testPlusSkipsAChatWithAPendingPrompt();
  console.log('\n8 passed, 0 failed');
}

main();
