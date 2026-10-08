// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// An ask_user_question blocks the agent until the user answers, so on
// every surface a NEW question
//  * switches the client to the asking tab, and
//  * in VS Code tells the host (`revealForQuestion`) so the sidebar view
//    or editor panel comes forward too.
// The Question panel in the transcript is the whole notice: no toast is
// raised in the webview. A replayed copy of a question already pending
// here changes nothing, and the remote webapp never posts the host-only
// message to the daemon.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  const remote = !!(opts && opts.remote);
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) html = html.replace('<body', '<body class="remote-chat"');

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
  win.matchMedia = function (query) {
    return {
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };

  const posted = [];
  let state = opts && opts.initialState;
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

  return {win, posted, getState: () => state};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// The tab's entry on the group strip (#tab-list): rendered for the chat
// on screen and its group only, so a background chat has none.
function tabElement(win, tabId) {
  return win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
}

// The record of an open tab, or null once it is closed.
function tabRecord(win, tabId) {
  return win._testApi.openTabs().find(t => t.id === tabId) || null;
}

// A tab on the strip is clicked there; a background chat is picked the
// way the Chats panel does it.
function clickTab(win, tabId) {
  assert.ok(tabRecord(win, tabId), `tab ${tabId} must exist`);
  const el = tabElement(win, tabId);
  if (el) el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  else win._testApi.switchToTab(tabId);
}

// The composer is in answer mode while the tab on screen has a question.
function answering(win) {
  return win.document.body.classList.contains('ask-answering');
}

// Whether the tab is flagged as waiting for an answer (the strip's "?"
// mark when the tab is rendered there).
function askPending(win, tabId) {
  const rec = tabRecord(win, tabId);
  assert.ok(rec, `tab ${tabId} must exist`);
  return rec.askPending;
}

// Every toast on screen: a question must never raise one.
function toasts(win) {
  return win.document.querySelectorAll('.kiss-notification');
}

function assertNoToast(win, where) {
  assert.strictEqual(toasts(win).length, 0, 'no toast ' + where);
}

// Messages come from the JSDOM realm (a different Object prototype), so
// they are copied before deepStrictEqual.
function hostMessages(posted, type) {
  return JSON.parse(JSON.stringify(posted.filter(m => m.type === type)));
}

function testNewQuestionSwitchesTabWithoutToast() {
  const {win, posted} = makeWebview();
  const api = win._testApi;
  const questionTab = api.getActiveTabId();
  api.createNewTab();
  const otherTab = api.getActiveTabId();
  assert.notStrictEqual(otherTab, questionTab);
  // The user has been working on the other tab since their last submit.
  win.document
    .getElementById('task-input')
    .dispatchEvent(new win.KeyboardEvent('keydown', {key: 'a', bubbles: true}));

  send(win, {
    type: 'askUser',
    question: 'Please provide the deployment token.',
    tabId: questionTab,
  });

  assert.strictEqual(
    api.getActiveTabId(),
    questionTab,
    'a new question switches to its tab, user activity notwithstanding',
  );
  assert.ok(answering(win), 'the composer answers the question at once');
  assert.ok(askPending(win, questionTab));
  assertNoToast(win, 'when the question arrives');
  assert.deepStrictEqual(hostMessages(posted, 'revealForQuestion'), [
    {type: 'revealForQuestion'},
  ]);
  assert.ok(
    !posted.some(m => m.type === 'askWaiting' || m.type === 'askWaitingDone'),
    'the retired waiting-notice messages are never posted',
  );

  // The user goes back to their own tab: the waiting tab stays flagged
  // and nothing else nags them.
  clickTab(win, otherTab);
  assert.strictEqual(api.getActiveTabId(), otherTab);
  assert.ok(!answering(win));
  assert.ok(askPending(win, questionTab));
  assertNoToast(win, 'after leaving the asking tab');

  // A replay of the same pending question (another client reloaded)
  // neither switches tabs again nor re-notifies the host.
  send(win, {
    type: 'askUser',
    question: 'Please provide the deployment token.',
    tabId: questionTab,
  });
  assert.strictEqual(api.getActiveTabId(), otherTab);
  assert.strictEqual(hostMessages(posted, 'revealForQuestion').length, 1);
  assertNoToast(win, 'on a replay of the pending question');

  // Answering retires the question.
  clickTab(win, questionTab);
  assert.ok(answering(win));
  win.document.getElementById('task-input').value = 'tok_live_123';
  win.document.getElementById('send-btn').click();
  assert.ok(
    posted.some(
      m =>
        m.type === 'userAnswer' &&
        m.tabId === questionTab &&
        m.answer === 'tok_live_123',
    ),
  );
  assert.ok(!answering(win));
  assert.ok(!askPending(win, questionTab), 'the flag goes with the answer');
  assertNoToast(win, 'after answering');

  win.close();
  console.log('  ok - a new question switches tabs and raises no toast');
}

function testTaskEndAndAskUserDoneRaiseNoToast() {
  for (const retire of [
    {type: 'askUserDone'},
    {type: 'task_done', success: true},
    {type: 'task_error', error: 'boom'},
  ]) {
    const {win, posted} = makeWebview();
    const api = win._testApi;
    const tab = api.getActiveTabId();
    send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
    assert.ok(answering(win), retire.type + ': answer mode entered');
    assertNoToast(win, retire.type + ': while the question is pending');

    send(win, Object.assign({tabId: tab}, retire));
    assert.ok(!answering(win), retire.type + ' retires the question');
    assertNoToast(win, retire.type + ': after the question is retired');
    assert.ok(
      !posted.some(m => m.type === 'askWaitingDone'),
      retire.type + ': nothing to tell the host',
    );
    win.close();
  }
  console.log('  ok - answering or ending the task raises no toast either');
}

function testClosingTheAskingTabPostsNothingExtra() {
  const {win, posted} = makeWebview();
  const api = win._testApi;
  const askTab = api.getActiveTabId();
  api.createNewTab();
  send(win, {type: 'askUser', question: 'Close me?', tabId: askTab});
  assert.strictEqual(api.getActiveTabId(), askTab);

  const closeBtn = tabElement(win, askTab).querySelector('.chat-tab-close');
  assert.ok(closeBtn, 'the tab has a close button');
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(tabRecord(win, askTab), null, 'the tab is gone');
  assert.ok(!answering(win));
  assertNoToast(win, 'after closing the asking tab');
  assert.ok(!posted.some(m => m.type === 'askWaitingDone'));
  win.close();
  console.log('  ok - closing the asking tab leaves nothing behind');
}

function testAskUserForUnknownTabIsIgnored() {
  const {win, posted} = makeWebview();
  const api = win._testApi;
  const activeBefore = api.getActiveTabId();

  send(win, {
    type: 'askUser',
    question: 'This belongs to another VS Code window.',
    tabId: 'foreign-window-tab',
  });

  assert.strictEqual(api.getActiveTabId(), activeBefore);
  assert.ok(!answering(win));
  assertNoToast(win, 'for a foreign tab');
  assert.strictEqual(hostMessages(posted, 'revealForQuestion').length, 0);
  win.close();
  console.log('  ok - foreign-window askUser is ignored');
}

// A reload while the user was typing an answer in another tab: the
// persisted draft comes back without yanking them off the tab they were
// on.
function testReloadWithAnswerDraftRestoresInPlace() {
  const first = makeWebview({remote: true});
  send(first.win, {type: 'daemonStatus', connected: true});
  send(first.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 't1', chatId: 'c1', title: 't1', workDir: ''},
      {tabId: 't2', chatId: 'c2', title: 't2', workDir: ''},
    ],
  });
  send(first.win, {type: 'askUser', tabId: 't1', question: 'Deploy?'});
  assert.strictEqual(first.win._testApi.getActiveTabId(), 't1');
  first.win.document.getElementById('task-input').value = 'yes, to staging';
  clickTab(first.win, 't2');
  first.win.dispatchEvent(new first.win.Event('pagehide'));
  const persisted = first.getState();
  first.win.close();

  const second = makeWebview({remote: true, initialState: persisted});
  send(second.win, {type: 'daemonStatus', connected: true});
  send(second.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 't1', chatId: 'c1', title: 't1', workDir: ''},
      {tabId: 't2', chatId: 'c2', title: 't2', workDir: ''},
    ],
  });
  assert.strictEqual(second.win._testApi.getActiveTabId(), 't2');
  send(second.win, {type: 'askUser', tabId: 't1', question: 'Deploy?'});
  assert.strictEqual(
    second.win._testApi.getActiveTabId(),
    't2',
    'a question already being answered here does not switch tabs',
  );
  assertNoToast(second.win, 'after the reload');
  clickTab(second.win, 't1');
  assert.strictEqual(
    second.win.document.getElementById('task-input').value,
    'yes, to staging',
  );
  second.win.close();
  console.log('  ok - a reload restores a half-typed answer in place');
}

// The same reload without anything typed: the pending question is still
// recognised as seen, so the user stays on their tab.
function testReloadWithoutAnswerDraftDoesNotSwitch() {
  const first = makeWebview({remote: true});
  send(first.win, {type: 'daemonStatus', connected: true});
  send(first.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 't1', chatId: 'c1', title: 't1', workDir: ''},
      {tabId: 't2', chatId: 'c2', title: 't2', workDir: ''},
    ],
  });
  send(first.win, {type: 'askUser', tabId: 't1', question: 'Deploy?'});
  clickTab(first.win, 't2');
  first.win.dispatchEvent(new first.win.Event('pagehide'));
  const persisted = first.getState();
  first.win.close();

  const second = makeWebview({remote: true, initialState: persisted});
  send(second.win, {type: 'daemonStatus', connected: true});
  send(second.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 't1', chatId: 'c1', title: 't1', workDir: ''},
      {tabId: 't2', chatId: 'c2', title: 't2', workDir: ''},
    ],
  });
  send(second.win, {type: 'askUser', tabId: 't1', question: 'Deploy?'});
  assert.strictEqual(second.win._testApi.getActiveTabId(), 't2');
  assertNoToast(second.win, 'after a reload without a draft');
  // A different question on that tab is new again.
  send(second.win, {type: 'askUserDone', tabId: 't1'});
  send(second.win, {type: 'askUser', tabId: 't1', question: 'Rollback?'});
  assert.strictEqual(second.win._testApi.getActiveTabId(), 't1');
  assertNoToast(second.win, 'for the next question');
  second.win.close();
  console.log('  ok - a reload without a typed answer stays put');
}

// A tab closed from another surface arrives as a tabs_state snapshot
// without it.
function testMirroredCloseRemovesTheTab() {
  const {win, posted} = makeWebview();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 't1', chatId: 'c1', title: 't1', workDir: ''},
      {tabId: 't2', chatId: 'c2', title: 't2', workDir: ''},
    ],
  });
  send(win, {type: 'askUser', tabId: 't1', question: 'Still there?'});
  assert.ok(answering(win));
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 't2', chatId: 'c2', title: 't2', workDir: ''}],
  });
  assert.strictEqual(tabRecord(win, 't1'), null, 'the tab was removed');
  assert.ok(!answering(win), 'and its question with it');
  assertNoToast(win, 'after a mirrored close');
  assert.ok(!posted.some(m => m.type === 'askWaitingDone'));
  win.close();
  console.log('  ok - a close mirrored from another surface removes the tab');
}

function testRemoteWebappPostsNothingToTheDaemon() {
  const {win, posted} = makeWebview({remote: true});
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 't1', chatId: 'c1', title: 't1', workDir: ''},
      {tabId: 't2', chatId: 'c2', title: 't2', workDir: ''},
    ],
  });
  clickTab(win, 't2');
  send(win, {type: 'askUser', tabId: 't1', question: 'Remote?'});
  assert.strictEqual(win._testApi.getActiveTabId(), 't1', 'switched');
  assertNoToast(win, 'on the remote webapp');
  send(win, {type: 'askUserDone', tabId: 't1'});
  assert.strictEqual(
    hostMessages(posted, 'revealForQuestion').length,
    0,
    'the host-only message never goes to the daemon',
  );
  win.close();
  console.log('  ok - remote webapp: switch only, no host message');
}

function runTests() {
  testNewQuestionSwitchesTabWithoutToast();
  testTaskEndAndAskUserDoneRaiseNoToast();
  testClosingTheAskingTabPostsNothingExtra();
  testAskUserForUnknownTabIsIgnored();
  testReloadWithAnswerDraftRestoresInPlace();
  testReloadWithoutAnswerDraftDoesNotSwitch();
  testMirroredCloseRemovesTheTab();
  testRemoteWebappPostsNothingToTheDaemon();
}

try {
  runTests();
  console.log('\n8 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
