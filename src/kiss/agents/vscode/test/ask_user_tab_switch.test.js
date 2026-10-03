// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// An ask_user_question blocks the agent until the user answers, so on
// every surface a NEW question
//  * switches the client to the asking tab,
//  * raises a sticky "Waiting for your answer" toast (cleared by the
//    user's X, or when the question is answered, its task ends or its
//    tab closes), and
//  * in VS Code tells the host (`askWaiting` / `askWaitingDone`) so the
//    sidebar view or editor panel comes forward too.
// A replayed copy of a question already pending here changes nothing,
// and the remote webapp never posts the host-only messages to the daemon.

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

function tabElement(win, tabId) {
  return win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
}

function clickTab(win, tabId) {
  const el = tabElement(win, tabId);
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

// The composer is in answer mode while the tab on screen has a question.
function answering(win) {
  return win.document.body.classList.contains('ask-answering');
}

function attentionGlyph(win, tabId) {
  const el = tabElement(win, tabId);
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  const marker = el.querySelector('.chat-tab-attention');
  return marker ? marker.textContent : '';
}

function waitingToast(win, tabId) {
  return win.document.querySelector(
    `.kiss-notification[data-notification-id=${JSON.stringify('ask:' + tabId)}]`,
  );
}

// Messages come from the JSDOM realm (a different Object prototype), so
// they are copied before deepStrictEqual.
function hostMessages(posted, type) {
  return JSON.parse(JSON.stringify(posted.filter(m => m.type === type)));
}

function testNewQuestionSwitchesTabAndRaisesToast() {
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
  assert.strictEqual(
    win.document.querySelector('.chat-tab.active').dataset.tabId,
    questionTab,
  );
  assert.ok(answering(win), 'the composer answers the question at once');
  assert.strictEqual(attentionGlyph(win, questionTab), '');

  const toast = waitingToast(win, questionTab);
  assert.ok(toast, 'a waiting toast is on screen');
  assert.strictEqual(toast.dataset.notificationSticky, 'true');
  assert.strictEqual(
    toast.querySelector('.kiss-notification-title').textContent,
    'Waiting for your answer',
  );
  assert.strictEqual(
    toast.querySelector('.kiss-notification-message').textContent,
    'Please provide the deployment token.',
  );
  assert.strictEqual(
    toast.getAttribute('aria-label'),
    'Waiting for your answer: Please provide the deployment token.',
  );
  assert.deepStrictEqual(hostMessages(posted, 'askWaiting'), [
    {
      type: 'askWaiting',
      tabId: questionTab,
      question: 'Please provide the deployment token.',
    },
  ]);

  // The user goes back to their own tab: the toast stays, the waiting
  // tab is flagged, and the toast's button brings them back.
  clickTab(win, otherTab);
  assert.strictEqual(api.getActiveTabId(), otherTab);
  assert.ok(!answering(win));
  assert.strictEqual(attentionGlyph(win, questionTab), '?');
  assert.ok(waitingToast(win, questionTab), 'the toast outlives the switch');

  // A replay of the same pending question (another client reloaded)
  // neither switches tabs again nor re-notifies the host.
  send(win, {
    type: 'askUser',
    question: 'Please provide the deployment token.',
    tabId: questionTab,
  });
  assert.strictEqual(api.getActiveTabId(), otherTab);
  assert.strictEqual(hostMessages(posted, 'askWaiting').length, 1);

  const button = waitingToast(win, questionTab).querySelector(
    '.kiss-notification-action',
  );
  assert.strictEqual(button.textContent, 'Show question');
  button.click();
  assert.strictEqual(api.getActiveTabId(), questionTab);
  assert.ok(answering(win));
  assert.strictEqual(
    waitingToast(win, questionTab),
    null,
    'using the toast button dismisses the toast',
  );
  assert.strictEqual(hostMessages(posted, 'askWaitingDone').length, 0);

  // Answering retires the question everywhere.
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
  assert.deepStrictEqual(hostMessages(posted, 'askWaitingDone'), [
    {type: 'askWaitingDone', tabId: questionTab},
  ]);
  assert.ok(!answering(win));

  win.close();
  console.log('  ok - a new question switches tabs and raises the toast');
}

function testDismissedToastKeepsTheQuestion() {
  const {win, posted} = makeWebview();
  const api = win._testApi;
  const tab = api.getActiveTabId();
  send(win, {type: 'askUser', question: 'Continue?', tabId: tab});
  const toast = waitingToast(win, tab);
  assert.ok(toast);

  toast.querySelector('.kiss-notification-close').click();
  assert.strictEqual(waitingToast(win, tab), null, 'the X clears the toast');
  assert.ok(answering(win), 'the question itself is still pending');
  assert.strictEqual(
    hostMessages(posted, 'askWaitingDone').length,
    0,
    'clearing the toast is not an answer',
  );

  send(win, {type: 'askUserDone', tabId: tab});
  assert.ok(!answering(win));
  assert.deepStrictEqual(hostMessages(posted, 'askWaitingDone'), [
    {type: 'askWaitingDone', tabId: tab},
  ]);
  win.close();
  console.log('  ok - the X clears the toast but keeps the question');
}

function testTaskEndAndAskUserDoneClearTheToast() {
  for (const retire of [
    {type: 'askUserDone'},
    {type: 'task_done', success: true},
    {type: 'task_error', error: 'boom'},
  ]) {
    const {win, posted} = makeWebview();
    const api = win._testApi;
    const tab = api.getActiveTabId();
    send(win, {type: 'askUser', question: 'Which branch?', tabId: tab});
    assert.ok(waitingToast(win, tab), retire.type + ': toast shown first');

    send(win, Object.assign({tabId: tab}, retire));
    assert.strictEqual(
      waitingToast(win, tab),
      null,
      retire.type + ' must clear the waiting toast',
    );
    assert.deepStrictEqual(
      hostMessages(posted, 'askWaitingDone'),
      [{type: 'askWaitingDone', tabId: tab}],
      retire.type + ' must tell the host once',
    );
    win.close();
  }
  console.log('  ok - answering or ending the task clears the toast');
}

function testClosingTheAskingTabClearsTheToast() {
  const {win, posted} = makeWebview();
  const api = win._testApi;
  const askTab = api.getActiveTabId();
  api.createNewTab();
  send(win, {type: 'askUser', question: 'Close me?', tabId: askTab});
  assert.strictEqual(api.getActiveTabId(), askTab);
  assert.ok(waitingToast(win, askTab));

  const closeBtn = tabElement(win, askTab).querySelector('.chat-tab-close');
  assert.ok(closeBtn, 'the tab has a close button');
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(tabElement(win, askTab), null, 'the tab is gone');
  assert.strictEqual(waitingToast(win, askTab), null, 'and so is its toast');
  assert.deepStrictEqual(hostMessages(posted, 'askWaitingDone'), [
    {type: 'askWaitingDone', tabId: askTab},
  ]);
  win.close();
  console.log('  ok - closing the asking tab clears the toast');
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
  assert.strictEqual(waitingToast(win, 'foreign-window-tab'), null);
  assert.strictEqual(hostMessages(posted, 'askWaiting').length, 0);
  win.close();
  console.log('  ok - foreign-window askUser is ignored');
}

// A reload while the user was typing an answer in another tab: the
// persisted draft comes back without yanking them off the tab they were
// on, and the toast reminds them the agent is still waiting.
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
  assert.ok(waitingToast(second.win, 't1'), 'the reminder toast is back');
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
  assert.ok(waitingToast(second.win, 't1'));
  // A different question on that tab is new again.
  send(second.win, {type: 'askUserDone', tabId: 't1'});
  send(second.win, {type: 'askUser', tabId: 't1', question: 'Rollback?'});
  assert.strictEqual(second.win._testApi.getActiveTabId(), 't1');
  second.win.close();
  console.log('  ok - a reload without a typed answer stays put');
}

// A tab closed from another surface arrives as a tabs_state snapshot
// without it; its toast must not outlive it.
function testMirroredCloseClearsTheToast() {
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
  assert.ok(waitingToast(win, 't1'));
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 't2', chatId: 'c2', title: 't2', workDir: ''}],
  });
  assert.strictEqual(tabElement(win, 't1'), null, 'the tab was removed');
  assert.strictEqual(waitingToast(win, 't1'), null, 'and its toast with it');
  assert.deepStrictEqual(hostMessages(posted, 'askWaitingDone'), [
    {type: 'askWaitingDone', tabId: 't1'},
  ]);
  win.close();
  console.log('  ok - a close mirrored from another surface clears the toast');
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
  assert.ok(waitingToast(win, 't1'), 'toast shown');
  send(win, {type: 'askUserDone', tabId: 't1'});
  assert.strictEqual(waitingToast(win, 't1'), null);
  assert.strictEqual(
    hostMessages(posted, 'askWaiting').length +
      hostMessages(posted, 'askWaitingDone').length,
    0,
    'host-only messages never go to the daemon',
  );
  win.close();
  console.log('  ok - remote webapp: toast and switch, no host messages');
}

function runTests() {
  testNewQuestionSwitchesTabAndRaisesToast();
  testDismissedToastKeepsTheQuestion();
  testTaskEndAndAskUserDoneClearTheToast();
  testClosingTheAskingTabClearsTheToast();
  testAskUserForUnknownTabIsIgnored();
  testReloadWithAnswerDraftRestoresInPlace();
  testReloadWithoutAnswerDraftDoesNotSwitch();
  testMirroredCloseClearsTheToast();
  testRemoteWebappPostsNothingToTheDaemon();
}

try {
  runTests();
  console.log('\n9 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
