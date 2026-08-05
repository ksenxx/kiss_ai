// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
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
  win.acquireVsCodeApi = function () {
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
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
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

function askModal(win) {
  return win.document.getElementById('ask-user-modal');
}

function attentionGlyph(win, tabId) {
  const el = tabElement(win, tabId);
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  const marker = el.querySelector('.chat-tab-attention');
  return marker ? marker.textContent : '';
}

function testBackgroundAskDoesNotSwitchTabs() {
  const {win, posted} = makeWebview();
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');

  const questionTab = api.getActiveTabId();
  assert.ok(questionTab, 'initial tab id must exist');

  api.createNewTab();
  const otherTab = api.getActiveTabId();
  assert.ok(otherTab && otherTab !== questionTab, 'second tab must be active');

  send(win, {
    type: 'askUser',
    question: 'Please provide the deployment token.',
    tabId: questionTab,
  });

  assert.strictEqual(
    api.getActiveTabId(),
    otherTab,
    'askUser for a still-running background task must not switch tabs',
  );
  const activeEl = win.document.querySelector('.chat-tab.active');
  assert.ok(activeEl, 'one tab must be marked active in the DOM');
  assert.strictEqual(
    activeEl.dataset.tabId,
    otherTab,
    'the active tab DOM class must stay on the tab the user is using',
  );
  assert.notStrictEqual(
    askModal(win).style.display,
    'flex',
    'a background question must not pop a modal over the active tab',
  );
  assert.strictEqual(
    attentionGlyph(win, questionTab),
    '?',
    'the waiting background tab must show an attention indicator',
  );
  assert.strictEqual(
    attentionGlyph(win, otherTab),
    '',
    'the active tab must not show an attention indicator',
  );

  clickTab(win, questionTab);
  assert.strictEqual(
    api.getActiveTabId(),
    questionTab,
    'clicking the waiting tab must switch to it',
  );
  const modal = askModal(win);
  assert.strictEqual(
    modal.style.display,
    'flex',
    'the stored question must be shown once the user visits the tab',
  );
  assert.ok(
    modal.textContent.includes('Please provide the deployment token.'),
    'the remounted modal must show the original question text',
  );
  assert.strictEqual(
    attentionGlyph(win, questionTab),
    '',
    'visiting the tab must clear its attention indicator',
  );

  const input = modal.querySelector('.ask-user-input');
  assert.ok(input, 'ask-user input must be mounted for the active tab');
  input.value = 'tok_live_123';
  modal.querySelector('.ask-user-submit').click();
  assert.ok(
    posted.some(
      msg =>
        msg.type === 'userAnswer' &&
        msg.tabId === questionTab &&
        msg.answer === 'tok_live_123',
    ),
    'submitting the remounted modal must answer the question tab',
  );

  win.close();
  console.log('  ok - background askUser never steals focus');
}

function testActiveTabAskShowsModalImmediately() {
  const {win} = makeWebview();
  const api = win._testApi;
  const activeTab = api.getActiveTabId();

  send(win, {
    type: 'askUser',
    question: 'Which branch should I push?',
    tabId: activeTab,
  });

  assert.strictEqual(
    api.getActiveTabId(),
    activeTab,
    'the active tab must stay active',
  );
  const modal = askModal(win);
  assert.strictEqual(
    modal.style.display,
    'flex',
    'askUser for the active tab must show the modal immediately',
  );
  assert.ok(
    modal.textContent.includes('Which branch should I push?'),
    'the modal must show the question text',
  );
  assert.strictEqual(
    attentionGlyph(win, activeTab),
    '',
    'the active tab shows the modal, so it needs no tab-bar indicator',
  );

  win.close();
  console.log('  ok - active-tab askUser still shows the modal at once');
}

function testAnsweringClearsBackgroundIndicator() {
  const {win} = makeWebview();
  const api = win._testApi;
  const questionTab = api.getActiveTabId();
  api.createNewTab();

  send(win, {
    type: 'askUser',
    question: 'Continue with the risky migration?',
    tabId: questionTab,
  });
  assert.strictEqual(
    attentionGlyph(win, questionTab),
    '?',
    'the background tab must be flagged as waiting',
  );

  send(win, {type: 'askUserDone', tabId: questionTab});
  assert.strictEqual(
    attentionGlyph(win, questionTab),
    '',
    'a retracted question must clear the attention indicator',
  );

  win.close();
  console.log('  ok - askUserDone clears the background attention indicator');
}

function testAskUserForUnknownTabIsIgnored() {
  const {win} = makeWebview();
  const api = win._testApi;
  const activeBefore = api.getActiveTabId();

  send(win, {
    type: 'askUser',
    question: 'This belongs to another VS Code window.',
    tabId: 'foreign-window-tab',
  });

  assert.strictEqual(
    api.getActiveTabId(),
    activeBefore,
    'askUser for an unknown foreign tab must not switch local tabs',
  );
  assert.notStrictEqual(
    askModal(win).style.display,
    'flex',
    'askUser for an unknown foreign tab must not show a modal locally',
  );

  win.close();
  console.log('  ok - foreign-window askUser is ignored');
}

function runTests() {
  testBackgroundAskDoesNotSwitchTabs();
  testActiveTabAskShowsModalImmediately();
  testAnsweringClearsBackgroundIndicator();
  testAskUserForUnknownTabIsIgnored();
}

try {
  runTests();
  console.log('\n4 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
