// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A path-only prompt (`./src/.../review_paper_sea.py`) opens the file
// instead of starting a task.  The webview cannot know that when the user
// hits Enter, so sendMessage() stamps the tab as it does for any run:
// `pendingTaskId` (the claim that lets it own the task's first output) and
// `unackedPrompt` (put back into the composer after a reload).  The host
// that opened the file answers `promptOpened`; this test drives the real
// webview (jsdom) and checks that the answer lifts both, and that a tab
// that IS running a task keeps its adopted id.

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
  const store = {state: undefined};
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => store.state,
      setState: s => {
        store.state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win._testApi.endLaunch();
  return {win, posted, store};
}

// The drafts the webview would hand back after a reload: what the
// `pagehide` handler persists (persistTabState).
function persistedDrafts(win, store) {
  win.dispatchEvent(new win.Event('pagehide'));
  return (store.state && store.state.inputDrafts) || {};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function typeAndEnter(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

// The task the visible tab has claimed, as main.js reports it to voice.js.
function ownedTaskId(win) {
  return win.kissVoiceOwner().taskId;
}

function submits(posted) {
  return posted.filter(m => m.type === 'submit');
}

const PROMPT = './src/kiss/agents/seas/review_paper/review_paper_sea.py';

function testAckLiftsTheClaim() {
  const {win, posted, store} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  typeAndEnter(win, PROMPT);
  assert.strictEqual(submits(posted).length, 1, 'the prompt was submitted');
  assert.strictEqual(submits(posted)[0].prompt, PROMPT);
  assert.strictEqual(
    ownedTaskId(win),
    'pending:' + tabId,
    'until the host answers, the tab claims the task it may have started',
  );
  assert.strictEqual(
    persistedDrafts(win, store)[tabId],
    PROMPT,
    'until the host answers, a reload would give the prompt back',
  );

  send(win, {type: 'promptOpened', tabId});

  assert.strictEqual(
    ownedTaskId(win),
    '',
    'BUG: the tab still claims a task after its prompt opened a file',
  );
  // A bare task id off the wire must no longer be adopted.
  send(win, {type: 'system_output', taskId: 'task-FOREIGN', text: 'foreign'});
  assert.strictEqual(
    ownedTaskId(win),
    '',
    'a lifted claim must not let the tab adopt a foreign task id',
  );
  assert.ok(
    !win.document.getElementById('output').textContent.includes('foreign'),
    "a foreign task's words must not render in the tab",
  );

  // The prompt is acknowledged: a reload gives back no unsent draft.
  assert.strictEqual(
    persistedDrafts(win, store)[tabId],
    undefined,
    'BUG: the opened path would come back as an unsent draft after a reload',
  );
  win.close();
  console.log('  ok - promptOpened lifts the pending claim and the prompt');
}

function testAckKeepsAnAdoptedTask() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  typeAndEnter(win, 'summarize the repo');
  send(win, {type: 'setTaskText', text: 'summarize the repo', tabId});
  send(win, {type: 'system_output', text: 'working', tabId, taskId: 'task-A'});
  assert.strictEqual(ownedTaskId(win), 'task-A', 'the tab adopted its task');

  // A stray ack (a race with a follow-up, or a stale reply) must not
  // unbind a tab from the task it is showing.
  send(win, {type: 'promptOpened', tabId});
  assert.strictEqual(
    ownedTaskId(win),
    'task-A',
    'promptOpened must not disown a task the tab has adopted',
  );
  win.close();
  console.log('  ok - promptOpened leaves an adopted task alone');
}

function testUnaddressedAckTargetsTheVisibleTab() {
  const {win, posted} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  typeAndEnter(win, PROMPT);
  assert.strictEqual(submits(posted).length, 1);
  send(win, {type: 'promptOpened'});
  assert.strictEqual(
    ownedTaskId(win),
    '',
    'an ack without a tabId is for the tab on screen, like the echoes',
  );
  assert.strictEqual(tabId, win._testApi.getActiveTabId());
  win.close();
  console.log('  ok - an unaddressed promptOpened is for the visible tab');
}

testAckLiftsTheClaim();
testAckKeepsAnAdoptedTask();
testUnaddressedAckTargetsTheVisibleTab();
console.log('promptOpenedLiftsClaim: all tests passed');
