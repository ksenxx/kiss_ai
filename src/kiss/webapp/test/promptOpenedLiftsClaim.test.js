// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A path-only prompt (`./src/.../review_paper_sea.py`) opens the file
// instead of starting a task.  The page cannot know that when the user
// hits Enter, so sendMessage() stamps the tab as it does for any run:
// `pendingTaskId` (the claim that lets it own the task's first output) and
// `unackedPrompt` (put back into the composer after a reload).  The host
// that opened the file answers `promptOpened`; this test drives the real
// page (jsdom) and checks that the answer lifts both, and that a tab
// that IS running a task keeps its adopted id.

const assert = require('assert');
const {typeAndEnter, submits} = require('./attachment_helpers');
const {makePage: makeChatPage, send, stubRaf} = require('./simplify2_harness');

function makePage() {
  const {win, posted, getState} = makeChatPage({beforeScripts: stubRaf});
  win._testApi.endLaunch();
  return {win, posted, getState};
}

// The drafts the page would hand back after a reload: what the
// `pagehide` handler persists (persistTabState).
function persistedDrafts(win, getState) {
  win.dispatchEvent(new win.Event('pagehide'));
  const state = getState();
  return (state && state.inputDrafts) || {};
}

// The task the visible tab has claimed, as main.js reports it to voice.js.
function ownedTaskId(win) {
  return win.kissVoiceOwner().taskId;
}

const PROMPT = './src/kiss/agents/seas/review_paper/review_paper_sea.py';

function testAckLiftsTheClaim() {
  const {win, posted, getState} = makePage();
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
    persistedDrafts(win, getState)[tabId],
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
    persistedDrafts(win, getState)[tabId],
    undefined,
    'BUG: the opened path would come back as an unsent draft after a reload',
  );
  win.close();
  console.log('  ok - promptOpened lifts the pending claim and the prompt');
}

function testAckKeepsAnAdoptedTask() {
  const {win} = makePage();
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
  const {win, posted} = makePage();
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
