// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The `/ask` side channel's nested tab, driven the way the daemon drives
// it.  Unlike a run_parallel / run_agent child there is NO tool-call
// panel in the parent that owns the child, so the daemon is the only
// thing that decides whether the tab exists:
//
//  1. live: `new_tab` (parent_tab_id = the asking tab) opens a nested
//     background tab and resumes the child; `subagentDone` closes it and
//     the daemon is told (`closeTab`).
//  2. replay: a finished side-channel child is announced as
//     `subagentDone` for its deterministic `<parent>__sub_<task>` id,
//     never as `openSubagentTab`.  That close is a no-op when the tab is
//     already gone, and closes a tab restored from saved state.
//  3. the parent keeps the answer: the `ask_answer` event renders in the
//     asking tab and is untouched by the child's close.
//
// Regression guard for the other direction too: an
// `openSubagentTab{isDone:true}` with no owning panel DOES re-open a
// finished tab in the webview (that is the run_agent/merge-resolver
// behaviour), which is exactly why the daemon must not send one for a
// finished side channel.

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
  for (const f of ['panelCopy.js', 'api.js', 'main.js']) {
    win.eval(fs.readFileSync(path.join(MEDIA, f), 'utf8'));
  }
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  return {win, posted, parentId: ready.tabId};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

function askAnswerPanels(win) {
  return win.document.querySelectorAll('#output .ask-answer');
}

const CHILD = 'ask-child-task-1';
const PARENT_TASK = 'parent-task-1';
const QUESTION = 'what is the parent doing?';

/** A running parent that just received `/ask` (prompt echo included). */
function runningParentWithAsk() {
  const {win, posted, parentId} = makeWebview();
  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
    taskId: PARENT_TASK,
  });
  send(win, {
    type: 'prompt',
    text: '/ask ' + QUESTION,
    tabId: parentId,
    taskId: PARENT_TASK,
  });
  return {win, posted, parentId};
}

function testLiveAskChildClosesOnDone() {
  const {win, posted, parentId} = runningParentWithAsk();
  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: CHILD,
    parent_tab_id: parentId,
    taskId: '',
  });
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === CHILD);
  assert.ok(resume, 'new_tab must make the webview resume the child');
  const subTabId = resume.tabId;
  assert.strictEqual(
    subTabId,
    parentId + '__sub_' + CHILD,
    'the nested tab uses the daemon-deterministic <parent>__sub_<task> id',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'the answering child gets one nested tab while it runs',
  );
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabId,
    parent_tab_id: parentId,
    description: QUESTION,
    task_id: CHILD,
    isSubagentTab: true,
    isDone: false,
  });
  send(win, {type: 'status', running: true, tabId: subTabId, startTs: Date.now()});
  assert.strictEqual(subagentTabEls(win).length, 1, 'still one tab');

  send(win, {type: 'result', text: 'the answer', taskId: CHILD, tabId: subTabId});
  send(win, {type: 'subagentDone', tab_id: subTabId, tabId: ''});
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'the child tab closes on subagentDone without any owning panel',
  );
  assert.ok(
    posted.some(m => m.type === 'closeTab' && m.tabId === subTabId),
    'the daemon is told about the close',
  );

  // The answer lands in the PARENT's transcript afterwards.
  send(win, {
    type: 'ask_answer',
    question: QUESTION,
    text: 'the answer',
    success: true,
    tabId: parentId,
    taskId: PARENT_TASK,
  });
  assert.strictEqual(askAnswerPanels(win).length, 1, 'answer panel rendered');
  assert.strictEqual(subagentTabEls(win).length, 0, 'no tab came back');
  win.close();
  console.log('  ok - live: /ask child tab closes on subagentDone');
}

function testReplayCloseIsNoopWhenTabAlreadyGone() {
  const {win, posted, parentId} = runningParentWithAsk();
  const subTabId = parentId + '__sub_' + CHILD;
  const tabsBefore = win.document.querySelectorAll('#tab-list .chat-tab').length;
  const postedBefore = posted.length;
  // What the daemon now sends on a parent replay for a finished side
  // channel: a close for the deterministic id, no openSubagentTab.
  send(win, {type: 'subagentDone', tab_id: subTabId, tabId: ''});
  assert.strictEqual(
    win.document.querySelectorAll('#tab-list .chat-tab').length,
    tabsBefore,
    'a close for an absent tab touches nothing',
  );
  assert.ok(
    !posted.slice(postedBefore).some(m => m.type === 'closeTab'),
    'no closeTab is echoed for a tab that was not open',
  );
  win.close();
  console.log('  ok - replay: subagentDone for an already-closed /ask tab is a no-op');
}

function testReplayCloseRemovesRestoredTab() {
  const {win, parentId} = runningParentWithAsk();
  // A tab that survived a reload (restored from saved webview state, or
  // opened by new_tab moments before the answer came back).
  send(win, {
    type: 'new_tab',
    task_id: CHILD,
    parent_tab_id: parentId,
    taskId: '',
  });
  const subTabId = parentId + '__sub_' + CHILD;
  assert.strictEqual(subagentTabEls(win).length, 1, 'tab open before replay');
  // The child's own replay (the late resumeSession) answers with a
  // close instead of the transcript.
  send(win, {type: 'subagentDone', tab_id: subTabId, tabId: ''});
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'the replay-time close removes the lingering tab',
  );
  win.close();
  console.log('  ok - replay: subagentDone closes a lingering /ask tab');
}

function testOpenAnnouncementWouldReopenFinishedTab() {
  const {win, parentId} = runningParentWithAsk();
  const subTabId = parentId + '__sub_' + CHILD;
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabId,
    parent_tab_id: parentId,
    description: QUESTION,
    task_id: CHILD,
    isSubagentTab: true,
    isDone: true,
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'an isDone openSubagentTab with no owning panel re-opens the tab -- ' +
      'the daemon must not send it for a finished side channel',
  );
  win.close();
  console.log('  ok - control: openSubagentTab{isDone} without a panel re-opens');
}

function main() {
  testLiveAskChildClosesOnDone();
  testReplayCloseIsNoopWhenTabAlreadyGone();
  testReplayCloseRemovesRestoredTab();
  testOpenAnnouncementWouldReopenFinishedTab();
  console.log('askSubagentTabAutoClose: all tests passed');
}

main();
process.exit(0);
