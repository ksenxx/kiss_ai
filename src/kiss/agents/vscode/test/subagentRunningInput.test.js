// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: a sub-agent chat tab must show the input textbox
// and the buttons below it (#input-container) WHILE its sub-agent
// task is RUNNING — so the user can inject follow-up prompts into the
// running sub-agent and stop ONLY that sub-agent — and must hide the
// input as soon as the sub-agent task completes.
//
// Behaviors under test (``media/main.js``):
//
//   1. Switching to a RUNNING sub-agent tab (``restoreTab``) shows
//      ``#input-container`` and the Stop button.
//   2. ``openSubagentTab`` converting the ACTIVE tab shows the input
//      while the sub-agent is running (live spawn / history click on
//      a still-running sub-agent) and hides it when the event carries
//      ``isDone`` (history click on a finished sub-agent).
//   3. Switching between a DONE sub-agent tab and a RUNNING sibling
//      toggles the input off/on (``restoreTab`` both branches).
//   4. ``subagentDone`` removes the finished sub-agent's input
//      surface immediately (the tab auto-closes and the input
//      visibility re-resolves for the newly active tab).
//   5. Typing + Send on a running sub-agent tab posts
//      ``appendUserMessage`` with the SUB-AGENT's tab id (prompt
//      injection routed to the sub-agent, not the parent).
//   6. Clicking Stop on a running sub-agent tab posts
//      ``{type:'stop', tabId:<subagent tab id>}`` (stops only the
//      sub-agent's task).
//
// The tests drive the real ``media/main.js`` against the real
// ``media/chat.html`` markup in jsdom — the exact production webview
// code — mirroring the harness of ``subagentTabAutoCloseOnDone.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/subagentRunningInput.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
 */
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** All sub-agent tabs currently rendered in the tab bar. */
function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

/** The currently active tab element in the tab bar. */
function activeTabEl(win) {
  return win.document.querySelector('#tab-list .chat-tab.active');
}

/** Click the tab bar entry for *tabId* exactly like a user would. */
function clickTab(win, tabId) {
  const el = Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab'),
  ).find(e => e.dataset.tabId === tabId);
  assert.ok(el, 'tab ' + tabId + ' must be in the tab bar');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    activeTabEl(win).dataset.tabId,
    tabId,
    'clicking tab ' + tabId + ' must activate it',
  );
}

/** True when ``#input-container`` is visible on the active tab. */
function inputVisible(win) {
  const c = win.document.getElementById('input-container');
  assert.ok(c, '#input-container must exist');
  return c.style.display !== 'none';
}

/**
 * Boot a webview with a running parent task whose agent called
 * ``run_parallel`` and spawned *n* sub-agents.  Replays the exact
 * backend broadcast sequence: ``status running`` → ``tool_call
 * run_parallel`` → per sub-agent ``new_tab`` (which makes the webview
 * post ``resumeSession``) → ``openSubagentTab``.
 */
function bootParallelRun(n) {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const parentId = ready.tabId;

  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
  });
  const taskNames = [];
  for (let i = 0; i < n; i++) taskNames.push('sub ' + (i + 1));
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(taskNames)},
  });

  const subTabIds = [];
  for (let i = 0; i < n; i++) {
    const taskId = 'sub-task-' + (i + 1);
    const before = posted.length;
    send(win, {
      type: 'new_tab',
      task_id: taskId,
      parent_tab_id: parentId,
      taskId: '',
    });
    const resume = posted
      .slice(before)
      .find(m => m.type === 'resumeSession' && m.taskId === taskId);
    assert.ok(resume, 'new_tab must make the webview post resumeSession');
    subTabIds.push(resume.tabId);
    send(win, {
      type: 'openSubagentTab',
      tab_id: resume.tabId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
    });
  }
  assert.strictEqual(
    subagentTabEls(win).length,
    n,
    'each spawned sub-agent must get its own tab',
  );
  return {win, posted, parentId, subTabIds};
}

// ---------------------------------------------------------------------------
// 1. Switching to a RUNNING sub-agent tab shows the input + Stop button.
// ---------------------------------------------------------------------------
function testRunningSubagentTabShowsInput() {
  const {win, subTabIds} = bootParallelRun(2);

  clickTab(win, subTabIds[0]);

  assert.ok(
    inputVisible(win),
    'the input textbox and the buttons below it must be VISIBLE on a ' +
      'sub-agent tab whose task is still running — the user must be ' +
      'able to inject prompts into the running sub-agent',
  );
  const stopBtn = win.document.getElementById('stop-btn');
  assert.strictEqual(
    stopBtn.style.display,
    'flex',
    'the Stop button must be visible on a running sub-agent tab so ' +
      'the user can stop ONLY the sub-agent task',
  );
  win.close();
  console.log('  ok - running sub-agent tab shows input + stop button');
}

// ---------------------------------------------------------------------------
// 2. openSubagentTab on the ACTIVE tab: running shows input, isDone hides it.
// ---------------------------------------------------------------------------
function testOpenSubagentTabActiveRespectsRunningState() {
  const {win, parentId, subTabIds} = bootParallelRun(1);

  clickTab(win, subTabIds[0]);
  // Re-broadcast of the same running sub-agent (idempotent update,
  // e.g. history replay of a STILL-RUNNING sub-agent).
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabIds[0],
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
  });
  assert.ok(
    inputVisible(win),
    'openSubagentTab for the ACTIVE tab must show the input while the ' +
      'sub-agent is running',
  );

  // History replay of a FINISHED sub-agent (isDone) must hide it.
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabIds[0],
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
    isDone: true,
  });
  assert.ok(
    !inputVisible(win),
    'openSubagentTab with isDone for the ACTIVE tab must hide the ' +
      'input — the sub-agent task already completed',
  );
  win.close();
  console.log('  ok - openSubagentTab on active tab respects running state');
}

// ---------------------------------------------------------------------------
// 3. restoreTab: DONE sub-agent tab hides input, RUNNING sibling shows it.
// ---------------------------------------------------------------------------
function testTabSwitchTogglesInputByRunningState() {
  const {win, parentId, subTabIds} = bootParallelRun(2);

  // Mark the first sub-agent as done via a history-style replay so
  // its tab STAYS OPEN (live ``subagentDone`` would auto-close it).
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabIds[0],
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
    isDone: true,
  });

  clickTab(win, subTabIds[0]);
  assert.ok(
    !inputVisible(win),
    'switching to a DONE sub-agent tab must hide the input textbox ' +
      'and the buttons below it',
  );

  clickTab(win, subTabIds[1]);
  assert.ok(
    inputVisible(win),
    'switching to a still-RUNNING sibling sub-agent tab must show ' +
      'the input again',
  );

  clickTab(win, parentId);
  assert.ok(
    inputVisible(win),
    'switching back to the parent (regular) tab must show the input',
  );
  win.close();
  console.log('  ok - tab switch toggles input by sub-agent running state');
}

// ---------------------------------------------------------------------------
// 4. subagentDone removes the finished sub-agent's input surface at once.
// ---------------------------------------------------------------------------
function testSubagentDoneRemovesInput() {
  const {win, subTabIds} = bootParallelRun(2);

  clickTab(win, subTabIds[0]);
  assert.ok(inputVisible(win), 'input must be visible while running');

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  // The finished sub-agent tab auto-closed; no input surface may be
  // left addressing the finished sub-agent.
  assert.ok(
    !subagentTabEls(win).some(e => e.dataset.tabId === subTabIds[0]),
    'the finished sub-agent tab must close on subagentDone',
  );
  const active = activeTabEl(win);
  assert.notStrictEqual(
    active.dataset.tabId,
    subTabIds[0],
    'the finished sub-agent tab must not stay active',
  );
  win.close();
  console.log('  ok - subagentDone removes the finished sub-agent input');
}

// ---------------------------------------------------------------------------
// 5. Send on a running sub-agent tab injects the prompt to the sub-agent.
// ---------------------------------------------------------------------------
function testSendInjectsPromptWithSubagentTabId() {
  const {win, posted, subTabIds} = bootParallelRun(2);

  clickTab(win, subTabIds[1]);
  assert.ok(inputVisible(win), 'input must be visible while running');

  const inp = win.document.getElementById('task-input');
  inp.value = 'focus on the tests';
  const before = posted.length;
  win.document
    .getElementById('send-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const injected = posted
    .slice(before)
    .find(m => m.type === 'appendUserMessage');
  assert.ok(
    injected,
    'sending while the sub-agent runs must post appendUserMessage ' +
      '(prompt injection into the live sub-agent)',
  );
  assert.strictEqual(
    injected.prompt,
    'focus on the tests',
    'the injected prompt must carry the typed text',
  );
  assert.strictEqual(
    injected.tabId,
    subTabIds[1],
    'the injected prompt must be routed with the SUB-AGENT tab id, ' +
      'not the parent tab id',
  );
  assert.ok(
    !posted.slice(before).some(m => m.type === 'submit'),
    'no fresh-task submit may be posted while the sub-agent runs',
  );
  assert.strictEqual(inp.value, '', 'the input clears after injecting');
  win.close();
  console.log('  ok - send on running sub-agent tab injects with sub tab id');
}

// ---------------------------------------------------------------------------
// 6. Stop on a running sub-agent tab stops ONLY that sub-agent's task.
// ---------------------------------------------------------------------------
function testStopPostsStopWithSubagentTabId() {
  const {win, posted, subTabIds} = bootParallelRun(2);

  clickTab(win, subTabIds[0]);
  const before = posted.length;
  win.document
    .getElementById('stop-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const stop = posted.slice(before).find(m => m.type === 'stop');
  assert.ok(stop, 'clicking Stop on a sub-agent tab must post stop');
  assert.strictEqual(
    stop.tabId,
    subTabIds[0],
    'the stop must carry the SUB-AGENT tab id so only the sub-agent ' +
      "task is stopped — not the parent's",
  );
  win.close();
  console.log('  ok - stop on running sub-agent tab targets the sub tab id');
}

// ---------------------------------------------------------------------------
// 7. ``status running:false`` on the ACTIVE sub-agent tab removes the input
//    at once (lifecycle fallback when ``subagentDone`` is delayed/lost).
// ---------------------------------------------------------------------------
function testStatusNotRunningRemovesInputOnActiveSubTab() {
  const {win, subTabIds} = bootParallelRun(2);

  clickTab(win, subTabIds[0]);
  assert.ok(inputVisible(win), 'input must be visible while running');

  send(win, {type: 'status', running: false, tabId: subTabIds[0]});

  assert.ok(
    !inputVisible(win),
    'a running:false status for the active sub-agent tab must remove ' +
      'the input textbox and the buttons below it immediately',
  );
  // A later running:true for the parent must not resurrect the input
  // on the still-active (finished) sub-agent tab.
  const stopBtn = win.document.getElementById('stop-btn');
  assert.strictEqual(
    stopBtn.style.display,
    'none',
    'the Stop button must hide once the sub-agent task ended',
  );
  win.close();
  console.log('  ok - status running:false removes active sub tab input');
}

function main() {
  testRunningSubagentTabShowsInput();
  testOpenSubagentTabActiveRespectsRunningState();
  testTabSwitchTogglesInputByRunningState();
  testSubagentDoneRemovesInput();
  testSendInjectsPromptWithSubagentTabId();
  testStopPostsStopWithSubagentTabId();
  testStatusNotRunningRemovesInputOnActiveSubTab();
  console.log('subagentRunningInput.test.js: all tests passed');
}

main();
