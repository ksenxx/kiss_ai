// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: a prompt injected into a RUNNING sub-agent tab
// must show up as a "Prompt" panel in the sub-agent's trajectory —
// and must SURVIVE tab switches and a ``task_events`` replay (the
// backend now records + persists the echoed prompt event under the
// sub-agent's task, stamped with ``taskId``).  Also locks the Stop
// wiring: Stop on a sub-agent tab posts ``stop`` with the SUB-AGENT's
// tab id only.
//
// Drives the real ``media/main.js`` against the real
// ``media/chat.html`` markup in jsdom with the exact backend event
// shapes captured from a live daemon run (new_tab → resumeSession →
// openSubagentTab → status → prompt echo), mirroring the harness of
// ``subagentRunningInput.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/subagentPromptTrajectory.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/** Build a jsdom window running the production chat webview. */
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

/** Click the tab bar entry for *tabId* exactly like a user would. */
function clickTab(win, tabId) {
  const el = Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab'),
  ).find(e => e.dataset.tabId === tabId);
  assert.ok(el, 'tab ' + tabId + ' must be in the tab bar');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** All rendered Prompt panels whose body contains *text*. */
function promptPanels(win, text) {
  return Array.from(win.document.querySelectorAll('.ev.prompt')).filter(el =>
    (el.textContent || '').includes(text),
  );
}

/**
 * Boot a webview with a running parent whose agent spawned one
 * sub-agent via ``run_parallel`` — the exact live broadcast sequence.
 */
function bootWithSubagent() {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const parentId = ready.tabId;

  send(win, {type: 'status', running: true, tabId: parentId,
    startTs: Date.now()});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 1'])},
  });
  const before = posted.length;
  send(win, {type: 'new_tab', task_id: 'sub-task-1',
    parent_tab_id: parentId, taskId: ''});
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === 'sub-task-1');
  assert.ok(resume, 'new_tab must make the webview post resumeSession');
  const subTabId = resume.tabId;
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    isSubagentTab: true,
    isDone: false,
  });
  send(win, {type: 'status', running: true, tabId: subTabId, startTs: 0});
  return {win, posted, parentId, subTabId};
}

// ---------------------------------------------------------------------------
// 1. Typing + Send on a running sub-agent tab posts appendUserMessage
//    with the SUB-AGENT's tab id.
// ---------------------------------------------------------------------------
function testSendPostsAppendUserMessageForSubTab() {
  const {win, posted, subTabId} = bootWithSubagent();
  clickTab(win, subTabId);

  const inp = win.document.getElementById('task-input');
  assert.ok(inp, '#task-input must exist');
  inp.value = 'HELLO SUBAGENT INJECTION';
  const before = posted.length;
  win.document.getElementById('send-btn').click();
  const appended = posted
    .slice(before)
    .find(m => m.type === 'appendUserMessage');
  assert.ok(appended, 'Send on a running sub-agent tab must post ' +
    'appendUserMessage (not submit)');
  assert.strictEqual(appended.tabId, subTabId,
    'the injected prompt must be routed to the SUB-AGENT tab');
  assert.strictEqual(appended.prompt, 'HELLO SUBAGENT INJECTION');
  assert.ok(
    !posted.slice(before).some(m => m.type === 'submit'),
    'no fresh submit must be posted for a running sub-agent tab',
  );
  console.log('ok - send posts appendUserMessage with the sub tab id');
}

// ---------------------------------------------------------------------------
// 2. The backend's echoed prompt event (tabId + taskId stamped) renders
//    a Prompt panel in the ACTIVE sub-agent tab's trajectory.
// ---------------------------------------------------------------------------
function testEchoedPromptRendersInSubagentTrajectory() {
  const {win, subTabId} = bootWithSubagent();
  clickTab(win, subTabId);

  send(win, {
    type: 'prompt',
    text: 'HELLO SUBAGENT INJECTION',
    tabId: subTabId,
    taskId: 'sub-task-1',
  });
  assert.strictEqual(
    promptPanels(win, 'HELLO SUBAGENT INJECTION').length,
    1,
    'the injected prompt must appear as a Prompt panel in the ' +
      'sub-agent tab trajectory',
  );
  console.log('ok - echoed prompt renders in the sub-agent trajectory');
}

// ---------------------------------------------------------------------------
// 3. The injected prompt SURVIVES switching away and back (tab restore).
// ---------------------------------------------------------------------------
function testInjectedPromptSurvivesTabSwitch() {
  const {win, parentId, subTabId} = bootWithSubagent();
  clickTab(win, subTabId);
  send(win, {
    type: 'prompt',
    text: 'HELLO SUBAGENT INJECTION',
    tabId: subTabId,
    taskId: 'sub-task-1',
  });

  clickTab(win, parentId);
  clickTab(win, subTabId);
  assert.strictEqual(
    promptPanels(win, 'HELLO SUBAGENT INJECTION').length,
    1,
    'the injected prompt must still be in the sub-agent trajectory ' +
      'after switching tabs away and back',
  );
  console.log('ok - injected prompt survives a tab switch');
}

// ---------------------------------------------------------------------------
// 4. A task_events replay containing the recorded prompt re-renders it
//    (backend now records the echo under the sub-agent's task).
// ---------------------------------------------------------------------------
function testReplayRendersRecordedPrompt() {
  const {win, subTabId} = bootWithSubagent();
  clickTab(win, subTabId);

  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: 'sub-task-1',
    events: [
      {type: 'prompt', text: 'sub 1', taskId: 'sub-task-1'},
      {
        type: 'prompt',
        text: 'HELLO SUBAGENT INJECTION',
        taskId: 'sub-task-1',
      },
    ],
  });
  assert.ok(
    promptPanels(win, 'HELLO SUBAGENT INJECTION').length >= 1,
    'a task_events replay must re-render the recorded injected prompt',
  );
  console.log('ok - task_events replay renders the recorded prompt');
}

// ---------------------------------------------------------------------------
// 5. Stop on a running sub-agent tab posts stop with the SUB tab id.
// ---------------------------------------------------------------------------
function testStopPostsSubagentTabId() {
  const {win, posted, subTabId} = bootWithSubagent();
  clickTab(win, subTabId);

  const stopBtn = win.document.getElementById('stop-btn');
  assert.ok(stopBtn, '#stop-btn must exist');
  const before = posted.length;
  stopBtn.click();
  const stops = posted.slice(before).filter(m => m.type === 'stop');
  assert.strictEqual(stops.length, 1, 'exactly one stop must be posted');
  assert.strictEqual(
    stops[0].tabId,
    subTabId,
    'Stop on a sub-agent tab must stop ONLY that sub-agent',
  );
  console.log('ok - stop posts the sub-agent tab id only');
}

function main() {
  testSendPostsAppendUserMessageForSubTab();
  testEchoedPromptRendersInSubagentTrajectory();
  testInjectedPromptSurvivesTabSwitch();
  testReplayRendersRecordedPrompt();
  testStopPostsSubagentTabId();
  console.log('All subagentPromptTrajectory tests passed.');
}

main();
// main.js installs webview timers (status clock, ghost-text debounce)
// that keep the node event loop alive; exit explicitly once every
// assertion above has passed.
process.exit(0);
