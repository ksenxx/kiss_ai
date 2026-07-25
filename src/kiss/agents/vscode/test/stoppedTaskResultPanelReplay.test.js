// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end chat-webview regression test: a stopped/failed task's
// terminal Result panel must survive replay.  Drives the real
// chat.html + panelCopy.js + main.js in jsdom.
//
// Backend context: task_runner's failure/stop paths broadcast their
// terminal {"type":"result"} event stamped with tabId; before the fix
// WebPrinter.broadcast never persisted tabId-stamped events, so the
// task's persisted stream ended `tool_call -> task_stopped ->
// followup_suggestion` with NO result row and every replay (webview
// reload, history load, adjacent-task scroll) showed no Result panel.
// The fix persists a tabId-stripped copy (with taskId) of that event;
// these tests assert the webview renders it in every replay path and
// documents the pre-fix symptom.

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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage() {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

const TASK_ID = 'stopped-task-1';

// The persisted stream of a user-stopped task BEFORE the fix (mirrors
// the real rows observed in ~/.kiss/sorcar.db): no result event.
function stoppedTaskEventsWithoutResult() {
  return [
    {type: 'prompt', text: 'review the repo', ts: 1000, taskId: TASK_ID},
    {type: 'text_delta', delta: 'Looking at the repo…', ts: 1500, taskId: TASK_ID},
    {
      type: 'tool_call',
      name: 'run_parallel',
      extras: {tasks: '["review A", "review B"]'},
      ts: 2000,
      taskId: TASK_ID,
    },
    {type: 'task_stopped', ts: 3000},
    {type: 'followup_suggestion', text: 'Resume the stopped task.', ts: 3001},
  ];
}

// The same stream AFTER the fix: the tabId-stripped terminal result
// (exactly what task_runner broadcasts and WebPrinter now persists)
// appears before the task_stopped marker.
function stoppedTaskEventsWithResult() {
  const evs = stoppedTaskEventsWithoutResult();
  evs.splice(3, 0, {
    type: 'result',
    text: 'Task stopped by user',
    success: false,
    total_tokens: 0,
    cost: '$0.0000',
    step_count: 0,
    ts: 2500,
    taskId: TASK_ID,
  });
  return evs;
}

function resultPanels(win) {
  return Array.from(win.document.querySelectorAll('.ev.rc'));
}

function testSymptomNoResultRowMeansNoPanel() {
  // Documents the pre-fix persisted data: replaying a stopped task's
  // stream WITHOUT a result row renders no Result panel — the exact
  // "why is the result event panel not shown in the last task" bug.
  const win = makeWebview();
  send(win, {
    type: 'task_events',
    task: 'review the repo',
    task_id: TASK_ID,
    events: stoppedTaskEventsWithoutResult(),
  });
  assert.strictEqual(
    resultPanels(win).length,
    0,
    'a stream with no persisted result row must render no Result panel',
  );
  console.log('  ok - pre-fix stream (no result row) reproduces the symptom');
}

function testHistoryReplayShowsStoppedResultPanel() {
  const win = makeWebview();
  send(win, {
    type: 'task_events',
    task: 'review the repo',
    task_id: TASK_ID,
    events: stoppedTaskEventsWithResult(),
  });
  const panels = resultPanels(win);
  assert.strictEqual(panels.length, 1, 'history replay must render 1 Result panel');
  const txt = panels[0].textContent;
  assert.ok(txt.includes('Task stopped by user'), 'panel shows the stop summary');
  assert.ok(txt.includes('Status: FAILED'), 'panel shows the FAILED status');
  console.log('  ok - history replay renders the stopped-task Result panel');
}

function testLiveStopShowsResultPanel() {
  // Live path: the tabId+taskId-stamped terminal result reaches the
  // active tab after streaming events adopted the task id.
  const win = makeWebview();
  for (const ev of stoppedTaskEventsWithResult()) {
    send(win, Object.assign({}, ev));
  }
  const panels = resultPanels(win);
  assert.strictEqual(panels.length, 1, 'live stop must render 1 Result panel');
  assert.ok(panels[0].textContent.includes('Task stopped by user'));
  console.log('  ok - live stop renders the stopped-task Result panel');
}

function testAdjacentTaskReplayShowsStoppedResultPanel() {
  // Adjacent-task overscroll uses the same persisted stream; the
  // stopped task's Result panel must render inside the adjacent block.
  const win = makeWebview();
  send(win, {
    type: 'task_events',
    task: 'current task',
    task_id: 'current-1',
    events: [
      {type: 'prompt', text: 'current task', ts: 5000, taskId: 'current-1'},
      {
        type: 'result',
        summary: 'done',
        success: true,
        total_tokens: 1,
        cost: '$0.01',
        ts: 5001,
        taskId: 'current-1',
      },
    ],
  });
  send(win, {
    type: 'adjacent_task_events',
    direction: 'prev',
    task: 'review the repo',
    task_id: TASK_ID,
    events: stoppedTaskEventsWithResult(),
  });
  const adjacent = win.document.querySelector('.adjacent-task');
  assert.ok(adjacent, 'adjacent task block rendered');
  const panels = adjacent.querySelectorAll('.ev.rc');
  assert.strictEqual(
    panels.length,
    1,
    'adjacent-task replay must render the stopped-task Result panel',
  );
  assert.ok(panels[0].textContent.includes('Task stopped by user'));
  console.log('  ok - adjacent-task replay renders the stopped-task Result panel');
}

function main() {
  testSymptomNoResultRowMeansNoPanel();
  testHistoryReplayShowsStoppedResultPanel();
  testLiveStopShowsResultPanel();
  testAdjacentTaskReplayShowsStoppedResultPanel();
  console.log('stoppedTaskResultPanelReplay.test.js: all assertions passed.');
}

// Explicit exit: each jsdom window keeps main.js interval timers
// alive, so without process.exit the node process (and the npm test
// chain that runs this file) would hang forever after the assertions
// pass.
try {
  main();
  process.exit(0);
} catch (err) {
  console.error(err);
  process.exit(1);
}
