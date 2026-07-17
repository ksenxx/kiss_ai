// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: adjacent-task overscroll navigation ("side scrolling
// of chats") in the chat webview.  When the tab's chat-id has multiple
// tasks, overscrolling at the top/bottom of ``#output`` must request
// and render the previous/next task of the chat (Cursor-style).
//
// The regression: commit df31a0e8 added ``_broadcast_early_prompts``
// (task_runner.py) which streams optimistic ``system_prompt``/``prompt``
// panel events with ``taskId: ''`` (EMPTY STRING) at submit time, BEFORE
// the task's DB row exists.  In ``media/main.js`` the message-switch
// default-case taskId-adoption block only guarded against ``undefined``
// and ``null``, so the empty string was adopted into ``currentTaskId``
// and — because both scroll anchors were still null right after
// ``setTaskText`` reset them — seeded ``oldestLoadedTaskId`` and
// ``newestLoadedTaskId`` to ``''``.  The real taskId that streams in
// moments later only re-seeds the anchors when BOTH are still null, so
// they stayed ``''`` forever.  Every subsequent overscroll then posted
// ``getAdjacentTask`` with ``taskId: ''``, which the backend resolves
// to "no such task" and answers with an EMPTY ``adjacent_task_events``,
// which in turn latched ``noPrevTask``/``noNextTask`` — permanently
// killing adjacent-task scrolling for the tab.
//
// A secondary bug is also covered: ``task_events`` replays that carry a
// valid ``task_id`` but an empty/missing ``task`` title (server.py's
// resume-race replay path) never synced ``currentTaskId`` nor re-seeded
// the anchors, so overscroll stayed blocked after such a replay.
//
// This test exercises the real ``media/main.js`` against the real
// ``media/chat.html`` in jsdom, the same harness as
// ``sideScrollWhileRunning.test.js`` / ``bashHeaderCyan.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/adjacentTaskScroll.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom webview running the real chat.html + panelCopy.js +
 * main.js, with a stubbed acquireVsCodeApi that records every message
 * posted to the extension host.
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
  // jsdom has no Element.scrollTo; main.js's rAF auto-scroll calls it.
  win.Element.prototype.scrollTo = function () {};
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: (msg) => posted.push(msg),
      getState: () => state,
      setState: (s) => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

/** Deliver a message-event from the extension host to the webview. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Give #output fixed scroll geometry (jsdom has no layout engine). */
function fakeGeometry(el) {
  Object.defineProperty(el, 'scrollWidth', {value: 2000, configurable: true});
  Object.defineProperty(el, 'clientWidth', {value: 400, configurable: true});
  Object.defineProperty(el, 'scrollHeight', {value: 3000, configurable: true});
  Object.defineProperty(el, 'clientHeight', {value: 500, configurable: true});
}

/** Fire `n` vertical wheel events on `O` (negative deltaY = up). */
function wheel(win, O, deltaY, n) {
  for (let i = 0; i < n; i++) {
    O.dispatchEvent(
      new win.WheelEvent('wheel', {deltaY, bubbles: true, cancelable: true}),
    );
  }
}

/** Boot a webview with one loaded history task (task_id '42'). */
function setupWithHistoryTask() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  const O = win.document.getElementById('output');
  fakeGeometry(O);
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-abc',
    task_id: '42',
    task: 'My old task',
    events: [
      {type: 'task_start', task: 'My old task'},
      {type: 'system_output', text: 'hello\n'},
    ],
  });
  return {win, posted, tabId, O};
}

function getAdjacent(posted) {
  return posted.filter((m) => m.type === 'getAdjacentTask');
}

// ---------------------------------------------------------------------------
// 1. REGRESSION: early prompt events with taskId:'' must not poison the
//    adjacent-task anchors.  Full live-task lifecycle, then overscroll up
//    must request the previous task relative to the REAL task id '123'.
// ---------------------------------------------------------------------------
function testEarlyPromptPoisoning() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  const O = win.document.getElementById('output');
  fakeGeometry(O);
  // Live submit flow as broadcast by the backend:
  send(win, {type: 'setTaskText', text: 'My new task', tabId});
  send(win, {type: 'status', running: true, tabId});
  // EARLY optimistic prompt panels (_broadcast_early_prompts) with
  // taskId:'' — the DB row does not exist yet.
  send(win, {type: 'system_prompt', text: 'sys', tabId, taskId: '', early: true});
  send(win, {type: 'prompt', text: 'My new task', tabId, taskId: '', early: true});
  // Real streamed events with the actual DB row id.
  send(win, {type: 'system_prompt', text: 'sys-real', tabId, taskId: '123'});
  send(win, {type: 'prompt', text: 'My new task', tabId, taskId: '123'});
  send(win, {type: 'system_output', text: 'working\n', tabId, taskId: '123'});
  send(win, {type: 'taskExecuted', tabId, taskId: '123'});
  send(win, {type: 'task_done', tabId, taskId: '123'});
  send(win, {type: 'status', running: false, tabId});
  // Overscroll up at top.
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  const adj = getAdjacent(posted);
  assert.ok(
    adj.length > 0,
    'overscroll after a live task with early taskId:"" prompts must still ' +
      'request the previous task (anchors were poisoned to "")',
  );
  assert.strictEqual(
    adj[0].taskId,
    '123',
    `getAdjacentTask must carry the REAL task id, got ${JSON.stringify(adj[0])}`,
  );
  assert.strictEqual(adj[0].direction, 'prev');
  win.close();
  console.log('PASS early-prompt taskId:"" does not poison adjacent anchors');
}

// ---------------------------------------------------------------------------
// 2. Never post getAdjacentTask with an empty taskId: an empty-string
//    anchor must not produce a request the backend resolves to "no such
//    task" (whose empty reply would latch noPrevTask forever).
// ---------------------------------------------------------------------------
function testNoEmptyTaskIdRequest() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  const O = win.document.getElementById('output');
  fakeGeometry(O);
  send(win, {type: 'setTaskText', text: 'My new task', tabId});
  send(win, {type: 'status', running: true, tabId});
  // ONLY the early empty-taskId events arrive (real id never streams —
  // worst case). Overscroll must post NOTHING rather than taskId:''.
  send(win, {type: 'system_prompt', text: 'sys', tabId, taskId: '', early: true});
  send(win, {type: 'prompt', text: 'My new task', tabId, taskId: '', early: true});
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  const bad = getAdjacent(posted).filter(
    (m) => m.taskId === '' || m.taskId === null || m.taskId === undefined,
  );
  assert.strictEqual(
    bad.length,
    0,
    'getAdjacentTask must never be posted with an empty/unknown taskId: ' +
      JSON.stringify(bad),
  );
  win.close();
  console.log('PASS no getAdjacentTask posted with empty taskId');
}

// ---------------------------------------------------------------------------
// 3. task_events replay with a valid task_id but NO task title (resume-race
//    replay path) must still enable adjacent overscroll.
// ---------------------------------------------------------------------------
function testTaskEventsWithoutTitle() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  const O = win.document.getElementById('output');
  fakeGeometry(O);
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '42',
    events: [{type: 'system_output', text: 'b\n'}],
  });
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  const adj = getAdjacent(posted);
  assert.ok(
    adj.length > 0 && adj[0].taskId === '42',
    'task_events with task_id but no task title must still allow ' +
      'adjacent overscroll; got ' + JSON.stringify(adj),
  );
  win.close();
  console.log('PASS task_events without task title still enables overscroll');
}

// ---------------------------------------------------------------------------
// 4. History-load + wheel-up overscroll requests prev, and the
//    adjacent_task_events reply renders an .adjacent-task container.
//    Then a further overscroll chains off the NEW oldest task id.
// ---------------------------------------------------------------------------
function testPrevRequestRenderAndChain() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  let adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'wheel overscroll at top must request prev task');
  assert.strictEqual(adj[0].taskId, '42');
  assert.strictEqual(adj[0].direction, 'prev');
  // Backend reply: render the previous task above the current one.
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Older task'},
      {type: 'system_output', text: 'older\n'},
    ],
  });
  const cont = O.querySelector('.adjacent-task[data-task]');
  assert.ok(cont, 'adjacent task container must render after reply');
  // Chained: next overscroll must key off the NEW oldest id '41'.
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  adj = getAdjacent(posted);
  assert.ok(adj.length >= 2, 'second overscroll must post another request');
  assert.strictEqual(
    adj[adj.length - 1].taskId,
    '41',
    'chained overscroll must use the newly loaded oldest task id',
  );
  win.close();
  console.log('PASS prev request + render + chained prev');
}

// ---------------------------------------------------------------------------
// 5. Overscroll down at the bottom requests the NEXT task.
// ---------------------------------------------------------------------------
function testNextAtBottom() {
  const {win, posted, O} = setupWithHistoryTask();
  O.scrollTop = O.scrollHeight - O.clientHeight; // at bottom
  wheel(win, O, 50, 10);
  const adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'overscroll at bottom must request next task');
  assert.strictEqual(adj[0].direction, 'next');
  assert.strictEqual(adj[0].taskId, '42');
  win.close();
  console.log('PASS next request at bottom');
}

// ---------------------------------------------------------------------------
// 6. Touch path: a downward finger drag at the top requests prev.
// ---------------------------------------------------------------------------
function testTouchPrev() {
  const {win, posted, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  function fire(type, y) {
    const e = new win.Event(type, {bubbles: true});
    Object.defineProperty(e, 'touches', {value: [{clientY: y, clientX: 100}]});
    O.dispatchEvent(e);
  }
  fire('touchstart', 500);
  for (let y = 520; y <= 720; y += 20) fire('touchmove', y);
  const adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'touch pull-down at top must request prev task');
  assert.strictEqual(adj[0].direction, 'prev');
  assert.strictEqual(adj[0].taskId, '42');
  win.close();
  console.log('PASS touch pull-down requests prev');
}

// ---------------------------------------------------------------------------
// 7. REGRESSION: a previous task that EXISTS (valid task_id) but has a
//    very short/empty trajectory (events: []) must NOT latch noPrevTask.
//    It must render a placeholder container and a further overscroll
//    must chain PAST it (request the task before the short one) — the
//    reported bug: "if the previous task has a very short trajectory,
//    I cannot scroll to the last task".
// ---------------------------------------------------------------------------
function testShortPrevTaskDoesNotBlockChaining() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  let adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'wheel overscroll at top must request prev task');
  assert.strictEqual(adj[0].taskId, '42');
  // Backend reply: the previous task exists but recorded NO events.
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Short task',
    task_id: '41',
    events: [],
  });
  const cont = O.querySelector('.adjacent-task[data-task-id="41"]');
  assert.ok(
    cont,
    'an adjacent task with an empty trajectory must still render a ' +
      'container (placeholder), not be silently dropped',
  );
  assert.ok(
    cont.querySelector('.adjacent-task-placeholder'),
    'empty-trajectory adjacent task must render a visible placeholder',
  );
  // The critical part: overscroll again — navigation must chain past
  // the short task using its id '41', not be dead (noPrevTask latched).
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  adj = getAdjacent(posted);
  assert.ok(
    adj.length >= 2,
    'overscroll after a short-trajectory prev task must request the ' +
      'task before it (noPrevTask must NOT be latched); got ' +
      JSON.stringify(adj),
  );
  assert.strictEqual(
    adj[adj.length - 1].taskId,
    '41',
    'chained overscroll must key off the short task id',
  );
  assert.strictEqual(adj[adj.length - 1].direction, 'prev');
  win.close();
  console.log('PASS short-trajectory prev task does not block chaining');
}

// ---------------------------------------------------------------------------
// 8. Same for the NEXT direction: a next task with an empty trajectory
//    must not latch noNextTask; further overscroll-down chains past it.
// ---------------------------------------------------------------------------
function testShortNextTaskDoesNotBlockChaining() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = O.scrollHeight - O.clientHeight; // at bottom
  wheel(win, O, 50, 10);
  let adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'overscroll at bottom must request next task');
  assert.strictEqual(adj[0].taskId, '42');
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'next',
    task: 'Short next task',
    task_id: '43',
    events: [],
  });
  assert.ok(
    O.querySelector('.adjacent-task[data-task-id="43"]'),
    'empty-trajectory next task must still render a container',
  );
  O.scrollTop = O.scrollHeight - O.clientHeight;
  wheel(win, O, 50, 10);
  adj = getAdjacent(posted);
  assert.ok(
    adj.length >= 2,
    'overscroll after a short-trajectory next task must request the ' +
      'task after it (noNextTask must NOT be latched); got ' +
      JSON.stringify(adj),
  );
  assert.strictEqual(adj[adj.length - 1].taskId, '43');
  assert.strictEqual(adj[adj.length - 1].direction, 'next');
  win.close();
  console.log('PASS short-trajectory next task does not block chaining');
}

// ---------------------------------------------------------------------------
// 9. Genuine end-of-chat (task:'' AND task_id:null) must STILL latch
//    noPrevTask: further overscroll must not spam getAdjacentTask.
// ---------------------------------------------------------------------------
function testGenuineEndOfChatStillLatches() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  let adj = getAdjacent(posted);
  assert.strictEqual(adj.length, 1);
  // Backend found no adjacent row: task '' and task_id null.
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: '',
    task_id: null,
    events: [],
  });
  assert.strictEqual(
    O.querySelector('.adjacent-task'),
    null,
    'a genuine no-more-tasks reply must not render a container',
  );
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  adj = getAdjacent(posted);
  assert.strictEqual(
    adj.length,
    1,
    'after a genuine end-of-chat reply, further overscroll must NOT ' +
      'post more getAdjacentTask requests',
  );
  win.close();
  console.log('PASS genuine end-of-chat still latches noPrevTask');
}

// ---------------------------------------------------------------------------
// 10. taskDeleted removes an empty-trajectory placeholder container too.
// ---------------------------------------------------------------------------
function testTaskDeletedRemovesPlaceholder() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Short task',
    task_id: '41',
    events: [],
  });
  assert.ok(O.querySelector('.adjacent-task[data-task-id="41"]'));
  send(win, {type: 'taskDeleted', chatId: 'chat-abc', taskId: '41', chatHasMoreTasks: true});
  assert.strictEqual(
    O.querySelector('.adjacent-task[data-task-id="41"]'),
    null,
    'taskDeleted must remove the placeholder adjacent-task container',
  );
  win.close();
  console.log('PASS taskDeleted removes placeholder container');
}

// ---------------------------------------------------------------------------
// 11. A real adjacent row with a valid id but an EMPTY title must still
//     render (label '(untitled task)'), not latch noPrevTask, and must
//     not target every panel via applyChevronState(…, '').
// ---------------------------------------------------------------------------
function testEmptyTitleTaskStillChains() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: '',
    task_id: '41',
    events: [],
  });
  const cont = O.querySelector('.adjacent-task[data-task-id="41"]');
  assert.ok(
    cont,
    'a real adjacent row with an empty title must still render a container',
  );
  assert.strictEqual(
    cont.dataset.task,
    '(untitled task)',
    'empty-title adjacent task must get a non-empty display label',
  );
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  const adj = getAdjacent(posted);
  assert.ok(
    adj.length >= 2 && adj[adj.length - 1].taskId === '41',
    'empty-title real task must not latch noPrevTask; got ' +
      JSON.stringify(adj),
  );
  win.close();
  console.log('PASS empty-title adjacent task still chains');
}

// ---------------------------------------------------------------------------
// 12. A trajectory whose only events are non-rendering terminal markers
//     (e.g. just task_done) must render the placeholder too — the
//     replay loop skips terminal events, leaving zero visible content.
// ---------------------------------------------------------------------------
function testTerminalOnlyTrajectoryRendersPlaceholder() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Terminal-only task',
    task_id: '41',
    events: [{type: 'task_done'}],
  });
  const cont = O.querySelector('.adjacent-task[data-task-id="41"]');
  assert.ok(cont, 'terminal-only trajectory must still render a container');
  assert.ok(
    cont.querySelector('.adjacent-task-placeholder'),
    'terminal-only trajectory must render the placeholder (replay ' +
      'produced no visible content)',
  );
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  const adj = getAdjacent(posted);
  assert.ok(
    adj.length >= 2 && adj[adj.length - 1].taskId === '41',
    'terminal-only trajectory must not block chaining; got ' +
      JSON.stringify(adj),
  );
  win.close();
  console.log('PASS terminal-only trajectory renders placeholder + chains');
}

testEarlyPromptPoisoning();
testNoEmptyTaskIdRequest();
testTaskEventsWithoutTitle();
testPrevRequestRenderAndChain();
testNextAtBottom();
testTouchPrev();
testShortPrevTaskDoesNotBlockChaining();
testShortNextTaskDoesNotBlockChaining();
testGenuineEndOfChatStillLatches();
testTaskDeletedRemovesPlaceholder();
testEmptyTitleTaskStillChains();
testTerminalOnlyTrajectoryRendersPlaceholder();
console.log('All adjacentTaskScroll tests passed');
