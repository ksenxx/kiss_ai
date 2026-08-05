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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function fakeGeometry(el) {
  Object.defineProperty(el, 'scrollWidth', {value: 2000, configurable: true});
  Object.defineProperty(el, 'clientWidth', {value: 400, configurable: true});
  Object.defineProperty(el, 'scrollHeight', {value: 3000, configurable: true});
  Object.defineProperty(el, 'clientHeight', {value: 500, configurable: true});
}

function wheel(win, O, deltaY, n) {
  for (let i = 0; i < n; i++) {
    O.dispatchEvent(
      new win.WheelEvent('wheel', {deltaY, bubbles: true, cancelable: true}),
    );
  }
}

function setupWithHistoryTask() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
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

function testEarlyPromptPoisoning() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  const O = win.document.getElementById('output');
  fakeGeometry(O);
  send(win, {type: 'setTaskText', text: 'My new task', tabId});
  send(win, {type: 'status', running: true, tabId});
  send(win, {type: 'system_prompt', text: 'sys', tabId, taskId: '', early: true});
  send(win, {type: 'prompt', text: 'My new task', tabId, taskId: '', early: true});
  send(win, {type: 'system_prompt', text: 'sys-real', tabId, taskId: '123'});
  send(win, {type: 'prompt', text: 'My new task', tabId, taskId: '123'});
  send(win, {type: 'system_output', text: 'working\n', tabId, taskId: '123'});
  send(win, {type: 'taskExecuted', tabId, taskId: '123'});
  send(win, {type: 'task_done', tabId, taskId: '123'});
  send(win, {type: 'status', running: false, tabId});
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

function testNoEmptyTaskIdRequest() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  const O = win.document.getElementById('output');
  fakeGeometry(O);
  send(win, {type: 'setTaskText', text: 'My new task', tabId});
  send(win, {type: 'status', running: true, tabId});
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

function testTaskEventsWithoutTitle() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
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

function testPrevRequestRenderAndChain() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  let adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'wheel overscroll at top must request prev task');
  assert.strictEqual(adj[0].taskId, '42');
  assert.strictEqual(adj[0].direction, 'prev');
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

function testNextAtBottom() {
  const {win, posted, O} = setupWithHistoryTask();
  O.scrollTop = O.scrollHeight - O.clientHeight;
  wheel(win, O, 50, 10);
  const adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'overscroll at bottom must request next task');
  assert.strictEqual(adj[0].direction, 'next');
  assert.strictEqual(adj[0].taskId, '42');
  win.close();
  console.log('PASS next request at bottom');
}

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

function testShortPrevTaskDoesNotBlockChaining() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  let adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'wheel overscroll at top must request prev task');
  assert.strictEqual(adj[0].taskId, '42');
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

function testShortNextTaskDoesNotBlockChaining() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = O.scrollHeight - O.clientHeight;
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

function testGenuineEndOfChatStillLatches() {
  const {win, posted, tabId, O} = setupWithHistoryTask();
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  let adj = getAdjacent(posted);
  assert.strictEqual(adj.length, 1);
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
testEmptyTitleTaskStillChains();
testTerminalOnlyTrajectoryRendersPlaceholder();
console.log('All adjacentTaskScroll tests passed');
