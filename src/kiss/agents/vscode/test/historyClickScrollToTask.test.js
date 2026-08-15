// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Clicking a task in the Task History sidebar must land the reader ON that
// task: after switching to (or loading) the tab that shows the task's chat,
// the transcript is scrolled so the clicked task's region is what the reader
// sees, the static task panel (#task-panel-text) names the clicked task, and
// the chat shows the clicked task's events.
//
// Three scenarios:
//   1. The clicked task is already spliced into the open tab's transcript as
//      an `.adjacent-task` neighbour -> scroll to it, no resumeSession.
//   2. The clicked task's events are NOT loaded in the open tab -> the tab is
//      replayed at that task via resumeSession({id, taskId, tabId}).
//   3. The clicked task IS the tab's own task -> scroll back to it, and no
//      resumeSession is issued (parity with historyClickSwitchExistingChat).

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function disableWorkspaceFilter(win) {
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  const ws = win.document.getElementById('hf-workspace');
  if (ws && ws.checked) {
    ws.checked = false;
    ws.dispatchEvent(new win.Event('change', {bubbles: true}));
  }
}

function chatTabs(win) {
  return Array.from(win.document.querySelectorAll('#tab-list .chat-tab'));
}

function taskPanelText(win) {
  const el = win.document.getElementById('task-panel-text');
  return el ? el.textContent : '';
}

function resumeMessages(posted) {
  return posted.filter(msg => msg && msg.type === 'resumeSession');
}

function historyRow(win, idx) {
  const rows = win.document.querySelectorAll('#history-list .sidebar-item');
  return rows[idx];
}

function historySession(id, taskId, title) {
  return {
    id,
    task_id: taskId,
    title,
    preview: title,
    has_events: true,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    timestamp: 1_700_000_000,
    work_dir: '',
    startTs: 1_700_000_000_000,
    endTs: 1_700_000_001_000,
  };
}

// jsdom computes no layout, so give the transcript a fake geometry that
// tracks O.scrollTop: the neighbour occupies content rows [0, 1000) and the
// tab's own task rows [1000, 2000), with a 500px viewport.
function fakeRegionGeometry(win, O) {
  Object.defineProperty(O, 'scrollHeight', {value: 2000, configurable: true});
  Object.defineProperty(O, 'clientHeight', {value: 500, configurable: true});
  O.getBoundingClientRect = () => ({top: 0, bottom: 500, left: 0, right: 400});
  const container = O.querySelector('.adjacent-task');
  container.getBoundingClientRect = () => ({
    top: 0 - O.scrollTop,
    bottom: 1000 - O.scrollTop,
    left: 0,
    right: 400,
  });
  for (const el of O.children) {
    if (el === container) continue;
    el.getBoundingClientRect = () => ({
      top: 1000 - O.scrollTop,
      bottom: 2000 - O.scrollTop,
      left: 0,
      right: 400,
    });
  }
}

// One open tab on chat-1 showing its own "Task B" (id 42) with a previous
// "Task A" (id 41) spliced in above it as an adjacent region.
function setupTabWithNeighbour() {
  const {win, posted} = makeWebview();
  disableWorkspaceFilter(win);
  const tabId = posted.find(msg => msg && msg.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '42',
    task: 'Task B',
    events: [
      {type: 'task_start', task: 'Task B'},
      {type: 'system_output', text: 'output of task B\n'},
    ],
  });
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Task A',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Task A'},
      {type: 'system_output', text: 'output of task A\n'},
    ],
  });
  const O = win.document.getElementById('output');
  assert.ok(
    O.querySelector('.adjacent-task[data-task-id="41"]'),
    'sanity: the neighbour task must be spliced into the transcript',
  );
  // Park the reader on the tab's own task, as after a real splice.
  fakeRegionGeometry(win, O);
  O.scrollTop = 1200;
  O.dispatchEvent(new win.Event('scroll'));
  assert.strictEqual(
    taskPanelText(win),
    'Task B',
    'sanity: the panel names the tab own task before the history click',
  );
  return {win, posted, tabId, O};
}

function testClickScrollsToSplicedNeighbour() {
  const {win, posted, tabId, O} = setupTabWithNeighbour();
  const resumeBefore = resumeMessages(posted).length;

  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [historySession('chat-1', 41, 'Task A')],
  });
  historyRow(win, 0).click();

  assert.strictEqual(
    chatTabs(win).length,
    1,
    'clicking a task of the already-open chat must not open a new tab',
  );
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    'the open tab showing the chat must stay active',
  );
  assert.strictEqual(
    resumeMessages(posted).length,
    resumeBefore,
    'a task already spliced into the transcript needs no resumeSession',
  );
  assert.strictEqual(
    O.scrollTop,
    0,
    'the transcript must scroll the clicked task region to the top',
  );
  assert.strictEqual(
    taskPanelText(win),
    'Task A',
    'the static task panel must name the clicked task after the scroll',
  );
  win.close();
  console.log('PASS history click scrolls to the spliced neighbour task');
}

function testClickScrollsBackToOwnTask() {
  const {win, posted, tabId, O} = setupTabWithNeighbour();
  // Park the reader on the neighbour first, as a real scroll would.
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      historySession('chat-1', 41, 'Task A'),
      historySession('chat-1', 42, 'Task B'),
    ],
  });
  historyRow(win, 0).click();
  assert.strictEqual(taskPanelText(win), 'Task A', 'sanity: parked on Task A');
  const resumeBefore = resumeMessages(posted).length;

  historyRow(win, 1).click();

  assert.strictEqual(
    resumeMessages(posted).length,
    resumeBefore,
    'the tab own task is already loaded: no resumeSession',
  );
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    'the open tab showing the chat must stay active',
  );
  assert.strictEqual(
    O.scrollTop,
    1000,
    'the transcript must scroll the own task region to the top',
  );
  assert.strictEqual(
    taskPanelText(win),
    'Task B',
    'clicking the tab own task must scroll back and rename the panel',
  );
  win.close();
  console.log('PASS history click scrolls back to the tab own task');
}

function testClickLoadsTaskNotInTranscript() {
  const {win, posted, tabId} = setupTabWithNeighbour();
  const resumeBefore = resumeMessages(posted).length;

  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [historySession('chat-1', 40, 'Task Z')],
  });
  historyRow(win, 0).click();

  assert.strictEqual(
    chatTabs(win).length,
    1,
    'loading another task of the open chat must reuse its tab',
  );
  const resumes = resumeMessages(posted);
  assert.strictEqual(
    resumes.length,
    resumeBefore + 1,
    'a task whose events are not in the transcript must be fetched',
  );
  const resume = resumes[resumes.length - 1];
  assert.strictEqual(resume.id, 'chat-1', 'resumeSession must name the chat');
  assert.strictEqual(resume.taskId, 40, 'resumeSession must name the task');
  assert.strictEqual(
    resume.tabId,
    tabId,
    'resumeSession must replay into the tab already showing the chat',
  );

  // The daemon answers with the clicked task's events; the tab must show
  // them and the panel must name the clicked task.
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '40',
    task: 'Task Z',
    events: [
      {type: 'task_start', task: 'Task Z'},
      {type: 'system_output', text: 'output of task Z\n'},
    ],
  });
  assert.strictEqual(
    taskPanelText(win),
    'Task Z',
    'the static task panel must name the loaded task',
  );
  const out = win.document.getElementById('output').textContent;
  assert.ok(
    out.includes('output of task Z'),
    'the chat must show the loaded task events',
  );
  win.close();
  console.log('PASS history click loads a task missing from the transcript');
}

function testClickRowWithoutTaskIdJustSwitches() {
  const {win, posted, tabId} = setupTabWithNeighbour();
  const resumeBefore = resumeMessages(posted).length;

  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [historySession('chat-1', null, 'Untitled chat')],
  });
  historyRow(win, 0).click();

  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    'a row without a task id still switches to the open chat tab',
  );
  assert.strictEqual(
    resumeMessages(posted).length,
    resumeBefore,
    'a row without a task id has nothing to fetch: no resumeSession',
  );
  win.close();
  console.log('PASS history row without a task id only switches tabs');
}

// The tab's own task produced no output, so it has no rendered region,
// while a spliced-in neighbour fills the transcript and the reader has
// scrolled into it.  Clicking the own task's history row must reclaim
// the static panel and the status row from the neighbour — without
// issuing a resumeSession, since the own task is already loaded.
function testClickOwnTaskWithoutRegionReclaimsPanel() {
  const {win, posted} = makeWebview();
  disableWorkspaceFilter(win);
  const tabId = posted.find(msg => msg && msg.type === 'ready').tabId;
  win._testApi.hideWelcome();
  // The tab's own "Task B" (id 42) recorded no output at all.
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '42',
    task: 'Task B',
    events: [],
  });
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Task A',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Task A'},
      {type: 'system_output', text: 'output of task A\n'},
    ],
  });
  const O = win.document.getElementById('output');
  const container = O.querySelector('.adjacent-task[data-task-id="41"]');
  assert.ok(
    container,
    'sanity: the neighbour task must be spliced into the transcript',
  );
  // The neighbour is the only rendered region: fake its geometry and give
  // it borrowable metrics, then scroll into it as a real reader would.
  Object.defineProperty(O, 'scrollHeight', {value: 1000, configurable: true});
  Object.defineProperty(O, 'clientHeight', {value: 500, configurable: true});
  O.getBoundingClientRect = () => ({top: 0, bottom: 500, left: 0, right: 400});
  container.getBoundingClientRect = () => ({
    top: 0 - O.scrollTop,
    bottom: 1000 - O.scrollTop,
    left: 0,
    right: 400,
  });
  container.dataset.metricTokens = 'Tokens: 999';
  O.scrollTop = 100;
  O.dispatchEvent(new win.Event('scroll'));
  assert.strictEqual(
    taskPanelText(win),
    'Task A',
    'sanity: the panel names the neighbour once the reader scrolls into it',
  );
  const statusTokens = win.document.getElementById('status-tokens');
  assert.strictEqual(
    statusTokens.textContent,
    'Tokens: 999',
    'sanity: the status row is lent to the neighbour before the click',
  );
  const resumeBefore = resumeMessages(posted).length;

  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [historySession('chat-1', 42, 'Task B')],
  });
  historyRow(win, 0).click();

  assert.strictEqual(
    chatTabs(win).length,
    1,
    'clicking the own task of the open chat must not open a new tab',
  );
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabId,
    'the open tab showing the chat must stay active',
  );
  assert.strictEqual(
    resumeMessages(posted).length,
    resumeBefore,
    'the tab own task is already loaded: no resumeSession',
  );
  assert.strictEqual(
    taskPanelText(win),
    'Task B',
    'the static task panel must be reclaimed for the clicked own task',
  );
  assert.strictEqual(
    statusTokens.textContent,
    '',
    'the status row must show the own task metrics again after the click',
  );
  win.close();
  console.log('PASS history click reclaims the panel for a region-less task');
}

testClickScrollsToSplicedNeighbour();
testClickScrollsBackToOwnTask();
testClickLoadsTaskNotInTranscript();
testClickRowWithoutTaskIdJustSwitches();
testClickOwnTaskWithoutRegionReclaimsPanel();
console.log('All historyClickScrollToTask tests passed');
