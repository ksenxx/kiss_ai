// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for EDITOR-TABS MODE task-status mirroring
// in the chat webview (media/main.js):
//  - `panelTitle` messages carry a `state` field mirroring the internal
//    tab strip's status dot ('' before any task, 'running' while one
//    runs, 'ok'/'fail' after it ends) so the extension host can paint
//    the circle into the EDITOR tab title;
//  - a finishing task posts `revealPanel` so the host brings the
//    panel's editor tab forward (sidebar mode's finished-task switch);
//  - terminal error/stop events post `revealPanel` too;
//  - a task ending in a tab this panel does not own posts nothing;
//  - sidebar (non-editor) mode never posts `revealPanel`.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs || '');
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
  let state;
  win.acquireVsCodeApi = function () {
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
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=editor-tabs-status-main.js',
  );

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byType(posted, type) {
  return posted.filter(m => m.type === type);
}

function lastTitle(posted) {
  const titles = byType(posted, 'panelTitle');
  assert.ok(titles.length > 0, 'at least one panelTitle must be posted');
  return titles[titles.length - 1];
}

const ROOT = 'root-tab-0001';

function editorAttrs() {
  return (
    ' class="editor-tab-mode"' +
    ` data-kiss-tab-id="${ROOT}"` +
    ' data-kiss-tab-title="My chat"'
  );
}

// The daemon's live sequence for one task in *tabId*, exactly as
// commands.py / task_runner.py / json_printer.py broadcast it (the
// terminal task_done carries no success field; the verdict lives on
// the preceding result event).
function startTask(win, tabId, taskId) {
  send(win, {type: 'setTaskText', text: 'do the thing', tabId});
  send(win, {type: 'clear', chat_id: 'chat-' + tabId, tabId});
  send(win, {
    type: 'status',
    running: true,
    tabId,
    startTs: Date.now() - 1000,
    taskId,
  });
}

function endTask(win, tabId, taskId, success) {
  send(win, {
    type: 'result',
    text: success ? 'done' : 'it broke',
    summary: success ? '<p>done</p>' : '<p>it broke</p>',
    success,
    is_continue: false,
    total_tokens: 10,
    cost: '$0.0001',
    step_count: 1,
    tabId,
    taskId,
  });
  send(win, {
    type: 'task_done',
    tabId,
    startTs: Date.now() - 1000,
    endTs: Date.now(),
  });
  send(win, {type: 'status', running: false, tabId, taskId});
}

function testStatusStatesReachTheHost() {
  const {win, posted} = makeWebview(editorAttrs());

  // Boot: no task has run yet.
  assert.strictEqual(
    lastTitle(posted).state,
    '',
    'boot must report the no-task-yet state',
  );

  startTask(win, ROOT, 'task-1');
  assert.strictEqual(
    lastTitle(posted).state,
    'running',
    'a running task must report state "running"',
  );
  assert.strictEqual(lastTitle(posted).tabId, ROOT);

  endTask(win, ROOT, 'task-1', true);
  assert.strictEqual(
    lastTitle(posted).state,
    'ok',
    'a successful task must report state "ok"',
  );

  // A later failed task flips the state to 'fail' (via 'running').
  startTask(win, ROOT, 'task-2');
  assert.strictEqual(lastTitle(posted).state, 'running');
  endTask(win, ROOT, 'task-2', false);
  assert.strictEqual(
    lastTitle(posted).state,
    'fail',
    'a failed task must report state "fail"',
  );

  // ... and a following success flips it back.
  startTask(win, ROOT, 'task-3');
  endTask(win, ROOT, 'task-3', true);
  assert.strictEqual(lastTitle(posted).state, 'ok');

  win.close();
  console.log('  ok - panelTitle carries the running/ok/fail states');
}

function testFinishedTaskRevealsThePanel() {
  const {win, posted} = makeWebview(editorAttrs());

  startTask(win, ROOT, 'task-1');
  assert.strictEqual(
    byType(posted, 'revealPanel').length,
    0,
    'starting a task must not reveal the panel',
  );

  endTask(win, ROOT, 'task-1', true);
  assert.strictEqual(
    byType(posted, 'revealPanel').length,
    1,
    'the finishing task must reveal the panel exactly once',
  );

  win.close();
  console.log('  ok - task_done posts revealPanel');
}

function testTerminalErrorAndStopRevealThePanel() {
  const {win, posted} = makeWebview(editorAttrs());

  startTask(win, ROOT, 'task-1');
  send(win, {
    type: 'task_error',
    tabId: ROOT,
    startTs: Date.now(),
    endTs: Date.now(),
  });
  send(win, {type: 'status', running: false, tabId: ROOT});
  assert.strictEqual(
    byType(posted, 'revealPanel').length,
    1,
    'task_error must reveal the panel',
  );
  assert.strictEqual(
    lastTitle(posted).state,
    'fail',
    'task_error must paint the fail state',
  );

  startTask(win, ROOT, 'task-2');
  send(win, {
    type: 'task_stopped',
    tabId: ROOT,
    startTs: Date.now(),
    endTs: Date.now(),
  });
  send(win, {type: 'status', running: false, tabId: ROOT});
  assert.strictEqual(
    byType(posted, 'revealPanel').length,
    2,
    'task_stopped must reveal the panel too',
  );

  win.close();
  console.log('  ok - task_error / task_stopped post revealPanel');
}

function testForeignTabTaskDoneDoesNotReveal() {
  const {win, posted} = makeWebview(editorAttrs());

  // Another panel's root tab finishing reaches every client, but this
  // panel does not own that tab, so it must stay quiet.
  send(win, {
    type: 'task_done',
    tabId: 'some-other-panels-tab',
    startTs: Date.now(),
    endTs: Date.now(),
  });
  assert.strictEqual(
    byType(posted, 'revealPanel').length,
    0,
    'a foreign tab finishing must not reveal this panel',
  );

  win.close();
  console.log('  ok - a foreign tab finishing does not reveal the panel');
}

function testReplayedTaskRestoresState() {
  const {win, posted} = makeWebview(editorAttrs());

  // A revived/resumed panel gets no live status events — the persisted
  // transcript replays via task_events. The replayed verdict must
  // repaint the editor title's circle.
  send(win, {
    type: 'task_events',
    tabId: ROOT,
    task: 'old failed task',
    task_id: 'task-hist-1',
    events: [
      {type: 'prompt', text: 'old failed task'},
      {
        type: 'result',
        text: 'it broke',
        summary: '<p>it broke</p>',
        success: false,
        is_continue: false,
        total_tokens: 10,
        cost: '$0.0001',
        step_count: 1,
      },
    ],
  });
  assert.strictEqual(
    lastTitle(posted).state,
    'fail',
    'a replayed failed task must restore the fail state',
  );

  send(win, {
    type: 'task_events',
    tabId: ROOT,
    task: 'old good task',
    task_id: 'task-hist-2',
    events: [
      {type: 'prompt', text: 'old good task'},
      {
        type: 'result',
        text: 'done',
        summary: '<p>done</p>',
        success: true,
        is_continue: false,
        total_tokens: 10,
        cost: '$0.0001',
        step_count: 1,
      },
    ],
  });
  assert.strictEqual(
    lastTitle(posted).state,
    'ok',
    'a replayed successful task must restore the ok state',
  );

  win.close();
  console.log('  ok - a task_events replay restores the ok/fail state');
}

function testChatBindingRestoresRunState() {
  const {win, posted} = makeWebview(editorAttrs());

  // A revived panel whose chat already exists (chat ids are allocated
  // by the first run) must show a circle even before any replay
  // arrives: the registry snapshot's chat binding proves a task ran.
  send(win, {
    type: 'tabs_state',
    tabs: [
      {
        tabId: ROOT,
        chatId: 'chat-bound-1',
        title: 'My chat',
        workDir: '',
        scopeWorkDir: '',
      },
    ],
  });
  assert.strictEqual(
    lastTitle(posted).state,
    'ok',
    'a chat-bound root tab must report a run state after reconcile',
  );

  win.close();
  console.log('  ok - a registry chat binding restores the run state');
}

function testSidebarModeNeverPostsReveal() {
  const {win, posted} = makeWebview('');
  const tabId = win._testApi.getActiveTabId();

  startTask(win, tabId, 'task-1');
  endTask(win, tabId, 'task-1', true);
  assert.strictEqual(
    byType(posted, 'revealPanel').length,
    0,
    'sidebar mode must never post revealPanel',
  );
  assert.strictEqual(
    byType(posted, 'panelTitle').length,
    0,
    'sidebar mode must never post panelTitle either',
  );

  win.close();
  console.log('  ok - sidebar mode posts neither revealPanel nor panelTitle');
}

testStatusStatesReachTheHost();
testFinishedTaskRevealsThePanel();
testTerminalErrorAndStopRevealThePanel();
testForeignTabTaskDoneDoesNotReveal();
testReplayedTaskRestoresState();
testChatBindingRestoresRunState();
testSidebarModeNeverPostsReveal();
console.log('editorTabsStatusReveal: all tests passed');
