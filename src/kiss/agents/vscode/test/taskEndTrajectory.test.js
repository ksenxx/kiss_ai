// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// When a task ends, the chat webview folds every event panel of the
// task into one collapsible panel called "Trajectory": the transcript
// then reads as the task's text, the Trajectory and the result.  The
// terminal events (task_done, task_error, task_stopped,
// task_interrupted) fold the tab the event names, a status-only end
// folds too, and a transcript REBUILT from stored events (a reload or
// reattach replay, a neighbouring task spliced in) is folded by its
// replay when the task has ended.  A running task's panels stay in the
// open.  These tests drive the real webview end to end through window
// messages, exactly as the daemon does.

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
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=trajectory-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function readyTabId(posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  return ready.tabId;
}

function injectCss(win) {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const styleEl = win.document.createElement('style');
  styleEl.textContent = css;
  win.document.head.appendChild(styleEl);
}

/** Whether *el* and every ancestor up to #output is displayed. */
function isDisplayed(win, el) {
  for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
    if (win.getComputedStyle(n).getPropertyValue('display').trim() === 'none')
      return false;
    if (n.id === 'output') break;
  }
  return true;
}

/** The direct children of *root* that are event panels. */
function eventPanels(root) {
  return Array.from(root.children).filter(
    el => el.classList.contains('ev') || el.classList.contains('llm-panel'),
  );
}

/**
 * Check that *root* holds exactly one Trajectory panel standing
 * between the task's text and its result, collapsed, with every other
 * event panel of the task nested in it.  Returns the panel.
 */
function assertFolded(root, why) {
  const trajectories = Array.from(root.children).filter(el =>
    el.classList.contains('trajectory'),
  );
  assert.strictEqual(trajectories.length, 1, why + ': one Trajectory panel');
  const traj = trajectories[0];
  assert.ok(traj.classList.contains('collapsed'), why + ': collapsed');
  assert.ok(
    traj.classList.contains('collapsible') &&
      traj.classList.contains('ev') &&
      !traj.classList.contains('tc'),
    why + ': a collapsible event panel, not a tool-call panel',
  );
  const hdr = traj.querySelector(':scope > .trajectory-h.collapse-header');
  assert.ok(hdr, why + ': header is the collapse header');
  assert.ok(
    hdr.textContent.startsWith('Trajectory'),
    why + ': header reads Trajectory, got ' + JSON.stringify(hdr.textContent),
  );
  const sub = traj.querySelector(':scope > .trajectory-sub');
  assert.ok(sub && sub.children.length > 0, why + ': holds the event panels');
  const n = sub.children.length;
  assert.strictEqual(
    hdr.querySelector('.collapse-preview').textContent,
    n + (n === 1 ? ' event' : ' events'),
    why + ': preview counts the events',
  );
  for (const el of eventPanels(root)) {
    assert.ok(
      el === traj ||
        el.classList.contains('task-panel') ||
        el.classList.contains('rc'),
      why + ': outside the Trajectory only the task text and the result',
    );
  }
  for (const el of Array.from(sub.children)) {
    assert.ok(
      !el.classList.contains('rc') && !el.classList.contains('task-panel'),
      why + ': neither the result nor the task text is folded',
    );
  }
  const after = [];
  for (let el = traj.nextElementSibling; el; el = el.nextElementSibling) {
    after.push(el);
  }
  assert.ok(
    after.every(
      el =>
        !el.classList.contains('ev') ||
        el.classList.contains('rc') ||
        el.classList.contains('adjacent-task'),
    ),
    why + ': the result follows the Trajectory',
  );
  return traj;
}

/**
 * Stream a small real task: prompt, thoughts, three tools, finish. The
 * stream keeps its newest two panels open, so the Bash panel only folds
 * because two more panels (the Read call and the thoughts its result
 * arms) follow it.
 */
function streamTask(win, tabId) {
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {type: 'setTaskText', text: 'do something', tabId});
  send(win, {type: 'clear', tabId});
  send(win, {type: 'prompt', text: 'do something', tabId});
  send(win, {type: 'thinking_start', tabId});
  send(win, {type: 'thinking_delta', text: 'planning', tabId});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId});
  send(win, {type: 'tool_result', name: 'Bash', content: 'ok', tabId});
  send(win, {type: 'thinking_delta', text: 'checking', tabId});
  send(win, {type: 'tool_call', name: 'Read', path: 'a.txt', tabId});
  send(win, {type: 'tool_result', name: 'Read', content: 'aaa', tabId});
  send(win, {type: 'thinking_delta', text: 'wrapping up', tabId});
  send(win, {
    type: 'tool_call',
    name: 'finish',
    tabId,
    extras: {summary: 'done'},
  });
}

function finishTask(win, tabId) {
  send(win, {type: 'result', summary: '<p>done</p>', success: true, tabId});
  send(win, {
    type: 'task_done',
    tabId,
    startTs: Date.now() - 5000,
    endTs: Date.now(),
  });
  send(win, {type: 'status', running: false, tabId});
}

function clickHeader(win, panel) {
  panel
    .querySelector(':scope > .collapse-header')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

// A task finishing on the visible tab folds every event panel into one
// collapsed Trajectory panel between the task's text and its result;
// a click on its header opens it and shows the panels it holds, a
// second click folds it again.  A trailing event (usage_info) leaves
// the fold alone.
function testVisibleFinishFoldsTrajectory() {
  const {win, posted} = makeWebview();
  injectCss(win);
  const tabId = readyTabId(posted);
  const O = win.document.getElementById('output');
  streamTask(win, tabId);
  const streamed = eventPanels(O).filter(
    el => !el.classList.contains('task-panel'),
  );
  assert.ok(streamed.length >= 4, 'precondition: the stream rendered panels');
  assert.ok(
    !O.querySelector('.trajectory'),
    'a running task has no Trajectory panel',
  );

  finishTask(win, tabId);
  send(win, {type: 'usage_info', tabId, total_tokens: 9, cost: '$0.01'});

  const traj = assertFolded(O, 'visible finish');
  const sub = traj.querySelector(':scope > .trajectory-sub');
  assert.deepStrictEqual(
    Array.from(sub.children),
    streamed,
    'every streamed panel moved into the Trajectory, in order',
  );
  assert.ok(
    O.querySelector(':scope > .task-panel') &&
      O.querySelector(':scope > .task-panel').nextElementSibling === traj,
    'the Trajectory follows the task text',
  );
  assert.ok(
    traj.nextElementSibling && traj.nextElementSibling.classList.contains('rc'),
    'the result follows the Trajectory',
  );
  const bash = sub.querySelector('.tc-bash');
  assert.ok(!isDisplayed(win, bash), 'a folded Trajectory hides its panels');
  assert.ok(isDisplayed(win, O.querySelector(':scope > .rc')), 'result shown');

  clickHeader(win, traj);
  assert.ok(!traj.classList.contains('collapsed'), 'the header click opens it');
  assert.ok(isDisplayed(win, bash), 'an open Trajectory shows its panels');
  assert.strictEqual(
    traj.querySelector('.collapse-preview').textContent,
    '',
    'an open Trajectory shows no preview',
  );
  send(win, {type: 'usage_info', tabId, total_tokens: 9, cost: '$0.01'});
  assert.ok(
    !traj.classList.contains('collapsed'),
    'a trailing event must not re-fold a Trajectory the user opened',
  );
  clickHeader(win, traj);
  assert.ok(traj.classList.contains('collapsed'), 'the second click folds it');
  assert.ok(!isDisplayed(win, bash), 'folded again: panels hidden');
  win.close();
  console.log('  ok - a visible finish folds the panels into a Trajectory');
}

// A terminal event without a tabId (an old daemon) folds the visible
// transcript.
function testUntaggedFinishFoldsVisibleTab() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  const O = win.document.getElementById('output');
  streamTask(win, tabId);
  send(win, {type: 'result', summary: 'd', success: true, tabId});
  send(win, {type: 'task_done'});
  assertFolded(O, 'untagged task_done');
  send(win, {type: 'status', running: false, tabId});
  send(win, {type: 'usage_info', tabId, total_tokens: 9, cost: '$0.01'});
  assertFolded(O, 'untagged task_done, after status and usage_info');
  win.close();
  console.log('  ok - an untagged finish folds the visible transcript');
}

// A task that ends in error on a BACKGROUND tab folds that tab's
// detached fragment, so the error's switch onto it shows the Trajectory.
function testBackgroundErrorFinishFoldsFragment() {
  const {win, posted} = makeWebview();
  const rootId = readyTabId(posted);
  const O = win.document.getElementById('output');

  send(win, {type: 'new_tab', task_id: 'bg-task', taskId: ''});
  const resume = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 'bg-task',
  );
  assert.ok(resume, 'new_tab must open a background tab');
  const bgId = resume.tabId;
  assert.notStrictEqual(bgId, rootId, 'the new tab is a background tab');

  streamTask(win, bgId);
  assert.ok(
    !O.querySelector('.trajectory'),
    'the background stream leaves the visible tab alone',
  );
  send(win, {type: 'task_error', tabId: bgId, text: 'boom'});
  send(win, {type: 'status', running: false, tabId: bgId});

  // The error pulled the user onto the finished tab (focusFinishedTab).
  const traj = assertFolded(O, 'background task_error');
  assert.ok(
    traj.querySelector('.trajectory-sub .tc-bash'),
    'the fragment folded the Bash panel',
  );
  win.close();
  console.log('  ok - a background task_error folds its fragment');
}

// A task that ends without a terminal event this client saw (the
// daemon's status-only end) folds on the status; a late event panel
// joins the existing Trajectory at the next end instead of opening a
// second one.
function testStatusOnlyEndFoldsAndRefolds() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  const O = win.document.getElementById('output');
  streamTask(win, tabId);
  send(win, {type: 'result', summary: 'd', success: true, tabId});
  send(win, {type: 'status', running: false, tabId});
  const traj = assertFolded(O, 'status-only end');
  const sub = traj.querySelector('.trajectory-sub');
  assert.strictEqual(sub.querySelectorAll('.tc-bash').length, 1);

  send(win, {type: 'tool_call', name: 'Bash', command: 'late', tabId});
  send(win, {type: 'tool_result', name: 'Bash', content: 'late', tabId});
  assert.ok(
    eventPanels(O).some(el => el.classList.contains('tc-bash')),
    'the late panel lands on the transcript',
  );
  send(win, {type: 'task_done', tabId});
  const again = assertFolded(O, 'terminal event after a status-only end');
  assert.strictEqual(again, traj, 'the same Trajectory panel adopts');
  assert.strictEqual(
    sub.querySelectorAll('.tc-bash').length,
    2,
    'the late panel joined the Trajectory',
  );
  const count = sub.children.length;
  send(win, {type: 'status', running: false, tabId});
  assert.strictEqual(
    sub.children.length,
    count,
    'an end with nothing new to fold changes nothing',
  );
  win.close();
  console.log('  ok - a status-only end folds; a later end reuses the panel');
}

// A transcript rebuilt from stored events (reload / reattach replay) of
// a task that has ended is folded by the replay itself, before any
// status event: the panels keep their collapsed digest inside the
// Trajectory, and the result stays outside.
function testReplayOfFinishedTaskFolds() {
  const {win, posted} = makeWebview();
  injectCss(win);
  const tabId = readyTabId(posted);
  const O = win.document.getElementById('output');
  send(win, {
    type: 'task_events',
    tabId,
    task: 'replayed task',
    task_id: 'replayed-1',
    events: [
      {type: 'prompt', text: 'do something'},
      {type: 'tool_call', name: 'Bash', command: 'ls'},
      {type: 'tool_result', name: 'Bash', content: 'ok'},
      {type: 'result', summary: '<p>done</p>', success: true},
      {type: 'task_done'},
    ],
  });
  const traj = assertFolded(O, 'replayed finished task');
  const bash = traj.querySelector('.trajectory-sub .tc-bash');
  assert.ok(bash, 'the replay folded the tool panel into the Trajectory');
  assert.ok(bash.classList.contains('collapsed'), 'digest kept inside');
  assert.ok(!isDisplayed(win, bash), 'hidden until the Trajectory opens');
  clickHeader(win, traj);
  assert.ok(isDisplayed(win, bash), 'one click away');
  send(win, {type: 'status', running: false, tabId});
  assert.ok(
    !traj.classList.contains('collapsed'),
    'the status that follows the replay must not re-fold an opened panel',
  );
  win.close();
  console.log('  ok - a replayed finished task is folded by the replay');
}

// The replay of a RUNNING task (a reload mid-run: the daemon's
// status running:true precedes task_events) keeps its panels in the
// open; the terminal event folds them when the task ends.
function testRunningReplayStaysOpenUntilTheEnd() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  const O = win.document.getElementById('output');
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {
    type: 'task_events',
    tabId,
    task: 'running task',
    task_id: 'running-1',
    events: [
      {type: 'prompt', text: 'do something'},
      {type: 'tool_call', name: 'Bash', command: 'ls'},
      {type: 'tool_result', name: 'Bash', content: 'ok'},
    ],
  });
  assert.ok(
    !O.querySelector('.trajectory'),
    'a running task replays without a Trajectory panel',
  );
  send(win, {type: 'tool_call', name: 'Read', path: 'b.txt', tabId});
  send(win, {type: 'tool_result', name: 'Read', content: 'bbb', tabId});
  finishTask(win, tabId);
  const traj = assertFolded(O, 'running replay, then finish');
  assert.ok(
    traj.querySelector('.trajectory-sub .tc-bash') &&
      traj.querySelector('.trajectory-sub .tc-path'),
    'the replayed and the live panels fold together',
  );
  win.close();
  console.log('  ok - a running replay folds only when the task ends');
}

// A neighbouring task's replayed transcript, spliced in while the live
// task runs, gets its own Trajectory in its own container; the live
// task's panels stay open until its finish, which folds only them.
function testAdjacentReplayFoldsItsOwnContainer() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  const O = win.document.getElementById('output');
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {type: 'setTaskText', text: 'do something', tabId});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId});
  send(win, {type: 'tool_result', name: 'Bash', content: 'ok', tabId});
  send(win, {
    type: 'adjacent_task_events',
    direction: 'prev',
    task: 'do something',
    task_id: 'earlier-run',
    events: [
      {type: 'task_start', task: 'do something'},
      {type: 'tool_call', name: 'Bash', command: 'echo old'},
      {type: 'tool_result', name: 'Bash', content: 'old'},
      {type: 'result', summary: 'old done', success: true},
      {type: 'task_done'},
    ],
  });
  const adjacent = O.querySelector('.adjacent-task[data-task="do something"]');
  assert.ok(adjacent, 'the adjacent task container must render');
  assertFolded(adjacent, 'adjacent replay');
  assert.ok(
    !Array.from(O.children).some(el => el.classList.contains('trajectory')),
    'the live task has no Trajectory while it runs',
  );
  assert.ok(
    O.querySelector(':scope > .tc-bash'),
    'the live Bash panel is still on the transcript',
  );

  send(win, {type: 'result', summary: 'new done', success: true, tabId});
  send(win, {type: 'task_done', tabId});
  const live = assertFolded(O, 'live finish beside an adjacent task');
  assert.ok(
    !live.querySelector('.adjacent-task'),
    'the neighbour is not folded into the live Trajectory',
  );
  assert.strictEqual(
    adjacent.querySelectorAll('.trajectory').length,
    1,
    'the neighbour keeps its own single Trajectory',
  );
  win.close();
  console.log('  ok - an adjacent replay folds its own container');
}

// A terminal event for a tab this client no longer has, or for a
// background tab with nothing streamed, is a no-op; the visible tab
// folds only on its own end.
function testTerminalEventsForOtherTabsAreNoOps() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  const O = win.document.getElementById('output');
  streamTask(win, tabId);

  send(win, {type: 'task_stopped', tabId: 'no-such-tab'});
  send(win, {type: 'status', running: false, tabId: 'no-such-tab'});
  send(win, {type: 'new_tab', task_id: 'idle-task', taskId: ''});
  const resume = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 'idle-task',
  );
  // The stop names a background tab with no streamed fragment; its
  // focusFinishedTab switch is undone by clicking home again.
  send(win, {type: 'task_stopped', tabId: resume.tabId});
  const homeTabEl =
    win.document.querySelector(
      '#tab-list .chat-tab[data-tab-id="' + tabId + '"]',
    ) ||
    win.document.querySelector(
      '#main-tab-list .chat-tab[data-tab-id="' + tabId + '"]',
    );
  assert.ok(homeTabEl, 'the original tab is still listed');
  homeTabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !O.querySelector('.trajectory'),
    "other tabs' terminal events must not fold the visible transcript",
  );
  finishTask(win, tabId);
  assertFolded(O, 'own finish after other tabs ended');
  win.close();
  console.log('  ok - terminal events for other tabs are no-ops');
}

testVisibleFinishFoldsTrajectory();
testUntaggedFinishFoldsVisibleTab();
testBackgroundErrorFinishFoldsFragment();
testStatusOnlyEndFoldsAndRefolds();
testReplayOfFinishedTaskFolds();
testRunningReplayStaysOpenUntilTheEnd();
testAdjacentReplayFoldsItsOwnContainer();
testTerminalEventsForOtherTabsAreNoOps();
console.log('taskEndTrajectory: all tests passed');
