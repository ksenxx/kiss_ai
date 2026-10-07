// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// When a task finishes, the chat webview must not explicitly collapse
// any event panel: the transcript keeps the exact state the live
// stream left it in.  The terminal events (task_done, task_error,
// task_stopped, task_interrupted) stamp every panel that is on screen,
// and the not-running branch of applyChevronState leaves stamped
// panels alone.  A transcript REBUILT from stored events (a reload or
// reattach replay) carries no stamps and keeps the finished-task
// digest: everything but the result collapsed, but every panel stays
// on screen.  These tests drive the real webview end to end through
// window messages, exactly as the daemon does.

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
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

function outputPanels(win) {
  return Array.from(
    win.document.querySelectorAll('#output .collapsible'),
  );
}

function panelState(p) {
  return {collapsed: p.classList.contains('collapsed')};
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

/**
 * Stream a small real task: prompt, thoughts, three tools, finish. The
 * stream keeps its newest two panels open, so the Bash panel only folds
 * because two more panels (the Read call and the thoughts its result
 * arms) follow it.
 */
function streamTask(win, tabId) {
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
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

// A task finishing on the visible tab leaves every panel exactly as
// the stream left it: the pass a trailing event (usage_info) triggers
// with the running flag off must not collapse anything, and the panel
// the stream left open stays open.
function testVisibleFinishLeavesPanelsAsIs() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  streamTask(win, tabId);

  const before = outputPanels(win).map(panelState);
  const last = outputPanels(win)[before.length - 1];
  assert.ok(
    !last.classList.contains('collapsed'),
    'precondition: the stream leaves its latest panel open',
  );

  finishTask(win, tabId);
  send(win, {type: 'usage_info', tabId, total_tokens: 9, cost: '$0.01'});

  const after = outputPanels(win).map(panelState);
  assert.deepStrictEqual(
    after,
    before,
    'the finish must not change any panel: no explicit collapse',
  );
  win.close();
  console.log('  ok - a visible finish leaves every panel as the stream left it');
}

// A terminal event without a tabId (an old daemon) stamps the visible
// transcript.
function testUntaggedFinishStampsVisibleTab() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  streamTask(win, tabId);
  send(win, {type: 'result', summary: 'd', success: true, tabId});
  send(win, {type: 'task_done'});
  send(win, {type: 'status', running: false, tabId});
  send(win, {type: 'usage_info', tabId, total_tokens: 9, cost: '$0.01'});
  assert.ok(
    outputPanels(win).every(p => p._liveFinished),
    'an untagged task_done must stamp (and so keep) the visible panels',
  );
  win.close();
  console.log('  ok - an untagged finish stamps the visible transcript');
}

// A panel the user expanded mid-run stays expanded through the finish
// and through later repaints (applyChevronState passes).
function testUserExpandedPanelSurvivesFinish() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  streamTask(win, tabId);

  const bash = win.document.querySelector('#output .tc-bash');
  assert.ok(bash.classList.contains('collapsed'), 'older panel collapsed');
  bash
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(!bash.classList.contains('collapsed'), 'user expanded it');

  finishTask(win, tabId);
  send(win, {type: 'usage_info', tabId, total_tokens: 9, cost: '$0.01'});

  assert.ok(
    !bash.classList.contains('collapsed'),
    'the finish must not re-collapse a panel the user expanded',
  );
  win.close();
  console.log('  ok - a user-expanded panel survives the finish');
}

// A task that ends in error on a BACKGROUND tab stamps that tab's
// detached fragment, so switching to it shows the transcript exactly
// as it streamed instead of the collapsed digest.
function testBackgroundErrorFinishKeepsPanels() {
  const {win, posted} = makeWebview();
  const rootId = readyTabId(posted);

  send(win, {type: 'new_tab', task_id: 'bg-task', taskId: ''});
  const resume = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 'bg-task',
  );
  assert.ok(resume, 'new_tab must open a background tab');
  const bgId = resume.tabId;
  assert.notStrictEqual(bgId, rootId, 'the new tab is a background tab');

  streamTask(win, bgId);
  send(win, {type: 'task_error', tabId: bgId, text: 'boom'});
  send(win, {type: 'status', running: false, tabId: bgId});

  // The error pulled the user onto the finished tab (focusFinishedTab);
  // repaint once more and check nothing collapsed.
  const panels = outputPanels(win);
  assert.ok(panels.length >= 3, 'the background transcript is on screen');
  assert.ok(
    !panels[panels.length - 1].classList.contains('collapsed'),
    'the panel the stream left open stays open after task_error',
  );
  win.close();
  console.log('  ok - a background task_error finish keeps its panels');
}

// Switching away from a finished transcript and back must not apply
// the digest either: the stamped panels ride the detached fragment.
function testTabSwitchAfterFinishKeepsPanels() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  streamTask(win, tabId);
  finishTask(win, tabId);

  send(win, {type: 'new_tab', task_id: 'other-task', taskId: ''});
  const resume = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 'other-task',
  );
  const otherId = resume.tabId;
  const otherTabEl =
    win.document.querySelector(
      '#tab-list .chat-tab[data-tab-id="' + otherId + '"]',
    ) ||
    win.document.querySelector(
      '#main-tab-list .chat-tab[data-tab-id="' + otherId + '"]',
    );
  if (otherTabEl) {
    otherTabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  }
  const finishedTabEl =
    win.document.querySelector(
      '#tab-list .chat-tab[data-tab-id="' + tabId + '"]',
    ) ||
    win.document.querySelector(
      '#main-tab-list .chat-tab[data-tab-id="' + tabId + '"]',
    );
  assert.ok(finishedTabEl, 'the finished tab is still listed');
  finishedTabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const panels = outputPanels(win);
  assert.ok(panels.length >= 3, 'the finished transcript is back on screen');
  assert.ok(
    !panels[panels.length - 1].classList.contains('collapsed'),
    'switching back must not collapse the panel the stream left open',
  );
  win.close();
  console.log('  ok - tab switch after the finish keeps the panels');
}

// REGRESSION: a transcript rebuilt from stored events (reload /
// reattach replay) has no live-finish stamps and keeps the digest —
// panels collapsed by the replay pass, yet every one of them stays on
// screen, one click away.
function testReplayKeepsFinishedDigest() {
  const {win, posted} = makeWebview();
  injectCss(win);
  const tabId = readyTabId(posted);
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
  const bash = win.document.querySelector('#output .tc-bash');
  assert.ok(bash, 'replay rendered the tool panel');
  assert.ok(
    bash.classList.contains('collapsed'),
    'a replayed finished task keeps the collapsed digest',
  );
  assert.ok(
    isDisplayed(win, bash),
    'a replayed finished task must not hide its event panels',
  );
  win.close();
  console.log('  ok - a replayed finished task keeps the digest');
}

// REGRESSION: a neighbouring task's replayed digest, spliced into the
// transcript while the live task runs, takes no live-finish stamp —
// even when its label equals the live task's name (a rerun) — so the
// finish leaves its panels folded.
function testAdjacentReplayKeepsDigestThroughSameNameFinish() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {type: 'setTaskText', text: 'do something', tabId});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId});
  send(win, {type: 'tool_result', name: 'Bash', content: 'ok', tabId});
  send(win, {
    type: 'adjacent_task_events',
    direction: 'prev',
    task: 'do something',
    task_id: 'earlier-run',
    tabId,
    events: [
      {type: 'task_start', task: 'do something'},
      {type: 'tool_call', name: 'Bash', command: 'ls'},
      {type: 'tool_result', name: 'Bash', content: 'old'},
      {type: 'result', summary: 'old done', success: true},
    ],
  });
  const adjacent = win.document.querySelector('#output .adjacent-task');
  assert.ok(adjacent, 'the earlier task must be spliced in');
  const adjPanel = adjacent.querySelector(
    '.collapsible:not(.rc):not(.task-panel)',
  );
  assert.ok(
    adjPanel.classList.contains('collapsed'),
    'precondition: the adjacent digest panel is folded',
  );

  finishTask(win, tabId);
  send(win, {type: 'usage_info', tabId, total_tokens: 9, cost: '$0.01'});

  assert.ok(
    !adjPanel._liveFinished,
    'the finish must not stamp a neighbouring task\'s panels',
  );
  assert.ok(
    adjPanel.classList.contains('collapsed'),
    'the finish must not unfold the adjacent digest',
  );
  const livePanel = win.document.querySelector('#output > .tc-bash');
  assert.ok(
    livePanel && !livePanel.classList.contains('collapsed'),
    'the live task\'s own panels still keep their streamed state',
  );
  win.close();
  console.log('  ok - an adjacent digest survives a same-name finish');
}

// REGRESSION: an end announced only by `status running:false` (the
// attach path for an already-finished chat) is not a live finish and
// keeps the digest pass.
function testStatusOnlyEndStillDigests() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  streamTask(win, tabId);
  send(win, {type: 'result', summary: 'd', success: true, tabId});
  send(win, {type: 'status', running: false, tabId});
  send(win, {type: 'usage_info', tabId, total_tokens: 5, cost: '$0.01'});

  const panels = outputPanels(win).filter(p => !p.classList.contains('rc'));
  assert.ok(panels.length >= 3, 'the streamed event panels are on screen');
  assert.ok(
    panels.every(p => p.classList.contains('collapsed')),
    'a status-only end keeps the digest pass: every event panel folds',
  );
  win.close();
  console.log('  ok - a status-only end still digests');
}

// Terminal events for tabs this client cannot stamp are no-ops: an
// unknown tab, and a background tab that never streamed a fragment.
function testUnstampableTerminalEventsAreNoOps() {
  const {win, posted} = makeWebview();
  const tabId = readyTabId(posted);
  streamTask(win, tabId);

  send(win, {type: 'task_done', tabId: 'no-such-tab'});
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

  const panels = outputPanels(win);
  assert.ok(
    panels.length >= 3 && panels.every(p => !p._liveFinished),
    'terminal events for other tabs must not stamp the visible panels',
  );
  finishTask(win, tabId);
  send(win, {type: 'usage_info', tabId, total_tokens: 9, cost: '$0.01'});
  assert.ok(
    outputPanels(win).every(p => p._liveFinished),
    'the visible tab still stamps and keeps its own panels',
  );
  win.close();
  console.log('  ok - unstampable terminal events are no-ops');
}

testVisibleFinishLeavesPanelsAsIs();
testUntaggedFinishStampsVisibleTab();
testUserExpandedPanelSurvivesFinish();
testBackgroundErrorFinishKeepsPanels();
testTabSwitchAfterFinishKeepsPanels();
testReplayKeepsFinishedDigest();
testAdjacentReplayKeepsDigestThroughSameNameFinish();
testStatusOnlyEndStillDigests();
testUnstampableTerminalEventsAreNoOps();
console.log('taskEndNoCollapse: all tests passed');
