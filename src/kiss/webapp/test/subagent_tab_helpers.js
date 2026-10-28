// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// DOM helpers shared by the run_parallel / run_agent sub-agent tab suites
// (page built by simplify2_harness.makePage, or by any bootstrap that
// evals media/main.js into chat.html).

const assert = require('assert');
const {send} = require('./simplify2_harness');

/** Every sub-agent tab rendered in the tab bar, in DOM order. */
function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

/** The first tool-call panel whose header names run_parallel, or null. */
function runParallelPanel(win) {
  const headers = win.document.querySelectorAll('#output .ev.tc .tc-h');
  for (const h of headers) {
    const txt = (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt.startsWith('run_parallel')) return h.closest('.ev.tc');
  }
  return null;
}

/**
 * Click *panel*'s header chevron (collapse or expand it).
 *
 * @param {object} win The page window.
 * @param {Element} panel The `.ev.tc` panel element.
 * @param {Function} [drain] Called after the click, for pages on the real
 *   transport that collect the frames the click sent.
 */
function togglePanel(win, panel, drain) {
  panel
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  if (drain) drain();
}

/** The user clicks the close button of tab *tabId*. */
function clickClose(win, tabId) {
  const btn = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"] .chat-tab-close`,
  );
  assert.ok(btn, 'sub-agent tab ' + tabId + ' must render a close button');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** The daemon reports every sub-agent in *tabIds* finished. */
function finishFanOut(win, tabIds) {
  for (const id of tabIds) send(win, {type: 'subagentDone', tab_id: id});
}

/**
 * The daemon's replay burst for finished children: one `openSubagentTab`
 * per task, `isDone: true`, carrying the tab id `tabIdFor(taskId, i)`
 * gives (a stable id, or the id the fan-out opened).
 */
function announceFinished(win, parentId, taskIds, tabIdFor) {
  taskIds.forEach((taskId, i) =>
    send(win, {
      type: 'openSubagentTab',
      tab_id: tabIdFor(taskId, i),
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
      isDone: true,
    }),
  );
}

/** The tab ids of the sub-agent tabs in the strip, in strip order. */
function openSubTabIds(win) {
  return subagentTabEls(win).map(el => el.dataset.tabId);
}

/**
 * The default `deliver(msg)` for *page*: `send`, then (for a transport
 * client from `transport_page_helpers.makeWebappClient`) `drain()`, so
 * the frames the message provoked are in `posted` for the caller.
 */
function deliverTo(page) {
  return msg => {
    send(page.win, msg);
    if (page.drain) page.drain();
  };
}

/**
 * Boot a parent task that spawns *n* sub-agents through run_parallel on
 * *page* (`{win, posted}` from any chat.html builder, or a transport
 * client, whose frames are drained for you), exactly the way
 * the backend drives the page: `status running`, the run_parallel
 * `tool_call`, then `new_tab` + `openSubagentTab` per sub-agent.
 *
 * @param {{deliver?: Function, done?: boolean, declared?: number,
 *   replay?: boolean}} [opts] `deliver(msg)` replaces `send(win, msg)`
 *   for a page fed through a transport shim; `done` opens every
 *   sub-agent tab already finished (`isDone: true`); `declared` is how
 *   many tasks the run_parallel call lists (default *n*: a run may
 *   spawn fewer children than it declared); `replay` follows every
 *   `openSubagentTab` with the child's (empty) `task_events` transcript,
 *   the way the daemon does, which binds the tab to its task id.
 * @returns {{win: object, posted: Array<object>, parentId: string,
 *   panel: Element, taskIds: Array<string>, subTabIds: Array<string>}}
 */
function bootParallelRun(n, page, opts) {
  const {win, posted} = page;
  const {
    deliver = deliverTo(page),
    done = false,
    declared = n,
    replay = false,
  } = opts || {};
  if (page.drain) page.drain();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'the page must post ready with a tabId');
  const parentId = ready.tabId;

  deliver({
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
  });
  const taskNames = [];
  for (let i = 0; i < declared; i++) taskNames.push('sub ' + (i + 1));
  deliver({
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(taskNames)},
  });
  const panel = runParallelPanel(win);
  assert.ok(panel, 'run_parallel tool_call must render a .ev.tc panel');

  const taskIds = [];
  const subTabIds = [];
  for (let i = 0; i < n; i++) {
    const taskId = 'sub-task-' + (i + 1);
    taskIds.push(taskId);
    const tabId = spawnSubagent(page, parentId, taskId, 'sub ' + (i + 1), i, {
      deliver,
      extra: {isDone: done},
    });
    subTabIds.push(tabId);
    if (replay) {
      deliver({
        type: 'task_events',
        tabId,
        task: 'sub ' + (i + 1),
        task_id: taskId,
        events: [],
      });
    }
  }
  assert.strictEqual(
    subagentTabEls(win).length,
    n,
    'each spawned sub-agent must get its own tab',
  );
  assert.strictEqual(
    activeTabId(win),
    parentId,
    'precondition: spawning sub-agents must leave the user on the parent',
  );
  return {win, posted, parentId, panel, taskIds, subTabIds};
}

// A successful Write tool_result body, exactly as json_printer.py emits it.
function writeSuccess(filePath, content) {
  return (
    'Successfully wrote ' +
    String(content || '').length +
    ' characters to ' +
    filePath
  );
}

/** A sub-agent writes its report: the Write tool_call and its result. */
function writeReport(win, filePath, content, extra) {
  send(
    win,
    Object.assign(
      {type: 'tool_call', name: 'Write', path: filePath, content: content},
      extra || {},
    ),
  );
  send(
    win,
    Object.assign(
      {
        type: 'tool_result',
        content: writeSuccess(filePath, content),
        is_error: false,
        tool_name: 'Write',
        path: filePath,
      },
      extra || {},
    ),
  );
}

/** Every open content tab (file, report) of the tab registry. */
function contentTabs(win) {
  return win._testApi.openTabs().filter(t => t.isContentTab);
}

/** The open content tab whose title contains *name*, or undefined. */
function contentTabFor(win, name) {
  return contentTabs(win).find(t => t.title.indexOf(name) >= 0);
}

/** The content tabs the tab strip lists (the file screen only). */
function stripContentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.content-tab'),
  );
}

/** The tab-bar element of tab *tabId*, or null. */
function tabEl(win, tabId) {
  return win.document.querySelector(
    '#tab-list .chat-tab[data-tab-id=' + JSON.stringify(tabId) + ']',
  );
}

/** Every rendered report iframe of the content tab area. */
function reportFrames(win) {
  return Array.from(
    win.document.querySelectorAll(
      '#content-tab-area .content-tab-view iframe.content-html-frame',
    ),
  );
}

/** The srcdoc of the one report iframe on view (asserts there is one). */
function visibleSrcdoc(win) {
  const frames = reportFrames(win).filter(
    f => f.closest('.content-tab-view').style.display !== 'none',
  );
  assert.strictEqual(frames.length, 1, 'expected exactly one visible frame');
  return frames[0].getAttribute('srcdoc') || '';
}

/** The id of the tab on view. */
function activeTabId(win) {
  return win._testApi.getActiveTabId();
}

/** The user clicks tab element *el*. */
/** Whether the chat (`#output`) is the surface on screen. */
function chatSurfaceVisible(win) {
  const out = win.document.getElementById('output');
  return !!out && out.style.display !== 'none';
}

function touchEvent(win, type, target, points, changed) {
  const e = new win.Event(type, {bubbles: true, cancelable: true});
  Object.defineProperty(e, 'touches', {value: points});
  Object.defineProperty(e, 'changedTouches', {value: changed || points});
  target.dispatchEvent(e);
}

/** The user swipes left on the chat: the file screen comes on view. */
function swipeLeftToFileScreen(win) {
  const page = win.document.getElementById('output');
  touchEvent(win, 'touchstart', page, [{clientX: 300, clientY: 100}]);
  touchEvent(win, 'touchmove', page, [{clientX: 100, clientY: 100}]);
  touchEvent(win, 'touchend', page, [], [{clientX: 100, clientY: 100}]);
}

/**
 * Spawn one sub-agent the way the daemon does: `new_tab` for `taskId`
 * under `parentId`, which must make the page post a `resumeSession`
 * for that task, then `openSubagentTab` on the tab the page chose.
 *
 * @param {object} page `{win, posted}` (a harness page or a transport
 *   client, whose frames are drained into `posted` after each message).
 * @param {string} parentId The parent tab id.
 * @param {string} taskId The sub-agent's task id.
 * @param {string} desc The tab's description.
 * @param {number} idx The child's index in its fan-out.
 * @param {object} [opts] `deliver(msg)` plays a daemon frame (default
 *   `deliverTo(page)`); `extra` adds fields to the `openSubagentTab`
 *   frame.
 * @returns {string} The sub-agent's tab id.
 */
function spawnSubagent(page, parentId, taskId, desc, idx, opts) {
  const {win, posted} = page;
  const {deliver = deliverTo(page), extra = {}} = opts || {};
  const before = posted.length;
  deliver({
    type: 'new_tab',
    task_id: taskId,
    parent_tab_id: parentId,
    taskId: '',
  });
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === taskId);
  assert.ok(
    resume,
    'new_tab for ' +
      taskId +
      ' under parent ' +
      parentId +
      ' must make the page post resumeSession (a tab must open for ' +
      'EVERY sub-agent)',
  );
  deliver(
    Object.assign(
      {
        type: 'openSubagentTab',
        tab_id: resume.tabId,
        parent_tab_id: parentId,
        description: desc,
        task_id: taskId,
        taskIndex: idx,
      },
      extra,
    ),
  );
  assert.ok(
    subagentTabEls(win).some(el => el.dataset.tabId === resume.tabId),
    'sub-agent ' + taskId + ' must get its own OPEN tab',
  );
  return resume.tabId;
}

module.exports = {
  announceFinished,
  openSubTabIds,
  subagentTabEls,
  runParallelPanel,
  togglePanel,
  clickClose,
  finishFanOut,
  bootParallelRun,
  writeSuccess,
  writeReport,
  contentTabs,
  contentTabFor,
  stripContentTabEls,
  tabEl,
  reportFrames,
  visibleSrcdoc,
  activeTabId,
  chatSurfaceVisible,
  swipeLeftToFileScreen,
  spawnSubagent,
};
