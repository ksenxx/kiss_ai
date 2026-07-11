// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: a sub-agent tab must CLOSE as soon as its
// sub-agent finishes (the backend broadcasts ``subagentDone``).
//
// Behavior under test (``case 'subagentDone':`` in ``media/main.js``):
//
//   1. When one sub-agent of a fan-out finishes, ONLY its tab closes;
//      the still-running sibling tabs stay open and the owning
//      run_parallel panel stays uncollapsed.
//   2. When the LAST sub-agent finishes, its tab closes and the owning
//      run_parallel panel collapses (no open sub-agent tab remains).
//   3. When the finished sub-agent tab is the ACTIVE tab, the webview
//      switches to an adjacent tab instead of showing a dead view.
//   4. A ``subagentDone`` for an unknown tab id is a no-op.
//
// The tests drive the real ``media/main.js`` against the real
// ``media/chat.html`` markup in jsdom — the exact production webview
// code — mirroring the harness of ``runParallelPanelTabsSync.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/subagentTabAutoCloseOnDone.test.js

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

/** The run_parallel tool-call panel element in the active chat DOM. */
function runParallelPanel(win) {
  const headers = win.document.querySelectorAll('#output .ev.tc .tc-h');
  for (const h of headers) {
    const txt = (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt.startsWith('run_parallel')) return h.closest('.ev.tc');
  }
  return null;
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
  const panel = runParallelPanel(win);
  assert.ok(panel, 'run_parallel tool_call must render a .ev.tc panel');

  const taskIds = [];
  const subTabIds = [];
  for (let i = 0; i < n; i++) {
    const taskId = 'sub-task-' + (i + 1);
    taskIds.push(taskId);
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
    // The server replays the sub-agent row and converts the tab.
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
  return {win, posted, parentId, panel, taskIds, subTabIds};
}

// ---------------------------------------------------------------------------
// 1. subagentDone closes ONLY the finished sub-agent's tab.
// ---------------------------------------------------------------------------
function testDoneClosesOnlyFinishedTab() {
  const {win, posted, panel, subTabIds} = bootParallelRun(2);

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  const remaining = subagentTabEls(win);
  assert.strictEqual(
    remaining.length,
    1,
    'the finished sub-agent tab must close as soon as it finishes, ' +
      'leaving only the still-running sibling open',
  );
  assert.strictEqual(
    remaining[0].dataset.tabId,
    subTabIds[1],
    'the surviving tab must be the still-running sibling',
  );
  assert.ok(
    posted.some(m => m.type === 'closeTab' && m.tabId === subTabIds[0]),
    'the backend must be told to close the finished sub-agent tab',
  );
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the run_parallel panel must stay uncollapsed while a sibling ' +
      'sub-agent is still running',
  );
  win.close();
  console.log('  ok - subagentDone closes only the finished tab');
}

// ---------------------------------------------------------------------------
// 2. The LAST subagentDone closes its tab and collapses the panel.
// ---------------------------------------------------------------------------
function testLastDoneClosesTabAndCollapsesPanel() {
  const {win, posted, panel, subTabIds} = bootParallelRun(2);

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[1]});

  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'every finished sub-agent tab must be closed',
  );
  for (const id of subTabIds) {
    assert.ok(
      posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close sub-agent tab ' + id,
    );
  }
  assert.ok(
    panel.classList.contains('collapsed'),
    'the run_parallel panel must collapse once no open sub-agent ' +
      'tab remains',
  );
  win.close();
  console.log('  ok - last subagentDone closes its tab and collapses panel');
}

// ---------------------------------------------------------------------------
// 3. Finishing the ACTIVE sub-agent tab switches to an adjacent tab.
// ---------------------------------------------------------------------------
function testDoneOnActiveTabSwitchesAway() {
  const {win, parentId, subTabIds} = bootParallelRun(2);

  // Activate the first sub-agent tab exactly like a user click.
  const subEl = subagentTabEls(win).find(
    el => el.dataset.tabId === subTabIds[0],
  );
  assert.ok(subEl, 'the first sub-agent tab must be in the tab bar');
  subEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    activeTabEl(win).dataset.tabId,
    subTabIds[0],
    'clicking the sub-agent tab must activate it',
  );

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  const active = activeTabEl(win);
  assert.ok(active, 'a tab must be active after the finished tab closed');
  assert.notStrictEqual(
    active.dataset.tabId,
    subTabIds[0],
    'the closed sub-agent tab must not stay active',
  );
  assert.ok(
    [parentId, subTabIds[1]].includes(active.dataset.tabId),
    'an adjacent tab (parent or sibling) must become active',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'only the still-running sibling sub-agent tab must remain open',
  );
  win.close();
  console.log('  ok - subagentDone on the active tab switches away');
}

// ---------------------------------------------------------------------------
// 4. subagentDone for an unknown tab id is a no-op.
// ---------------------------------------------------------------------------
function testDoneUnknownTabIsNoop() {
  const {win, posted, subTabIds} = bootParallelRun(2);

  const before = posted.length;
  send(win, {type: 'subagentDone', tab_id: 'no-such-tab'});

  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'a subagentDone for an unknown tab must not touch open tabs',
  );
  assert.ok(
    !posted
      .slice(before)
      .some(m => m.type === 'closeTab' && subTabIds.includes(m.tabId)),
    'no open sub-agent tab may be closed for an unknown tab id',
  );
  win.close();
  console.log('  ok - subagentDone for an unknown tab id is a no-op');
}

function main() {
  testDoneClosesOnlyFinishedTab();
  testLastDoneClosesTabAndCollapsesPanel();
  testDoneOnActiveTabSwitchesAway();
  testDoneUnknownTabIsNoop();
  console.log('subagentTabAutoCloseOnDone.test.js: all tests passed');
}

main();
