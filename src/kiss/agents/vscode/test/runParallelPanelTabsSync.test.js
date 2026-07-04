// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the run_parallel panel ⇔ sub-agent tabs
// invariant in the chat webview (``media/main.js``):
//
//   * While a ``run_parallel`` tool-call panel is UNCOLLAPSED, the
//     tabs of its sub-agents MUST be open.
//   * While a ``run_parallel`` tool-call panel is COLLAPSED, the
//     tabs of its sub-agents MUST be closed.
//
// Violations reproduced (before the fix):
//
//   1. Collapsing the run_parallel panel (clicking its header) left
//      every sub-agent tab open.
//   2. Re-expanding the panel did not reopen the sub-agent tabs that
//      the collapse should have closed.
//   3. Closing a sub-agent tab by hand left the panel uncollapsed
//      while some of its sub-agent tabs were closed.
//   4. The automatic collapse passes (``collapseOlderPanels`` while
//      streaming, ``collapseAllExceptResult`` at task end) collapsed
//      the run_parallel panel while its sub-agent tabs stayed open.
//
// The tests drive the real ``media/main.js`` against the real
// ``media/chat.html`` markup in jsdom — the exact production webview
// code — mirroring the harness of ``tab_timer_per_tab.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/runParallelPanelTabsSync.test.js

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

/** Click the collapse header of *panel* (toggles .collapsed). */
function togglePanel(win, panel) {
  const hdr = panel.querySelector('.tc-h');
  hdr.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
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
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub one', 'sub two'])},
  });
  const panel = runParallelPanel(win);
  assert.ok(panel, 'run_parallel tool_call must render a .ev.tc panel');
  assert.ok(
    !panel.classList.contains('collapsed'),
    'run_parallel panel must start uncollapsed',
  );

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
// 1. Collapsing the run_parallel panel must close its sub-agent tabs.
// ---------------------------------------------------------------------------
function testCollapseClosesSubagentTabs() {
  const {win, posted, panel, subTabIds} = bootParallelRun(2);

  togglePanel(win, panel);
  assert.ok(
    panel.classList.contains('collapsed'),
    'clicking the header must collapse the run_parallel panel',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'INVARIANT VIOLATED: run_parallel panel is collapsed but its ' +
      'sub-agent tabs are still open',
  );
  for (const id of subTabIds) {
    assert.ok(
      posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close sub-agent tab ' + id,
    );
  }
  win.close();
  console.log('  ok - collapsing the run_parallel panel closes sub tabs');
}

// ---------------------------------------------------------------------------
// 2. Re-expanding the run_parallel panel must reopen its sub-agent tabs.
// ---------------------------------------------------------------------------
function testExpandReopensSubagentTabs() {
  const {win, posted, panel, taskIds} = bootParallelRun(2);

  togglePanel(win, panel); // collapse → tabs close
  const before = posted.length;
  togglePanel(win, panel); // expand → tabs must reopen
  assert.ok(
    !panel.classList.contains('collapsed'),
    'second click must uncollapse the run_parallel panel',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'INVARIANT VIOLATED: run_parallel panel is uncollapsed but its ' +
      'sub-agent tabs are not open',
  );
  for (const taskId of taskIds) {
    assert.ok(
      posted
        .slice(before)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      'reopened sub-agent tab must resume backend task ' + taskId,
    );
  }
  win.close();
  console.log('  ok - expanding the run_parallel panel reopens sub tabs');
}

// ---------------------------------------------------------------------------
// 3. Manually closing a sub-agent tab must restore consistency: the
//    panel collapses (and the remaining sibling sub tabs close).
// ---------------------------------------------------------------------------
function testManualSubTabCloseCollapsesPanel() {
  const {win, panel, subTabIds} = bootParallelRun(2);

  const firstEl = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${subTabIds[0]}"] .chat-tab-close`,
  );
  assert.ok(firstEl, 'sub-agent tab must render a close button');
  firstEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const collapsed = panel.classList.contains('collapsed');
  const openSubTabs = subagentTabEls(win).length;
  assert.ok(
    (collapsed && openSubTabs === 0) || (!collapsed && openSubTabs === 2),
    'INVARIANT VIOLATED: after closing a sub-agent tab by hand the ' +
      'panel state and tab state diverge (collapsed=' +
      collapsed +
      ', open sub tabs=' +
      openSubTabs +
      ')',
  );
  win.close();
  console.log('  ok - manual sub-tab close keeps panel/tabs consistent');
}

// ---------------------------------------------------------------------------
// 4. The automatic collapse passes must never leave the run_parallel
//    panel collapsed while its sub-agent tabs are open — neither while
//    streaming continues (collapseOlderPanels) nor at task end
//    (collapseAllExceptResult).
// ---------------------------------------------------------------------------
function testAutoCollapseKeepsInvariant() {
  const {win, panel, parentId} = bootParallelRun(2);

  // The parallel fan-out finished: tool_result arrives, the agent
  // thinks some more (new Thoughts panel → collapseOlderPanels), and
  // finally the task ends (result → collapseAllExceptResult).
  send(win, {
    type: 'tool_result',
    tabId: parentId,
    content: 'all sub-agents done',
  });
  send(win, {type: 'thinking_start', tabId: parentId});
  send(win, {type: 'thinking_delta', tabId: parentId, text: 'wrapping up'});
  send(win, {type: 'thinking_end', tabId: parentId});
  send(win, {
    type: 'tool_call',
    name: 'finish',
    tabId: parentId,
    extras: {summary: 'done'},
  });
  send(win, {type: 'result', tabId: parentId, summary: 'done', success: true});

  const collapsed = panel.classList.contains('collapsed');
  const openSubTabs = subagentTabEls(win).length;
  assert.ok(
    (collapsed && openSubTabs === 0) || (!collapsed && openSubTabs === 2),
    'INVARIANT VIOLATED: automatic collapse left the run_parallel ' +
      'panel collapsed=' +
      collapsed +
      ' while ' +
      openSubTabs +
      ' sub-agent tabs are open',
  );
  win.close();
  console.log('  ok - automatic collapse passes keep panel/tabs consistent');
}

// ---------------------------------------------------------------------------
// 5. A delayed ``openSubagentTab`` replay for a tab that was closed by
//    collapsing the panel must not recreate the tab while the panel is
//    collapsed.
// ---------------------------------------------------------------------------
function testDelayedOpenSubagentTabDoesNotReopenCollapsedPanel() {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const parentId = ready.tabId;

  send(win, {type: 'status', running: true, tabId: parentId});
  send(win, {type: 'tool_call', name: 'run_parallel', tabId: parentId});
  const panel = runParallelPanel(win);
  assert.ok(panel, 'run_parallel tool_call must render a panel');

  send(win, {
    type: 'new_tab',
    task_id: 'late-sub-task',
    parent_tab_id: parentId,
    taskId: '',
  });
  const resume = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 'late-sub-task',
  );
  assert.ok(resume, 'new_tab must request resumeSession');
  assert.strictEqual(subagentTabEls(win).length, 1, 'sanity: tab opened');

  togglePanel(win, panel);
  assert.ok(panel.classList.contains('collapsed'), 'panel collapsed');
  assert.strictEqual(subagentTabEls(win).length, 0, 'collapse closed tab');

  // This is the race the reviewer found: the server may still deliver
  // the conversion/replay for the now-closed tab id.  It must be
  // ignored/deferred, not allowed to recreate an open tab behind a
  // collapsed run_parallel panel.
  send(win, {
    type: 'openSubagentTab',
    tab_id: resume.tabId,
    parent_tab_id: parentId,
    description: 'late sub',
    task_id: 'late-sub-task',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'INVARIANT VIOLATED: delayed openSubagentTab recreated a sub-agent ' +
      'tab while the owning run_parallel panel is collapsed',
  );
  win.close();
  console.log('  ok - delayed openSubagentTab cannot reopen collapsed panel');
}

// ---------------------------------------------------------------------------
// 6. Replayed/persisted sub-agent tabs can be opened by
//    ``openSubagentTab`` alone (no preceding live ``new_tab`` in this
//    browser).  They must still be associated with the run_parallel
//    panel so collapse closes them and expand reopens them.
// ---------------------------------------------------------------------------
function testOpenSubagentTabOnlyPathIsAssociated() {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const parentId = ready.tabId;

  send(win, {type: 'status', running: true, tabId: parentId});
  send(win, {type: 'tool_call', name: 'run_parallel', tabId: parentId});
  const panel = runParallelPanel(win);
  assert.ok(panel, 'run_parallel tool_call must render a panel');

  send(win, {
    type: 'openSubagentTab',
    tab_id: parentId + '__sub_replayed-task',
    parent_tab_id: parentId,
    description: 'replayed sub',
    task_id: 'replayed-task',
    taskIndex: 0,
  });
  assert.strictEqual(subagentTabEls(win).length, 1, 'replayed sub tab open');

  togglePanel(win, panel);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'INVARIANT VIOLATED: openSubagentTab-only sub tab stayed open ' +
      'after collapsing the run_parallel panel',
  );

  const before = posted.length;
  togglePanel(win, panel);
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'expanding must reopen an openSubagentTab-only sub-agent tab',
  );
  assert.ok(
    posted
      .slice(before)
      .some(m => m.type === 'resumeSession' && m.taskId === 'replayed-task'),
    'reopening an openSubagentTab-only sub tab must resume its task id',
  );
  win.close();
  console.log('  ok - openSubagentTab-only path is associated with panel');
}

// ---------------------------------------------------------------------------
// 7. Sub-agents spawned while the panel is collapsed must NOT open
//    tabs; expanding the panel afterwards must open them all.
// ---------------------------------------------------------------------------
function testSpawnWhileCollapsedDefersTabs() {
  const {win, posted, panel, parentId} = bootParallelRun(2);

  togglePanel(win, panel); // collapse → both tabs close
  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'sub-task-3',
    parent_tab_id: parentId,
    taskId: '',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'INVARIANT VIOLATED: a sub-agent spawned while the run_parallel ' +
      'panel is collapsed must not open a tab',
  );
  assert.ok(
    !posted.slice(before).some(m => m.type === 'resumeSession'),
    'no resumeSession must be posted while the panel is collapsed',
  );

  togglePanel(win, panel); // expand → all three tabs open
  assert.strictEqual(
    subagentTabEls(win).length,
    3,
    'expanding the panel must open the deferred sub-agent tab too',
  );
  assert.ok(
    posted
      .slice(before)
      .some(m => m.type === 'resumeSession' && m.taskId === 'sub-task-3'),
    'the deferred sub-agent must be resumed when the panel expands',
  );
  win.close();
  console.log('  ok - spawns while collapsed are deferred until expand');
}

async function main() {
  const tests = [
    testCollapseClosesSubagentTabs,
    testExpandReopensSubagentTabs,
    testManualSubTabCloseCollapsesPanel,
    testAutoCollapseKeepsInvariant,
    testDelayedOpenSubagentTabDoesNotReopenCollapsedPanel,
    testOpenSubagentTabOnlyPathIsAssociated,
    testSpawnWhileCollapsedDefersTabs,
  ];
  for (const t of tests) {
    await t();
  }
  console.log('runParallelPanelTabsSync.test.js: all tests passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
