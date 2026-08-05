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

function runParallelPanel(win) {
  const headers = win.document.querySelectorAll('#output .ev.tc .tc-h');
  for (const h of headers) {
    const txt = (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt.startsWith('run_parallel')) return h.closest('.ev.tc');
  }
  return null;
}

function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

function togglePanel(win, panel) {
  const hdr = panel.querySelector('.tc-h');
  hdr.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

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

function testExpandReopensSubagentTabs() {
  const {win, posted, panel, taskIds} = bootParallelRun(2);

  togglePanel(win, panel);
  const before = posted.length;
  togglePanel(win, panel);
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

function testManualSubTabCloseClosesOnlyThatTab() {
  const {win, panel, subTabIds} = bootParallelRun(2);

  const firstEl = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${subTabIds[0]}"] .chat-tab-close`,
  );
  assert.ok(firstEl, 'sub-agent tab must render a close button');
  firstEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const collapsed = panel.classList.contains('collapsed');
  const openSubTabs = subagentTabEls(win).length;
  assert.ok(
    !collapsed && openSubTabs === 1,
    'closing one sub-agent tab by hand must leave the sibling tab ' +
      'open and the panel uncollapsed (collapsed=' +
      collapsed +
      ', open sub tabs=' +
      openSubTabs +
      ')',
  );
  win.close();
  console.log('  ok - manual sub-tab close keeps panel/tabs consistent');
}

function testManualSubTabCloseKeepsSiblingsOpen() {
  const {win, posted, panel, parentId, subTabIds} = bootParallelRun(3);

  const firstEl = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${subTabIds[0]}"] .chat-tab-close`,
  );
  assert.ok(firstEl, 'sub-agent tab must render a close button');
  firstEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const openIds = subagentTabEls(win).map(el => el.dataset.tabId);
  assert.deepStrictEqual(
    openIds.sort(),
    [subTabIds[1], subTabIds[2]].sort(),
    'BUG: closing one sub-agent tab closed its sibling sub-agent ' +
      'tabs too (open sub tabs after close: ' +
      JSON.stringify(openIds) +
      ')',
  );
  for (const id of [subTabIds[1], subTabIds[2]]) {
    assert.ok(
      !posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must NOT be told to close sibling sub-agent tab ' + id,
    );
  }
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the run_parallel panel must stay uncollapsed while sibling ' +
      'sub-agent tabs are open',
  );

  send(win, {type: 'thinking_start', tabId: parentId});
  send(win, {type: 'thinking_delta', tabId: parentId, text: 'waiting'});
  send(win, {type: 'thinking_end', tabId: parentId});
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabIds[1],
    parent_tab_id: parentId,
    description: 'sub 2',
    task_id: 'sub-task-2',
    taskIndex: 1,
  });
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabIds[0],
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
  });
  const openAfter = subagentTabEls(win).map(el => el.dataset.tabId);
  assert.deepStrictEqual(
    openAfter.sort(),
    [subTabIds[1], subTabIds[2]].sort(),
    'a later sync must not reopen the user-closed sub-agent tab or ' +
      'close the surviving siblings (open sub tabs: ' +
      JSON.stringify(openAfter) +
      ')',
  );
  win.close();
  console.log('  ok - manual sub-tab close keeps sibling sub tabs open');
}

function testManualCloseOfAllSubTabsThenExpandReopensAll() {
  const {win, posted, panel, taskIds, subTabIds} = bootParallelRun(2);

  for (const id of subTabIds) {
    const btn = win.document.querySelector(
      `#tab-list .chat-tab[data-tab-id="${id}"] .chat-tab-close`,
    );
    assert.ok(btn, 'sub-agent tab ' + id + ' must render a close button');
    btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  }
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'closing every sub-agent tab by hand must leave none open',
  );
  assert.ok(
    panel.classList.contains('collapsed'),
    'with no open sub-agent tabs left the panel must be collapsed',
  );

  const before = posted.length;
  togglePanel(win, panel);
  assert.ok(
    !panel.classList.contains('collapsed'),
    'clicking the header must uncollapse the run_parallel panel',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'expanding the panel must reopen every sub-agent tab, including ' +
      'those previously closed by hand',
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
  console.log('  ok - closing all sub tabs by hand, expand reopens all');
}

function testAutoCollapseKeepsInvariant() {
  const {win, panel, parentId} = bootParallelRun(2);

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

function testSpawnWhileCollapsedDefersTabs() {
  const {win, posted, panel, parentId} = bootParallelRun(2);

  togglePanel(win, panel);
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

  togglePanel(win, panel);
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

function testTaskEndCollapsePassClosesSubTabs() {
  const {win, panel, parentId} = bootParallelRun(2);

  send(win, {
    type: 'tool_result',
    tabId: parentId,
    content: 'all sub-agents done',
  });
  send(win, {type: 'result', tabId: parentId, summary: 'done', success: true});
  send(win, {type: 'status', running: false, tabId: parentId});
  send(win, {type: 'usage_info', tabId: parentId});
  assert.ok(
    panel.classList.contains('chv-hidden'),
    'the task-end collapse pass must hide the run_parallel panel',
  );
  assert.ok(
    panel.classList.contains('collapsed'),
    'a hidden run_parallel panel must also be marked collapsed',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'the task-end collapse of the finished run_parallel panel must ' +
      'close its sub-agent tabs',
  );
  assert.strictEqual(
    win.document.getElementById('task-panel-collapse-btn'),
    null,
    'the removed Collapse/Uncollapse Chats button must not exist',
  );
  win.close();
  console.log('  ok - task-end collapse pass closes sub tabs');
}

function testRunParallelFinishAutoCollapseClosesSubTabs() {
  const {win, posted, panel, parentId, taskIds, subTabIds} =
    bootParallelRun(2);

  send(win, {
    type: 'tool_result',
    tabId: parentId,
    content: 'all sub-agents done',
  });
  send(win, {type: 'thinking_start', tabId: parentId});
  send(win, {type: 'thinking_delta', tabId: parentId, text: 'wrapping up'});
  send(win, {type: 'thinking_end', tabId: parentId});

  assert.ok(
    panel.classList.contains('collapsed'),
    'BUG REPRODUCED: after the run_parallel tool finished and the ' +
      'agent moved on, the auto-collapse pass must collapse the ' +
      'run_parallel panel like every other tool panel',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'INVARIANT VIOLATED: the agent collapsed the finished ' +
      'run_parallel panel but its sub-agent tabs remain open',
  );
  for (const id of subTabIds) {
    assert.ok(
      posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close sub-agent tab ' + id,
    );
  }

  send(win, {
    type: 'tool_call',
    name: 'finish',
    tabId: parentId,
    extras: {summary: 'done'},
  });
  send(win, {type: 'result', tabId: parentId, summary: 'done', success: true});
  send(win, {type: 'status', running: false, tabId: parentId});
  assert.ok(
    panel.classList.contains('collapsed'),
    'the finished run_parallel panel must stay collapsed at task end',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'sub-agent tabs must stay closed at task end',
  );

  const before = posted.length;
  togglePanel(win, panel);
  assert.ok(!panel.classList.contains('collapsed'), 'panel expanded');
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'expanding the finished run_parallel panel must reopen its tabs',
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
  console.log(
    '  ok - finished run_parallel auto-collapse closes sub tabs',
  );
}

function testRunningFanOutStaysExemptFromAutoCollapse() {
  const {win, panel, parentId} = bootParallelRun(2);

  send(win, {type: 'thinking_start', tabId: parentId});
  send(win, {type: 'thinking_delta', tabId: parentId, text: 'waiting'});
  send(win, {type: 'thinking_end', tabId: parentId});

  assert.ok(
    !panel.classList.contains('collapsed'),
    'a run_parallel panel whose fan-out is still running must stay ' +
      'uncollapsed',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'the live sub-agent tabs must stay open while the fan-out runs',
  );
  win.close();
  console.log('  ok - running fan-out stays exempt from auto-collapse');
}

function testParentReplayAdoptsOpenSubTabsBeforeFinishedCollapse() {
  const {win, posted, panel, parentId, taskIds, subTabIds} =
    bootParallelRun(2);

  send(win, {
    type: 'task_events',
    tabId: parentId,
    task: 'parent replay',
    task_id: 'parent-task',
    events: [
      {type: 'tool_call', name: 'run_parallel', tabId: parentId},
      {
        type: 'tool_result',
        tabId: parentId,
        content: 'all sub-agents done',
      },
      {type: 'result', tabId: parentId, summary: 'done', success: true},
    ],
  });

  const replayedPanel = runParallelPanel(win);
  assert.ok(replayedPanel, 'replay must render a run_parallel panel');
  assert.notStrictEqual(
    replayedPanel,
    panel,
    'task_events replay must replace the old panel DOM element',
  );
  assert.ok(
    replayedPanel.classList.contains('collapsed'),
    'replay collapse must collapse the finished run_parallel panel',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'INVARIANT VIOLATED: replay collapse replaced the run_parallel ' +
      'panel and left its already-open sub-agent tabs open',
  );
  for (const id of subTabIds) {
    assert.ok(
      posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'replay collapse must close adopted sub-agent tab ' + id,
    );
  }

  const before = posted.length;
  togglePanel(win, replayedPanel);
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'expanding the replayed panel must reopen adopted sub-agent tabs',
  );
  for (const taskId of taskIds) {
    assert.ok(
      posted
        .slice(before)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      'reopened adopted sub-agent tab must resume backend task ' + taskId,
    );
  }
  win.close();
  console.log(
    '  ok - parent replay adopts open sub tabs before finished collapse',
  );
}

function testParentReplayKeepsRunningFanOutOpen() {
  const {win, panel, parentId} = bootParallelRun(2);

  send(win, {
    type: 'task_events',
    tabId: parentId,
    task: 'parent replay running',
    task_id: 'parent-task',
    events: [
      {type: 'tool_call', name: 'run_parallel', tabId: parentId},
      {type: 'thinking_start', tabId: parentId},
      {type: 'thinking_delta', tabId: parentId, text: 'waiting'},
      {type: 'thinking_end', tabId: parentId},
    ],
  });

  const replayedPanel = runParallelPanel(win);
  assert.ok(replayedPanel, 'replay must render a run_parallel panel');
  assert.notStrictEqual(
    replayedPanel,
    panel,
    'task_events replay must replace the old panel DOM element',
  );
  assert.ok(
    !replayedPanel.classList.contains('collapsed'),
    'a replayed run_parallel panel whose fan-out is still running ' +
      'must stay uncollapsed',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'live sub-agent tabs must stay open after a running fan-out replay',
  );
  win.close();
  console.log('  ok - parent replay keeps running fan-out open');
}

async function main() {
  const tests = [
    testCollapseClosesSubagentTabs,
    testExpandReopensSubagentTabs,
    testManualSubTabCloseClosesOnlyThatTab,
    testManualSubTabCloseKeepsSiblingsOpen,
    testManualCloseOfAllSubTabsThenExpandReopensAll,
    testAutoCollapseKeepsInvariant,
    testDelayedOpenSubagentTabDoesNotReopenCollapsedPanel,
    testOpenSubagentTabOnlyPathIsAssociated,
    testSpawnWhileCollapsedDefersTabs,
    testTaskEndCollapsePassClosesSubTabs,
    testRunParallelFinishAutoCollapseClosesSubTabs,
    testRunningFanOutStaysExemptFromAutoCollapse,
    testParentReplayAdoptsOpenSubTabsBeforeFinishedCollapse,
    testParentReplayKeepsRunningFanOutOpen,
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
