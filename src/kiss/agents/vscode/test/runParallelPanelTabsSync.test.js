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

/** The daemon reports every child of the fan-out finished. */
function finishFanOut(win, subTabIds) {
  for (const id of subTabIds) send(win, {type: 'subagentDone', tab_id: id});
}

/** The daemon's reply to resuming finished children: history tabs. */
function announceFinished(win, parentId, taskIds, subTabIds) {
  taskIds.forEach((taskId, i) =>
    send(win, {
      type: 'openSubagentTab',
      tab_id: subTabIds[i],
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isDone: true,
    }),
  );
}

function clickClose(win, tabId) {
  const btn = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"] .chat-tab-close`,
  );
  assert.ok(btn, 'sub-agent tab ' + tabId + ' must render a close button');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
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
    2,
    'collapsing the panel of a RUNNING fan-out must keep its sub-agent ' +
      'tabs open (they are the children\'s question/answer surface)',
  );
  assert.ok(
    !posted.some(m => m.type === 'closeTab'),
    'the backend must not be told to close a running sub-agent tab',
  );

  finishFanOut(win, subTabIds);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'subagentDone must close every finished sub-agent tab',
  );
  for (const id of subTabIds) {
    assert.ok(
      posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close finished sub-agent tab ' + id,
    );
  }
  assert.ok(
    panel.classList.contains('collapsed'),
    'the panel stays collapsed once its children are gone',
  );
  win.close();
  console.log('  ok - collapse keeps running sub tabs; subagentDone closes them');
}

function testExpandReopensSubagentTabs() {
  const {win, posted, panel, taskIds, subTabIds} = bootParallelRun(2);

  togglePanel(win, panel);
  let before = posted.length;
  togglePanel(win, panel);
  assert.ok(
    !panel.classList.contains('collapsed'),
    'second click must uncollapse the run_parallel panel',
  );
  assert.deepStrictEqual(
    subagentTabEls(win).map(el => el.dataset.tabId).sort(),
    [...subTabIds].sort(),
    'the running sub-agent tabs stayed open across collapse/expand',
  );
  assert.ok(
    !posted.slice(before).some(m => m.type === 'resumeSession'),
    'expanding must not resubscribe tabs that never closed',
  );

  finishFanOut(win, subTabIds);
  assert.strictEqual(subagentTabEls(win).length, 0, 'children closed');
  assert.ok(
    panel.classList.contains('collapsed'),
    'a fan-out whose last child tab closed collapses on its own',
  );

  before = posted.length;
  togglePanel(win, panel);
  assert.ok(
    !panel.classList.contains('collapsed'),
    'clicking the header must uncollapse the finished run_parallel panel',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'INVARIANT VIOLATED: run_parallel panel is uncollapsed but its ' +
      'finished sub-agent tabs are not open',
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
  console.log('  ok - expanding the run_parallel panel reopens finished tabs');
}

function testManualSubTabCloseClosesOnlyThatTab() {
  const {win, posted, panel, subTabIds} = bootParallelRun(2);

  // A running sub-agent's tab stays open unless the USER closes it: the
  // hand close removes that tab alone and is echoed to the daemon, which
  // mirrors it to every other surface.
  clickClose(win, subTabIds[0]);
  assert.ok(
    posted.some(m => m.type === 'closeTab' && m.tabId === subTabIds[0]),
    'the backend must be told about the hand close of the running tab',
  );
  const collapsed = panel.classList.contains('collapsed');
  const openIds = subagentTabEls(win).map(el => el.dataset.tabId);
  assert.ok(
    !collapsed && openIds.length === 1 && openIds[0] === subTabIds[1],
    'the hand close must close only its tab and leave the sibling ' +
      'open and the panel uncollapsed (collapsed=' +
      collapsed +
      ', open sub tabs=' +
      JSON.stringify(openIds) +
      ')',
  );
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  assert.deepStrictEqual(
    subagentTabEls(win).map(el => el.dataset.tabId),
    [subTabIds[1]],
    'the closed sub-agent finishing changes nothing',
  );
  win.close();
  console.log('  ok - hand-close of a running sub tab closes only that tab');
}

function testManualSubTabCloseKeepsSiblingsOpen() {
  const {win, posted, panel, parentId, subTabIds} = bootParallelRun(3);

  clickClose(win, subTabIds[0]);
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'a hand close of a running sub-agent tab closes that tab',
  );
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

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
    isDone: true,
  });
  const openAfter = subagentTabEls(win).map(el => el.dataset.tabId);
  assert.deepStrictEqual(
    openAfter.sort(),
    [subTabIds[1], subTabIds[2]].sort(),
    'a later sync must not reopen the finished, closed sub-agent tab or ' +
      'close the surviving siblings (open sub tabs: ' +
      JSON.stringify(openAfter) +
      ')',
  );
  win.close();
  console.log('  ok - one child finishing keeps sibling sub tabs open');
}

function testManualCloseOfAllSubTabsThenExpandReopensAll() {
  const {win, posted, panel, parentId, taskIds, subTabIds} =
    bootParallelRun(2);

  for (const id of subTabIds) clickClose(win, id);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'hand-closing running sub-agent tabs closes them',
  );
  assert.ok(
    panel.classList.contains('collapsed'),
    'a fan-out whose every child tab the user closed collapses',
  );

  finishFanOut(win, subTabIds);
  togglePanel(win, panel);
  assert.strictEqual(subagentTabEls(win).length, 2, 'finished tabs reopen');
  announceFinished(win, parentId, taskIds, subTabIds);

  for (const id of subTabIds) clickClose(win, id);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'closing every finished sub-agent tab by hand must leave none open',
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
  console.log('  ok - closing all finished sub tabs by hand, expand reopens all');
}

function testAutoCollapseKeepsInvariant() {
  const {win, panel, parentId, subTabIds} = bootParallelRun(2);

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

  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'INVARIANT VIOLATED: an automatic collapse pass closed the tabs of ' +
      'sub-agents that are still running (panel collapsed=' +
      panel.classList.contains('collapsed') +
      ')',
  );
  finishFanOut(win, subTabIds);
  assert.ok(
    panel.classList.contains('collapsed') && subagentTabEls(win).length === 0,
    'once the children finish their tabs close and the panel collapses',
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
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'collapse keeps the running sub-agent tab open',
  );

  send(win, {
    type: 'openSubagentTab',
    tab_id: resume.tabId,
    parent_tab_id: parentId,
    description: 'late sub',
    task_id: 'late-sub-task',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'a delayed announcement of the running child lands on its open tab',
  );

  send(win, {type: 'subagentDone', tab_id: resume.tabId});
  assert.strictEqual(subagentTabEls(win).length, 0, 'done closes the tab');
  send(win, {
    type: 'openSubagentTab',
    tab_id: resume.tabId,
    parent_tab_id: parentId,
    description: 'late sub',
    task_id: 'late-sub-task',
    isDone: true,
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'INVARIANT VIOLATED: delayed openSubagentTab recreated a finished ' +
      'sub-agent tab while the owning run_parallel panel is collapsed',
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
    isDone: true,
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
    3,
    'a sub-agent spawned while the run_parallel panel is collapsed ' +
      'must still get its tab: a live child is never left tabless',
  );
  assert.ok(
    posted
      .slice(before)
      .some(m => m.type === 'resumeSession' && m.taskId === 'sub-task-3'),
    'the spawned sub-agent must be resumed right away',
  );
  assert.ok(
    panel.classList.contains('collapsed'),
    'the spawn does not unfold the panel the user collapsed',
  );

  const beforeExpand = posted.length;
  togglePanel(win, panel);
  assert.strictEqual(
    subagentTabEls(win).length,
    3,
    'expanding the panel keeps exactly one tab per running sub-agent',
  );
  assert.ok(
    !posted.slice(beforeExpand).some(m => m.type === 'resumeSession'),
    'expanding must not resubscribe tabs that never closed',
  );
  win.close();
  console.log('  ok - spawns while collapsed open their tab immediately');
}

function testTaskEndPassKeepsUserOpenedFanOut() {
  const {win, panel, parentId, taskIds, subTabIds} = bootParallelRun(2);

  // The children finish and the user reopens their history tabs.
  finishFanOut(win, subTabIds);
  togglePanel(win, panel);
  assert.strictEqual(subagentTabEls(win).length, 2, 'finished tabs reopen');
  announceFinished(win, parentId, taskIds, subTabIds);

  send(win, {
    type: 'tool_result',
    tabId: parentId,
    content: 'all sub-agents done',
  });
  send(win, {type: 'result', tabId: parentId, summary: 'done', success: true});
  send(win, {type: 'status', running: false, tabId: parentId});
  send(win, {type: 'usage_info', tabId: parentId});
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the task-end pass must leave a run_parallel panel the user opened ' +
      'open',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'a fan-out panel the user keeps open keeps its sub-agent tabs',
  );
  assert.strictEqual(
    win.document.getElementById('task-panel-collapse-btn'),
    null,
    'the removed Collapse/Uncollapse Chats button must not exist',
  );
  win.close();
  console.log('  ok - task-end pass keeps a user-opened fan-out and its tabs');
}

function testRunParallelFinishAutoCollapseClosesSubTabs() {
  const {win, posted, panel, parentId, taskIds, subTabIds} =
    bootParallelRun(2);

  // run_parallel returns only after its last child finished.
  finishFanOut(win, subTabIds);
  assert.ok(
    panel.classList.contains('collapsed'),
    'a fan-out whose children all finished collapses on its own',
  );
  send(win, {
    type: 'tool_result',
    tabId: parentId,
    content: 'all sub-agents done',
  });
  send(win, {type: 'thinking_start', tabId: parentId});
  send(win, {type: 'thinking_delta', tabId: parentId, text: 'wrapping up'});
  send(win, {type: 'thinking_end', tabId: parentId});
  // The stream keeps its newest two panels open, so the agent has to
  // move on by one more panel before the fan-out folds.
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: parentId});

  assert.ok(
    panel.classList.contains('collapsed'),
    'after the run_parallel tool finished and the agent moved on, the ' +
      'auto-collapse pass must leave the run_parallel panel collapsed ' +
      'like every other tool panel',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'INVARIANT VIOLATED: the finished run_parallel panel is collapsed ' +
      'but its sub-agent tabs remain open',
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

  // A finished parent has finished children: reopen their history tabs.
  finishFanOut(win, subTabIds);
  togglePanel(win, panel);
  assert.strictEqual(subagentTabEls(win).length, 2, 'finished tabs reopen');
  announceFinished(win, parentId, taskIds, subTabIds);
  const closesBefore = posted.filter(m => m.type === 'closeTab').length;

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
  const replayCloses = posted
    .filter(m => m.type === 'closeTab')
    .slice(closesBefore)
    .map(m => m.tabId);
  for (const id of subTabIds) {
    assert.ok(
      replayCloses.includes(id),
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
    testTaskEndPassKeepsUserOpenedFanOut,
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
