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

function runParallelPanels(win) {
  return Array.from(win.document.querySelectorAll('#output .tc-run-parallel'));
}

function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

function openSubTabIds(win) {
  return subagentTabEls(win)
    .map(el => el.dataset.tabId)
    .sort();
}

function togglePanel(win, panel) {
  const hdr = panel.querySelector('.tc-h');
  hdr.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function switchToTabEl(win, tabId) {
  const el = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"]`,
  );
  assert.ok(el, 'tab ' + tabId + ' must be rendered in the tab bar');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function spawnSub(win, posted, parentId, taskId, desc, idx) {
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
  assert.ok(
    resume,
    'new_tab for ' +
      taskId +
      ' under parent ' +
      parentId +
      ' must make the webview post resumeSession (a tab must open for ' +
      'EVERY sub-agent, irrespective of how many run_parallel calls ' +
      'the agent already made)',
  );
  send(win, {
    type: 'openSubagentTab',
    tab_id: resume.tabId,
    parent_tab_id: parentId,
    description: desc,
    task_id: taskId,
    taskIndex: idx,
  });
  assert.ok(
    subagentTabEls(win).some(el => el.dataset.tabId === resume.tabId),
    'sub-agent ' + taskId + ' must get its own OPEN tab',
  );
  return resume.tabId;
}

function runParallelCall(win, posted, agentTabId, taskIds, descPrefix) {
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: agentTabId,
    extras: {tasks: JSON.stringify(taskIds)},
  });
  const subTabIds = [];
  for (let i = 0; i < taskIds.length; i++) {
    subTabIds.push(
      spawnSub(win, posted, agentTabId, taskIds[i], descPrefix + (i + 1), i),
    );
  }
  return subTabIds;
}

function bootRunningRoot() {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const rootId = ready.tabId;
  send(win, {
    type: 'status',
    running: true,
    tabId: rootId,
    startTs: Date.now(),
  });
  return {win, posted, rootId};
}

function testThreeSequentialRunParallelCallsOpenTabs() {
  const {win, posted, rootId} = bootRunningRoot();

  for (let k = 1; k <= 3; k++) {
    const taskIds = ['call' + k + '-sub-1', 'call' + k + '-sub-2'];
    const subTabIds = runParallelCall(
      win,
      posted,
      rootId,
      taskIds,
      'c' + k + ' sub ',
    );
    assert.strictEqual(
      subagentTabEls(win).length,
      2,
      'run_parallel call #' +
        k +
        ' must open one tab per sub-agent (got ' +
        subagentTabEls(win).length +
        ')',
    );
    const panels = runParallelPanels(win);
    assert.strictEqual(panels.length, k, 'call #' + k + ' renders panel #' + k);
    const panel = panels[k - 1];
    assert.ok(
      !panel.classList.contains('collapsed'),
      'panel #' + k + ' must start uncollapsed',
    );

    send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
    assert.deepStrictEqual(
      openSubTabIds(win),
      [subTabIds[1]],
      'subagentDone for ' +
        taskIds[0] +
        ' must close ONLY the corresponding tab',
    );
    assert.ok(
      !panel.classList.contains('collapsed'),
      'panel #' + k + ' must stay uncollapsed while a sibling tab is open',
    );
    send(win, {type: 'subagentDone', tab_id: subTabIds[1]});
    assert.strictEqual(
      subagentTabEls(win).length,
      0,
      'all sub-agent tabs of call #' + k + ' must be closed when done',
    );
    assert.ok(
      panel.classList.contains('collapsed'),
      'panel #' + k + ' must collapse once its whole fan-out finished',
    );

    send(win, {
      type: 'tool_result',
      tabId: rootId,
      content: 'call ' + k + ' done',
    });
    send(win, {type: 'thinking_start', tabId: rootId});
    send(win, {type: 'thinking_delta', tabId: rootId, text: 'next'});
    send(win, {type: 'thinking_end', tabId: rootId});
  }
  assert.strictEqual(
    runParallelPanels(win).length,
    3,
    'three run_parallel calls render three panels',
  );
  win.close();
  console.log('  ok - three sequential run_parallel calls each open tabs');
}

function testPerPanelExpandCollapseIndependence() {
  const {win, posted, rootId} = bootRunningRoot();

  const callTaskIds = [];
  for (let k = 1; k <= 3; k++) {
    const taskIds = ['call' + k + '-sub-1', 'call' + k + '-sub-2'];
    callTaskIds.push(taskIds);
    const subTabIds = runParallelCall(
      win,
      posted,
      rootId,
      taskIds,
      'c' + k + ' sub ',
    );
    send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
    send(win, {type: 'subagentDone', tab_id: subTabIds[1]});
    send(win, {
      type: 'tool_result',
      tabId: rootId,
      content: 'call ' + k + ' done',
    });
  }
  send(win, {
    type: 'tool_call',
    name: 'finish',
    tabId: rootId,
    extras: {summary: 'done'},
  });
  send(win, {type: 'result', tabId: rootId, summary: 'done', success: true});
  send(win, {type: 'status', running: false, tabId: rootId});
  assert.strictEqual(subagentTabEls(win).length, 0, 'all fan-outs closed');

  const panels = runParallelPanels(win);
  assert.strictEqual(panels.length, 3, 'three panels rendered');

  let before = posted.length;
  togglePanel(win, panels[0]);
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'expanding panel #1 must reopen exactly its own 2 sub-agent tabs',
  );
  for (const taskId of callTaskIds[0]) {
    assert.ok(
      posted
        .slice(before)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      'expanding panel #1 must resume its own sub-agent ' + taskId,
    );
  }
  const call1TabIds = openSubTabIds(win);

  before = posted.length;
  togglePanel(win, panels[2]);
  assert.strictEqual(
    subagentTabEls(win).length,
    4,
    "expanding panel #3 must open its own 2 tabs and leave panel #1's " +
      '2 tabs open',
  );
  for (const taskId of callTaskIds[2]) {
    assert.ok(
      posted
        .slice(before)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      'expanding panel #3 must resume its own sub-agent ' + taskId,
    );
  }
  for (const taskId of callTaskIds[0]) {
    assert.ok(
      !posted
        .slice(before)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      "expanding panel #3 must NOT touch panel #1's sub-agent " + taskId,
    );
  }

  togglePanel(win, panels[2]);
  assert.deepStrictEqual(
    openSubTabIds(win),
    call1TabIds,
    'BUG: collapsing panel #3 must close ONLY the tabs spawned by ' +
      "run_parallel call #3 — call #1's tabs must stay open",
  );
  assert.ok(
    !panels[0].classList.contains('collapsed'),
    'panel #1 must stay uncollapsed (its tabs are open)',
  );

  togglePanel(win, panels[0]);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'collapsing panel #1 must close its own sub-agent tabs',
  );

  before = posted.length;
  togglePanel(win, panels[1]);
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'expanding panel #2 must reopen exactly its own 2 sub-agent tabs',
  );
  for (const taskId of callTaskIds[1]) {
    assert.ok(
      posted
        .slice(before)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      'expanding panel #2 must resume its own sub-agent ' + taskId,
    );
  }
  win.close();
  console.log('  ok - per-panel expand/collapse touches only its own tabs');
}

function testThreeLevelNestedRunParallel() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(
    win,
    posted,
    rootId,
    ['l1-a', 'l1-b', 'l1-c'],
    'L1 sub ',
  );
  assert.strictEqual(subagentTabEls(win).length, 3, 'level-1 tabs open');

  send(win, {type: 'thinking_start', tabId: l1[0]});
  send(win, {type: 'thinking_delta', tabId: l1[0], text: 'fanning out'});
  send(win, {type: 'thinking_end', tabId: l1[0]});
  const l2 = runParallelCall(
    win,
    posted,
    l1[0],
    ['l2-a', 'l2-b', 'l2-c'],
    'L2 sub ',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    6,
    "a sub-agent's run_parallel must open tabs for ITS sub-agents too",
  );

  send(win, {type: 'thinking_start', tabId: l2[0]});
  send(win, {type: 'thinking_delta', tabId: l2[0], text: 'fanning out'});
  send(win, {type: 'thinking_end', tabId: l2[0]});
  const l3 = runParallelCall(
    win,
    posted,
    l2[0],
    ['l3-a', 'l3-b', 'l3-c'],
    'L3 sub ',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    9,
    'a 3rd-level run_parallel must open tabs for its sub-agents too',
  );

  send(win, {type: 'subagentDone', tab_id: l3[0]});
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1, ...l2, l3[1], l3[2]].sort(),
    "finishing l3-a must close ONLY l3-a's tab",
  );
  send(win, {type: 'subagentDone', tab_id: l3[1]});
  send(win, {type: 'subagentDone', tab_id: l3[2]});
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1, ...l2].sort(),
    'all level-3 tabs closed, levels 1–2 untouched',
  );

  send(win, {type: 'tool_result', tabId: l2[0], content: 'l3 done'});
  send(win, {type: 'result', tabId: l2[0], summary: 'done', success: true});
  send(win, {type: 'subagentDone', tab_id: l2[0]});
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1, l2[1], l2[2]].sort(),
    "finishing l2-a must close ONLY l2-a's tab",
  );
  send(win, {type: 'subagentDone', tab_id: l2[1]});
  send(win, {type: 'subagentDone', tab_id: l2[2]});
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1].sort(),
    'all level-2 tabs closed, level 1 untouched',
  );

  send(win, {type: 'tool_result', tabId: l1[0], content: 'l2 done'});
  send(win, {type: 'result', tabId: l1[0], summary: 'done', success: true});
  send(win, {type: 'subagentDone', tab_id: l1[0]});
  assert.deepStrictEqual(
    openSubTabIds(win),
    [l1[1], l1[2]].sort(),
    "finishing l1-a must close ONLY l1-a's tab",
  );
  send(win, {type: 'subagentDone', tab_id: l1[1]});
  send(win, {type: 'subagentDone', tab_id: l1[2]});
  assert.strictEqual(subagentTabEls(win).length, 0, 'all fan-outs closed');
  assert.ok(
    runParallelPanels(win)[0].classList.contains('collapsed'),
    'root panel collapses once its whole fan-out finished',
  );
  win.close();
  console.log('  ok - 3-level nested run_parallel opens/closes per level');
}

function testNestedPanelCollapseExpand() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a', 'l1-b'], 'L1 ');
  runParallelCall(win, posted, l1[0], ['l2-a', 'l2-b'], 'L2 ');
  assert.strictEqual(subagentTabEls(win).length, 4, 'both levels open');

  switchToTabEl(win, l1[0]);
  const nestedPanels = runParallelPanels(win);
  assert.strictEqual(
    nestedPanels.length,
    1,
    'the sub-agent tab shows its own run_parallel panel',
  );
  const nestedPanel = nestedPanels[0];
  assert.ok(
    !nestedPanel.classList.contains('collapsed'),
    'nested panel starts uncollapsed',
  );
  togglePanel(win, nestedPanel);
  assert.ok(
    nestedPanel.classList.contains('collapsed'),
    'clicking the nested header collapses the nested panel',
  );
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1].sort(),
    'collapsing the NESTED panel must close ONLY the sub-sub-agent ' +
      'tabs it spawned (level-1 tabs stay open)',
  );

  const before = posted.length;
  togglePanel(win, nestedPanel);
  assert.strictEqual(
    subagentTabEls(win).length,
    4,
    'expanding the nested panel must reopen its sub-sub-agent tabs',
  );
  for (const taskId of ['l2-a', 'l2-b']) {
    assert.ok(
      posted
        .slice(before)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      'reopened sub-sub-agent tab must resume backend task ' + taskId,
    );
  }
  const l2New = openSubTabIds(win).filter(id => !l1.includes(id));
  assert.strictEqual(l2New.length, 2, 'two fresh level-2 tabs');

  switchToTabEl(win, rootId);
  const rootPanel = runParallelPanels(win)[0];
  assert.ok(rootPanel, 'root panel present in the root chat DOM');
  togglePanel(win, rootPanel);
  assert.ok(rootPanel.classList.contains('collapsed'), 'root collapsed');
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'collapsing the root panel must close the level-1 tabs and their ' +
      'still-open descendants',
  );
  win.close();
  console.log('  ok - nested panel collapse/expand closes/reopens its tabs');
}

function testSubagentMakesMultipleRunParallelCalls() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a'], 'L1 ');

  const c1 = runParallelCall(win, posted, l1[0], ['n1-a', 'n1-b'], 'N1 ');
  send(win, {type: 'subagentDone', tab_id: c1[0]});
  send(win, {type: 'subagentDone', tab_id: c1[1]});
  send(win, {type: 'tool_result', tabId: l1[0], content: 'call 1 done'});
  send(win, {type: 'thinking_start', tabId: l1[0]});
  send(win, {type: 'thinking_delta', tabId: l1[0], text: 'next call'});
  send(win, {type: 'thinking_end', tabId: l1[0]});
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1].sort(),
    'nested call #1 fan-out fully closed after both subagentDone',
  );

  const c2 = runParallelCall(win, posted, l1[0], ['n2-a', 'n2-b'], 'N2 ');
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1, ...c2].sort(),
    "the sub-agent's SECOND run_parallel call must open one tab per " +
      'sub-agent, exactly like the first call',
  );

  send(win, {type: 'subagentDone', tab_id: c2[0]});
  send(win, {type: 'subagentDone', tab_id: c2[1]});
  send(win, {type: 'tool_result', tabId: l1[0], content: 'call 2 done'});
  send(win, {type: 'thinking_start', tabId: l1[0]});
  send(win, {type: 'thinking_delta', tabId: l1[0], text: 'next call'});
  send(win, {type: 'thinking_end', tabId: l1[0]});
  const c3 = runParallelCall(win, posted, l1[0], ['n3-a', 'n3-b'], 'N3 ');
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1, ...c3].sort(),
    "the sub-agent's THIRD run_parallel call must open one tab per " +
      'sub-agent',
  );

  switchToTabEl(win, l1[0]);
  const panels = runParallelPanels(win);
  assert.strictEqual(panels.length, 3, 'three nested panels rendered');
  const before = posted.length;
  togglePanel(win, panels[0]);
  assert.ok(!panels[0].classList.contains('collapsed'), 'panel #1 expanded');
  for (const taskId of ['n1-a', 'n1-b']) {
    assert.ok(
      posted
        .slice(before)
        .some(m => m.type === 'resumeSession' && m.taskId === taskId),
      'expanding nested panel #1 must resume its own sub-agent ' + taskId,
    );
  }
  assert.strictEqual(
    subagentTabEls(win).length,
    5,
    "panel #1's 2 reopened tabs + panel #3's 2 live tabs + l1-a",
  );
  togglePanel(win, panels[2]);
  assert.strictEqual(
    subagentTabEls(win).length,
    3,
    'BUG: collapsing nested panel #3 must close ONLY its own 2 tabs — ' +
      "panel #1's reopened tabs must stay open",
  );
  assert.ok(
    !panels[0].classList.contains('collapsed'),
    'nested panel #1 must stay uncollapsed (its tabs are open)',
  );
  win.close();
  console.log("  ok - a sub-agent's repeated run_parallel calls open tabs");
}

function testSubagentResultAutoCollapseClosesNestedTabs() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a', 'l1-b'], 'L1 ');
  const l2 = runParallelCall(win, posted, l1[0], ['l2-a', 'l2-b'], 'L2 ');
  assert.strictEqual(subagentTabEls(win).length, 4, 'both levels open');

  send(win, {type: 'tool_result', tabId: l1[0], content: 'nested done'});
  send(win, {type: 'result', tabId: l1[0], summary: 'done', success: true});
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1].sort(),
    "the sub-agent's result must auto-collapse its finished nested " +
      "run_parallel panel and close the nested fan-out's tabs " +
      '(open now: ' +
      JSON.stringify(openSubTabIds(win)) +
      ', expected only level-1: ' +
      JSON.stringify([...l1].sort()) +
      ')',
  );
  for (const id of l2) {
    assert.ok(
      posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close nested sub-agent tab ' + id,
    );
  }
  win.close();
  console.log('  ok - sub-agent result auto-collapse closes nested tabs');
}

function testAdjacentHistoryRunParallelPanelIsInert() {
  const {win, posted, rootId} = bootRunningRoot();
  send(win, {type: 'status', running: false, tabId: rootId});

  send(win, {
    type: 'adjacent_task_events',
    tabId: rootId,
    direction: 'prev',
    task: 'Older parallel task',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Older parallel task'},
      {
        type: 'tool_call',
        name: 'run_parallel',
        tabId: 'ghost-tab-from-old-session',
        extras: {tasks: JSON.stringify(['old sub 1'])},
      },
      {type: 'tool_result', content: 'done'},
      {type: 'result', summary: 'done', success: true},
    ],
  });
  const adj = win.document.querySelector('#output .adjacent-task');
  assert.ok(adj, 'adjacent task container must render');
  const panel = adj.querySelector('.tc-run-parallel');
  assert.ok(panel, 'the history block renders its run_parallel panel');
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'rendering a history run_parallel panel must not open tabs',
  );

  const before = posted.length;
  togglePanel(win, panel);
  assert.ok(!panel.classList.contains('collapsed'), 'panel expanded');
  togglePanel(win, panel);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'toggling a history run_parallel panel must not open tabs',
  );
  assert.ok(
    !posted.slice(before).some(m => m.type === 'resumeSession'),
    'toggling a history run_parallel panel must not resume anything',
  );
  win.close();
  console.log('  ok - adjacent-history run_parallel panel is inert');
}

function testSpawnUnderFragmentlessParentStillOpensTab() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a'], 'L1 ');
  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'g-task',
    parent_tab_id: l1[0],
    taskId: '',
  });
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === 'g-task');
  assert.ok(
    resume,
    'a sub-agent spawned under a DOM-less parent tab must still open ' +
      'a tab (resumeSession posted)',
  );
  send(win, {
    type: 'openSubagentTab',
    tab_id: resume.tabId,
    parent_tab_id: l1[0],
    description: 'grandchild',
    task_id: 'g-task',
    taskIndex: 0,
  });
  assert.deepStrictEqual(
    openSubTabIds(win),
    [l1[0], resume.tabId].sort(),
    'the grandchild tab must be open next to its parent sub-agent tab',
  );

  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: l1[0],
    extras: {tasks: JSON.stringify(['g-task'])},
  });
  send(win, {type: 'tool_result', tabId: l1[0], content: 'done'});
  send(win, {type: 'result', tabId: l1[0], summary: 'done', success: true});
  assert.deepStrictEqual(
    openSubTabIds(win),
    [l1[0]],
    'the late-rendered nested panel must adopt the unregistered ' +
      'grandchild tab and close it when the panel auto-collapses',
  );
  win.close();
  console.log('  ok - spawn under a DOM-less parent still opens a tab');
}

function testMultiPanelParentReplayAdoptsPerCall() {
  const {win, posted, rootId} = bootRunningRoot();

  const groups = [];
  for (let k = 1; k <= 3; k++) {
    groups.push(
      runParallelCall(win, posted, rootId, ['replay-c' + k], 'RC' + k + ' '),
    );
  }
  assert.strictEqual(subagentTabEls(win).length, 3, 'three live fan-outs');

  const rpEv = k => ({
    type: 'tool_call',
    name: 'run_parallel',
    tabId: rootId,
    extras: {tasks: JSON.stringify(['replay-c' + k])},
  });
  send(win, {
    type: 'task_events',
    tabId: rootId,
    task: 'multi replay',
    task_id: 'parent-task',
    events: [
      rpEv(1),
      {type: 'tool_result', tabId: rootId, content: 'c1 done'},
      rpEv(2),
      {type: 'tool_result', tabId: rootId, content: 'c2 done'},
      rpEv(3),
      {type: 'tool_result', tabId: rootId, content: 'c3 done'},
      {type: 'result', tabId: rootId, summary: 'done', success: true},
    ],
  });
  const panels = runParallelPanels(win);
  assert.strictEqual(panels.length, 3, 'replay renders three panels');
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    "the replay collapse must adopt and close EVERY call's tabs",
  );

  for (let k = 1; k <= 3; k++) {
    const before = posted.length;
    togglePanel(win, panels[k - 1]);
    const resumed = posted
      .slice(before)
      .filter(m => m.type === 'resumeSession')
      .map(m => m.taskId);
    assert.deepStrictEqual(
      resumed,
      ['replay-c' + k],
      'BUG: expanding replayed panel #' +
        k +
        " must resume only ITS call's sub-agent (resumed: " +
        JSON.stringify(resumed) +
        ')',
    );
    togglePanel(win, panels[k - 1]);
    assert.strictEqual(subagentTabEls(win).length, 0, 'group closed');
  }
  win.close();
  console.log('  ok - multi-panel parent replay adopts per call');
}

function testHistoryReopenGroupsPersistedSubsByCall() {
  const {win, posted, rootId} = bootRunningRoot();
  send(win, {type: 'status', running: false, tabId: rootId});

  const rpEv = names => ({
    type: 'tool_call',
    name: 'run_parallel',
    tabId: rootId,
    extras: {tasks: JSON.stringify(names)},
  });
  send(win, {
    type: 'task_events',
    tabId: rootId,
    task: 'history parent',
    task_id: 'hist-parent',
    events: [
      rpEv(['h1', 'h2']),
      {type: 'tool_result', tabId: rootId, content: 'c1 done'},
      rpEv(['h3', 'h4', 'h5']),
      {type: 'tool_result', tabId: rootId, content: 'c2 done'},
      {
        type: 'tool_call',
        name: 'run_parallel',
        tabId: rootId,
        extras: {tasks: '[truncated garba'},
      },
      {type: 'tool_result', tabId: rootId, content: 'c3 done'},
      {type: 'result', tabId: rootId, summary: 'done', success: true},
    ],
  });
  const panels = runParallelPanels(win);
  assert.strictEqual(panels.length, 3, 'replay renders three panels');
  for (const p of panels) {
    assert.ok(p.classList.contains('collapsed'), 'panels start collapsed');
  }

  for (let i = 1; i <= 6; i++) {
    send(win, {
      type: 'openSubagentTab',
      tab_id: rootId + '__sub_h' + i,
      parent_tab_id: rootId,
      description: 'hist sub ' + i,
      task_id: 'h' + i,
      taskIndex: i - 1,
      isDone: true,
    });
  }
  send(win, {
    type: 'openSubagentTab',
    tab_id: rootId + '__sub_extra',
    parent_tab_id: rootId,
    description: 'extra row',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'rows behind collapsed panels must not open tabs',
  );

  const expectGroup = (panelIdx, taskIds) => {
    const before = posted.length;
    togglePanel(win, panels[panelIdx]);
    const resumed = posted
      .slice(before)
      .filter(m => m.type === 'resumeSession')
      .map(m => m.taskId)
      .sort();
    assert.deepStrictEqual(
      resumed,
      taskIds.slice().sort(),
      'BUG: expanding history panel #' +
        (panelIdx + 1) +
        ' must reopen exactly its own persisted fan-out (resumed: ' +
        JSON.stringify(resumed) +
        ')',
    );
    assert.strictEqual(
      subagentTabEls(win).length,
      taskIds.length,
      'panel #' + (panelIdx + 1) + ' opens one tab per persisted row',
    );
    togglePanel(win, panels[panelIdx]);
    assert.strictEqual(subagentTabEls(win).length, 0, 'group closed');
  };
  expectGroup(1, ['h3', 'h4', 'h5']);
  expectGroup(0, ['h1', 'h2']);
  expectGroup(2, ['h6']);
  win.close();
  console.log('  ok - history reopen groups persisted subs per call');
}

function testAdjacentHistoryPanelDoesNotStealLiveFanout() {
  const {win, posted, rootId} = bootRunningRoot();

  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: rootId,
    extras: {tasks: JSON.stringify(['live sub'])},
  });
  const livePanel = runParallelPanels(win)[0];
  assert.ok(livePanel, 'live panel rendered');

  send(win, {
    type: 'adjacent_task_events',
    tabId: rootId,
    direction: 'next',
    task: 'Newer old task',
    task_id: '77',
    events: [
      {type: 'task_start', task: 'Newer old task'},
      {
        type: 'tool_call',
        name: 'run_parallel',
        tabId: 'stale-session-tab',
        extras: {tasks: '"not-a-list"'},
      },
      {type: 'tool_result', content: 'done'},
      {type: 'result', summary: 'done', success: true},
    ],
  });
  const adjPanel = win.document.querySelector(
    '#output .adjacent-task .tc-run-parallel',
  );
  assert.ok(adjPanel, 'adjacent history panel rendered');

  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'live-sub-task',
    parent_tab_id: rootId,
    taskId: '',
  });
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === 'live-sub-task');
  assert.ok(
    resume,
    'BUG: the live sub-agent must open its tab — the collapsed ' +
      'adjacent-history run_parallel panel must not defer/own it',
  );
  assert.strictEqual(subagentTabEls(win).length, 1, 'live sub tab open');

  togglePanel(win, livePanel);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'collapsing the live panel must close its sub-agent tab',
  );
  win.close();
  console.log('  ok - adjacent history panel cannot steal a live fan-out');
}

function testDelayedOpenSubagentAttachesToOwningCall() {
  const {win, posted, rootId} = bootRunningRoot();

  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: rootId,
    extras: {tasks: ['late sub']},
  });
  send(win, {
    type: 'new_tab',
    task_id: 't-late',
    parent_tab_id: rootId,
    taskId: '',
  });
  const lateResume = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 't-late',
  );
  assert.ok(lateResume, 'call #1 sub-agent opened');
  const panel1 = runParallelPanels(win)[0];
  togglePanel(win, panel1);
  assert.strictEqual(subagentTabEls(win).length, 0, 'call #1 tab closed');
  send(win, {type: 'tool_result', tabId: rootId, content: 'c1 done'});

  const c2 = runParallelCall(win, posted, rootId, ['t2-a'], 'C2 ');
  const panel2 = runParallelPanels(win)[1];
  assert.strictEqual(subagentTabEls(win).length, 1, 'call #2 tab open');

  send(win, {
    type: 'openSubagentTab',
    tab_id: lateResume.tabId,
    parent_tab_id: rootId,
    description: 'late sub',
    task_id: 't-late',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'the stale conversion must not reopen a tab behind collapsed #1',
  );

  togglePanel(win, panel2);
  let before = posted.length;
  togglePanel(win, panel2);
  let resumed = posted
    .slice(before)
    .filter(m => m.type === 'resumeSession')
    .map(m => m.taskId);
  assert.deepStrictEqual(
    resumed,
    ['t2-a'],
    "BUG: the delayed conversion for call #1's sub-agent leaked into " +
      "call #2's fan-out (panel #2 resumed: " +
      JSON.stringify(resumed) +
      ')',
  );
  assert.strictEqual(subagentTabEls(win).length, 1, 'only t2-a reopened');
  assert.ok(c2.length === 1, 'sanity: one call-#2 sub-agent');

  before = posted.length;
  togglePanel(win, panel1);
  resumed = posted
    .slice(before)
    .filter(m => m.type === 'resumeSession')
    .map(m => m.taskId);
  assert.deepStrictEqual(
    resumed,
    ['t-late'],
    'expanding panel #1 must resume its own delayed sub-agent',
  );
  win.close();
  console.log('  ok - delayed openSubagentTab attaches to the owning call');
}

function testUnregisteredTabAdoptsIntoNewestPanelOnly() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a'], 'L1 ');
  send(win, {
    type: 'new_tab',
    task_id: 'g-task',
    parent_tab_id: l1[0],
    taskId: '',
  });
  const g = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 'g-task',
  );
  assert.ok(g, 'grandchild tab opened');

  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: l1[0],
    extras: {tasks: 'not json {'},
  });
  send(win, {type: 'tool_result', tabId: l1[0], content: 'c1 done'});
  send(win, {type: 'tool_call', name: 'run_parallel', tabId: l1[0]});
  send(win, {type: 'result', tabId: l1[0], summary: 'done', success: true});
  assert.ok(
    subagentTabEls(win).some(el => el.dataset.tabId === g.tabId),
    "the unregistered grandchild must survive call #1's collapse " +
      '(it belongs to the newest, still-running call)',
  );

  switchToTabEl(win, l1[0]);
  const nested = runParallelPanels(win);
  assert.strictEqual(nested.length, 2, 'two nested panels rendered');
  assert.ok(
    nested[0].classList.contains('collapsed'),
    'finished call #1 collapsed at task end',
  );
  togglePanel(win, nested[1]);
  assert.ok(
    !subagentTabEls(win).some(el => el.dataset.tabId === g.tabId),
    'collapsing the newest panel must close the adopted grandchild',
  );
  win.close();
  console.log('  ok - unregistered tab adopts into the newest panel only');
}

async function main() {
  const tests = [
    testThreeSequentialRunParallelCallsOpenTabs,
    testPerPanelExpandCollapseIndependence,
    testThreeLevelNestedRunParallel,
    testNestedPanelCollapseExpand,
    testSubagentMakesMultipleRunParallelCalls,
    testSubagentResultAutoCollapseClosesNestedTabs,
    testAdjacentHistoryRunParallelPanelIsInert,
    testSpawnUnderFragmentlessParentStillOpensTab,
    testMultiPanelParentReplayAdoptsPerCall,
    testHistoryReopenGroupsPersistedSubsByCall,
    testAdjacentHistoryPanelDoesNotStealLiveFanout,
    testDelayedOpenSubagentAttachesToOwningCall,
    testUnregisteredTabAdoptsIntoNewestPanelOnly,
  ];
  for (const t of tests) {
    await t();
  }
  console.log('runParallelMultiCallNested.test.js: all tests passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
