// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the run_parallel ⇔ sub-agent tabs invariant
// with MULTIPLE run_parallel calls per agent and NESTED (3-level)
// run_parallel fan-outs (``media/main.js`` — the exact production
// webview code shared by the VS Code extension and the remote web
// app):
//
//   * Every run_parallel call an agent OR a sub-agent makes must open
//     one tab per spawned sub-agent — irrespective of how many
//     run_parallel calls that agent/sub-agent already made.
//   * When a sub-agent finishes (``subagentDone``), ONLY its tab
//     closes.
//   * Collapsing a run_parallel panel (by the user or by the agent's
//     automatic collapse passes) closes the tabs of ALL sub-agents
//     spawned by THAT call — and only that call.
//   * Uncollapsing a run_parallel panel reopens the tabs of ALL
//     sub-agents spawned by THAT call — and only that call.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/runParallelMultiCallNested.test.js

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

/** All run_parallel panels in the ACTIVE chat DOM, oldest first. */
function runParallelPanels(win) {
  return Array.from(win.document.querySelectorAll('#output .tc-run-parallel'));
}

/** All sub-agent tabs currently rendered in the tab bar. */
function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

/** Ids of all open sub-agent tabs (sorted). */
function openSubTabIds(win) {
  return subagentTabEls(win)
    .map(el => el.dataset.tabId)
    .sort();
}

/** Click the collapse header of *panel* (toggles .collapsed). */
function togglePanel(win, panel) {
  const hdr = panel.querySelector('.tc-h');
  hdr.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** Click tab *tabId* in the tab bar to switch to it. */
function switchToTabEl(win, tabId) {
  const el = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"]`,
  );
  assert.ok(el, 'tab ' + tabId + ' must be rendered in the tab bar');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/**
 * Replay the exact backend broadcast sequence that spawns ONE
 * sub-agent under *parentId*: ``new_tab`` (webview posts
 * ``resumeSession`` with the freshly allocated tab id) followed by the
 * server's ``openSubagentTab`` conversion.  Returns the sub-agent's
 * frontend tab id.
 */
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

/**
 * Emit the tool_call for one ``run_parallel`` invocation by the agent
 * of *agentTabId* and spawn *taskIds.length* sub-agents under it.
 * Returns the sub-agents' frontend tab ids.
 */
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

/** Boot a webview with a running root task; returns its tab id. */
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

// ---------------------------------------------------------------------------
// 1. An agent makes THREE sequential run_parallel calls.  Every call
//    must open one tab per sub-agent; every subagentDone must close
//    ONLY the finished sub-agent's tab; once a call's whole fan-out is
//    done its panel collapses.
// ---------------------------------------------------------------------------
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

    // First sub-agent finishes → ONLY its tab closes.
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
    // Second (last) sub-agent finishes → its tab closes, panel collapses.
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

// ---------------------------------------------------------------------------
// 2. With THREE finished run_parallel panels in one chat, expanding /
//    collapsing each panel must open / close ONLY the sub-agent tabs
//    spawned by THAT call — never a sibling call's tabs.
// ---------------------------------------------------------------------------
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

  // Expand panel #1 → ONLY call #1's two sub-agents reopen.
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

  // Expand panel #3 → call #3's tabs open IN ADDITION; call #1's stay.
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

  // Collapse panel #3 → ONLY call #3's tabs close; call #1's tabs stay
  // open and panel #1 stays uncollapsed.
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

  // Collapse panel #1 → its tabs close too.
  togglePanel(win, panels[0]);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'collapsing panel #1 must close its own sub-agent tabs',
  );

  // Expand panel #2 → only call #2's fan-out reopens.
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

// ---------------------------------------------------------------------------
// 3. Three-level nesting: the root agent spawns 3 sub-agents; one
//    sub-agent (a background tab) itself calls run_parallel and spawns
//    3 sub-sub-agents; one of THOSE calls run_parallel and spawns 3
//    more.  Every level must open its own tabs, and every
//    subagentDone must close only the corresponding tab.
// ---------------------------------------------------------------------------
function testThreeLevelNestedRunParallel() {
  const {win, posted, rootId} = bootRunningRoot();

  // Level 1: root agent fans out 3 sub-agents.
  const l1 = runParallelCall(
    win,
    posted,
    rootId,
    ['l1-a', 'l1-b', 'l1-c'],
    'L1 sub ',
  );
  assert.strictEqual(subagentTabEls(win).length, 3, 'level-1 tabs open');

  // Level 2: sub-agent l1-a (a BACKGROUND tab) calls run_parallel.
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

  // Level 3: sub-sub-agent l2-a (also a background tab) fans out.
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

  // Level-3 sub-agents finish one by one: each closes ONLY its tab.
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

  // l2-a's run_parallel returns; l2-a finishes → only l2-a's tab closes.
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

  // l1-a's run_parallel returns; level-1 sub-agents finish.
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

// ---------------------------------------------------------------------------
// 4. Collapsing a NESTED run_parallel panel (inside a sub-agent's tab)
//    closes the tabs of that call's sub-sub-agents only; expanding it
//    reopens them.  Collapsing the ROOT panel closes the sub-agent tab
//    AND (by cascade) its still-open descendants.
// ---------------------------------------------------------------------------
function testNestedPanelCollapseExpand() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a', 'l1-b'], 'L1 ');
  runParallelCall(win, posted, l1[0], ['l2-a', 'l2-b'], 'L2 ');
  assert.strictEqual(subagentTabEls(win).length, 4, 'both levels open');

  // The user switches to sub-agent l1-a's tab and collapses its nested
  // run_parallel panel by clicking its header.
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

  // Expanding the nested panel reopens its sub-sub-agent tabs.
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

  // Back on the root tab, collapsing the ROOT panel closes the level-1
  // tabs and cascades to their descendants.
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

// ---------------------------------------------------------------------------
// 5. A SUB-AGENT makes multiple run_parallel calls: the second call
//    must open tabs exactly like the first (irrespective of how many
//    calls were already made), and each nested panel controls only its
//    own fan-out.
// ---------------------------------------------------------------------------
function testSubagentMakesMultipleRunParallelCalls() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a'], 'L1 ');

  // Nested call #1 by sub-agent l1-a; both sub-subs finish.
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

  // Nested call #2 by the SAME sub-agent must open tabs again.
  const c2 = runParallelCall(win, posted, l1[0], ['n2-a', 'n2-b'], 'N2 ');
  assert.deepStrictEqual(
    openSubTabIds(win),
    [...l1, ...c2].sort(),
    "the sub-agent's SECOND run_parallel call must open one tab per " +
      'sub-agent, exactly like the first call',
  );

  // Nested call #3 after #2 completes — still must open tabs.
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

  // The user views l1-a's chat: two collapsed panels (#1, #2) and the
  // live panel #3.  Expanding panel #1 must reopen ONLY its fan-out;
  // collapsing panel #3 must close ONLY its fan-out.
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

// ---------------------------------------------------------------------------
// 6. The agent/sub-agent itself collapses a finished nested panel via
//    the automatic collapse pass at the sub-agent's task end
//    (``result`` → collapseAllExceptResult on the bg tab's DOM): the
//    nested fan-out's still-open tabs must close with it.
// ---------------------------------------------------------------------------
function testSubagentResultAutoCollapseClosesNestedTabs() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a', 'l1-b'], 'L1 ');
  const l2 = runParallelCall(win, posted, l1[0], ['l2-a', 'l2-b'], 'L2 ');
  assert.strictEqual(subagentTabEls(win).length, 4, 'both levels open');

  // The nested run_parallel finishes (its sub-agents' tabs are still
  // open — no subagentDone was delivered, e.g. the sub-sub-agents were
  // interrupted) and l1-a's own task ends: the bg result pass must
  // collapse the finished nested panel, closing its tabs.
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

// ---------------------------------------------------------------------------
// 7. A run_parallel panel rendered inside an ADJACENT-TASK history
//    block carries the tabId of a long-gone session as its parent.
//    Its collapse/expand must be inert: no sub-agent tabs open, no
//    resumeSession is posted, and nothing crashes (the parent tab's
//    chat DOM cannot be resolved).
// ---------------------------------------------------------------------------
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
  togglePanel(win, panel); // expand
  assert.ok(!panel.classList.contains('collapsed'), 'panel expanded');
  togglePanel(win, panel); // collapse
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

// ---------------------------------------------------------------------------
// 8. A sub-agent whose tab has NO chat DOM yet (no event ever streamed
//    to it, so its outputFragment is still null) spawns a sub-sub-
//    agent: the grandchild tab must still open (there is just no panel
//    to associate it with yet).
// ---------------------------------------------------------------------------
function testSpawnUnderFragmentlessParentStillOpensTab() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a'], 'L1 ');
  // No event was ever streamed to l1-a's tab: its chat DOM is empty.
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

  // The run_parallel tool_call reaches l1-a's tab only NOW (late
  // stream): the freshly rendered panel must adopt the already-open,
  // never-registered grandchild tab, so l1-a's task-end collapse pass
  // closes it (collapsed panel ⇒ its sub-agent tabs are closed).
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

// ---------------------------------------------------------------------------
// 9. A parent with THREE still-running run_parallel fan-outs is
//    replayed (task_events re-renders all panels as fresh DOM
//    elements).  Each replayed panel must adopt the open tabs of ITS
//    OWN call (matched by call ordinal) — expanding a replayed panel
//    must resume only that call's sub-agents.
// ---------------------------------------------------------------------------
function testMultiPanelParentReplayAdoptsPerCall() {
  const {win, posted, rootId} = bootRunningRoot();

  const groups = [];
  for (let k = 1; k <= 3; k++) {
    groups.push(
      runParallelCall(win, posted, rootId, ['replay-c' + k], 'RC' + k + ' '),
    );
  }
  assert.strictEqual(subagentTabEls(win).length, 3, 'three live fan-outs');

  // Replay the parent task: every panel is re-rendered (finished, so
  // the replay collapse closes each adopted group).
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

  // Expanding each replayed panel resumes ONLY its own call's task.
  for (let k = 1; k <= 3; k++) {
    const before = posted.length;
    togglePanel(win, panels[k - 1]); // expand
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
    togglePanel(win, panels[k - 1]); // collapse again
    assert.strictEqual(subagentTabEls(win).length, 0, 'group closed');
  }
  win.close();
  console.log('  ok - multi-panel parent replay adopts per call');
}

// ---------------------------------------------------------------------------
// 10. Fresh history reopen: a finished parent with three run_parallel
//     calls (2, 3, and 1 sub-agents) is replayed, then the persisted
//     sub-agent rows arrive via openSubagentTab in spawn order.  The
//     rows must be grouped per call (each panel's expected task count
//     from ``extras.tasks``), so expanding a panel reopens exactly its
//     own fan-out.
// ---------------------------------------------------------------------------
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
      // Call #3 with an unparseable tasks payload: its expected count
      // is unknown, so grouping falls back to the newest panel.
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

  // The server replays the persisted sub-agent rows in spawn order.
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
  // A defensive row without a task id must not break the grouping.
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
    togglePanel(win, panels[panelIdx]); // expand
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
    togglePanel(win, panels[panelIdx]); // collapse again
    assert.strictEqual(subagentTabEls(win).length, 0, 'group closed');
  };
  expectGroup(1, ['h3', 'h4', 'h5']);
  expectGroup(0, ['h1', 'h2']);
  expectGroup(2, ['h6']);
  win.close();
  console.log('  ok - history reopen groups persisted subs per call');
}

// ---------------------------------------------------------------------------
// 11. An adjacent-task history block containing an OLD run_parallel
//     panel is loaded while a LIVE run_parallel call is spawning: the
//     live sub-agent must open its tab and register with the LIVE
//     panel — the collapsed history panel must not defer or own it.
// ---------------------------------------------------------------------------
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

  // The user overscrolled: an adjacent (later) task renders BELOW the
  // live chat — its own old run_parallel panel arrives collapsed and
  // is now the LAST .tc-run-parallel in document order.
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

  // The tab belongs to the LIVE panel: collapsing it closes the tab.
  togglePanel(win, livePanel);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'collapsing the live panel must close its sub-agent tab',
  );
  win.close();
  console.log('  ok - adjacent history panel cannot steal a live fan-out');
}

// ---------------------------------------------------------------------------
// 12. A delayed openSubagentTab for a sub-agent of call #1 (whose tab
//     was closed by collapsing panel #1) arrives AFTER call #2's panel
//     exists.  It must re-attach to panel #1 (which registered the
//     task id) — not leak into call #2's fan-out.
// ---------------------------------------------------------------------------
function testDelayedOpenSubagentAttachesToOwningCall() {
  const {win, posted, rootId} = bootRunningRoot();

  // Call #1 (tasks passed as a raw array — tolerated for robustness).
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
  togglePanel(win, panel1); // collapse call #1 → its tab closes
  assert.strictEqual(subagentTabEls(win).length, 0, 'call #1 tab closed');
  send(win, {type: 'tool_result', tabId: rootId, content: 'c1 done'});

  // Call #2 spawns its own sub-agent.
  const c2 = runParallelCall(win, posted, rootId, ['t2-a'], 'C2 ');
  const panel2 = runParallelPanels(win)[1];
  assert.strictEqual(subagentTabEls(win).length, 1, 'call #2 tab open');

  // The server's delayed conversion for call #1's closed tab arrives.
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

  // Call #2's fan-out must NOT have absorbed t-late: collapsing and
  // re-expanding panel #2 must touch only t2-a.
  togglePanel(win, panel2); // collapse → closes t2-a
  let before = posted.length;
  togglePanel(win, panel2); // expand
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

  // Panel #1 still owns t-late: expanding it resumes only t-late.
  before = posted.length;
  togglePanel(win, panel1); // expand call #1
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

// ---------------------------------------------------------------------------
// 13. A never-registered open sub-agent tab (spawned before its
//     parent's DOM existed) is adopted by the NEWEST panel only: an
//     older sibling panel's collapse pass must leave it alone.
// ---------------------------------------------------------------------------
function testUnregisteredTabAdoptsIntoNewestPanelOnly() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a'], 'L1 ');
  // Grandchild spawned while l1-a's tab has NO chat DOM: unregistered.
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

  // l1-a's stream arrives late: TWO run_parallel panels render (call
  // #1 finished with an unparseable tasks payload, call #2 running).
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: l1[0],
    extras: {tasks: 'not json {'},
  });
  send(win, {type: 'tool_result', tabId: l1[0], content: 'c1 done'});
  send(win, {type: 'tool_call', name: 'run_parallel', tabId: l1[0]});
  // l1-a's task ends: the collapse pass must adopt the unregistered
  // grandchild into the NEWEST panel (call #2, still running → kept
  // open), never into finished call #1.
  send(win, {type: 'result', tabId: l1[0], summary: 'done', success: true});
  assert.ok(
    subagentTabEls(win).some(el => el.dataset.tabId === g.tabId),
    "the unregistered grandchild must survive call #1's collapse " +
      '(it belongs to the newest, still-running call)',
  );

  // Collapsing the newest panel (via the user viewing l1-a's chat)
  // closes the adopted grandchild.
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
