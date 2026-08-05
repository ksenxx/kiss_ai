// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests for the tab that is picked when a finished
// sub-agent tab CLOSES while the user is standing on it.
//
// `subagentDone` parks the sub-agent's report in a background content
// tab and then closes the sub-agent tab.  Closing the tab the user is
// on has to hand the user to some other tab - but the parent task is
// still running, so that other tab must never be the report that was
// just appended.  It must be a pre-existing chat tab, ideally the
// parent chat tab the sub-agent was spawned from.
//
// The real media/chat.html + panelCopy.js + api.js + main.js are booted
// in jsdom; nothing under test is mocked.

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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
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

function contentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.content-tab'),
  );
}

function contentTabFor(win, name) {
  return contentTabEls(win).find(
    el => (el.textContent || '').indexOf(name) >= 0,
  );
}

function tabEl(win, tabId) {
  return win.document.querySelector(
    '#tab-list .chat-tab[data-tab-id=' + JSON.stringify(tabId) + ']',
  );
}

function activeTabId(win) {
  return win._testApi.getActiveTabId();
}

function clickTab(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function chatSurfaceVisible(win) {
  const out = win.document.getElementById('output');
  return !!out && out.style.display !== 'none';
}

// Boot a parent task that spawns `n` sub-agents through run_parallel,
// exactly the way the backend drives the webview.
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

  const subTabIds = [];
  for (let i = 0; i < n; i++) {
    const taskId = 'sub-task-' + (i + 1);
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
    activeTabId(win),
    parentId,
    'precondition: spawning sub-agents must leave the user on the parent',
  );
  return {win, posted, parentId, subTabIds};
}

function testSoleSubagentReportDoesNotBecomeActiveOnClose() {
  // The reproduction: ONE sub-agent, the user is watching it, it writes
  // a report and finishes.  Its tab closes and the freshly appended
  // report tab sits right where the sub-agent tab used to be, so the
  // adjacency rule hands the user the report.
  const {win, parentId, subTabIds} = bootParallelRun(1);

  const subEl = tabEl(win, subTabIds[0]);
  assert.ok(subEl, 'the sub-agent tab must be in the tab bar');
  clickTab(win, subEl);
  assert.strictEqual(
    activeTabId(win),
    subTabIds[0],
    'precondition: the user is watching the sub-agent tab',
  );

  writeReport(win, 'reports/sub1.md', '# From sub-agent one', {
    tabId: subTabIds[0],
  });
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  const reportTab = contentTabFor(win, 'sub1.md');
  assert.ok(reportTab, 'the sub-agent report must open a content tab');
  assert.notStrictEqual(
    activeTabId(win),
    reportTab.dataset.tabId,
    'BUG - closing the finished sub-agent tab must not hand the user ' +
      'the report it just parked: the parent task is still running',
  );
  assert.ok(
    !reportTab.classList.contains('active'),
    'the background report tab must not be marked active',
  );
  assert.strictEqual(
    activeTabId(win),
    parentId,
    'the user must land on the parent chat tab the sub-agent came from',
  );
  assert.ok(
    chatSurfaceVisible(win),
    'the chat surface of the still-running parent must be visible',
  );

  win.close();
  console.log('  ok - a closing sole sub-agent tab never yields the report');
}

function testSubagentReportNotActiveWithSiblingStillRunning() {
  // Two sub-agents, the user is on the first.  It finishes with a
  // report; the user must land on a chat tab (parent or the running
  // sibling), never on the report.
  const {win, parentId, subTabIds} = bootParallelRun(2);

  clickTab(win, tabEl(win, subTabIds[0]));
  assert.strictEqual(activeTabId(win), subTabIds[0], 'precondition');

  writeReport(win, 'reports/sub1.md', '# one', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  const reportTab = contentTabFor(win, 'sub1.md');
  assert.ok(reportTab, 'the report content tab must exist');
  assert.notStrictEqual(
    activeTabId(win),
    reportTab.dataset.tabId,
    'BUG - the parked report must not become active',
  );
  assert.ok(
    [parentId, subTabIds[1]].includes(activeTabId(win)),
    'a surviving chat tab must become active',
  );
  assert.ok(chatSurfaceVisible(win), 'the chat surface must be visible');

  win.close();
  console.log('  ok - a sibling still running keeps focus on a chat tab');
}

function testDeepSubagentLandsOnItsParentChatTab() {
  // A sub-agent of a sub-agent: when the grandchild finishes while the
  // user is on it, the user should land on its own parent chat tab, not
  // on the report and not on some unrelated tab.
  const {win, posted, parentId, subTabIds} = bootParallelRun(1);

  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'grandchild-task',
    parent_tab_id: subTabIds[0],
    taskId: '',
  });
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === 'grandchild-task');
  assert.ok(resume, 'new_tab must make the webview post resumeSession');
  const grandId = resume.tabId;
  send(win, {
    type: 'openSubagentTab',
    tab_id: grandId,
    parent_tab_id: subTabIds[0],
    description: 'grandchild',
    task_id: 'grandchild-task',
    taskIndex: 0,
  });

  clickTab(win, tabEl(win, grandId));
  assert.strictEqual(activeTabId(win), grandId, 'precondition');

  writeReport(win, 'reports/grand.md', '# grand', {tabId: grandId});
  send(win, {type: 'subagentDone', tab_id: grandId});

  const reportTab = contentTabFor(win, 'grand.md');
  assert.ok(reportTab, 'the grandchild report must open a content tab');
  assert.notStrictEqual(
    activeTabId(win),
    reportTab.dataset.tabId,
    'BUG - the grandchild report must not become active',
  );
  assert.strictEqual(
    activeTabId(win),
    subTabIds[0],
    'the user must land on the grandchild\'s own parent chat tab',
  );
  assert.notStrictEqual(
    activeTabId(win),
    parentId,
    'the direct parent, not the root, must take over',
  );

  win.close();
  console.log('  ok - a grandchild hands the user to its own parent tab');
}

function testClosingActiveSubagentWithoutReportStillPicksChatTab() {
  // No report at all: the plain close path must keep behaving, landing
  // the user on a chat tab.
  const {win, parentId, subTabIds} = bootParallelRun(1);

  clickTab(win, tabEl(win, subTabIds[0]));
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  assert.strictEqual(
    contentTabEls(win).length,
    0,
    'no report was written, so no content tab may appear',
  );
  assert.strictEqual(
    activeTabId(win),
    parentId,
    'the user must land back on the parent chat tab',
  );
  assert.ok(chatSurfaceVisible(win), 'the chat surface must be visible');

  win.close();
  console.log('  ok - closing a report-less sub-agent tab picks the parent');
}

function testUserOpenedContentTabSurvivesAndIsNotStolen() {
  // The user had deliberately opened a file tab earlier.  A sub-agent
  // finishing must not shove them onto ANY content tab - existing or
  // new - while the parent task runs.
  const {win, parentId, subTabIds} = bootParallelRun(1);

  send(win, {
    type: 'fileContent',
    path: '/tmp/notes.txt',
    name: 'notes.txt',
    content: 'user opened this',
  });
  const fileTab = contentTabFor(win, 'notes.txt');
  assert.ok(fileTab, 'the user-opened file tab must exist');
  assert.strictEqual(
    activeTabId(win),
    fileTab.dataset.tabId,
    'precondition: a user-opened file focuses its tab',
  );

  clickTab(win, tabEl(win, subTabIds[0]));
  assert.strictEqual(activeTabId(win), subTabIds[0], 'precondition');

  writeReport(win, 'reports/sub1.md', '# one', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  assert.strictEqual(
    activeTabId(win),
    parentId,
    'BUG - the user must land on the parent chat tab, not on any ' +
      'content tab',
  );
  assert.ok(
    contentTabFor(win, 'notes.txt'),
    'the user-opened file tab must survive',
  );
  assert.ok(contentTabFor(win, 'sub1.md'), 'the report tab must exist');

  win.close();
  console.log('  ok - no content tab is stolen when a sub-agent closes');
}

function testReportStaysBackgroundWhenSubagentTabIsNotActive() {
  // Anti-over-fix guard: the original invariant still holds when the
  // user is NOT on the finishing sub-agent tab.
  const {win, parentId, subTabIds} = bootParallelRun(1);

  assert.strictEqual(activeTabId(win), parentId, 'precondition');
  writeReport(win, 'reports/quiet.md', '# quiet', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  assert.strictEqual(
    activeTabId(win),
    parentId,
    'REGRESSION - the user must stay on the parent tab',
  );
  const reportTab = contentTabFor(win, 'quiet.md');
  assert.ok(reportTab, 'the report tab must be reachable in the tab bar');
  assert.ok(
    !reportTab.classList.contains('active'),
    'the report tab must not be active',
  );

  win.close();
  console.log('  ok - an inactive sub-agent report still stays background');
}

function testUserFileOpenStillFocusesAfterSubagentDone() {
  // Anti-over-fix guard: a genuine user file open still switches tabs,
  // even right after a sub-agent finished.
  const {win, subTabIds} = bootParallelRun(1);

  writeReport(win, 'reports/bg.md', '# bg', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  send(win, {
    type: 'fileContent',
    path: '/tmp/opened.txt',
    name: 'opened.txt',
    content: 'hello from the editor',
  });
  const fileTab = contentTabFor(win, 'opened.txt');
  assert.ok(fileTab, 'clicking a file path must open a content tab');
  assert.strictEqual(
    activeTabId(win),
    fileTab.dataset.tabId,
    'REGRESSION - a user-requested file must still switch to its tab',
  );

  win.close();
  console.log('  ok - a user-opened file still focuses after subagentDone');
}

function testReportTabRemainsClickableAfterClose() {
  // The parked report must still be openable by the user afterwards.
  const {win, subTabIds} = bootParallelRun(1);

  clickTab(win, tabEl(win, subTabIds[0]));
  writeReport(win, 'reports/late.md', '# Late Report', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  const reportTab = contentTabFor(win, 'late.md');
  assert.ok(reportTab, 'the report tab must exist');
  clickTab(win, reportTab);
  assert.strictEqual(
    activeTabId(win),
    contentTabFor(win, 'late.md').dataset.tabId,
    'clicking the parked report tab must open it',
  );
  const frames = Array.from(
    win.document.querySelectorAll(
      '#content-tab-area .content-tab-view iframe.content-html-frame',
    ),
  ).filter(f => f.closest('.content-tab-view').style.display !== 'none');
  assert.strictEqual(frames.length, 1, 'exactly one frame must be visible');
  assert.ok(
    /<h1[^>]*>Late Report<\/h1>/.test(frames[0].getAttribute('srcdoc') || ''),
    'the clicked report must render its markdown',
  );

  win.close();
  console.log('  ok - the parked report tab is still clickable afterwards');
}

function main() {
  console.log('subagentDoneAdjacencyNoReport.test.js');
  testSoleSubagentReportDoesNotBecomeActiveOnClose();
  testSubagentReportNotActiveWithSiblingStillRunning();
  testDeepSubagentLandsOnItsParentChatTab();
  testClosingActiveSubagentWithoutReportStillPicksChatTab();
  testUserOpenedContentTabSurvivesAndIsNotStolen();
  testReportStaysBackgroundWhenSubagentTabIsNotActive();
  testUserFileOpenStillFocusesAfterSubagentDone();
  testReportTabRemainsClickableAfterClose();
  console.log('subagentDoneAdjacencyNoReport.test.js: all tests passed');
}

main();
