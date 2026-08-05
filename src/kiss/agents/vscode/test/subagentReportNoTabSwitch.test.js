// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests for the "never steal focus while a task is
// running" rule, applied to reports written by SUB-AGENTS.
//
// A sub-agent finishing (`subagentDone`) is NOT the task finishing: the
// parent task is still running.  When a sub-agent writes a report into a
// `reports/` folder and then finishes, the chat webview must open the
// report in a content tab (reachable in the tab bar) but must NOT yank
// the user onto it.  Focus may only be taken on a genuinely terminal
// event of the tab the user is on (task_done / task_error /
// task_interrupted / task_stopped) or on a real user action.
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

function reportFrames(win) {
  return Array.from(
    win.document.querySelectorAll(
      '#content-tab-area .content-tab-view iframe.content-html-frame',
    ),
  );
}

function visibleSrcdoc(win) {
  const frames = reportFrames(win).filter(
    f => f.closest('.content-tab-view').style.display !== 'none',
  );
  assert.strictEqual(frames.length, 1, 'expected exactly one visible frame');
  return frames[0].getAttribute('srcdoc') || '';
}

function activeTabId(win) {
  return win._testApi.getActiveTabId();
}

function clickTab(win, tabEl) {
  tabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
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

function testSubagentReportDoesNotStealFocus() {
  const {win, parentId, subTabIds} = bootParallelRun(2);

  writeReport(win, 'reports/sub1.md', '# From sub-agent one', {
    tabId: subTabIds[0],
  });
  assert.strictEqual(
    contentTabEls(win).length,
    0,
    'a report must not open while the sub-agent is still running',
  );

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  assert.strictEqual(
    activeTabId(win),
    parentId,
    'BUG - a sub-agent finishing is not the task finishing: the parent ' +
      'task is still running, so its report must not switch tabs',
  );
  const out = win.document.getElementById('output');
  assert.notStrictEqual(
    out.style.display,
    'none',
    'the chat surface of the still-running parent must stay visible',
  );

  const reportTab = contentTabFor(win, 'sub1.md');
  assert.ok(
    reportTab,
    'the sub-agent report must still open a content tab in the tab bar',
  );
  assert.ok(
    !reportTab.classList.contains('active'),
    'the background report tab must not be marked active',
  );

  // ...and the user can reach it whenever they want.
  clickTab(win, reportTab);
  assert.strictEqual(
    activeTabId(win),
    contentTabFor(win, 'sub1.md').dataset.tabId,
    'clicking the report tab must open it',
  );
  assert.ok(
    /<h1[^>]*>From sub-agent one<\/h1>/.test(visibleSrcdoc(win)),
    'the clicked report tab must render the converted markdown report',
  );

  win.close();
  console.log('  ok - a sub-agent report opens in the background only');
}

function testSubagentReportWhileOnAnotherSubagentTab() {
  // The user is deliberately watching sub-agent 2 while sub-agent 1
  // finishes and drops a report: their view must not move either.
  const {win, subTabIds} = bootParallelRun(2);

  const sub2El = win.document.querySelector(
    '#tab-list .chat-tab[data-tab-id=' + JSON.stringify(subTabIds[1]) + ']',
  );
  assert.ok(sub2El, 'the second sub-agent tab must be in the tab bar');
  clickTab(win, sub2El);
  assert.strictEqual(
    activeTabId(win),
    subTabIds[1],
    'precondition: the user is on the second sub-agent tab',
  );

  writeReport(win, 'reports/sub1.html', '<!DOCTYPE html><p>raw html</p>', {
    tabId: subTabIds[0],
  });
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  assert.strictEqual(
    activeTabId(win),
    subTabIds[1],
    'a sibling sub-agent report must not pull the user off the tab ' +
      'they chose to watch',
  );
  assert.ok(
    contentTabFor(win, 'sub1.html'),
    'the report tab must still be created for later viewing',
  );

  win.close();
  console.log('  ok - a sibling sub-agent report keeps the chosen tab');
}

function testExistingContentTabIsRefreshedWithoutFocus() {
  // Same report path written twice by two sub-agents: the second write
  // must reuse (and re-render) the existing content tab, still without
  // taking focus while the parent task runs.
  const {win, parentId, subTabIds} = bootParallelRun(2);

  writeReport(win, 'reports/shared.md', '# v1', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  const firstTab = contentTabFor(win, 'shared.md');
  assert.ok(firstTab, 'the first sub-agent report must open a content tab');
  const reportTabId = firstTab.dataset.tabId;

  writeReport(win, 'reports/shared.md', '# v2', {tabId: subTabIds[1]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[1]});

  assert.strictEqual(
    activeTabId(win),
    parentId,
    'regenerating a report from a sub-agent must not switch tabs either',
  );
  const tabsNow = contentTabEls(win);
  assert.strictEqual(
    tabsNow.length,
    1,
    'the regenerated report must reuse its tab, not open a second one',
  );
  assert.strictEqual(
    tabsNow[0].dataset.tabId,
    reportTabId,
    'the reused content tab must keep its identity',
  );

  clickTab(win, tabsNow[0]);
  assert.ok(
    /<h1[^>]*>v2<\/h1>/.test(visibleSrcdoc(win)),
    'the reused tab must show the regenerated content',
  );

  win.close();
  console.log('  ok - an existing report tab is refreshed without focus');
}

function testRefreshOfTheActiveReportTabStillRenders() {
  // If the user is ALREADY looking at the report tab, a sub-agent
  // rewriting that report must refresh what they see in place.
  const {win, subTabIds} = bootParallelRun(2);

  writeReport(win, 'reports/live.md', '# first', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  const reportTab = contentTabFor(win, 'live.md');
  clickTab(win, reportTab);
  const reportTabId = activeTabId(win);
  assert.strictEqual(
    reportTabId,
    reportTab.dataset.tabId,
    'precondition: the user is on the report tab',
  );

  writeReport(win, 'reports/live.md', '# second', {tabId: subTabIds[1]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[1]});

  assert.strictEqual(
    activeTabId(win),
    reportTabId,
    'the user must stay on the report tab they are reading',
  );
  assert.ok(
    /<h1[^>]*>second<\/h1>/.test(visibleSrcdoc(win)),
    'the visible report must be refreshed in place',
  );

  win.close();
  console.log('  ok - refreshing the active report tab renders in place');
}

function testTaskDoneStillSwitchesToTheReport() {
  // Anti-over-fix guard: a terminal event of the tab itself MUST still
  // focus the report it produced.
  const {win, parentId} = bootParallelRun(1);

  writeReport(win, 'reports/final.md', '# Final Report', {tabId: parentId});
  send(win, {type: 'task_done', success: true, tabId: parentId});

  const reportTab = contentTabFor(win, 'final.md');
  assert.ok(reportTab, 'task_done must open the report tab');
  assert.ok(
    reportTab.classList.contains('active'),
    'REGRESSION - a report opened at task_done must still become active',
  );
  assert.strictEqual(
    activeTabId(win),
    reportTab.dataset.tabId,
    'REGRESSION - task_done must switch to the report tab',
  );
  assert.ok(
    /<h1[^>]*>Final Report<\/h1>/.test(visibleSrcdoc(win)),
    'the focused report must be rendered',
  );

  win.close();
  console.log('  ok - task_done still switches to the report tab');
}

function testTerminalErrorEventsStillSwitchToTheReport() {
  ['task_error', 'task_interrupted', 'task_stopped'].forEach(terminal => {
    const {win, parentId} = bootParallelRun(1);
    writeReport(win, 'reports/salvaged.md', '# Salvaged', {tabId: parentId});
    send(win, {type: terminal, tabId: parentId});
    const reportTab = contentTabFor(win, 'salvaged.md');
    assert.ok(reportTab, terminal + ' must open the report tab');
    assert.strictEqual(
      activeTabId(win),
      reportTab.dataset.tabId,
      'REGRESSION - ' + terminal + ' must still switch to the report tab',
    );
    win.close();
  });
  console.log('  ok - error/interrupted/stopped still switch to the report');
}

function testParentTaskDoneAfterSubagentReportSwitches() {
  // The full realistic flow: a sub-agent quietly parks its report, and
  // when the PARENT task finally finishes the user is taken to it.
  const {win, parentId, subTabIds} = bootParallelRun(1);

  writeReport(win, 'reports/sub.md', '# Sub report', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  assert.strictEqual(
    activeTabId(win),
    parentId,
    'the sub-agent report must stay in the background',
  );

  writeReport(win, 'reports/parent.md', '# Parent report', {tabId: parentId});
  send(win, {type: 'task_done', success: true, tabId: parentId});

  const parentReport = contentTabFor(win, 'parent.md');
  assert.ok(parentReport, 'the parent report tab must open at task_done');
  assert.strictEqual(
    activeTabId(win),
    parentReport.dataset.tabId,
    'the parent task finishing must focus its own report',
  );
  assert.strictEqual(
    contentTabEls(win).length,
    2,
    'both the background sub-agent report and the parent report must ' +
      'be reachable',
  );

  win.close();
  console.log('  ok - the parent task_done focuses its report as before');
}

function testClickingAFilePathStillSwitches() {
  // handleFileContent is also the "user clicked a file path" path; that
  // is a genuine user action and must keep focusing even mid-task.
  const {win} = bootParallelRun(1);

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
  console.log('  ok - a user-opened file still switches to its tab');
}

function main() {
  console.log('subagentReportNoTabSwitch.test.js');
  testSubagentReportDoesNotStealFocus();
  testSubagentReportWhileOnAnotherSubagentTab();
  testExistingContentTabIsRefreshedWithoutFocus();
  testRefreshOfTheActiveReportTabStillRenders();
  testTaskDoneStillSwitchesToTheReport();
  testTerminalErrorEventsStillSwitchToTheReport();
  testParentTaskDoneAfterSubagentReportSwitches();
  testClickingAFilePathStillSwitches();
  console.log('subagentReportNoTabSwitch.test.js: all tests passed');
}

main();
