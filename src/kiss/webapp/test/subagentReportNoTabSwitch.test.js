// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests for the "never steal focus while a task is
// running" rule, applied to reports written by SUB-AGENTS.
//
// A sub-agent finishing (`subagentDone`) is NOT the task finishing: the
// parent task is still running.  When a sub-agent writes a report into a
// `reports/` folder and then finishes, the chat page must open the
// report in a content tab (registered, reachable on the file screen)
// but must NOT yank the user onto it.  Focus may only be taken on a
// genuinely terminal event of the tab the user is on (task_done /
// task_error / task_interrupted / task_stopped) or on a real user
// action.
//
// The page is the stacked (phone-width) remote webapp: a content tab
// shown replaces the chat on screen, and the tab strip lists content
// tabs on the file screen only.  A parked report therefore shows up in
// the tab registry (and on the file screen, one swipe left) but never
// on the chat screen's strip.
//
// The real media/chat.html + panelCopy.js + api.js + mobileSwipe.js +
// main.js are booted in jsdom; nothing under test is mocked.

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send, stubRaf} = require('./simplify2_harness');
const {
  bootParallelRun,
  writeReport,
  contentTabs,
  contentTabFor,
  stripContentTabEls,
  tabEl,
  visibleSrcdoc,
  activeTabId,
  clickTab,
  swipeLeftToFileScreen,
} = require('./subagent_tab_helpers');

function makePage() {
  return makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js', 'mobileSwipe.js'],
    beforeScripts: stubRaf,
  });
}

function testSubagentReportDoesNotStealFocus() {
  const {win, parentId, subTabIds} = bootParallelRun(2, makePage());

  writeReport(win, 'reports/sub1.md', '# From sub-agent one', {
    tabId: subTabIds[0],
  });
  assert.strictEqual(
    contentTabs(win).length,
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
    'the sub-agent report must still open a content tab in the registry',
  );
  assert.strictEqual(
    stripContentTabEls(win).length,
    0,
    'the chat screen stays on view: the strip lists no content tab',
  );

  // ...and the user can reach it whenever they want: one swipe left
  // puts the file screen, and the parked report, on view.
  swipeLeftToFileScreen(win);
  assert.strictEqual(
    activeTabId(win),
    reportTab.id,
    'the swipe to the file screen must open the parked report',
  );
  const stripEl = tabEl(win, reportTab.id);
  assert.ok(
    stripEl && stripEl.classList.contains('active'),
    'the file screen strip lists the report as the active tab',
  );
  assert.ok(
    /<h1[^>]*>From sub-agent one<\/h1>/.test(visibleSrcdoc(win)),
    'the opened report tab must render the converted markdown report',
  );

  win.close();
  console.log('  ok - a sub-agent report opens in the background only');
}

function testSubagentReportWhileOnAnotherSubagentTab() {
  // The user is deliberately watching sub-agent 2 while sub-agent 1
  // finishes and drops a report: their view must not move either.
  const {win, subTabIds} = bootParallelRun(2, makePage());

  const sub2El = tabEl(win, subTabIds[1]);
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
  const {win, parentId, subTabIds} = bootParallelRun(2, makePage());

  writeReport(win, 'reports/shared.md', '# v1', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  const firstTab = contentTabFor(win, 'shared.md');
  assert.ok(firstTab, 'the first sub-agent report must open a content tab');
  const reportTabId = firstTab.id;

  writeReport(win, 'reports/shared.md', '# v2', {tabId: subTabIds[1]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[1]});

  assert.strictEqual(
    activeTabId(win),
    parentId,
    'regenerating a report from a sub-agent must not switch tabs either',
  );
  const tabsNow = contentTabs(win);
  assert.strictEqual(
    tabsNow.length,
    1,
    'the regenerated report must reuse its tab, not open a second one',
  );
  assert.strictEqual(
    tabsNow[0].id,
    reportTabId,
    'the reused content tab must keep its identity',
  );

  swipeLeftToFileScreen(win);
  assert.strictEqual(activeTabId(win), reportTabId, 'the swipe opens it');
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
  const {win, subTabIds} = bootParallelRun(2, makePage());

  writeReport(win, 'reports/live.md', '# first', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  const reportTab = contentTabFor(win, 'live.md');
  swipeLeftToFileScreen(win);
  const reportTabId = activeTabId(win);
  assert.strictEqual(
    reportTabId,
    reportTab.id,
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
  const {win, parentId} = bootParallelRun(1, makePage());

  writeReport(win, 'reports/final.md', '# Final Report', {tabId: parentId});
  send(win, {type: 'task_done', success: true, tabId: parentId});

  const reportTab = contentTabFor(win, 'final.md');
  assert.ok(reportTab, 'task_done must open the report tab');
  const stripEl = tabEl(win, reportTab.id);
  assert.ok(
    stripEl && stripEl.classList.contains('active'),
    'REGRESSION - a report opened at task_done must still become active',
  );
  assert.strictEqual(
    activeTabId(win),
    reportTab.id,
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
    const {win, parentId} = bootParallelRun(1, makePage());
    writeReport(win, 'reports/salvaged.md', '# Salvaged', {tabId: parentId});
    send(win, {type: terminal, tabId: parentId});
    const reportTab = contentTabFor(win, 'salvaged.md');
    assert.ok(reportTab, terminal + ' must open the report tab');
    assert.strictEqual(
      activeTabId(win),
      reportTab.id,
      'REGRESSION - ' + terminal + ' must still switch to the report tab',
    );
    win.close();
  });
  console.log('  ok - error/interrupted/stopped still switch to the report');
}

function testParentTaskDoneAfterSubagentReportSwitches() {
  // The full realistic flow: a sub-agent quietly parks its report, and
  // when the PARENT task finally finishes the user is taken to it.
  const {win, parentId, subTabIds} = bootParallelRun(1, makePage());

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
    parentReport.id,
    'the parent task finishing must focus its own report',
  );
  assert.strictEqual(
    contentTabs(win).length,
    2,
    'both the background sub-agent report and the parent report must ' +
      'be open',
  );
  assert.strictEqual(
    stripContentTabEls(win).length,
    2,
    'the file screen strip lists both reports, so both are reachable',
  );

  win.close();
  console.log('  ok - the parent task_done focuses its report as before');
}

// The "a user-opened file still switches to its tab" guard lives in
// subagentDoneAdjacencyNoReport.test.js
// (testUserFileOpenStillFocusesAfterSubagentDone).

function main() {
  console.log('subagentReportNoTabSwitch.test.js');
  testSubagentReportDoesNotStealFocus();
  testSubagentReportWhileOnAnotherSubagentTab();
  testExistingContentTabIsRefreshedWithoutFocus();
  testRefreshOfTheActiveReportTabStillRenders();
  testTaskDoneStillSwitchesToTheReport();
  testTerminalErrorEventsStillSwitchToTheReport();
  testParentTaskDoneAfterSubagentReportSwitches();
  console.log('subagentReportNoTabSwitch.test.js: all tests passed');
}

main();
