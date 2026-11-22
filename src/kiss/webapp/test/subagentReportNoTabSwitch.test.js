// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests for the "never steal focus while a task is
// running" rule, applied to reports written by SUB-AGENTS.
//
// A sub-agent finishing (`subagentDone`) is NOT the task finishing: the
// parent task is still running.  A report a sub-agent wrote into a
// `reports/` folder is part of its parent's work: when the sub-agent
// finishes, the chat page asks the daemon for nothing, opens no content
// tab and leaves the user where they are; the report is asked for with
// the parent task's own when that task ends.  Focus may only be taken
// on a genuinely terminal event of the user's own task (task_done /
// task_error / task_interrupted / task_stopped), by the reports of that
// task (the daemon's replies to the page's requests), or on a real user
// action.
//
// The page is the stacked (phone-width) remote webapp: a content tab
// shown replaces the chat on screen, and the tab strip lists content
// tabs on the file screen only, one swipe left of the chat.
//
// The real media/chat.html + panelCopy.js + api.js + mobileSwipe.js +
// main.js are booted in jsdom; nothing under test is mocked.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  click: clickTab,
} = require('./simplify2_harness');
const {
  bootParallelRun,
  writeReport,
  reportRequests,
  answerReport,
  contentTabs,
  contentTabFor,
  stripContentTabEls,
  tabEl,
  visibleSrcdoc,
  activeTabId,
  chatSurfaceVisible,
  swipeLeftToFileScreen,
} = require('./subagent_tab_helpers');

function makePage() {
  return makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js', 'mobileSwipe.js'],
    beforeScripts: stubRaf,
  });
}

// A task ends with the terminal event *ev*, and the daemon answers the
// page's request for the one report that task wrote, *text*.
function endTaskWithReport(win, posted, ev, text) {
  send(win, ev);
  const reqs = reportRequests(posted);
  assert.strictEqual(reqs.length, 1, ev.type + ' must ask for its report');
  assert.strictEqual(reqs[0].tabId, ev.tabId, "asked for as the task's tab");
  answerReport(win, reqs[0], text);
}

function testSubagentReportDoesNotStealFocus() {
  const {win, posted, parentId, subTabIds} = bootParallelRun(2, makePage());

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
  assert.ok(
    chatSurfaceVisible(win),
    'the chat surface of the still-running parent must stay visible',
  );

  assert.deepStrictEqual(
    reportRequests(posted),
    [],
    "the daemon must not be asked for a sub-agent's report",
  );
  assert.strictEqual(
    contentTabs(win).length,
    0,
    "a sub-agent's report must open no content tab, not even in the registry",
  );
  assert.strictEqual(
    stripContentTabEls(win).length,
    0,
    'the chat screen stays on view: the strip lists no content tab',
  );

  // ...and nothing is parked on the file screen either: with no file
  // open, a swipe left leaves the user on the chat.
  swipeLeftToFileScreen(win);
  assert.strictEqual(
    activeTabId(win),
    parentId,
    'no report waits on the file screen for the swipe to open',
  );

  win.close();
  console.log("  ok - a sub-agent's report is not opened at subagentDone");
}

function testSubagentReportWhileOnAnotherSubagentTab() {
  // The user is deliberately watching sub-agent 2 while sub-agent 1
  // finishes and drops a report: their view must not move either.
  const {win, posted, subTabIds} = bootParallelRun(2, makePage());

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
  assert.deepStrictEqual(reportRequests(posted), [], 'nothing is asked for');
  assert.strictEqual(contentTabs(win).length, 0, 'no report tab is created');

  win.close();
  console.log('  ok - a sibling sub-agent report keeps the chosen tab');
}

function testSubagentRewriteLeavesAnOpenReportAlone() {
  // The user has a report open (they opened the file themselves) while
  // two sub-agents rewrite it: neither rewrite is asked for, so the
  // tab keeps what it shows and the user stays where they are, on the
  // report or back on the chat.
  const {win, posted, parentId, subTabIds} = bootParallelRun(2, makePage());
  send(win, {
    type: 'fileContent',
    tabId: parentId,
    path: '/ws/reports/live.md',
    name: 'live.md',
    content: '# first',
    version: '1:7',
  });
  const reportTabId = activeTabId(win);
  assert.strictEqual(
    contentTabFor(win, 'live.md').id,
    reportTabId,
    'precondition: the user is on the report tab',
  );

  writeReport(win, 'reports/live.md', '# second', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  assert.strictEqual(
    activeTabId(win),
    reportTabId,
    'the user must stay on the report tab they are reading',
  );
  assert.ok(
    /<h1[^>]*>first<\/h1>/.test(visibleSrcdoc(win)),
    "a sub-agent's rewrite must not replace what the tab shows",
  );

  win._testApi.switchToTab(parentId);
  writeReport(win, 'reports/live.md', '# third', {tabId: subTabIds[1]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[1]});
  assert.strictEqual(
    activeTabId(win),
    parentId,
    'regenerating a report from a sub-agent must not switch tabs either',
  );
  assert.strictEqual(
    contentTabs(win)
      .map(t => t.id)
      .join(),
    reportTabId,
    'the open report tab is the only content tab, and the same one',
  );
  assert.deepStrictEqual(reportRequests(posted), [], 'nothing is asked for');

  win.close();
  console.log("  ok - a sub-agent's rewrite leaves an open report alone");
}

function testTaskDoneStillSwitchesToTheReport() {
  // Anti-over-fix guard: a terminal event of the tab itself MUST still
  // focus the report it produced.
  const {win, posted, parentId} = bootParallelRun(1, makePage());

  writeReport(win, 'reports/final.md', '# Final Report', {tabId: parentId});
  const done = {type: 'task_done', success: true, tabId: parentId};
  endTaskWithReport(win, posted, done, '# Final Report');

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
    const {win, posted, parentId} = bootParallelRun(1, makePage());
    writeReport(win, 'reports/salvaged.md', '# Salvaged', {tabId: parentId});
    const end = {type: terminal, tabId: parentId};
    endTaskWithReport(win, posted, end, '# Salvaged');
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
  // The full realistic flow: a sub-agent's report waits quietly, and
  // when the PARENT task finally finishes both reports are asked for,
  // as that task's, and the user is taken to the one written last, the
  // parent's own.
  const {win, posted, parentId, subTabIds} = bootParallelRun(1, makePage());

  writeReport(win, 'reports/sub.md', '# Sub report', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  assert.strictEqual(
    activeTabId(win),
    parentId,
    'the sub-agent report must not move the user',
  );

  writeReport(win, 'reports/parent.md', '# Parent report', {tabId: parentId});
  send(win, {type: 'task_done', success: true, tabId: parentId});
  const reqs = reportRequests(posted);
  assert.deepStrictEqual(
    reqs.map(r => [r.path, r.tabId]),
    [
      ['reports/sub.md', parentId],
      ['reports/parent.md', parentId],
    ],
    "task_done must ask for the sub-agent's report and its own, as its own",
  );
  answerReport(win, reqs[0], '# Sub report');
  answerReport(win, reqs[1], '# Parent report');

  const parentReport = contentTabFor(win, 'parent.md');
  assert.ok(parentReport, 'the parent report tab must open at task_done');
  assert.strictEqual(
    activeTabId(win),
    parentReport.id,
    'the parent task finishing must focus its own report',
  );
  assert.ok(
    contentTabFor(win, 'sub.md'),
    "the sub-agent's report is open too, behind the parent's",
  );
  assert.strictEqual(
    stripContentTabEls(win).length,
    2,
    'the file screen strip lists both reports',
  );

  win.close();
  console.log("  ok - the parent task_done opens the sub-agent's report too");
}

// The "a user-opened file still switches to its tab" guard lives in
// subagentDoneAdjacencyNoReport.test.js
// (testUserFileOpenStillFocusesAfterSubagentDone).

function main() {
  console.log('subagentReportNoTabSwitch.test.js');
  testSubagentReportDoesNotStealFocus();
  testSubagentReportWhileOnAnotherSubagentTab();
  testSubagentRewriteLeavesAnOpenReportAlone();
  testTaskDoneStillSwitchesToTheReport();
  testTerminalErrorEventsStillSwitchToTheReport();
  testParentTaskDoneAfterSubagentReportSwitches();
  console.log('subagentReportNoTabSwitch.test.js: all tests passed');
}

main();
