// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A run_agent child's own terminal event (`task_done` / `task_error`
// addressed to its `<parent>__sub_<task>` tab) arrives BEFORE the
// daemon's `subagentDone` that closes the tab.  That event is the
// child's end, not the user's task finishing: the page must not switch
// onto the sub-agent tab, nor open the child's report (it waits for the
// parent task's end: the daemon is not asked for it), only for
// `subagentDone` to close the tab a moment later (a visible bounce).
// The parent's own task_done keeps its switch to the report the parent
// wrote.
//
// The real chat.html + panelCopy.js + api.js + marked.min.js + main.js
// run in jsdom; nothing is mocked.

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {
  spawnSubagent,
  writeReport,
  reportRequests,
  answerReport,
  contentTabs,
  contentTabFor,
  activeTabId,
  subagentTabEls,
} = require('./subagent_tab_helpers');

const {send, readyTabId} = harness;

function makePage() {
  const {win, posted} = harness.makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
  });
  return {win, posted, parentId: readyTabId(posted)};
}

/** A running parent that just spawned one run_agent child. */
function parentWithChild() {
  const page = makePage();
  const {win, posted, parentId} = page;
  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
    taskId: 'parent-task',
  });
  const subTabId = spawnSubagent(page, parentId, 'child-task', 'sub 1', 0);
  send(win, {
    type: 'status',
    running: true,
    tabId: subTabId,
    startTs: Date.now(),
    taskId: 'child-task',
  });
  return {win, posted, parentId, subTabId};
}

function testChildTaskDoneDoesNotSwitch(terminalType) {
  const {win, posted, parentId, subTabId} = parentWithChild();
  writeReport(win, 'reports/child.md', '# Child report', {tabId: subTabId});
  assert.strictEqual(activeTabId(win), parentId, 'parent is on screen');

  send(win, {type: terminalType, success: true, tabId: subTabId});

  assert.strictEqual(
    activeTabId(win),
    parentId,
    'BUG - a sub-agent tab\'s own ' +
      terminalType +
      ' must not switch the page onto the sub-agent tab',
  );

  // The daemon then closes the child's tab: the parent stays on screen.
  send(win, {type: 'subagentDone', tab_id: subTabId});
  assert.strictEqual(subagentTabEls(win).length, 0, 'the child tab closed');
  assert.strictEqual(activeTabId(win), parentId, 'the parent is still shown');
  assert.deepStrictEqual(
    reportRequests(posted),
    [],
    "BUG - the daemon must not be asked for the child's report",
  );
  assert.strictEqual(
    contentTabs(win).length,
    0,
    "BUG - the child's report must open no tab",
  );

  win.close();
  console.log('  ok - a sub-agent ' + terminalType + ' leaves the active tab');
}

function testParentTaskDoneStillSwitchesToReport() {
  const {win, posted, parentId, subTabId} = parentWithChild();
  send(win, {type: 'task_done', success: true, tabId: subTabId});
  send(win, {type: 'subagentDone', tab_id: subTabId});
  writeReport(win, 'reports/final.md', '# Final', {tabId: parentId});
  send(win, {type: 'task_done', success: true, tabId: parentId});
  answerReport(win, reportRequests(posted)[0], '# Final');
  const report = contentTabFor(win, 'final.md');
  assert.ok(report, 'the parent report opens');
  assert.strictEqual(
    activeTabId(win),
    report.id,
    'REGRESSION - the parent\'s own task_done still focuses its report',
  );
  win.close();
  console.log('  ok - the parent task_done still switches to its report');
}

function main() {
  console.log('f12_subagent_task_done_no_switch.test.js');
  testChildTaskDoneDoesNotSwitch('task_done');
  testChildTaskDoneDoesNotSwitch('task_error');
  testParentTaskDoneStillSwitchesToReport();
  console.log('f12_subagent_task_done_no_switch.test.js: all tests passed');
}

main();
