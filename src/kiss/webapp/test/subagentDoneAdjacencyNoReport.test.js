// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests for the tab that is picked when a finished
// sub-agent tab CLOSES while the user is standing on it.
//
// `subagentDone` closes the sub-agent tab; the report the sub-agent
// wrote waits for the end of the task that runs it (the daemon is asked
// for nothing and no content tab opens).  Closing the tab the user is
// on has to hand the user to some other tab - but the parent task is
// still running, so that other tab must never be a content tab.  It
// must be a pre-existing chat tab, ideally the parent chat tab the
// sub-agent was spawned from.
//
// The page is the stacked (phone-width) remote webapp: a content tab
// shown replaces the chat on screen, and the tab strip lists content
// tabs on the file screen only.
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
  contentTabs,
  contentTabFor,
  stripContentTabEls,
  tabEl,
  activeTabId,
  chatSurfaceVisible,
  spawnSubagent,
} = require('./subagent_tab_helpers');

function makePage() {
  return makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: stubRaf,
    extraScripts: ['marked.min.js', 'mobileSwipe.js'],
  });
}

// The finished sub-agent's report waits for its task's end: the daemon
// is not asked for it, so no content tab can take the closed tab's
// place.
function assertReportNotOpened(win, posted) {
  assert.deepStrictEqual(
    reportRequests(posted),
    [],
    "the daemon must not be asked for a sub-agent's report",
  );
  assert.strictEqual(
    contentTabs(win).length,
    0,
    "BUG - a sub-agent's report must open no content tab: the parent " +
      'task is still running',
  );
}

function testSoleSubagentReportDoesNotBecomeActiveOnClose() {
  // The reproduction: ONE sub-agent, the user is watching it, it writes
  // a report and finishes.  Its tab closes; a report tab appended right
  // where the sub-agent tab used to be would be what the adjacency
  // rule hands the user.
  const {win, posted, parentId, subTabIds} = bootParallelRun(1, makePage());

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

  assertReportNotOpened(win, posted);
  assert.strictEqual(
    stripContentTabEls(win).length,
    0,
    'the chat screen stays on view: the strip lists no content tab',
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
  // report; the user must land on the parent chat tab (not on the
  // running sibling next to it), never on the report.
  const {win, posted, parentId, subTabIds} = bootParallelRun(2, makePage());

  clickTab(win, tabEl(win, subTabIds[0]));
  assert.strictEqual(activeTabId(win), subTabIds[0], 'precondition');

  writeReport(win, 'reports/sub1.md', '# one', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  assertReportNotOpened(win, posted);
  assert.strictEqual(
    activeTabId(win),
    parentId,
    'the parent chat tab must become active, not the running sibling',
  );
  assert.ok(chatSurfaceVisible(win), 'the chat surface must be visible');

  win.close();
  console.log('  ok - a sibling still running keeps focus on a chat tab');
}

function testDeepSubagentLandsOnItsParentChatTab() {
  // A sub-agent of a sub-agent: when the grandchild finishes while the
  // user is on it, the user should land on its own parent chat tab, not
  // on the report and not on some unrelated tab.
  const page = makePage();
  const {win, posted, parentId, subTabIds} = bootParallelRun(1, page);
  const grandId = spawnSubagent(
    page,
    subTabIds[0],
    'grandchild-task',
    'grandchild',
    0,
  );

  clickTab(win, tabEl(win, grandId));
  assert.strictEqual(activeTabId(win), grandId, 'precondition');

  writeReport(win, 'reports/grand.md', '# grand', {tabId: grandId});
  send(win, {type: 'subagentDone', tab_id: grandId});

  assertReportNotOpened(win, posted);
  assert.strictEqual(
    activeTabId(win),
    subTabIds[0],
    "the user must land on the grandchild's own parent chat tab",
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
  const {win, parentId, subTabIds} = bootParallelRun(1, makePage());

  clickTab(win, tabEl(win, subTabIds[0]));
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  assert.strictEqual(
    contentTabs(win).length,
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
  const {win, parentId, subTabIds} = bootParallelRun(1, makePage());

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
    fileTab.id,
    'precondition: a user-opened file focuses its tab',
  );

  // The file screen's strip lists files only: back on the chat (the
  // Chats panel's pick), the user steps onto the sub-agent tab.
  win._testApi.switchToTab(parentId);
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
  assert.strictEqual(
    contentTabs(win).length,
    1,
    "the user's file stays the only content tab: the report opens none",
  );

  win.close();
  console.log('  ok - no content tab is stolen when a sub-agent closes');
}

function testUserFileOpenStillFocusesAfterSubagentDone() {
  // Anti-over-fix guard: a genuine user file open still switches tabs,
  // even right after a sub-agent finished.
  const {win, subTabIds} = bootParallelRun(1, makePage());

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
    fileTab.id,
    'REGRESSION - a user-requested file must still switch to its tab',
  );

  win.close();
  console.log('  ok - a user-opened file still focuses after subagentDone');
}

function testFileOfAFinishedSubagentShowsItsParentsRun() {
  // A file reached from a sub-agent's tab stays on screen when the
  // sub-agent finishes and stands for the parent chat from then on: the
  // run under it (Task Info, usage, clock, Stop) is the parent's.
  const {win, parentId, subTabIds} = bootParallelRun(1, makePage());
  const text = id => win.document.getElementById(id).textContent;
  const stop = win.document.getElementById('stop-btn');
  const settings = {task_id: 'p1', model: 'parent-model'};
  send(win, {type: 'task_settings', tabId: parentId, settings});
  send(win, {type: 'usage_info', tabId: parentId, total_tokens: 1234, cost: 0.5});
  send(win, {type: 'status', running: true, tabId: subTabIds[0]});
  clickTab(win, tabEl(win, subTabIds[0]));
  assert.notStrictEqual(text('meta-task-id'), 'p1', 'precondition');
  const file = {path: '/tmp/notes.txt', name: 'notes.txt', content: 'x'};
  send(win, {type: 'fileContent', tabId: subTabIds[0], ...file});
  const fileTab = contentTabFor(win, 'notes.txt');
  assert.strictEqual(activeTabId(win), fileTab.id, 'precondition');

  send(win, {type: 'subagentDone', tab_id: subTabIds[0], task_id: 'sub-task-1'});
  assert.strictEqual(activeTabId(win), fileTab.id, 'the file stays on screen');
  assert.match(text('status-text'), /^Running/, "the parent's clock ticks");
  assert.strictEqual(stop.style.display, 'flex', "Stop is the parent's");
  assert.strictEqual(text('meta-task-id'), 'p1', "Task Info is the parent's");
  assert.strictEqual(text('meta-model'), 'parent-model');
  assert.strictEqual(text('status-tokens'), 'Tokens: 1.23K', 'and the numbers');

  send(win, {type: 'status', running: false, tabId: parentId});
  assert.doesNotMatch(text('status-text'), /^Running/, 'the clock stops with it');
  assert.strictEqual(stop.style.display, 'none');
  win.close();
  console.log("  ok - a finished sub-agent's file shows its parent's run");
}

// That no report is parked on the file screen for a swipe to find is
// pinned by subagentReportNoTabSwitch.test.js.

function main() {
  console.log('subagentDoneAdjacencyNoReport.test.js');
  testSoleSubagentReportDoesNotBecomeActiveOnClose();
  testSubagentReportNotActiveWithSiblingStillRunning();
  testDeepSubagentLandsOnItsParentChatTab();
  testClosingActiveSubagentWithoutReportStillPicksChatTab();
  testUserOpenedContentTabSurvivesAndIsNotStolen();
  testUserFileOpenStillFocusesAfterSubagentDone();
  testFileOfAFinishedSubagentShowsItsParentsRun();
  console.log('subagentDoneAdjacencyNoReport.test.js: all tests passed');
}

main();
