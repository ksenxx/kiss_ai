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
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  click: clickTab,
} = require('./simplify2_harness');
const {
  bootParallelRun,
  writeReport,
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

function testSoleSubagentReportDoesNotBecomeActiveOnClose() {
  // The reproduction: ONE sub-agent, the user is watching it, it writes
  // a report and finishes.  Its tab closes and the freshly appended
  // report tab sits right where the sub-agent tab used to be, so the
  // adjacency rule hands the user the report.
  const {win, parentId, subTabIds} = bootParallelRun(1, makePage());

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
    reportTab.id,
    'BUG - closing the finished sub-agent tab must not hand the user ' +
      'the report it just parked: the parent task is still running',
  );
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
  const {win, parentId, subTabIds} = bootParallelRun(2, makePage());

  clickTab(win, tabEl(win, subTabIds[0]));
  assert.strictEqual(activeTabId(win), subTabIds[0], 'precondition');

  writeReport(win, 'reports/sub1.md', '# one', {tabId: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  const reportTab = contentTabFor(win, 'sub1.md');
  assert.ok(reportTab, 'the report content tab must exist');
  assert.notStrictEqual(
    activeTabId(win),
    reportTab.id,
    'BUG - the parked report must not become active',
  );
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
  const {win, parentId, subTabIds} = bootParallelRun(1, page);
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

  const reportTab = contentTabFor(win, 'grand.md');
  assert.ok(reportTab, 'the grandchild report must open a content tab');
  assert.notStrictEqual(
    activeTabId(win),
    reportTab.id,
    'BUG - the grandchild report must not become active',
  );
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

  // The file screen's strip lists the chat group too: the user steps
  // onto the sub-agent tab from there.
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

// That the parked report stays reachable (one swipe left puts it on the
// file screen, rendered) is pinned by subagentReportNoTabSwitch.test.js.

function main() {
  console.log('subagentDoneAdjacencyNoReport.test.js');
  testSoleSubagentReportDoesNotBecomeActiveOnClose();
  testSubagentReportNotActiveWithSiblingStillRunning();
  testDeepSubagentLandsOnItsParentChatTab();
  testClosingActiveSubagentWithoutReportStillPicksChatTab();
  testUserOpenedContentTabSurvivesAndIsNotStolen();
  testUserFileOpenStillFocusesAfterSubagentDone();
  console.log('subagentDoneAdjacencyNoReport.test.js: all tests passed');
}

main();
