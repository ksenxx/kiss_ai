// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {bootParallelRun, subagentTabEls} = require('./subagent_tab_helpers');

function activeTabEl(win) {
  return win.document.querySelector('#tab-list .chat-tab.active');
}

function testDoneClosesOnlyFinishedTab() {
  const {win, posted, panel, subTabIds} = bootParallelRun(2, makePage());

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  const remaining = subagentTabEls(win);
  assert.strictEqual(
    remaining.length,
    1,
    'the finished sub-agent tab must close as soon as it finishes, ' +
      'leaving only the still-running sibling open',
  );
  assert.strictEqual(
    remaining[0].dataset.tabId,
    subTabIds[1],
    'the surviving tab must be the still-running sibling',
  );
  assert.ok(
    posted.some(m => m.type === 'closeTab' && m.tabId === subTabIds[0]),
    'the backend must be told to close the finished sub-agent tab',
  );
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the run_parallel panel must stay uncollapsed while a sibling ' +
      'sub-agent is still running',
  );
  win.close();
  console.log('  ok - subagentDone closes only the finished tab');
}

function testLastDoneClosesTabAndCollapsesPanel() {
  const {win, posted, panel, subTabIds} = bootParallelRun(2, makePage());

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
  send(win, {type: 'subagentDone', tab_id: subTabIds[1]});

  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'every finished sub-agent tab must be closed',
  );
  for (const id of subTabIds) {
    assert.ok(
      posted.some(m => m.type === 'closeTab' && m.tabId === id),
      'the backend must be told to close sub-agent tab ' + id,
    );
  }
  assert.ok(
    panel.classList.contains('collapsed'),
    'the run_parallel panel must collapse once no open sub-agent ' +
      'tab remains',
  );
  win.close();
  console.log('  ok - last subagentDone closes its tab and collapses panel');
}

function testDoneOnActiveTabSwitchesAway() {
  const {win, parentId, subTabIds} = bootParallelRun(2, makePage());

  const subEl = subagentTabEls(win).find(
    el => el.dataset.tabId === subTabIds[0],
  );
  assert.ok(subEl, 'the first sub-agent tab must be in the tab bar');
  subEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    activeTabEl(win).dataset.tabId,
    subTabIds[0],
    'clicking the sub-agent tab must activate it',
  );

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  const active = activeTabEl(win);
  assert.ok(active, 'a tab must be active after the finished tab closed');
  assert.notStrictEqual(
    active.dataset.tabId,
    subTabIds[0],
    'the closed sub-agent tab must not stay active',
  );
  assert.ok(
    [parentId, subTabIds[1]].includes(active.dataset.tabId),
    'an adjacent tab (parent or sibling) must become active',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'only the still-running sibling sub-agent tab must remain open',
  );
  win.close();
  console.log('  ok - subagentDone on the active tab switches away');
}

function testDoneUnknownTabIsNoop() {
  const {win, posted, subTabIds} = bootParallelRun(2, makePage());

  const before = posted.length;
  send(win, {type: 'subagentDone', tab_id: 'no-such-tab'});

  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'a subagentDone for an unknown tab must not touch open tabs',
  );
  assert.ok(
    !posted
      .slice(before)
      .some(m => m.type === 'closeTab' && subTabIds.includes(m.tabId)),
    'no open sub-agent tab may be closed for an unknown tab id',
  );
  win.close();
  console.log('  ok - subagentDone for an unknown tab id is a no-op');
}

function main() {
  testDoneClosesOnlyFinishedTab();
  testLastDoneClosesTabAndCollapsesPanel();
  testDoneOnActiveTabSwitchesAway();
  testDoneUnknownTabIsNoop();
  console.log('subagentTabAutoCloseOnDone.test.js: all tests passed');
}

main();
