// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {bootParallelRun, subagentTabEls} = require('./subagent_tab_helpers');

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
  testDoneUnknownTabIsNoop();
  console.log('subagentTabAutoCloseOnDone.test.js: all tests passed');
}

main();
