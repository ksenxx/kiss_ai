// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for a background tab's rendering swapping
// itself on screen (media/main.js processOutputEventForBgTab): the user
// is reading a finished sub-agent's tab while the parent, hidden, gets
// a `summary` tool_call.  The summary folds the fan-out panel, the fold
// closes the finished sub-agent tabs, and closing the tab on screen
// restores the parent (restoreTab moves its fragment into #output and
// loads the PRE-event state as the live stream).  The event must then
// finish as the visible tab's: rendered once (the dispatcher re-checks
// "for the active tab?" after the background pass), with the live
// stream continuing from the state that rendered it (the next thoughts
// open a new panel after the summary, not the sealed one before it).

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf, click, output} = require('./simplify2_harness');
const {
  bootParallelRun,
  finishFanOut,
  togglePanel,
  subagentTabEls,
  activeTabId,
  tabEl,
} = require('./subagent_tab_helpers');

/** Boot a parent with two finished, reopened sub-agent tabs; show sub 1. */
function bootReopened() {
  // Synchronous animation frames: streamed text is painted at once.
  const booted = bootParallelRun(2, makePage({beforeScripts: stubRaf}));
  const {win, panel, parentId, subTabIds} = booted;
  finishFanOut(win, subTabIds);
  assert.strictEqual(subagentTabEls(win).length, 0, 'children closed');
  togglePanel(win, panel);
  assert.strictEqual(subagentTabEls(win).length, 2, 'children reopened');
  const subIds = subagentTabEls(win).map(el => el.dataset.tabId);
  click(win, tabEl(win, subIds[0]));
  assert.strictEqual(activeTabId(win), subIds[0], 'sub 1 is on screen');
  send(win, {type: 'thinking_start', tabId: parentId});
  send(win, {
    type: 'thinking_delta',
    tabId: parentId,
    text: 'before the summary',
  });
  send(win, {type: 'thinking_end', tabId: parentId});
  return {win, parentId, subIds};
}

function testSummaryRenderedOnceAfterSwap() {
  const {win, parentId, subIds} = bootReopened();
  send(win, {
    type: 'tool_call',
    name: 'summary',
    tabId: parentId,
    extras: {description: 'folded the fan-out'},
  });
  assert.strictEqual(
    activeTabId(win),
    parentId,
    'closing the sub-agent tab on screen put the parent there',
  );
  assert.ok(
    !subIds.some(id => tabEl(win, id)),
    'the folded fan-out closed its finished sub-agent tabs',
  );
  const summaries = output(win).querySelectorAll('.ev.tc.tc-summary');
  assert.strictEqual(
    summaries.length,
    1,
    'the summary that swapped the tab on screen is rendered exactly once',
  );
  assert.ok(
    summaries[0].querySelector('.summary-sub .tc-run-parallel'),
    'the summary digest holds the folded fan-out',
  );

  // The live stream continues from the state that rendered the summary:
  // the next thoughts open a panel AFTER it, not in the sealed one.
  send(win, {type: 'tool_result', tabId: parentId, content: 'ok'});
  send(win, {type: 'thinking_start', tabId: parentId});
  send(win, {
    type: 'thinking_delta',
    tabId: parentId,
    text: 'after the summary',
  });
  const panels = Array.from(output(win).children);
  const summaryIdx = panels.indexOf(summaries[0]);
  const after = panels.slice(summaryIdx + 1);
  assert.ok(
    after.some(p => /after the summary/.test(p.textContent || '')),
    'thoughts streamed after the swap land in a panel below the summary',
  );
  assert.ok(
    !/after the summary/.test(summaries[0].textContent || ''),
    'they do not land in a panel the summary folded away',
  );
  assert.strictEqual(
    output(win).querySelectorAll('.ev.tc.tc-summary').length,
    1,
    'still one summary panel',
  );
  win.close();
  console.log(
    '  ok - a background render that swaps its tab on screen renders once',
  );
}

function testStepCountFollowsTheSwap() {
  const {win, parentId} = bootReopened();
  const steps = win.document.getElementById('status-steps');
  send(win, {
    type: 'tool_call',
    name: 'summary',
    tabId: parentId,
    extras: {description: 'folded'},
  });
  assert.strictEqual(activeTabId(win), parentId);
  send(win, {type: 'tool_result', tabId: parentId, content: 'ok'});
  send(win, {type: 'thinking_start', tabId: parentId});
  send(win, {type: 'thinking_delta', tabId: parentId, text: 'next'});
  // Steps so far: the pre-summary thoughts (1), this one (2).
  assert.strictEqual(
    steps.textContent,
    'Steps: 2',
    'the step counter continues from the state that rendered the summary',
  );
  win.close();
  console.log('  ok - the step counter follows the swapped-in tab');
}

// `clear` and `showWelcome` for a hidden tab fold its fan-outs first, so
// the same swap can happen there: the branch that empties the tab must
// then be the visible-tab one (the transcript is on screen now), not
// the hidden-tab one that nulls a fragment restoreTab already drained.
function testClearAfterSwapEmptiesTheTranscript() {
  const {win, parentId} = bootReopened();
  send(win, {type: 'clear', tabId: parentId, chat_id: 'chat-p'});
  assert.strictEqual(activeTabId(win), parentId, 'the parent is on screen');
  assert.strictEqual(
    output(win).querySelectorAll('.tc-run-parallel').length,
    0,
    'the cleared transcript shows no panel of the replaced run',
  );
  assert.strictEqual(
    output(win).querySelectorAll('.llm-panel').length,
    0,
    'nor the thoughts before it',
  );
  // The replaced run's events are gone for good: a later event for the
  // tab lands in the fresh transcript, nothing of the old one resurfaces.
  send(win, {type: 'thinking_start', tabId: parentId});
  send(win, {type: 'thinking_delta', tabId: parentId, text: 'fresh run'});
  assert.ok(/fresh run/.test(output(win).textContent));
  assert.strictEqual(
    output(win).querySelectorAll('.tc-run-parallel').length,
    0,
  );
  win.close();
  console.log(
    '  ok - clear that swaps its tab on screen empties the transcript',
  );
}

function testShowWelcomeAfterSwapShowsTheWelcome() {
  const {win, parentId} = bootReopened();
  send(win, {type: 'showWelcome', tabId: parentId});
  assert.strictEqual(activeTabId(win), parentId, 'the parent is on screen');
  assert.strictEqual(
    output(win).querySelectorAll('.tc-run-parallel').length,
    0,
    'the reset transcript shows no panel of the discarded run',
  );
  const welcome = win.document.getElementById('welcome');
  assert.ok(welcome, 'the welcome screen exists');
  assert.notStrictEqual(
    welcome.style.display,
    'none',
    'the welcome screen is shown',
  );
  win.close();
  console.log(
    '  ok - showWelcome that swaps its tab on screen shows the welcome',
  );
}

function main() {
  testSummaryRenderedOnceAfterSwap();
  testStepCountFollowsTheSwap();
  testClearAfterSwapEmptiesTheTranscript();
  testShowWelcomeAfterSwapShowsTheWelcome();
  console.log('bgRenderTabSwap_sweep1026.test.js: all tests passed');
}

main();
