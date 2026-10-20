// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {bootParallelRun, subagentTabEls} = require('./subagent_tab_helpers');

const {makePage, send} = harness;

function activeTabEl(win) {
  return win.document.querySelector('#tab-list .chat-tab.active');
}

function clickTab(win, tabId) {
  harness.clickTab(win, tabId);
  assert.strictEqual(
    activeTabEl(win).dataset.tabId,
    tabId,
    'clicking tab ' + tabId + ' must activate it',
  );
}

function inputVisible(win) {
  const c = win.document.getElementById('input-container');
  assert.ok(c, '#input-container must exist');
  return c.style.display !== 'none';
}

function testRunningSubagentTabShowsInput() {
  const {win, subTabIds} = bootParallelRun(2, makePage());

  clickTab(win, subTabIds[0]);

  assert.ok(
    inputVisible(win),
    'the input textbox and the buttons below it must be VISIBLE on a ' +
      'sub-agent tab whose task is still running — the user must be ' +
      'able to inject prompts into the running sub-agent',
  );
  const stopBtn = win.document.getElementById('stop-btn');
  assert.strictEqual(
    stopBtn.style.display,
    'flex',
    'the Stop button must be visible on a running sub-agent tab so ' +
      'the user can stop ONLY the sub-agent task',
  );
  win.close();
  console.log('  ok - running sub-agent tab shows input + stop button');
}

function testOpenSubagentTabActiveRespectsRunningState() {
  const {win, parentId, subTabIds} = bootParallelRun(1, makePage());

  clickTab(win, subTabIds[0]);
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabIds[0],
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
  });
  assert.ok(
    inputVisible(win),
    'openSubagentTab for the ACTIVE tab must show the input while the ' +
      'sub-agent is running',
  );

  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabIds[0],
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
    isDone: true,
  });
  assert.ok(
    !inputVisible(win),
    'openSubagentTab with isDone for the ACTIVE tab must hide the ' +
      'input — the sub-agent task already completed',
  );
  win.close();
  console.log('  ok - openSubagentTab on active tab respects running state');
}

function testTabSwitchTogglesInputByRunningState() {
  const {win, parentId, subTabIds} = bootParallelRun(2, makePage());

  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabIds[0],
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
    isDone: true,
  });

  clickTab(win, subTabIds[0]);
  assert.ok(
    !inputVisible(win),
    'switching to a DONE sub-agent tab must hide the input textbox ' +
      'and the buttons below it',
  );

  clickTab(win, subTabIds[1]);
  assert.ok(
    inputVisible(win),
    'switching to a still-RUNNING sibling sub-agent tab must show ' +
      'the input again',
  );

  clickTab(win, parentId);
  assert.ok(
    inputVisible(win),
    'switching back to the parent (regular) tab must show the input',
  );
  win.close();
  console.log('  ok - tab switch toggles input by sub-agent running state');
}

function testSubagentDoneRemovesInput() {
  const {win, subTabIds} = bootParallelRun(2, makePage());

  clickTab(win, subTabIds[0]);
  assert.ok(inputVisible(win), 'input must be visible while running');

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  assert.ok(
    !subagentTabEls(win).some(e => e.dataset.tabId === subTabIds[0]),
    'the finished sub-agent tab must close on subagentDone',
  );
  const active = activeTabEl(win);
  assert.notStrictEqual(
    active.dataset.tabId,
    subTabIds[0],
    'the finished sub-agent tab must not stay active',
  );
  win.close();
  console.log('  ok - subagentDone removes the finished sub-agent input');
}

function testSendInjectsPromptWithSubagentTabId() {
  const {win, posted, subTabIds} = bootParallelRun(2, makePage());

  clickTab(win, subTabIds[1]);
  assert.ok(inputVisible(win), 'input must be visible while running');

  const inp = win.document.getElementById('task-input');
  inp.value = 'focus on the tests';
  const before = posted.length;
  win.document
    .getElementById('send-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const injected = posted
    .slice(before)
    .find(m => m.type === 'appendUserMessage');
  assert.ok(
    injected,
    'sending while the sub-agent runs must post appendUserMessage ' +
      '(prompt injection into the live sub-agent)',
  );
  assert.strictEqual(
    injected.prompt,
    'focus on the tests',
    'the injected prompt must carry the typed text',
  );
  assert.strictEqual(
    injected.tabId,
    subTabIds[1],
    'the injected prompt must be routed with the SUB-AGENT tab id, ' +
      'not the parent tab id',
  );
  assert.ok(
    !posted.slice(before).some(m => m.type === 'submit'),
    'no fresh-task submit may be posted while the sub-agent runs',
  );
  assert.strictEqual(inp.value, '', 'the input clears after injecting');
  win.close();
  console.log('  ok - send on running sub-agent tab injects with sub tab id');
}

function testStopPostsStopWithSubagentTabId() {
  const {win, posted, subTabIds} = bootParallelRun(2, makePage());

  clickTab(win, subTabIds[0]);
  const before = posted.length;
  win.document
    .getElementById('stop-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  const stop = posted.slice(before).find(m => m.type === 'stop');
  assert.ok(stop, 'clicking Stop on a sub-agent tab must post stop');
  assert.strictEqual(
    stop.tabId,
    subTabIds[0],
    'the stop must carry the SUB-AGENT tab id so only the sub-agent ' +
      "task is stopped — not the parent's",
  );
  win.close();
  console.log('  ok - stop on running sub-agent tab targets the sub tab id');
}

function testStatusNotRunningRemovesInputOnActiveSubTab() {
  const {win, subTabIds} = bootParallelRun(2, makePage());

  clickTab(win, subTabIds[0]);
  assert.ok(inputVisible(win), 'input must be visible while running');

  send(win, {type: 'status', running: false, tabId: subTabIds[0]});

  assert.ok(
    !inputVisible(win),
    'a running:false status for the active sub-agent tab must remove ' +
      'the input textbox and the buttons below it immediately',
  );
  const stopBtn = win.document.getElementById('stop-btn');
  assert.strictEqual(
    stopBtn.style.display,
    'none',
    'the Stop button must hide once the sub-agent task ended',
  );
  win.close();
  console.log('  ok - status running:false removes active sub tab input');
}

function main() {
  testRunningSubagentTabShowsInput();
  testOpenSubagentTabActiveRespectsRunningState();
  testTabSwitchTogglesInputByRunningState();
  testSubagentDoneRemovesInput();
  testSendInjectsPromptWithSubagentTabId();
  testStopPostsStopWithSubagentTabId();
  testStatusNotRunningRemovesInputOnActiveSubTab();
  console.log('subagentRunningInput.test.js: all tests passed');
}

main();
