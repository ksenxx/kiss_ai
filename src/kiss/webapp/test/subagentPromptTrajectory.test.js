// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, clickTab} = require('./simplify2_harness');
const {bootParallelRun} = require('./subagent_tab_helpers');

// A prompt injected into a running sub-agent is echoed back as a
// `prompt` event tagged with the sub-agent tab: it renders in THAT tab's
// trajectory, survives a tab switch and comes back on a replay.  The
// send / stop routing of a sub-agent tab is pinned by
// subagentRunningInput.test.js.

function promptPanels(win, text) {
  return Array.from(win.document.querySelectorAll('.ev.prompt')).filter(el =>
    (el.textContent || '').includes(text),
  );
}

function bootWithSubagent() {
  const {win, parentId, subTabIds} = bootParallelRun(1, makePage());
  return {win, parentId, subTabId: subTabIds[0]};
}

function testEchoedPromptRendersInSubagentTrajectory() {
  const {win, subTabId} = bootWithSubagent();
  clickTab(win, subTabId);

  send(win, {
    type: 'prompt',
    text: 'HELLO SUBAGENT INJECTION',
    tabId: subTabId,
    taskId: 'sub-task-1',
  });
  assert.strictEqual(
    promptPanels(win, 'HELLO SUBAGENT INJECTION').length,
    1,
    'the injected prompt must appear as a Prompt panel in the ' +
      'sub-agent tab trajectory',
  );
  console.log('ok - echoed prompt renders in the sub-agent trajectory');
}

function testInjectedPromptSurvivesTabSwitch() {
  const {win, parentId, subTabId} = bootWithSubagent();
  clickTab(win, subTabId);
  send(win, {
    type: 'prompt',
    text: 'HELLO SUBAGENT INJECTION',
    tabId: subTabId,
    taskId: 'sub-task-1',
  });

  clickTab(win, parentId);
  clickTab(win, subTabId);
  assert.strictEqual(
    promptPanels(win, 'HELLO SUBAGENT INJECTION').length,
    1,
    'the injected prompt must still be in the sub-agent trajectory ' +
      'after switching tabs away and back',
  );
  console.log('ok - injected prompt survives a tab switch');
}

function testReplayRendersRecordedPrompt() {
  const {win, subTabId} = bootWithSubagent();
  clickTab(win, subTabId);

  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: 'sub-task-1',
    events: [
      {type: 'prompt', text: 'sub 1', taskId: 'sub-task-1'},
      {
        type: 'prompt',
        text: 'HELLO SUBAGENT INJECTION',
        taskId: 'sub-task-1',
      },
    ],
  });
  assert.ok(
    promptPanels(win, 'HELLO SUBAGENT INJECTION').length >= 1,
    'a task_events replay must re-render the recorded injected prompt',
  );
  console.log('ok - task_events replay renders the recorded prompt');
}

function main() {
  testEchoedPromptRendersInSubagentTrajectory();
  testInjectedPromptSurvivesTabSwitch();
  testReplayRendersRecordedPrompt();
  console.log('All subagentPromptTrajectory tests passed.');
}

main();
process.exit(0);
