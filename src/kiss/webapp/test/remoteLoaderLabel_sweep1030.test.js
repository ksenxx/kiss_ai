// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for the line a remote chat shows while its
// transcript is on the way (showRemoteChatLoader in media/main.js): it
// names the machine by its label, as every other machine name on the
// page does, not by its registered name.

'use strict';

const assert = require('assert');
const {makePage} = require('./simplify2_harness');
const {finishedSession, sendHistory} = require('./history_helpers');

function testLoaderUsesTheMachineLabel() {
  const {win, posted} = makePage({bodyClass: 'remote-chat'});
  const row = finishedSession('remote', 'Remote task', {
    machine: 'box',
    task_id: 'r-task',
    has_events: true,
  });
  sendHistory(win, posted, [row], {
    machines: ['box'],
    localMachine: 'local',
    machineLabels: {box: 'Build Box'},
  });
  win.document.querySelector('.history-chat-header').click();
  const loader = win.document.querySelector('#remote-chat-loader');
  assert.ok(loader, 'the clicked tab says the chat is on its way');
  assert.strictEqual(
    loader.textContent,
    'Loading the chat from Build Box\u2026',
  );
  win.close();
  console.log('  ok - the remote-chat loader names the machine by its label');
}

console.log('remoteLoaderLabel_sweep1030');
testLoaderUsesTheMachineLabel();
