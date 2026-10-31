// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end page tests: the top status bar (the one holding
// tokens, cost and steps; hidden on the remote page, whose task-info
// panel mirrors it) carries the server's machine name.  The daemon
// stamps `machine` into every configData reply
// (src/kiss/server/commands.py _cmd_get_config), which reaches
// media/main.js in the remote webapp.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function testMachineNameLandsInTheStatusBar() {
  const {win} = makePage();
  const el = win.document.getElementById('status-machine');
  assert.ok(el, '#status-machine must exist in the status bar');
  assert.strictEqual(
    el.closest('#tab-status-bar'),
    win.document.getElementById('tab-status-bar'),
    'the machine name lives in the tokens/cost/steps bar',
  );
  assert.strictEqual(el.textContent, '', 'empty until the daemon reports');
  send(win, {type: 'configData', config: {}, machine: 'build-box-7'});
  assert.strictEqual(el.textContent, 'build-box-7');
  win.close();
}

function testConfigDataWithoutMachineLeavesTheNameAlone() {
  const {win} = makePage();
  const el = win.document.getElementById('status-machine');
  send(win, {type: 'configData', config: {}, machine: 'build-box-7'});
  // A later reply without the field (an older daemon) must not blank
  // the name the user is looking at.
  send(win, {type: 'configData', config: {}});
  assert.strictEqual(el.textContent, 'build-box-7');
  win.close();
}

function main() {
  testMachineNameLandsInTheStatusBar();
  console.log('  ok - configData machine name shows in the status bar');
  testConfigDataWithoutMachineLeavesTheNameAlone();
  console.log('  ok - a reply without machine keeps the shown name');
  console.log('statusBarMachineName.test.js: all tests passed');
}

main();
