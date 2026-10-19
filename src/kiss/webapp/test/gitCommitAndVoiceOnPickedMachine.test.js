// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// The Git Commit button and the voice trigger follow the chat header's
// drop-downs: a tab picked onto a registered machine commits that
// machine's folder (the `autocommitAction` is stamped `machine` and
// carries the machine's working directory), and a dictated task
// submits to that machine; a tab back on this machine commits and
// submits here.

/* global require, console, process */

'use strict';

const assert = require('assert');
const {
  makePage,
  closeWindows,
  send,
  byId,
  msgs,
  assertJsonEqual,
} = require('./workdir_helpers');

const MACHINES = [
  {name: 'laptop', local: true, status: 'reachable', address: 'https://10.0.0.5:8787'},
  {name: 'gpu-box', status: 'reachable', address: 'https://gpu:8787'},
];
const BOX_DIRS = {
  type: 'machineWorkDirs',
  machine: 'gpu-box',
  workDir: '/srv/proj',
  recentWorkDirs: [{path: '/srv/proj', ts: Date.now() / 1000}],
  error: '',
};

function pickMachine(win, value) {
  const select = byId(win, 'chat-machine');
  select.value = value;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/** A page whose daemon knows its own folder and the machines, picked onto gpu-box. */
function bootOnGpuBox() {
  const made = makePage();
  send(made.win, {
    type: 'configData',
    config: {work_dir: '/work/ws', recent_work_dirs: []},
    machine: 'laptop',
  });
  send(made.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  pickMachine(made.win, 'gpu-box');
  send(made.win, BOX_DIRS);
  made.posted.length = 0;
  return made;
}

function tabId(win) {
  return win._testApi.getActiveTabId();
}

function testTheButtonCommitsTheMachinesFolder() {
  const {win, posted} = bootOnGpuBox();
  byId(win, 'autocommit-btn').click();
  assertJsonEqual(msgs(posted, 'autocommitAction'), [
    {tabId: tabId(win), workDir: '/srv/proj', machine: 'gpu-box', type: 'autocommitAction'},
  ]);
  assert.ok(byId(win, 'autocommit-btn').disabled, 'in flight until the outcome');

  // The outcome re-arms the button; back on this machine the next
  // commit is this daemon's folder, with no machine.
  send(win, {
    type: 'autocommit_done', success: true, committed: true, manual: true,
    message: 'gpu-box: Committed: x', tabId: tabId(win),
  });
  assert.ok(!byId(win, 'autocommit-btn').disabled, 're-armed');
  pickMachine(win, '');
  posted.length = 0;
  byId(win, 'autocommit-btn').click();
  assertJsonEqual(msgs(posted, 'autocommitAction'), [
    {tabId: tabId(win), workDir: '/work/ws', type: 'autocommitAction'},
  ]);
}

function testAMachineWhoseFolderIsUnknownCommitsItsOwn() {
  const {win, posted} = bootOnGpuBox();
  // A second pick onto another registered machine whose folder has
  // not been reported yet: the machine commits its own configured one.
  send(win, {
    type: 'machines',
    machines: [...MACHINES, {name: 'office', status: 'reachable', address: 'https://office:8787'}],
    checkedAt: 2,
  });
  pickMachine(win, 'office');
  posted.length = 0;
  byId(win, 'autocommit-btn').click();
  assertJsonEqual(msgs(posted, 'autocommitAction'), [
    {tabId: tabId(win), workDir: '', machine: 'office', type: 'autocommitAction'},
  ]);
}

function testTheVoiceTriggerSubmitsToTheMachine() {
  const {win, posted} = bootOnGpuBox();
  byId(win, 'task-input').value = 'Speaker #1 says that: run the tests';
  win.dispatchEvent(new win.CustomEvent('kiss-voice-submit', {detail: {tabId: tabId(win)}}));
  const submits = msgs(posted, 'submit');
  assert.strictEqual(submits.length, 1, 'one submit');
  assert.strictEqual(submits[0].machine, 'gpu-box', 'the dictated task goes to the machine');
  assert.strictEqual(submits[0].prompt, 'Speaker #1 says that: run the tests');
  assert.ok(!('workDir' in submits[0]), "the machine's own folder: no workDir");

  send(win, {type: 'status', running: false, tabId: tabId(win)});
  pickMachine(win, '');
  posted.length = 0;
  byId(win, 'task-input').value = 'Speaker #1 says that: and here';
  win.dispatchEvent(new win.CustomEvent('kiss-voice-submit', {detail: {}}));
  assert.strictEqual(msgs(posted, 'submit')[0].machine, '', 'back on this machine');
}

const tests = [
  ['the button commits the picked machine\'s folder', testTheButtonCommitsTheMachinesFolder],
  ['an unknown folder means the machine\'s own', testAMachineWhoseFolderIsUnknownCommitsItsOwn],
  ['the voice trigger submits to the picked machine', testTheVoiceTriggerSubmitsToTheMachine],
];

let failed = 0;
for (const [name, fn] of tests) {
  try {
    fn();
    console.log('ok - ' + name);
  } catch (e) {
    failed += 1;
    console.log('not ok - ' + name);
    console.log(e && e.stack ? e.stack : String(e));
  } finally {
    closeWindows();
  }
}
console.log(`\n${tests.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
