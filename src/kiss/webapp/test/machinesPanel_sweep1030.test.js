// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the right sidebar's Machines and Schedule
// subpanels of media/main.js when they show a registered machine:
//
// * a refused `machines` reply (the machine is down) is no baseline: the
//   machine's next answer adds nothing and leaves the drawer closed;
// * a poll reply that rebuilds the Schedule list while a Delete is in
//   flight keeps that Delete spent, so a second click posts nothing;
//   the reply to the removal re-arms it;
// * a finished task asks the machine the subpanel shows for its
//   registry too (a `/machine add` run there changed that one).

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  fakeMatchMedia,
  fakeIntervals,
} = require('./simplify2_harness');

const MACHINES = [
  {
    name: 'laptop',
    address: 'https://10.0.0.5:8787',
    local: true,
    status: 'reachable',
    error: '',
  },
  {
    name: 'gpu-box',
    address: 'ssh me@gpu',
    local: false,
    status: 'reachable',
    error: '',
  },
];

// gpu-box's own registry: the machine and one it reaches.
const GPU_MACHINES = [
  {
    name: 'gpu-box',
    address: 'https://10.0.0.9:8787',
    local: true,
    status: 'reachable',
    error: '',
  },
  {
    name: 'rack-1',
    address: 'ssh me@rack',
    local: false,
    status: 'reachable',
    error: '',
  },
];

/** A connected chat page (desktop unless *desktop* is false). */
function makePage(desktop) {
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts(w) {
      stubRaf(w);
      fakeIntervals(w);
      fakeMatchMedia(w, desktop !== false);
    },
  });
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  return {win, posted};
}

/** Pick *machine* in the chat header's drop-down. */
function pickMachine(win, machine) {
  const select = win.document.getElementById('chat-machine');
  select.value = machine;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/** gpu-box's forwarded `machines` reply. */
function gpuMachines(fields) {
  return Object.assign(
    {type: 'machines', machine: 'gpu-box', forwarded: true, checkedAt: 2},
    fields,
  );
}

function testRefusedReplyIsNoBaseline() {
  const {win} = makePage(false);
  const panel = win.document.getElementById('meta-panel');
  const input = win.document.getElementById('task-input');
  pickMachine(win, 'gpu-box');
  send(win, gpuMachines({machines: GPU_MACHINES}));
  input.focus();
  assert.ok(!panel.classList.contains('open'), 'the drawer starts closed');
  send(
    win,
    gpuMachines({
      machines: [],
      checkedAt: 0,
      error: 'gpu-box: the forward timed out',
    }),
  );
  send(win, gpuMachines({machines: GPU_MACHINES, checkedAt: 3}));
  assert.ok(!panel.classList.contains('open'), 'nothing was added');
  assert.strictEqual(win.document.activeElement, input, 'focus stays put');
  // A machine that is really new still brings the drawer up.
  const added = GPU_MACHINES.concat([
    {
      name: 'rack-2',
      address: 'ssh me@r2',
      local: false,
      status: 'reachable',
      error: '',
    },
  ]);
  send(win, gpuMachines({machines: added, checkedAt: 4}));
  assert.ok(panel.classList.contains('open'), 'rack-2 was added');
  win.close();
  console.log('  ok - a refused machines reply does not reset the baseline');
}

function testPollKeepsASpentDelete() {
  const {win, posted} = makePage(true);
  const job = {
    id: 'a',
    name: 'Job',
    schedule: 'every 5m',
    kind: 'prompt',
    what: 'do it',
    enabled: true,
    running: false,
    nextRunAt: 0,
    lastRunAt: 0,
    lastStatus: '',
    workDir: '',
  };
  const list = win.document.getElementById('meta-schedule-list');
  const row = () => list.querySelector('.sidebar-panel-row');
  const removals = () =>
    posted.filter(m => m.type === 'cronAction' && m.action === 'remove');
  send(win, {type: 'cronJobs', jobs: [job]});
  row().querySelector('[aria-label="Delete the job"]').click();
  row().querySelector('.sidebar-confirm-yes').click();
  assert.strictEqual(removals().length, 1);
  // The 30 s poll answers before the removal does.
  send(win, {type: 'cronJobs', jobs: [job]});
  const yes = row().querySelector('.sidebar-confirm-yes');
  assert.strictEqual(
    row().querySelector('.sidebar-item-confirm').style.display,
    '',
    'the question is asked again in the rebuilt row',
  );
  assert.ok(yes.disabled, 'its Delete is still spent');
  yes.click();
  assert.strictEqual(removals().length, 1, 'no second removal');
  // The daemon refuses the removal (its reply names the job): Delete is
  // live again.
  send(win, {
    type: 'cronJobs',
    jobs: [job],
    action: 'remove',
    jobId: 'a',
    error: 'busy',
  });
  assert.ok(!row().querySelector('.sidebar-confirm-yes').disabled);
  win.close();
  console.log('  ok - a poll reply keeps an answered Delete spent');
}

function testTaskDoneAsksTheShownMachine() {
  const {win, posted} = makePage(true);
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'tabMachine', tabId, machine: 'gpu-box'});
  send(win, gpuMachines({machines: GPU_MACHINES}));
  const from = posted.length;
  send(win, {type: 'status', running: true, tabId, startTs: 1000});
  send(win, {type: 'task_done', tabId, startTs: 1000, endTs: 7000});
  const asked = posted
    .slice(from)
    .filter(m => m.type === 'getMachines')
    .map(m => m.machine || '');
  assert.deepStrictEqual(asked, ['', 'gpu-box']);
  // gpu-box's answer lists the machine its task registered.
  const added = GPU_MACHINES.concat([
    {
      name: 'rack-2',
      address: 'ssh me@r2',
      local: false,
      status: 'reachable',
      error: '',
    },
  ]);
  send(win, gpuMachines({machines: added, checkedAt: 5}));
  const names = Array.from(
    win.document.querySelectorAll('#meta-machines-list .sidebar-panel-name'),
  ).map(n => n.textContent);
  assert.ok(names.includes('rack-2'), names.join(','));
  win.close();
  console.log('  ok - task_done asks the machine the subpanel shows');
}

function main() {
  console.log('machinesPanel_sweep1030');
  testRefusedReplyIsNoBaseline();
  testPollKeepsASpentDelete();
  testTaskDoneAsksTheShownMachine();
}

main();
