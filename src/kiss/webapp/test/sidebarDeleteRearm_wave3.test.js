// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The Schedule and Machines subpanels of media/main.js keep a row's
// inline Delete confirmation open across the list rebuild every reply
// causes, and keep its Delete spent (disabled) while the removal is in
// flight, so a second click cannot post it again.  Any `remove` reply
// used to re-arm it: with two confirmations answered (rows A and B), the
// reply to B's removal re-enabled A's Delete while A's removal was still
// in flight.  Only the reply naming the row (`jobId` for a job, `name`
// for a machine, as the daemon echoes the removal) re-arms it; a reply
// for another row, or one that names no row, leaves it spent, and
// Cancel followed by the trash icon asks afresh.
//
// Real chat.html + main.js on jsdom (simplify2_harness).

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  fakeMatchMedia,
  fakeIntervals,
} = require('./simplify2_harness');

const {test, report} = makeRunner();

/** A connected desktop chat page. */
function makePage() {
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts(w) {
      stubRaf(w);
      fakeIntervals(w);
      fakeMatchMedia(w, true);
    },
  });
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  return {win, posted};
}

function job(id) {
  return {
    id,
    name: 'Job ' + id,
    schedule: 'every 5m',
    kind: 'prompt',
    what: 'do ' + id,
    enabled: true,
    running: false,
    nextRunAt: 0,
    lastRunAt: 0,
    lastStatus: '',
    workDir: '',
  };
}

function machine(name, local) {
  return {
    name,
    address: local ? 'https://10.0.0.5:8787' : 'ssh me@' + name,
    local,
    status: 'reachable',
    error: '',
  };
}

/** The row of `list` keyed `key`. */
function row(win, listId, key) {
  return Array.from(
    win.document.getElementById(listId).querySelectorAll('.sidebar-panel-row'),
  ).find(r => r.dataset.rowKey === key);
}

/** Open the row's confirmation and answer Delete: the removal is posted. */
function confirmDelete(win, listId, key, label) {
  const r = row(win, listId, key);
  r.querySelector(`[aria-label="${label}"]`).click();
  r.querySelector('.sidebar-confirm-yes').click();
}

/** Whether the row's confirmation is open and its Delete spent. */
function spent(win, listId, key) {
  const r = row(win, listId, key);
  assert.ok(r, `row ${key} is listed`);
  assert.strictEqual(
    r.querySelector('.sidebar-item-confirm').style.display,
    '',
    `row ${key} still asks`,
  );
  return r.querySelector('.sidebar-confirm-yes').disabled;
}

const SCHEDULE = 'meta-schedule-list';
const MACHINES = 'meta-machines-list';

async function main() {
  await test(
    "Schedule: the reply to another job's removal never re-arms a spent " +
      'Delete; its own refusal does',
    () => {
      const {win, posted} = makePage();
      send(win, {type: 'cronJobs', jobs: [job('a'), job('b')]});
      confirmDelete(win, SCHEDULE, 'b', 'Delete the job');
      confirmDelete(win, SCHEDULE, 'a', 'Delete the job');
      const removals = () =>
        posted.filter(m => m.type === 'cronAction' && m.action === 'remove');
      assert.deepStrictEqual(
        removals().map(m => m.jobId),
        ['b', 'a'],
      );
      // B is removed; A's removal is still in flight.
      send(win, {
        type: 'cronJobs',
        jobs: [job('a')],
        action: 'remove',
        jobId: 'b',
        error: '',
      });
      assert.ok(spent(win, SCHEDULE, 'a'), "B's reply leaves A's Delete spent");
      row(win, SCHEDULE, 'a').querySelector('.sidebar-confirm-yes').click();
      assert.strictEqual(removals().length, 2, 'no second removal of A');
      // A reply that names no job (an older daemon) says nothing about A.
      send(win, {
        type: 'cronJobs',
        jobs: [job('a')],
        action: 'remove',
        error: 'busy',
      });
      assert.ok(spent(win, SCHEDULE, 'a'), 'an unnamed reply leaves it spent');
      // A's own refusal re-arms it.
      send(win, {
        type: 'cronJobs',
        jobs: [job('a')],
        action: 'remove',
        jobId: 'a',
        error: 'job is already running',
      });
      assert.ok(!spent(win, SCHEDULE, 'a'), "A's refusal re-arms its Delete");
      win.close();
    },
  );

  await test("Schedule: another action's reply naming the job never re-arms it", () => {
    const {win} = makePage();
    send(win, {type: 'cronJobs', jobs: [job('a')]});
    confirmDelete(win, SCHEDULE, 'a', 'Delete the job');
    send(win, {
      type: 'cronJobs',
      jobs: [job('a')],
      action: 'pause',
      jobId: 'a',
      error: '',
    });
    assert.ok(spent(win, SCHEDULE, 'a'));
    win.close();
  });

  await test(
    "Machines: the reply to another machine's removal never re-arms a " +
      'spent Delete; its own refusal does',
    () => {
      const {win, posted} = makePage();
      const all = [
        machine('laptop', true),
        machine('m-a', false),
        machine('m-b', false),
      ];
      send(win, {type: 'machines', machines: all, checkedAt: 1});
      confirmDelete(win, MACHINES, 'm-b', 'Delete the machine');
      confirmDelete(win, MACHINES, 'm-a', 'Delete the machine');
      assert.deepStrictEqual(
        posted
          .filter(m => m.type === 'machineAction' && m.action === 'remove')
          .map(m => m.name),
        ['m-b', 'm-a'],
      );
      send(win, {
        type: 'machines',
        machines: [all[0], all[1]],
        checkedAt: 2,
        action: 'remove',
        name: 'm-b',
        error: '',
      });
      assert.ok(spent(win, MACHINES, 'm-a'), "m-b's reply leaves m-a spent");
      send(win, {
        type: 'machines',
        machines: [all[0], all[1]],
        checkedAt: 3,
        action: 'remove',
        name: 'm-a',
        error: 'tunnel busy',
      });
      assert.ok(!spent(win, MACHINES, 'm-a'), "m-a's refusal re-arms it");
      win.close();
    },
  );

  await test('a Delete left spent asks afresh after Cancel and the trash icon', () => {
    const {win, posted} = makePage();
    send(win, {type: 'cronJobs', jobs: [job('a')]});
    confirmDelete(win, SCHEDULE, 'a', 'Delete the job');
    send(win, {
      type: 'cronJobs',
      jobs: [job('a')],
      action: 'remove',
      error: 'busy',
    });
    assert.ok(spent(win, SCHEDULE, 'a'));
    row(win, SCHEDULE, 'a').querySelector('.sidebar-confirm-no').click();
    confirmDelete(win, SCHEDULE, 'a', 'Delete the job');
    assert.strictEqual(
      posted.filter(m => m.type === 'cronAction' && m.action === 'remove')
        .length,
      2,
      'the new confirmation posts the removal again',
    );
    win.close();
  });

  report('sidebarDeleteRearm_wave3');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
