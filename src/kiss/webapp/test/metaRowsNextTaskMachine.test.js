// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// jsdom tests: the task-info panel's Workdir / Max budget / Machine
// rows of a chat headed for a registered machine.
//
// A task's rows come from its own task_settings event, which the
// daemon running the task emits (a remote task's settings are the
// remote daemon's, relayed with its transcript), so a task that has
// settings needs no machine stamp.  Without settings (a new chat, a
// task whose settings have not arrived) the rows describe the tab's
// NEXT task: the machine it goes to (the drop-down pick or the
// daemon's binding), that machine's working directory and its default
// budget as the machineWorkDirs reply told them — never this daemon's
// directory and budget for a chat headed elsewhere, which the chat
// header's drop-downs would contradict.
//
// Run with: node src/kiss/webapp/test/metaRowsNextTaskMachine.test.js

/* global require, console, process */

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  fakeIntervals,
  fakeMatchMedia,
  readyTabId,
  pickMachine,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

const EMDASH = '\u2014';
const MACHINES = [
  {name: 'laptop', local: true, status: 'reachable', address: 'https://l:1'},
  {name: 'box', label: 'gpu.lab', status: 'reachable', address: 'https://b:1'},
];

function makePage(opts) {
  const {desktop = true} = opts || {};
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts(w) {
      fakeIntervals(w);
      stubRaf(w);
      fakeMatchMedia(w, desktop);
    },
    extraScripts: ['marked.min.js'],
  });
  send(win, {
    type: 'configData',
    config: {work_dir: '/local/dir', max_budget: 42, recent_work_dirs: []},
    machine: 'laptop',
    apiKeys: {},
  });
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  const TAB = readyTabId(posted);
  posted.length = 0;
  return {win, posted, TAB};
}

function rowText(win, id) {
  return win.document.getElementById(id).textContent;
}

function rows(win) {
  return {
    workDir: rowText(win, 'meta-workdir'),
    budget: rowText(win, 'meta-max-budget'),
    machine: rowText(win, 'meta-machine'),
  };
}

function boxDirs(extra) {
  return Object.assign(
    {
      type: 'machineWorkDirs',
      machine: 'box',
      workDir: '/box/dir',
      recentWorkDirs: [{path: '/box/dir', ts: 1}],
      maxBudget: 100,
      error: '',
    },
    extra || {},
  );
}

async function main() {
  await test('a home chat without settings shows this daemon\'s defaults', () => {
    const {win} = makePage();
    assert.deepStrictEqual(rows(win), {
      workDir: '/local/dir',
      budget: '$42.00',
      machine: 'laptop',
    });
  });

  await test(
    'a chat whose next task goes to a picked machine shows that ' +
      "machine, its directory and its default budget once known",
    () => {
      const {win, posted} = makePage();
      pickMachine(win, 'box');
      assert.deepStrictEqual(
        rows(win),
        {workDir: EMDASH, budget: EMDASH, machine: 'gpu.lab'},
        "nothing of the machine's is known until its reply: no local fallback",
      );
      assert.strictEqual(
        posted.filter(m => m.type === 'getMachineWorkDirs').length,
        1,
        'the pick asks the machine once',
      );
      send(win, boxDirs());
      assert.deepStrictEqual(rows(win), {
        workDir: '/box/dir',
        budget: '$100.00',
        machine: 'gpu.lab',
      });
      // A machine whose config names no budget shows none, not ours.
      send(win, boxDirs({maxBudget: null}));
      assert.strictEqual(rows(win).budget, EMDASH);
      // Home again: this daemon's defaults are back.
      pickMachine(win, '');
      assert.deepStrictEqual(rows(win), {
        workDir: '/local/dir',
        budget: '$42.00',
        machine: 'laptop',
      });
    },
  );

  await test(
    "a task's own settings win over the next-task defaults, and the " +
      'Machine row names the binding, not an unsent pick',
    () => {
      const {win, TAB} = makePage();
      send(win, {type: 'tabMachine', tabId: TAB, machine: 'box'});
      send(win, boxDirs());
      assert.deepStrictEqual(
        rows(win),
        {workDir: '/box/dir', budget: '$100.00', machine: 'gpu.lab'},
        "the daemon's binding shows the machine's defaults",
      );
      send(win, {type: 'setTaskText', text: 'do it', tabId: TAB});
      send(win, {
        type: 'task_settings',
        settings: {
          model: 'model-x',
          work_dir: '/box/task',
          max_budget: 7.5,
          chat_id: 'chat-1',
          task_id: 'task-1',
          host: 'box.lab',
        },
        tabId: TAB,
        taskId: 'task-1',
      });
      assert.deepStrictEqual(
        rows(win),
        {workDir: '/box/task', budget: '$7.50', machine: 'gpu.lab'},
        "the remote task's own settings paint the rows",
      );
      assert.strictEqual(rowText(win, 'meta-model'), 'model-x');
      // Picking another machine for the NEXT task leaves the rows on
      // the task the tab shows.
      pickMachine(win, '');
      assert.deepStrictEqual(rows(win), {
        workDir: '/box/task',
        budget: '$7.50',
        machine: 'gpu.lab',
      });
      // A new run clears the settings: the rows describe the next
      // task again, on the machine now picked.
      send(win, {type: 'clear', chat_id: 'chat-1', tabId: TAB});
      assert.deepStrictEqual(rows(win), {
        workDir: '/local/dir',
        budget: '$42.00',
        machine: 'laptop',
      });
    },
  );

  await test(
    'a machineWorkDirs reply for another machine leaves the rows alone',
    () => {
      const {win} = makePage();
      pickMachine(win, 'box');
      send(win, boxDirs({machine: 'other', workDir: '/other', maxBudget: 1}));
      assert.deepStrictEqual(rows(win), {
        workDir: EMDASH,
        budget: EMDASH,
        machine: 'gpu.lab',
      });
    },
  );

  await test(
    'a content tab shown on the stacked layout keeps the rows on the ' +
      'chat they were lent to, its machine included',
    () => {
      const {win, TAB} = makePage({desktop: false});
      send(win, {type: 'tabMachine', tabId: TAB, machine: 'box'});
      assert.deepStrictEqual(rows(win), {
        workDir: EMDASH,
        budget: EMDASH,
        machine: 'gpu.lab',
      });
      // A file view opens as the active tab on the stacked layout; it
      // has no machine of its own (the global pick is this machine).
      send(win, {
        type: 'fileContent',
        name: 'report.html',
        path: '/box/dir/report.html',
        content: '<h1>report</h1>',
      });
      const active = win._testApi.getActiveTabId();
      assert.notStrictEqual(active, TAB, 'the file view is the active tab');
      assert.ok(
        win._testApi.openTabs().some(t => t.id === active && t.isContentTab),
        'the active tab is a content tab',
      );
      // The machine's reply, arriving now, still paints the chat's rows.
      send(win, boxDirs());
      assert.deepStrictEqual(
        rows(win),
        {workDir: '/box/dir', budget: '$100.00', machine: 'gpu.lab'},
        "the reply repaints the lent chat's rows under a content tab",
      );
      // A repaint from elsewhere (the machines poll) keeps them too.
      send(win, {type: 'machines', machines: MACHINES, checkedAt: 2});
      assert.deepStrictEqual(rows(win), {
        workDir: '/box/dir',
        budget: '$100.00',
        machine: 'gpu.lab',
      });
    },
  );

  report('metaRowsNextTaskMachine');
}

main().catch(e => {
  console.error(e && e.stack ? e.stack : String(e));
  process.exit(1);
});
