// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// The @-mention picker follows the chat header's machine drop-down: a
// tab picked onto a registered machine asks for that machine's files
// (`getFiles` stamped `machine`, carrying the machine's working
// directory) and counts a picked mention there (`recordFileUsage`
// stamped `machine`); a tab back on this machine asks this daemon.  A
// list from another machine than the tab's is dropped, and a machine
// still indexing its folder when the daemon stopped waiting is asked
// again.
//
// Regression: the page sent `getFiles` with this daemon's folder and no
// machine, so this daemon searched its own disk for the machine's files,
// found none, and the empty reply kept the picker closed.

/* global require */

'use strict';

const assert = require('assert');
const {
  runTests,
  send,
  byId,
  msgs,
  assertJsonEqual,
} = require('./workdir_helpers');
const {pickMachine, typeInto} = require('./simplify2_harness');
const {key} = require('./ui_antipattern_harness');
const {MACHINES, bootOnGpuBox} = require('./picked_machine_helpers');
const {activeTabId: tabId} = require('./subagent_tab_helpers');

/** The mentions the open picker offers ([] when it is closed). */
function pickerItems(win) {
  const ac = byId(win, 'autocomplete');
  if (ac.style.display !== 'block') return [];
  return Array.from(ac.querySelectorAll('.ac-item')).map(d => d.dataset.text);
}

/** Answer the page's mention query *prefix* with *files*, as the daemon would. */
function answerFiles(win, prefix, files, machine) {
  const reply = {type: 'files', files, prefix, tabId: tabId(win)};
  if (machine) reply.machine = machine;
  send(win, reply);
}

function testAMentionListsAndCountsTheMachinesFiles() {
  const {win, posted} = bootOnGpuBox();
  typeInto(win, byId(win, 'task-input'), 'look at @ma');
  assertJsonEqual(msgs(posted, 'getFiles'), [
    {
      prefix: 'ma',
      workDir: '/srv/proj',
      tabId: tabId(win),
      machine: 'gpu-box',
      type: 'getFiles',
    },
  ]);
  // The machine's list (the daemon stamps the forwarded reply with the
  // machine) opens the picker.
  answerFiles(win, 'ma', [{type: 'file', text: './main.py'}], 'gpu-box');
  assertJsonEqual(pickerItems(win), ['./main.py']);
  byId(win, 'autocomplete').querySelector('.ac-item').click();
  assert.strictEqual(byId(win, 'task-input').value, 'look at ./main.py ');
  assertJsonEqual(msgs(posted, 'recordFileUsage'), [
    {
      path: './main.py',
      workDir: '/srv/proj',
      machine: 'gpu-box',
      type: 'recordFileUsage',
    },
  ]);
}

function testAMachineWhoseFolderIsUnknownListsItsOwn() {
  const {win, posted} = bootOnGpuBox();
  // A pick onto another registered machine whose folder has not been
  // reported yet: the machine lists its own configured one.
  send(win, {
    type: 'machines',
    machines: [
      ...MACHINES,
      {name: 'office', status: 'reachable', address: 'https://office:8787'},
    ],
    checkedAt: 2,
  });
  pickMachine(win, 'office');
  posted.length = 0;
  typeInto(win, byId(win, 'task-input'), '@');
  assertJsonEqual(msgs(posted, 'getFiles'), [
    {
      prefix: '',
      workDir: '',
      tabId: tabId(win),
      machine: 'office',
      type: 'getFiles',
    },
  ]);
}

function testATabOnThisMachineAsksThisDaemon() {
  const {win, posted} = bootOnGpuBox();
  pickMachine(win, '');
  posted.length = 0;
  typeInto(win, byId(win, 'task-input'), '@ma');
  assertJsonEqual(msgs(posted, 'getFiles'), [
    {prefix: 'ma', workDir: '/work/ws', tabId: tabId(win), type: 'getFiles'},
  ]);
  answerFiles(win, 'ma', [{type: 'file', text: './make.py'}], '');
  assertJsonEqual(pickerItems(win), ['./make.py']);
  byId(win, 'autocomplete').querySelector('.ac-item').click();
  assertJsonEqual(msgs(posted, 'recordFileUsage'), [
    {path: './make.py', workDir: '/work/ws', type: 'recordFileUsage'},
  ]);
}

function testAMentionPickedWithTheCaretInsideTheWordReplacesIt() {
  const {win, posted} = bootOnGpuBox();
  pickMachine(win, '');
  posted.length = 0;
  const inp = byId(win, 'task-input');
  // An edit of a typed mention: the caret sits at "@re|port.md".
  inp.value = '@report.md summarize';
  inp.setSelectionRange(3, 3);
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  assertJsonEqual(
    msgs(posted, 'getFiles').map(m => m.prefix),
    ['re'],
  );
  answerFiles(win, 're', [{type: 'file', text: './report.md'}], '');
  byId(win, 'autocomplete').querySelector('.ac-item').click();
  assert.strictEqual(
    inp.value,
    './report.md summarize',
    "BUG - the old mention's tail after the caret survived the completion",
  );
  // The space after the word is kept, not doubled.
  assert.strictEqual(inp.selectionStart, './report.md'.length);
}

function testALateListOfThePreviousMachineIsDropped() {
  const {win} = bootOnGpuBox();
  typeInto(win, byId(win, 'task-input'), '@ma');
  // The tab moves back to this machine before gpu-box answers.
  pickMachine(win, '');
  typeInto(win, byId(win, 'task-input'), '@ma');
  answerFiles(win, 'ma', [{type: 'file', text: './make_local.py'}], '');
  assertJsonEqual(pickerItems(win), ['./make_local.py']);
  // gpu-box's list arrives late: it does not answer this mention.
  answerFiles(win, 'ma', [{type: 'file', text: './main_remote.py'}], 'gpu-box');
  assertJsonEqual(pickerItems(win), ['./make_local.py']);
}

function testALateListDoesNotReopenADismissedPicker() {
  // Escape (or a blur) dismisses the picker: the list still on its way
  // for the mention typed before does not reopen it; the next keystroke
  // asks again and that answer shows.
  const {win} = bootOnGpuBox();
  const inp = byId(win, 'task-input');
  typeInto(win, inp, '@a');
  answerFiles(win, 'a', [{type: 'file', text: './alpha.js'}], 'gpu-box');
  assertJsonEqual(pickerItems(win), ['./alpha.js']);
  typeInto(win, inp, '@ab');
  key(win, inp, 'Escape');
  assertJsonEqual(pickerItems(win), [], 'Escape closes the picker');
  answerFiles(win, 'ab', [{type: 'file', text: './abc.js'}], 'gpu-box');
  assertJsonEqual(
    pickerItems(win),
    [],
    'BUG: the late list reopened the picker the user dismissed',
  );
  typeInto(win, inp, '@abc');
  answerFiles(win, 'abc', [{type: 'file', text: './abc.js'}], 'gpu-box');
  assertJsonEqual(pickerItems(win), ['./abc.js'], 'typing on asks again');
  inp.dispatchEvent(new win.FocusEvent('blur'));
  assertJsonEqual(pickerItems(win), [], 'a blur closes the picker');
  answerFiles(win, 'abc', [{type: 'file', text: './abcd.js'}], 'gpu-box');
  assertJsonEqual(pickerItems(win), [], 'and keeps it closed');
}

function testAMachineStillIndexingIsAskedAgain() {
  const {win, posted} = bootOnGpuBox();
  typeInto(win, byId(win, 'task-input'), '@ma');
  posted.length = 0;
  // The daemon stopped waiting for gpu-box's first scan of its folder.
  const loading = {type: 'files', files: [], loading: true, prefix: 'ma'};
  send(win, {...loading, tabId: tabId(win), machine: 'gpu-box'});
  assertJsonEqual(msgs(posted, 'getFiles'), [
    {
      prefix: 'ma',
      workDir: '/srv/proj',
      tabId: tabId(win),
      machine: 'gpu-box',
      type: 'getFiles',
    },
  ]);
  // This daemon sends its own list after its loading reply unasked.
  pickMachine(win, '');
  typeInto(win, byId(win, 'task-input'), '@ma');
  posted.length = 0;
  send(win, {...loading, tabId: tabId(win)});
  assertJsonEqual(msgs(posted, 'getFiles'), []);
}

const tests = [
  [
    "a mention lists and counts the picked machine's files",
    testAMentionListsAndCountsTheMachinesFiles,
  ],
  [
    'a machine still indexing is asked again',
    testAMachineStillIndexingIsAskedAgain,
  ],
  [
    'a late list of the previous machine is dropped',
    testALateListOfThePreviousMachineIsDropped,
  ],
  [
    'a late list does not reopen a dismissed picker',
    testALateListDoesNotReopenADismissedPicker,
  ],
  [
    'a mention picked with the caret inside the word replaces it',
    testAMentionPickedWithTheCaretInsideTheWordReplacesIt,
  ],
  [
    "an unknown folder means the machine's own",
    testAMachineWhoseFolderIsUnknownListsItsOwn,
  ],
  [
    'a tab on this machine asks this daemon',
    testATabOnThisMachineAsksThisDaemon,
  ],
];

runTests(tests);
