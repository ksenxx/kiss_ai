// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console, process */

// Audit 2026-10-16 (mainjs-1 F1): the machine a content tab is bound to
// is the one the daemon stamped on the `fileContent` reply, never the
// header drop-down's unsent machine pick.  Every reply a registered
// machine answers carries `machine` (sidebar_panels.py
// machine_workspace_reply), so an unstamped reply is a file of this
// machine; a report (a synthetic event, no reply) is on the machine the
// task ran on (the daemon's binding, `tabMachine`).  Before the fix the
// tab adopted the pick and its Save / Reload went to the picked machine.
// main.js against the real chat.html, through the ui_antipattern harness.

const assert = require('assert');
const H = require('./ui_antipattern_harness');

const harness = require('./simplify2_harness');

const {makePage, send, sleep, click} = H;

const MACHINES = [
  {name: 'laptop', local: true, status: 'reachable', address: 'https://a'},
  {name: 'gpu-box', status: 'reachable', address: 'ssh me@gpu'},
];

function setUp(ctx) {
  const {win} = ctx;
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: '/ws/a'}});
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', workDir: '/ws/a', chatId: 'chat-1'}],
  });
  send(win, {type: 'machines', machines: MACHINES});
}

function pickMachine(win, name) {
  harness.pickMachine(win, name);
  assert.strictEqual(
    harness.byId(win, 'chat-machine').value,
    name,
    'the drop-down shows the pick',
  );
}

async function openEditAndSave(ctx, fileEvent) {
  const {win, posted, created} = ctx;
  const before = created.length;
  send(win, Object.assign({type: 'fileContent', tabId: 'a1'}, fileEvent));
  for (let i = 0; i < 100 && created.length < before + 1; i++) await sleep(20);
  assert.strictEqual(created.length, before + 1, 'the file got an editor');
  created[before].editor._type('edited');
  posted.length = 0;
  const views = win.document.querySelectorAll('.content-tab-view');
  const view = views[views.length - 1];
  click(win, view.querySelector('.content-save-btn'));
  const saves = posted.filter(m => m.type === 'saveFile');
  assert.strictEqual(saves.length, 1, 'one saveFile was posted');
  return saves[0];
}

async function testUnstampedReplyStaysLocal() {
  const ctx = makePage();
  setUp(ctx);
  // The user picks gpu-box but submits nothing: the pick is where the
  // NEXT task goes, not where the transcript's files are.
  pickMachine(ctx.win, 'gpu-box');
  const save = await openEditAndSave(ctx, {
    path: '/shared/notes.txt',
    name: 'notes.txt',
    content: 'v1 on disk',
    version: 'ver-1',
  });
  assert.strictEqual(
    save.machine,
    undefined,
    'a file the local daemon served is saved locally, not on the pick: ' +
      JSON.stringify(save),
  );
  assert.strictEqual(save.path, '/shared/notes.txt');
  assert.strictEqual(save.workDir, '/ws/a');
  ctx.win.close();
  console.log('  ok - unstamped fileContent + unsent pick -> local save');
}

async function testStampedReplyStaysOnItsMachine() {
  const ctx = makePage();
  setUp(ctx);
  const save = await openEditAndSave(ctx, {
    path: '/shared/notes.txt',
    name: 'notes.txt',
    content: 'v1 on disk',
    version: 'ver-1',
    machine: 'gpu-box',
  });
  assert.strictEqual(
    save.machine,
    'gpu-box',
    'a file gpu-box served is saved on gpu-box: ' + JSON.stringify(save),
  );
  ctx.win.close();
  console.log('  ok - stamped fileContent -> save goes to that machine');
}

function openReport(win, filePath, content) {
  send(win, {type: 'tool_call', name: 'Write', path: filePath, content});
  send(win, {
    type: 'tool_result',
    content: 'Successfully wrote ' + content.length + ' characters to ' + filePath,
    is_error: false,
    tool_name: 'Write',
    path: filePath,
  });
  send(win, {type: 'task_done', success: true});
}

function contentTabCount(win) {
  return win._testApi.openTabs().filter(t => t.isContentTab).length;
}

async function testReportFollowsTheBinding() {
  const ctx = makePage();
  const {win} = ctx;
  setUp(ctx);
  // The daemon bound the tab to gpu-box (its task ran there); the user
  // then picked this machine ('' in the drop-down) for the next task
  // without submitting.
  send(win, {type: 'tabMachine', tabId: 'a1', machine: 'gpu-box'});
  pickMachine(win, '');
  openReport(win, '/ws/a/reports/out.html', '<p>report</p>');
  assert.strictEqual(contentTabCount(win), 1, 'the report tab opened');
  // The report is a file of gpu-box: a fileContent reply gpu-box
  // answers for the same path is the same file (one tab per path per
  // machine), while an unstamped one (this machine's copy) is another.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/a/reports/out.html',
    name: 'out.html',
    content: '<p>report</p>',
    version: 'v1',
    machine: 'gpu-box',
  });
  assert.strictEqual(
    contentTabCount(win),
    1,
    "the report tab is gpu-box's file: a gpu-box reply reuses it",
  );
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/a/reports/out.html',
    name: 'out.html',
    content: '<p>local copy</p>',
    version: 'v1',
  });
  assert.strictEqual(
    contentTabCount(win),
    2,
    "this machine's copy of the path is a second tab",
  );
  win.close();
  console.log('  ok - a report is on the machine the task ran on');
}

(async () => {
  await testUnstampedReplyStaysLocal();
  await testStampedReplyStaysOnItsMachine();
  await testReportFollowsTheBinding();
  console.log('audit1016b_file_content_machine: all passed');
  process.exit(0);
})().catch(e => {
  console.error(e);
  process.exit(1);
});
