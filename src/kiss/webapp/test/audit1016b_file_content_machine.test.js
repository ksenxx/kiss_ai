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
// machine; a finished task's report is asked of the machine the task
// ran on (the daemon's binding, `tabMachine`), whose stamped reply
// opens it.  Before the fix the tab adopted the pick and its Save /
// Reload went to the picked machine.
// main.js against the real chat.html, through the ui_antipattern harness.

const assert = require('assert');
const H = require('./ui_antipattern_harness');

const harness = require('./simplify2_harness');
const {MACHINES} = require('./settings_helpers');
const {
  writeReport,
  reportRequests,
  answerReport,
} = require('./subagent_tab_helpers');

const {makePage, send, sleep, click, ofType} = H;

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
  // The stubbed editor is created on the next microtask.
  await sleep(0);
  assert.strictEqual(created.length, before + 1, 'the file got an editor');
  created[before].editor._type('edited');
  posted.length = 0;
  const views = win.document.querySelectorAll('.content-tab-view');
  const view = views[views.length - 1];
  click(win, view.querySelector('.content-save-btn'));
  const saves = ofType(posted, 'saveFile');
  assert.strictEqual(saves.length, 1, 'one saveFile was posted');
  return saves[0];
}

async function testUnstampedReplyStaysLocal() {
  const ctx = makePage();
  setUp(ctx);
  // The user picks gpu-box: the pick is where new chats run (one opens
  // for it), not where the files of the chat left are.
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

function contentTabCount(win) {
  return win._testApi.openTabs().filter(t => t.isContentTab).length;
}

function testReportFollowsTheBinding() {
  const ctx = makePage();
  const {win, posted} = ctx;
  setUp(ctx);
  // The daemon bound the tab to gpu-box (its task runs there); the user
  // then picked this machine ('' in the drop-down), which opened a new
  // chat for this machine.  The gpu-box task ends behind that chat.
  send(win, {type: 'tabMachine', tabId: 'a1', machine: 'gpu-box'});
  pickMachine(win, '');
  assert.notStrictEqual(win._testApi.getActiveTabId(), 'a1', 'a new chat');
  writeReport(win, '/ws/a/reports/out.html', '<p>report</p>', {tabId: 'a1'});
  send(win, {type: 'task_done', success: true, tabId: 'a1'});
  const [req] = reportRequests(posted);
  assert.strictEqual(
    req.machine,
    'gpu-box',
    'the report is asked of the machine the task ran on, not of the pick: ' +
      JSON.stringify(req),
  );
  answerReport(win, req, '<p>report</p>', {machine: 'gpu-box'});
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
  testReportFollowsTheBinding();
  console.log('audit1016b_file_content_machine: all passed');
  process.exit(0);
})().catch(e => {
  console.error(e);
  process.exit(1);
});
