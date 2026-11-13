// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A transcript file link of a tab bound to a registered machine names
// a file on THAT machine: the daemon forwards the `checkPaths` and the
// click's `openFile` there unchanged, so the `workDir` they carry must
// be a directory of the machine.  main.js used to send this daemon's
// own working directory (`/Users/me/kiss` from a laptop), which the
// machine resolved `reports/x.txt` against and found nothing: the
// result's link was inert, or -- when a stale check had promoted it --
// the click answered "File not found".

const assert = require('assert');
const {makePage, send, click, ofType} = require('./simplify2_harness');

const HOME = 'home';
const BOX = 'box';
const LOCAL_WD = '/Users/me/kiss';
const BOX_WD = '/home/me/kiss';

// A page of a laptop daemon (working directory LOCAL_WD) whose tab BOX
// is bound to machine `box`, whose working directory is BOX_WD.  With
// `hidden`, a second tab HOME is on screen and BOX is parked.
function makeBoundPage(hidden) {
  const page = makePage();
  const tabs = [{tabId: BOX, chatId: 'chat-box', title: 'box', workDir: LOCAL_WD}];
  if (hidden) {
    tabs.unshift({tabId: HOME, chatId: 'chat-home', title: 'home', workDir: LOCAL_WD});
  }
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {type: 'configData', config: {work_dir: LOCAL_WD}, apiKeys: {}});
  send(page.win, {type: 'tabs_state', tabs});
  send(page.win, {type: 'tabMachine', tabId: BOX, machine: 'box'});
  send(page.win, {
    type: 'machineWorkDirs',
    machine: 'box',
    workDir: BOX_WD,
    recentWorkDirs: [],
    error: '',
  });
  if (hidden) {
    page.win._testApi.switchToTab(HOME);
    page.win._testApi.endLaunch();
    assert.strictEqual(page.win._testApi.getActiveTabId(), HOME);
  }
  page.posted.length = 0;
  return page;
}

function result(win, text) {
  send(win, {type: 'result', text, tabId: BOX});
}

function testCheckAndOpenUseTheMachinesDirectory() {
  const {win, posted} = makeBoundPage(false);
  result(win, 'Review: reports/round11-review.txt');
  const checks = ofType(posted, 'checkPaths');
  assert.strictEqual(checks.length, 1, 'one checkPaths for the result');
  assert.strictEqual(checks[0].machine, 'box');
  assert.strictEqual(
    checks[0].workDir,
    BOX_WD,
    "the check names the machine's directory, not this daemon's",
  );
  assert.deepStrictEqual(Array.from(checks[0].paths), ['reports/round11-review.txt']);
  const span = win.document.querySelector('#output [data-path-candidate]');
  assert.ok(span, 'the path became a candidate span');
  assert.strictEqual(
    span.getAttribute('data-path-machine'),
    'box',
    'the candidate span remembers the machine',
  );

  send(win, {
    type: 'pathsExist',
    tabId: BOX,
    machine: 'box',
    workDir: BOX_WD,
    results: {'reports/round11-review.txt': true},
  });
  const link = win.document.querySelector('#output [data-path]');
  assert.ok(link, 'the path is a clickable link');
  assert.ok(link.classList.contains('kiss-filelink'));
  assert.strictEqual(link.getAttribute('data-path-machine'), 'box');

  click(win, link);
  const opens = ofType(posted, 'openFile');
  assert.strictEqual(opens.length, 1, 'the click sends one openFile');
  assert.strictEqual(opens[0].path, 'reports/round11-review.txt');
  assert.strictEqual(opens[0].machine, 'box');
  assert.strictEqual(opens[0].tabId, BOX);
  assert.strictEqual(
    opens[0].workDir,
    BOX_WD,
    'the open resolves against the directory the link was checked under',
  );
  win.close();
  console.log("  ok - a bound tab's file links are checked and opened on the machine");
}

function testHiddenBoundTabChecksOnTheMachineToo() {
  const {win, posted} = makeBoundPage(true);
  result(win, 'Review: reports/round11-review.txt');
  const checks = ofType(posted, 'checkPaths');
  assert.strictEqual(checks.length, 1, 'one checkPaths for the parked tab');
  assert.strictEqual(checks[0].machine, 'box');
  assert.strictEqual(checks[0].tabId, BOX);
  assert.strictEqual(checks[0].workDir, BOX_WD);
  win.close();
  console.log("  ok - a parked bound tab's file links are checked on the machine");
}

function testUnknownMachineDirectoryFallsBackToItsDaemon() {
  // Before the machine's directories arrive the check carries no
  // directory: the machine's daemon then resolves relative paths
  // against its own working directory.
  const page = makePage();
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {type: 'configData', config: {work_dir: LOCAL_WD}, apiKeys: {}});
  send(page.win, {
    type: 'tabs_state',
    tabs: [{tabId: BOX, chatId: 'chat-box', title: 'box', workDir: LOCAL_WD}],
  });
  send(page.win, {type: 'tabMachine', tabId: BOX, machine: 'box'});
  page.posted.length = 0;
  result(page.win, 'see reports/round11-review.txt');
  const checks = ofType(page.posted, 'checkPaths');
  assert.strictEqual(checks.length, 1);
  assert.strictEqual(checks[0].machine, 'box');
  assert.strictEqual(checks[0].workDir, '', 'no local directory leaks to the machine');
  page.win.close();
  console.log("  ok - an unknown machine directory sends none, not this daemon's");
}

function testLocalTabKeepsThisDaemonsDirectory() {
  const page = makePage();
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {type: 'configData', config: {work_dir: LOCAL_WD}, apiKeys: {}});
  send(page.win, {
    type: 'tabs_state',
    tabs: [{tabId: BOX, chatId: 'chat-box', title: 'box', workDir: LOCAL_WD}],
  });
  page.posted.length = 0;
  result(page.win, 'see reports/round11-review.txt');
  const checks = ofType(page.posted, 'checkPaths');
  assert.strictEqual(checks.length, 1);
  assert.strictEqual(checks[0].machine, undefined);
  assert.strictEqual(checks[0].workDir, LOCAL_WD);
  send(page.win, {
    type: 'pathsExist',
    tabId: BOX,
    workDir: LOCAL_WD,
    results: {'reports/round11-review.txt': true},
  });
  const link = page.win.document.querySelector('#output [data-path]');
  assert.ok(link);
  click(page.win, link);
  const opens = ofType(page.posted, 'openFile');
  assert.strictEqual(opens.length, 1);
  assert.strictEqual(opens[0].workDir, LOCAL_WD);
  assert.strictEqual(opens[0].machine, undefined);
  page.win.close();
  console.log("  ok - a local tab's file links keep this daemon's directory");
}

function testInlineImagesOfABoundTabLoadFromTheMachine() {
  const {win, posted} = makeBoundPage(false);
  send(win, {
    type: 'result',
    tabId: BOX,
    text: 'done',
    success: true,
    summary: '<p><img src="plots/a.png" alt="Plot A"></p>',
  });
  const opens = ofType(posted, 'openFile').filter(m => /^img-\d+$/.test(m.token));
  assert.strictEqual(opens.length, 1, 'one tokened openFile for the picture');
  assert.strictEqual(opens[0].path, 'plots/a.png');
  assert.strictEqual(opens[0].machine, 'box');
  assert.strictEqual(opens[0].workDir, BOX_WD, "the picture is read in the machine's directory");
  win.close();
  console.log("  ok - a bound tab's inline images are read on the machine");
}

// A26_astra-1: a task's links are relative to the directory THAT task
// ran in on the machine (its replayed header says which), not to the
// directory the machine's drop-down shows by now: another surface may
// have moved the machine's working directory since the task started.
function testATasksOwnDirectoryWinsOverTheMachinesCurrentOne(hidden) {
  const {win, posted} = makeBoundPage(hidden);
  const OLD = '/home/me/old';
  send(win, {
    type: 'task_events',
    tabId: BOX,
    task: 'write the report',
    task_id: 't1',
    chat_id: 'chat-box',
    extra: JSON.stringify({work_dir: OLD, startTs: 1}),
    events: [],
  });
  send(win, {
    type: 'machineWorkDirs',
    machine: 'box',
    workDir: '/home/me/new',
    recentWorkDirs: [],
    error: '',
  });
  posted.length = 0;
  result(win, 'Review: reports/result.txt');
  const checks = ofType(posted, 'checkPaths');
  assert.strictEqual(checks.length, 1, 'one checkPaths for the result');
  assert.strictEqual(checks[0].machine, 'box');
  assert.strictEqual(
    checks[0].workDir,
    OLD,
    "BUG - the link was checked in the machine's current directory, not the task's",
  );
  if (!hidden) {
    send(win, {
      type: 'pathsExist',
      tabId: BOX,
      machine: 'box',
      workDir: OLD,
      results: {'reports/result.txt': true},
    });
    click(win, win.document.querySelector('#output [data-path]'));
    const opens = ofType(posted, 'openFile');
    assert.strictEqual(opens.length, 1, 'the click sends one openFile');
    assert.strictEqual(opens[0].workDir, OLD, "the open names the task's directory");
    assert.strictEqual(opens[0].machine, 'box');
  }
  // The daemon binds the tab elsewhere: the directory recorded for
  // `box` names nothing on `other`, whose own directory is used.
  send(win, {type: 'tabMachine', tabId: BOX, machine: 'other'});
  send(win, {
    type: 'machineWorkDirs',
    machine: 'other',
    workDir: '/srv/other',
    recentWorkDirs: [],
    error: '',
  });
  posted.length = 0;
  result(win, 'See: reports/next.txt');
  const again = ofType(posted, 'checkPaths');
  assert.strictEqual(again.length, 1);
  assert.strictEqual(again[0].machine, 'other');
  assert.strictEqual(again[0].workDir, '/srv/other', "the other machine's directory");
  win.close();
  console.log(
    "  ok - a task's own directory on the machine wins over its current one" +
      (hidden ? ' (parked tab)' : ''),
  );
}

// A transcript's paths are relative to the folder ITS task ran in, not
// to the folder the tab browses by now.  With `adjacent` the previous
// task of the chat is spliced in, whose task_settings name another
// folder than the current task's; without, the current task prints a
// result after the global working directory moved.
function testATranscriptResolvesInItsOwnTasksFolder(machine, adjacent) {
  const PREV = '/workspace/previous';
  const CUR = '/workspace/current';
  const {win, posted} = machine ? makeBoundPage(false) : makePage();
  if (!machine) {
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'configData', config: {work_dir: CUR}, apiKeys: {}});
    send(win, {type: 'tabs_state', tabs: [{tabId: BOX, chatId: 'chat-box', title: 'box'}]});
  }
  const settings = (id, dir) => ({
    type: 'task_settings',
    settings: {task_id: id, chat_id: 'chat-box', work_dir: dir},
  });
  send(win, {
    type: 'task_events',
    tabId: BOX,
    task: 'current',
    task_id: 'current',
    chat_id: 'chat-box',
    extra: JSON.stringify({work_dir: CUR, startTs: 1}),
    events: [settings('current', CUR)],
  });
  send(win, {type: 'configData', config: {work_dir: '/workspace/new'}, apiKeys: {}});
  posted.length = 0;
  const shown = {
    type: 'result',
    tabId: BOX,
    success: true,
    summary: '<p>See reports/a.txt</p><img src="plots/a.png">',
  };
  if (adjacent) {
    send(win, {
      type: 'adjacent_task_events',
      tabId: BOX,
      direction: 'prev',
      task: 'previous',
      task_id: 'previous',
      events: [settings('previous', PREV), shown],
    });
  } else {
    send(win, shown);
  }
  const want = adjacent ? PREV : CUR;
  const check = ofType(posted, 'checkPaths')[0];
  const image = ofType(posted, 'openFile').find(m => /^img-\d+$/.test(m.token));
  assert.strictEqual(check.workDir, want, "BUG - the link was checked outside its task's folder");
  assert.strictEqual(image.workDir, want, "BUG - the picture was read outside its task's folder");
  assert.strictEqual(check.machine, machine || undefined);
  win.close();
  console.log(
    '  ok - ' +
      (adjacent ? 'a spliced-in task' : 'the running task') +
      ' resolves its paths in its own folder' +
      (machine ? ' on the machine' : ''),
  );
}

testCheckAndOpenUseTheMachinesDirectory();
testHiddenBoundTabChecksOnTheMachineToo();
testInlineImagesOfABoundTabLoadFromTheMachine();
testUnknownMachineDirectoryFallsBackToItsDaemon();
testLocalTabKeepsThisDaemonsDirectory();
testATasksOwnDirectoryWinsOverTheMachinesCurrentOne(false);
testATasksOwnDirectoryWinsOverTheMachinesCurrentOne(true);
testATranscriptResolvesInItsOwnTasksFolder('', true);
testATranscriptResolvesInItsOwnTasksFolder('', false);
testATranscriptResolvesInItsOwnTasksFolder('box', true);
