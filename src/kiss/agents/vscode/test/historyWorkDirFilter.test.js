// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the History panel's WORKDIR filter: the daemon's
// `history` reply lists the chat header's working directories as
// `workDirs`, the <select id="hf-workdir"> offers All plus each of them
// in that order, choosing one hides the rows of other directories at
// once and refetches page one with `workDir` so the daemon applies it
// in SQL, the selection survives refreshes, and a directory the daemon
// stops listing falls back to All.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: m => posted.push(m),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function session(id, title, workDir, extra) {
  return Object.assign(
    {
      id: 'chat-' + id,
      task_id: 'task-' + id,
      title,
      timestamp: 1_700_000_000 + id * 100,
      preview: title,
      has_events: false,
      failed: false,
      is_running: false,
      tokens: 0,
      cost: 0,
      steps: 0,
      is_favorite: false,
      work_dir: workDir,
      startTs: (1_700_000_000 + id * 100) * 1000,
      endTs: (1_700_000_000 + id * 100) * 1000 + 10_000,
    },
    extra || {},
  );
}

const KISS = '/home/u/kiss';
const PAPER = '/home/u/paper';
const SESSIONS = [
  session(1, 'kiss one', KISS),
  session(2, 'paper one', PAPER),
  session(3, 'kiss two', KISS),
  session(4, 'nowhere', ''),
];
const WORK_DIRS = [KISS, PAPER];

// The reply to the webview's latest getHistory request: it must echo
// that request's generation (a filter change bumps it and refetches).
function historyReply(posted, sessions, workDirs) {
  const requests = posted.filter(m => m && m.type === 'getHistory');
  const last = requests[requests.length - 1];
  const reply = {
    type: 'history',
    sessions,
    offset: 0,
    generation: last ? last.generation : 0,
  };
  if (workDirs) reply.workDirs = workDirs;
  return reply;
}

function lastHistoryRequest(posted) {
  const requests = posted.filter(m => m && m.type === 'getHistory');
  return requests[requests.length - 1];
}

function rows(win) {
  return Array.from(
    win.document
      .getElementById('history-list')
      .querySelectorAll('.sidebar-item'),
  );
}

function visibleTitles(win) {
  return rows(win)
    .filter(r => r.style.display !== 'none')
    .map(r => r.querySelector('.sidebar-item-text').textContent)
    .sort();
}

function options(win) {
  return Array.from(win.document.getElementById('hf-workdir').options).map(
    o => [o.value, o.textContent],
  );
}

function selectWorkDir(win, value) {
  const sel = win.document.getElementById('hf-workdir');
  sel.value = value;
  sel.dispatchEvent(new win.Event('change', {bubbles: true}));
}

function testOptionsComeFromTheReply() {
  const {win, posted} = makeWebview();
  const sel = win.document.getElementById('hf-workdir');
  assert.ok(sel && sel.tagName === 'SELECT', 'hf-workdir select must exist');
  assert.ok(!sel.parentElement.hidden, 'the WORKDIR filter is always shown');
  assert.deepStrictEqual(options(win), [['', 'All']]);
  send(win, historyReply(posted, SESSIONS, WORK_DIRS));
  assert.deepStrictEqual(options(win), [
    ['', 'All'],
    [KISS, KISS],
    [PAPER, PAPER],
  ]);
  // Each path option carries the full path as its tooltip (the
  // select clamps its width).
  assert.strictEqual(sel.options[1].title, KISS);
  // The daemon's order (most recently used first) is kept, not sorted.
  send(win, historyReply(posted, SESSIONS, [PAPER, KISS, '/home/u/a']));
  assert.deepStrictEqual(
    options(win).map(o => o[0]),
    ['', PAPER, KISS, '/home/u/a'],
  );
  // A reply without the field (an older daemon) leaves the options alone;
  // non-string and empty entries are skipped.
  send(win, historyReply(posted, SESSIONS));
  assert.deepStrictEqual(
    options(win).map(o => o[0]),
    ['', PAPER, KISS, '/home/u/a'],
  );
  send(win, historyReply(posted, SESSIONS, [KISS, '', 7, null]));
  assert.deepStrictEqual(options(win), [
    ['', 'All'],
    [KISS, KISS],
  ]);
  win.close();
  console.log('  ok - WORKDIR options come from the history reply');
}

function testRowsCarryTheirWorkDir() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, WORK_DIRS));
  const seen = {};
  rows(win).forEach(r => {
    seen[r.querySelector('.sidebar-item-text').textContent] = r.dataset.workDir;
  });
  assert.deepStrictEqual(seen, {
    'kiss one': KISS,
    'paper one': PAPER,
    'kiss two': KISS,
    nowhere: '',
  });
  win.close();
  console.log('  ok - rows carry data-work-dir');
}

function testSelectingADirectoryFiltersRowsAndRefetches() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, WORK_DIRS));
  assert.strictEqual(visibleTitles(win).length, 4, 'All shows every row');
  const pill = win.document.getElementById('hf-workdir').parentElement;
  assert.ok(!pill.classList.contains('active'));
  assert.strictEqual(lastHistoryRequest(posted), undefined, 'no request yet');

  selectWorkDir(win, KISS);
  // The rows of other directories vanish at once, before the refetch.
  assert.deepStrictEqual(visibleTitles(win), ['kiss one', 'kiss two']);
  assert.ok(pill.classList.contains('active'), 'pill takes the accent');
  // The daemon applies the directory too (a project's rows live beyond
  // the loaded page): the change refetches page one with workDir.
  const request = lastHistoryRequest(posted);
  assert.strictEqual(request.workDir, KISS);
  assert.strictEqual(request.offset, 0);
  assert.strictEqual(request.tag, '');
  assert.strictEqual(request.machine, '');

  selectWorkDir(win, PAPER);
  assert.deepStrictEqual(visibleTitles(win), ['paper one']);

  selectWorkDir(win, '');
  assert.strictEqual(visibleTitles(win).length, 4, 'All restores every row');
  assert.ok(!pill.classList.contains('active'));
  assert.strictEqual(lastHistoryRequest(posted).workDir, '');

  // The directory combines with the other client-side filters (SEA).
  const fixture = [
    session(1, 'kiss paper', KISS, {sea: 'write_paper_sea'}),
    session(2, 'other paper', PAPER, {sea: 'write_paper_sea'}),
    session(3, 'kiss plain', KISS),
  ];
  send(win, historyReply(posted, fixture, WORK_DIRS));
  selectWorkDir(win, KISS);
  const sea = win.document.getElementById('hf-sea');
  sea.value = 'write_paper_sea';
  sea.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.deepStrictEqual(visibleTitles(win), ['kiss paper']);
  win.close();
  console.log('  ok - selecting a directory filters rows and refetches');
}

function testSelectionSurvivesRefreshesAndRemoval() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, WORK_DIRS));
  selectWorkDir(win, PAPER);
  const requestsBefore = posted.filter(m => m.type === 'getHistory').length;
  // A refresh listing the same directories keeps the selection, the
  // filter and the accent, and does not refetch.
  send(win, historyReply(posted, SESSIONS, WORK_DIRS));
  const sel = win.document.getElementById('hf-workdir');
  assert.strictEqual(sel.value, PAPER);
  assert.ok(sel.parentElement.classList.contains('active'));
  assert.deepStrictEqual(visibleTitles(win), ['paper one']);
  assert.strictEqual(
    posted.filter(m => m.type === 'getHistory').length,
    requestsBefore,
    'no refetch while the selection stands',
  );
  // The daemon no longer listing the chosen directory (its rows were
  // deleted) falls back to All and refetches for All.
  send(
    win,
    historyReply(
      posted,
      SESSIONS.filter(s => s.work_dir !== PAPER),
      [KISS],
    ),
  );
  assert.strictEqual(sel.value, '');
  assert.ok(!sel.parentElement.classList.contains('active'));
  assert.strictEqual(visibleTitles(win).length, 3, 'all rows after fallback');
  assert.strictEqual(
    lastHistoryRequest(posted).workDir,
    '',
    'refetched for All',
  );
  win.close();
  console.log(
    '  ok - selection survives refreshes; a removed directory resets',
  );
}

function testSelectingADirectoryKeepsChatsCollapsed() {
  const {win, posted} = makeWebview();
  // Two tasks of one chat make a collapsible chat group.
  const fixture = [
    session(1, 'kiss one', KISS, {id: 'chat-a'}),
    session(2, 'kiss two', KISS, {id: 'chat-a'}),
    session(3, 'paper one', PAPER, {id: 'chat-b'}),
  ];
  send(win, historyReply(posted, fixture, WORK_DIRS));
  const groups = () =>
    Array.from(
      win.document
        .getElementById('history-list')
        .querySelectorAll('.history-chat-group'),
    );
  assert.ok(groups().length >= 2, 'chats are grouped');
  const collapsedBefore = groups().map(g => g.classList.contains('collapsed'));
  // A directory scopes the list like a machine: it is no search, so
  // every chat stays as folded as it was.
  selectWorkDir(win, KISS);
  send(win, historyReply(posted, fixture.slice(0, 2), WORK_DIRS));
  assert.deepStrictEqual(
    groups()
      .filter(g => g.style.display !== 'none')
      .map(g => g.classList.contains('collapsed')),
    collapsedBefore.slice(0, 1),
  );
  win.close();
  console.log('  ok - selecting a directory keeps the chats collapsed');
}

function testStaleReplyLeavesTheSelectionAlone() {
  const {win, posted} = makeWebview();
  send(win, historyReply(posted, SESSIONS, WORK_DIRS));
  selectWorkDir(win, PAPER);
  const current = lastHistoryRequest(posted);
  assert.strictEqual(current.workDir, PAPER);
  const requestsBefore = posted.filter(m => m.type === 'getHistory').length;
  // A late reply of the previous view (an older generation) whose
  // directory list lacks the chosen one: renderHistory drops its rows,
  // and the options must stay as they are too, or the selection would
  // fall back to All and the pending current reply be dropped.
  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    generation: current.generation - 1,
    workDirs: [KISS],
  });
  const sel = win.document.getElementById('hf-workdir');
  assert.strictEqual(sel.value, PAPER, 'selection survives a stale reply');
  assert.deepStrictEqual(
    options(win).map(o => o[0]),
    ['', KISS, PAPER],
    'options survive a stale reply',
  );
  assert.strictEqual(
    posted.filter(m => m.type === 'getHistory').length,
    requestsBefore,
    'no refetch from a stale reply',
  );
  // The current reply then renders and is filtered as asked.
  send(win, historyReply(posted, [SESSIONS[1]], WORK_DIRS));
  assert.deepStrictEqual(visibleTitles(win), ['paper one']);
  win.close();
  console.log('  ok - a stale-generation reply leaves the selection alone');
}

function testADirectoryResetsAndLocksTheMachineFilter() {
  const {win, posted} = makeWebview();
  // A remote machine is registered and selected.
  send(
    win,
    Object.assign(historyReply(posted, SESSIONS, WORK_DIRS), {
      machines: ['box'],
      localMachine: 'myhost',
    }),
  );
  const machine = win.document.getElementById('hf-machine');
  machine.value = 'box';
  machine.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(lastHistoryRequest(posted).machine, 'box');
  assert.strictEqual(visibleTitles(win).length, 0, 'no local row is a box row');
  // Choosing a directory (a path on this machine) resets the machine
  // to All and locks it: the daemon answers with its own rows, and
  // they must show.
  selectWorkDir(win, KISS);
  assert.strictEqual(machine.value, '');
  assert.ok(
    machine.disabled,
    'machine filter locked while a directory is chosen',
  );
  const request = lastHistoryRequest(posted);
  assert.strictEqual(request.workDir, KISS);
  assert.strictEqual(request.machine, '');
  send(
    win,
    Object.assign(historyReply(posted, SESSIONS.slice(0, 1), WORK_DIRS), {
      machines: ['box'],
      localMachine: 'myhost',
    }),
  );
  assert.deepStrictEqual(visibleTitles(win), ['kiss one']);
  assert.strictEqual(machine.value, '', 'the rebuild keeps All');
  assert.ok(machine.disabled, 'still locked after a reply');
  // Back to every directory: the machine filter is usable again.
  selectWorkDir(win, '');
  assert.ok(!machine.disabled);
  win.close();
  console.log('  ok - a chosen directory resets and locks the machine filter');
}

// A directory adopted in the chat header (this surface's own pick, or
// another surface's pick arriving as workDirChanged) joins the WORKDIR
// filter at once, at the front, with no history reply in between; the
// client keeps its own copy of the header's list (configData's
// recent_work_dirs, moved to the front as directories are adopted), so
// successive picks all stay listed and the selection survives them.
function testAnAdoptedDirectoryJoinsTheOptionsAtOnce() {
  const {win, posted} = makeWebview();
  const sel = win.document.getElementById('hf-workdir');
  const NEW = '/home/u/new';
  const OTHER = '/home/u/other';
  // Before any history reply the daemon's configData already names the
  // header's directories: the filter offers them.
  send(win, {
    type: 'configData',
    config: {
      work_dir: KISS,
      recent_work_dirs: [
        {path: PAPER, ts: 100},
        {path: KISS, ts: 200},
      ],
    },
  });
  assert.deepStrictEqual(
    options(win).map(o => o[0]),
    ['', KISS, PAPER],
    'the filter offers the header directories from configData',
  );
  send(win, historyReply(posted, SESSIONS, WORK_DIRS));
  selectWorkDir(win, PAPER);
  const requestsBefore = posted.filter(m => m.type === 'getHistory').length;
  // Another surface adopts a new directory: the daemon's broadcast
  // alone puts it at the front, and the header lists it too.
  send(win, {type: 'workDirChanged', workDir: NEW});
  assert.deepStrictEqual(
    options(win).map(o => o[0]),
    ['', NEW, KISS, PAPER],
    'the adopted directory leads the options with no history reply',
  );
  assert.strictEqual(
    win.document.getElementById('chat-workdir').value,
    NEW,
    'the header follows',
  );
  assert.strictEqual(sel.value, PAPER, 'the selection survives the pick');
  assert.ok(sel.parentElement.classList.contains('active'));
  const requestsAfter = posted.filter(m => m.type === 'getHistory').length;
  assert.strictEqual(
    posted.filter(m => m.type === 'getHistory').slice(requestsBefore).every(
      m => m.workDir === PAPER,
    ),
    true,
    'any refetch of the re-scope keeps the selection',
  );
  // This surface's own pick (the VS Code host's workDirPicked after a
  // header pick): the previous directory stays listed, in second place,
  // like the daemon's recent_work_dirs would order it.
  send(win, {type: 'workDirPicked', path: OTHER});
  assert.deepStrictEqual(
    options(win).map(o => o[0]),
    ['', OTHER, NEW, KISS, PAPER],
    'successive picks all stay listed, newest first',
  );
  // Picking a listed directory again moves it to the front without a
  // duplicate, trailing separator or not.
  send(win, {type: 'workDirChanged', workDir: KISS + '/'});
  assert.deepStrictEqual(
    options(win).map(o => o[0]),
    ['', KISS + '/', OTHER, NEW, PAPER],
    'a re-adopted directory moves to the front, listed once',
  );
  assert.strictEqual(sel.value, PAPER, 'the selection still stands');
  // A fresh configData (the daemon's authoritative list) that no longer
  // names the chosen directory resets the filter to All and refetches.
  send(win, {
    type: 'configData',
    config: {work_dir: KISS, recent_work_dirs: [{path: KISS, ts: 300}]},
  });
  assert.deepStrictEqual(options(win).map(o => o[0]), ['', KISS]);
  assert.strictEqual(sel.value, '', 'the removed selection falls back to All');
  assert.ok(
    posted.filter(m => m.type === 'getHistory').length > requestsAfter,
    'refetched for All',
  );
  assert.strictEqual(lastHistoryRequest(posted).workDir, '');
  // The client's copy keeps the daemon's cap of 30 rows: adopting a
  // 31st directory drops the oldest, as the daemon does, so the next
  // reply changes nothing.
  const many = Array.from({length: 30}, (_, i) => ({
    path: '/home/u/d' + i,
    ts: 1000 - i,
  }));
  send(win, {
    type: 'configData',
    config: {work_dir: '/home/u/d0', recent_work_dirs: many},
  });
  assert.strictEqual(options(win).length, 31, 'All + 30 rows');
  send(win, {type: 'workDirChanged', workDir: NEW});
  const capped = options(win).map(o => o[0]);
  assert.strictEqual(capped.length, 31, 'All + 30 rows after the pick');
  assert.strictEqual(capped[1], NEW);
  assert.strictEqual(capped[30], '/home/u/d28', 'the oldest row is dropped');
  win.close();
  console.log('  ok - an adopted directory joins the options at once');
}

function runTests() {
  testOptionsComeFromTheReply();
  testRowsCarryTheirWorkDir();
  testSelectingADirectoryFiltersRowsAndRefetches();
  testSelectionSurvivesRefreshesAndRemoval();
  testSelectingADirectoryKeepsChatsCollapsed();
  testStaleReplyLeavesTheSelectionAlone();
  testADirectoryResetsAndLocksTheMachineFilter();
  testAnAdoptedDirectoryJoinsTheOptionsAtOnce();
}

try {
  runTests();
  console.log('\n8 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
