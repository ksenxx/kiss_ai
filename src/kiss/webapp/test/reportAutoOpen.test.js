// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests: whenever the agent generates a report (a .md or
// .html file written by the agent into a `reports` folder), the chat
// page must open that file in a content tab and switch to that tab --
// but only AFTER the task finishes (task_done for the owning chat tab).
// While the task is still running the report must NOT pop open.
// Reports of tasks that end in task_error / task_interrupted /
// task_stopped still open (the task finished).  The page asks the
// daemon for the file (an `openFile` with a `report:` token and the
// `report` flag) and the reply opens as any file does.  A report whose
// worktree is not merged gets no file from the daemon and stays
// pending: it is asked for again when that worktree is merged and
// whenever the daemon is reached anew (the pending reports are kept in
// localStorage, so also after a reload), and only the reply to its
// latest request counts.  A sub-agent's reports are never opened.

'use strict';

const assert = require('assert');
const {randomUUID} = require('crypto');
const {
  makePage: makeChatPage,
  send,
  readyTabId,
  clickTab,
  sendWorkDir,
  sendTabsState,
  storageSnapshot,
  plain,
} = require('./simplify2_harness');
// Content tabs on the group strip (#tab-list).  The strip lists them on
// the file screen only, which an opened report brings on view; the tab
// registry (registryContentTabs) holds them wherever the user is.
const {
  writeSuccess,
  writeReport,
  reportRequests,
  answerReport,
  stripContentTabEls: contentTabs,
  contentTabs: registryContentTabs,
  reportFrames,
  visibleSrcdoc: activeSrcdoc,
  spawnSubagent,
  bootParallelRun,
} = require('./subagent_tab_helpers');

function brokenMarked(win) {
  win.marked = {
    parse: function () {
      throw new Error('marked exploded');
    },
  };
}

// `reloadOf`: the page (a makePage result) this one is the reload of.
// It starts from the tabs that page persists on its way out and from
// its localStorage.  `storage`: localStorage items to start from
// instead.  `url`: null for a page whose localStorage throws.
// `shareWith`: the page of another window of the same origin, whose
// localStorage this one shares (jsdom gives every window its own).
function makePage(opts) {
  opts = opts || {};
  const before = opts.reloadOf;
  if (before) before.win.dispatchEvent(new before.win.Event('pagehide'));
  const storage = before ? storageSnapshot(before.win) : opts.storage || {};
  return makeChatPage({
    bodyClass: 'remote-chat',
    url: opts.url,
    initialState: before ? plain(before.getState()) : undefined,
    beforeScripts: win => {
      if (opts.brokenMarked) brokenMarked(win);
      if (opts.shareWith) {
        const shared = opts.shareWith.win.localStorage;
        Object.defineProperty(win, 'localStorage', {value: shared});
      }
      Object.keys(storage).forEach(key => {
        win.localStorage.setItem(key, storage[key]);
      });
    },
    extraScripts: opts.withMarked ? ['marked.min.js'] : [],
  });
}

// The names of the localStorage items that tell of the report at
// *path*: the ones it is stored in, and the one that says it is
// settled.
function reportItemNames(win, path) {
  return Object.keys(storageSnapshot(win)).filter(name => {
    return name.includes(JSON.stringify(path));
  });
}

// The name of an item a window stores a pending report in: no two
// are alike, also for the same report.
function storedItemName(tabId, workDir, path, since) {
  const report = JSON.stringify([tabId, workDir, path, since]);
  return 'kiss-pending-report:' + report + randomUUID();
}

// What the page keeps in localStorage of its pending reports, by tab,
// in the order they became pending.  A report has an item of its own
// for each window that stored it: the item's name tells the tab, the
// folder, the path and when the report was written, then an id of its
// own, and the item holds when the report became pending there.  A
// report that several windows stored became pending in the first.
function storedReports(win) {
  const prefix = 'kiss-pending-report:';
  const items = storageSnapshot(win);
  const reps = {};
  const told = new Set();
  Object.keys(items)
    .filter(name => name.startsWith(prefix))
    .sort((a, b) => items[a] - items[b])
    .forEach(name => {
      assert.ok(Number(items[name]) > 0, 'a time is held: ' + items[name]);
      const report = name.slice(prefix.length, name.lastIndexOf(']') + 1);
      assert.match(
        name.slice(prefix.length + report.length),
        /^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/,
        'the name ends in an id of its own: ' + name,
      );
      if (told.has(report)) return;
      told.add(report);
      const [tabId, workDir, path, since, ...more] = JSON.parse(report);
      assert.deepStrictEqual(more, [], 'the name tells nothing else');
      (reps[tabId] = reps[tabId] || []).push({path, workDir, since});
    });
  return reps;
}

// The daemon emits task_done when the task finishes; only then may
// stashed report tabs open.
function finishTask(win, extra) {
  send(win, Object.assign({type: 'task_done', success: true}, extra || {}));
}

// The daemon's outcome of a Merge, Discard or "Do nothing" (`action`
// merge / discard / nothing) of the worktree of tab *tabId*.
function worktreeResult(win, tabId, action, extra) {
  const ev = {type: 'worktree_result', tabId, action, success: true};
  send(win, Object.assign(ev, {message: action}, extra));
}

// Nothing is open, and the page asked the daemon for *asked* reports.
function assertNoReportTab(win, posted, label, asked = 0) {
  assert.strictEqual(
    reportRequests(posted).length,
    asked,
    label + ': the daemon must have been asked for ' + asked + ' report(s)',
  );
  assert.strictEqual(
    registryContentTabs(win).length,
    0,
    label + ': no content tab must be opened, but found one',
  );
  assert.strictEqual(
    reportFrames(win).length,
    0,
    label + ': no report iframe must be rendered',
  );
}

function assertReportTabActive(win, label) {
  const tabsFound = contentTabs(win);
  assert.strictEqual(
    tabsFound.length,
    1,
    label +
      ': BUG -- the agent generated a report but the chat page did ' +
      'not open it as a content tab',
  );
  assert.ok(
    tabsFound[0].classList.contains('active'),
    label + ': the report tab must become the active tab',
  );
  const out = win.document.getElementById('output');
  assert.strictEqual(
    out.style.display,
    'none',
    label + ': the chat surface must be hidden while the report is shown',
  );
  assert.ok(
    win.document.querySelector(
      '#content-tab-area .content-save-bar .content-mode-btn',
    ),
    label + ': the report is the file itself: Save bar, Edit source toggle',
  );
  return tabsFound[0];
}

function testMdReportWaitsForTaskDone() {
  const {win, posted} = makePage({withMarked: true});
  assert.ok(
    win.document.body.classList.contains('remote-chat'),
    'precondition: the web app marks its body with remote-chat',
  );
  // The task started in /ws/task; the global working folder moved since.
  send(win, {type: 'task_settings', settings: {work_dir: '/ws/task'}});
  sendWorkDir(win, '/ws/now');
  const text = '# Big Title\n\nHello **world**';
  // The daemon stamps every event with its clock (`ts`).
  writeReport(win, 'reports/summary.md', text, {ts: 1700000000000});
  // Reproduces the bug: the report used to pop open right on the Write
  // tool_result, stealing focus while the task was still running.
  assertNoReportTab(win, posted, 'md report before task_done');
  finishTask(win);
  // The page asks the daemon for the report the task wrote: the path
  // as the Write named it, the task's folder, and when it was written.
  assert.deepStrictEqual(plain(reportRequests(posted)), [
    {
      type: 'openFile',
      path: 'reports/summary.md',
      workDir: '/ws/task',
      tabId: win._testApi.getActiveTabId(),
      token: 'report:1',
      report: true,
      since: 1700000000000,
    },
  ]);
  assertNoReportTab(win, posted, 'md report before the daemon answers', 1);
  answerReport(win, reportRequests(posted)[0], text);
  const tab = assertReportTabActive(win, 'md report after task_done');
  assert.ok(
    tab.textContent.indexOf('summary.md') >= 0,
    'report tab must be titled after the report file',
  );
  const srcdoc = activeSrcdoc(win);
  assert.ok(
    srcdoc.indexOf('<!DOCTYPE html>') === 0,
    'markdown report must be wrapped in a full HTML document',
  );
  assert.ok(
    /<h1[^>]*>Big Title<\/h1>/.test(srcdoc),
    'markdown heading must be converted to <h1>, got: ' + srcdoc.slice(0, 200),
  );
  assert.ok(
    /<strong>world<\/strong>/.test(srcdoc),
    'markdown bold must be converted to <strong>',
  );
  win.close();
  console.log('  ok - .md report opens as converted HTML only at task_done');
}

function testHtmlReportOpensVerbatim() {
  const {win, posted} = makePage({withMarked: true});
  const htmlDoc = '<!DOCTYPE html><html><body><h2>Perf</h2></body></html>';
  writeReport(win, 'reports/perf.html', htmlDoc);
  assertNoReportTab(win, posted, 'html report before task_done');
  finishTask(win);
  // The page's former `isReport` flag means nothing any more.
  answerReport(win, reportRequests(posted)[0], htmlDoc, {isReport: true});
  assertReportTabActive(win, 'html report');
  // The editable preview puts its Save-key bridge after the doctype.
  assert.ok(
    activeSrcdoc(win).endsWith('<html><body><h2>Perf</h2></body></html>'),
    '.html report content must be rendered verbatim in the iframe',
  );
  win.close();
  console.log('  ok - .html report opens verbatim at task_done');
}

function testMarkdownFallbackWithoutMarked() {
  const {win, posted} = makePage();
  writeReport(win, 'reports/plain.md', '# Raw <stuff>');
  finishTask(win);
  answerReport(win, reportRequests(posted)[0], '# Raw <stuff>');
  assertReportTabActive(win, 'no-marked fallback');
  const srcdoc = activeSrcdoc(win);
  assert.ok(
    srcdoc.indexOf('<pre># Raw &lt;stuff&gt;</pre>') >= 0,
    'without marked the markdown must be shown escaped in a <pre>, got: ' +
      srcdoc,
  );
  win.close();
  console.log('  ok - markdown falls back to <pre> when marked is missing');
}

function testMarkedThrowFallsBack() {
  const {win, posted} = makePage({brokenMarked: true});
  writeReport(win, 'reports/crash.md', 'boom & bust');
  finishTask(win);
  answerReport(win, reportRequests(posted)[0], 'boom & bust');
  assertReportTabActive(win, 'broken-marked fallback');
  assert.ok(
    activeSrcdoc(win).indexOf('<pre>boom &amp; bust</pre>') >= 0,
    'a throwing marked.parse must fall back to the escaped <pre> body',
  );
  win.close();
  console.log('  ok - markdown falls back to <pre> when marked throws');
}

function testLaterToolWorkKeepsReportForTaskDone() {
  // Once the Write succeeded, the report is confirmed; the agent keeps
  // working (more tool calls) and the report must still open when the
  // task eventually finishes.
  const {win, posted} = makePage({withMarked: true});
  writeReport(win, 'reports/kept.md', '# kept');
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/other.txt'});
  send(win, {
    type: 'tool_result',
    content: 'file body',
    is_error: false,
    tool_name: 'Read',
    path: '/tmp/other.txt',
  });
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls'});
  send(win, {
    type: 'tool_result',
    content: 'ok',
    is_error: false,
    tool_name: 'Bash',
  });
  assertNoReportTab(win, posted, 'report while later tools run');
  finishTask(win);
  answerReport(win, reportRequests(posted)[0], '# kept');
  assertReportTabActive(win, 'report after later tool work');
  assert.ok(
    /<h1[^>]*>kept<\/h1>/.test(activeSrcdoc(win)),
    'the confirmed report must survive later tool calls',
  );
  win.close();
  console.log('  ok - later tool calls do not drop a confirmed report');
}

function testTaskDoneWithoutReportOpensNothing() {
  const {win, posted} = makePage({withMarked: true});
  finishTask(win);
  assertNoReportTab(win, posted, 'task_done without any report');
  win.close();
  console.log('  ok - task_done without a report opens nothing');
}

function testTaskDoneOpensReportOnlyOnce() {
  const {win, posted} = makePage({withMarked: true});
  writeReport(win, 'reports/once.md', '# once');
  finishTask(win);
  const req = reportRequests(posted)[0];
  answerReport(win, req, '# once');
  assertReportTabActive(win, 'first task_done');
  // A second terminal event (e.g. replayed or duplicated) must not
  // re-open or duplicate the already-opened report.
  const before = contentTabs(win).length;
  finishTask(win);
  assert.strictEqual(
    reportRequests(posted).length,
    1,
    'a duplicate task_done must not ask for the report again',
  );
  // Nor does a reply that answers no pending request: the first
  // reply's token is spent, and report:7 was never issued.
  for (const token of [req.token, 'report:7']) {
    const stray = {token, tabId: req.tabId, path: 'reports/stray.md'};
    answerReport(win, stray, '# stray');
  }
  assert.strictEqual(
    contentTabs(win).length,
    before,
    'a duplicate task_done must not open the report again',
  );
  win.close();
  console.log('  ok - a report opens only once per task completion');
}

function testFailedWriteDoesNotOpen() {
  const {win, posted} = makePage({withMarked: true});
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/broken.md',
    content: '# nope',
  });
  send(win, {
    type: 'tool_result',
    content: 'disk full',
    is_error: true,
    tool_name: 'Write',
    path: 'reports/broken.md',
  });
  // A later Write-success-shaped tool_result must not resurrect the
  // failed report (the stash must have been cleared).
  send(win, {
    type: 'tool_result',
    content: writeSuccess('reports/broken.md', '# nope'),
    is_error: false,
    tool_name: 'Write',
    path: 'reports/broken.md',
  });
  finishTask(win);
  assertNoReportTab(win, posted, 'failed Write followed by stray success');
  win.close();
  console.log('  ok - failed Write does not open a report tab');
}

function testWriteFailureStringsDoNotOpen() {
  // UsefulTools.Write returns "Error: ..." strings and DockerTools.Write
  // returns "[exit code: N] ..." strings without raising, so the
  // tool_result arrives with is_error:false even though the Write failed.
  // Only the canonical "Successfully wrote " result may open the report.
  const {win, posted} = makePage({withMarked: true});
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/denied.md',
    content: '# nope',
  });
  send(win, {
    type: 'tool_result',
    content: "Error: [Errno 13] Permission denied: 'reports/denied.md'",
    is_error: false,
    tool_name: 'Write',
    path: 'reports/denied.md',
  });
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/docker.md',
    content: '# nope',
  });
  send(win, {
    type: 'tool_result',
    content: '[exit code: 1]\nsh: cannot create reports/docker.md',
    is_error: false,
    tool_name: 'Write',
    path: 'reports/docker.md',
  });
  finishTask(win);
  assertNoReportTab(win, posted, 'Write failure strings with is_error:false');
  win.close();
  console.log('  ok - Write failure strings with is_error:false do not open');
}

function testMismatchedResultDoesNotOpen() {
  const {win, posted} = makePage({withMarked: true});
  // Result from a different tool, even with a Write-success-shaped body.
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/a.md',
    content: '# a',
  });
  send(win, {
    type: 'tool_result',
    content: writeSuccess('reports/a.md', '# a'),
    is_error: false,
    tool_name: 'Bash',
  });
  // Write result for a different file than the stashed report.
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/b.md',
    content: '# b',
  });
  send(win, {
    type: 'tool_result',
    content: writeSuccess('other/b.md', '# b'),
    is_error: false,
    tool_name: 'Write',
    path: 'other/b.md',
  });
  finishTask(win);
  assertNoReportTab(win, posted, 'mismatched tool_name/path results');
  win.close();
  console.log('  ok - mismatched tool_name/path results do not open');
}

function testPathlessWriteSuccessOpens() {
  // json_printer's multi-block message route can omit tool_input (and
  // therefore path) on a tool_result; the path check applies only when
  // the field is present, so a path-less canonical Write success must
  // still open the stashed report at task_done.
  const {win, posted} = makePage({withMarked: true});
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/pathless.md',
    content: '# no path on result',
  });
  send(win, {
    type: 'tool_result',
    content: writeSuccess('reports/pathless.md', '# no path on result'),
    is_error: false,
    tool_name: 'Write',
  });
  assertNoReportTab(win, posted, 'path-less Write success before task_done');
  finishTask(win);
  answerReport(win, reportRequests(posted)[0], '# no path on result');
  assertReportTabActive(win, 'path-less Write success');
  win.close();
  console.log('  ok - a path-less canonical Write success still opens');
}

// task_error / task_interrupted / task_stopped opening the report is
// pinned by subagentReportNoTabSwitch.testTerminalErrorEventsStillSwitchToTheReport.

function testCloseTabDiscardsQueuedReport() {
  // Closing a chat tab means the user no longer cares about that
  // task's pending report; it must not pop open later and must not
  // leak in the queue.
  const {win, posted} = makePage({withMarked: true});
  const firstTabId = win._testApi.getActiveTabId();
  // The daemon's task-text echo makes the first chat a used one ("+" on
  // a chat that is still new only focuses the composer).
  send(win, {type: 'setTaskText', text: 'report task', tabId: firstTabId});
  // The first tab confirms a report while the user works in another
  // chat (opened without leaving: "+" would retire the idle first one)...
  win._testApi.createNewTab();
  assert.notStrictEqual(win._testApi.getActiveTabId(), firstTabId);
  writeReport(win, 'reports/closed.md', '# never shown', {tabId: firstTabId});
  // ...then the user comes back to that chat and leaves it with "+",
  // which closes it.
  win._testApi.switchToTab(firstTabId);
  const addBtn = win.document.querySelector('#new-chat-btn');
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !win._testApi.openTabs().some(t => t.id === firstTabId),
    'the first chat is closed',
  );
  finishTask(win, {tabId: firstTabId});
  assertNoReportTab(win, posted, 'report of a closed tab');
  win.close();
  console.log('  ok - closing a chat tab discards its queued report');
}

function testClearDiscardsQueuedReport() {
  // `clear` marks the start of a NEW task in the tab; a report queued
  // by a previous task that never reached a terminal event is stale
  // and must not open at the new task's completion.
  const {win, posted} = makePage({withMarked: true});
  writeReport(win, 'reports/stale.md', '# stale');
  send(win, {type: 'clear'});
  finishTask(win);
  assertNoReportTab(win, posted, 'stale report across a clear');
  win.close();
  console.log('  ok - a clear (new task) discards the queued report');
}

function testTraversalPathsDoNotOpen() {
  const {win, posted} = makePage({withMarked: true});
  writeReport(win, 'reports/../escape.md', '# escaped');
  writeReport(win, 'reports/sub/../../escape2.md', '# escaped again');
  finishTask(win);
  assertNoReportTab(win, posted, 'lexical ".." traversal out of reports/');
  // Harmless "." segments must still be recognized as reports paths.
  writeReport(win, './reports/./ok.md', '# fine');
  finishTask(win);
  const req = reportRequests(posted)[0];
  assert.strictEqual(
    req.path,
    './reports/./ok.md',
    'the daemon is asked for the path exactly as the Write named it',
  );
  answerReport(win, req, '# fine', {path: '/ws/reports/ok.md'});
  assertReportTabActive(win, 'dot-segment report');
  win.close();
  console.log('  ok - reports/../ paths do not open; ./ segments still do');
}

function testNonReportPathsDoNotOpen() {
  const {win, posted} = makePage({withMarked: true});
  writeReport(win, 'notes/summary.md', '# not a report');
  writeReport(win, 'reports/data.txt', 'plain text');
  writeReport(win, 'reports.md', '# file named reports.md');
  writeReport(win, 'myreports/x.html', '<p>nope</p>');
  writeReport(win, 'reports', 'extensionless file named reports');
  finishTask(win);
  assertNoReportTab(win, posted, 'non-report paths');
  win.close();
  console.log('  ok - md/html outside a reports folder do not open');
}

function testNonWriteToolsDoNotOpen() {
  const {win, posted} = makePage({withMarked: true});
  send(win, {
    type: 'tool_call',
    name: 'Edit',
    path: 'reports/summary.md',
    old_string: 'a',
    new_string: 'b',
  });
  send(win, {
    type: 'tool_result',
    content: 'Successfully replaced 1 occurrence(s)',
    is_error: false,
    tool_name: 'Edit',
    path: 'reports/summary.md',
  });
  send(win, {type: 'tool_call', name: 'Read', path: 'reports/summary.html'});
  send(win, {
    type: 'tool_result',
    content: '<p>hi</p>',
    is_error: false,
    tool_name: 'Read',
    path: 'reports/summary.html',
  });
  finishTask(win);
  assertNoReportTab(win, posted, 'non-Write tools');
  win.close();
  console.log('  ok - non-Write tools touching reports/ do not open');
}

function testInterveningToolCallClearsStash() {
  const {win, posted} = makePage({withMarked: true});
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/summary.md',
    content: '# pending',
  });
  // No tool_result for the Write; another tool call starts instead.
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/other.txt'});
  send(win, {
    type: 'tool_result',
    content: 'file body',
    is_error: false,
    tool_name: 'Read',
    path: '/tmp/other.txt',
  });
  finishTask(win);
  assertNoReportTab(win, posted, 'intervening tool_call');
  win.close();
  console.log('  ok - an intervening tool_call clears the pending report');
}

function testReplayDoesNotOpen() {
  const {win, posted} = makePage({withMarked: true});
  send(win, {
    type: 'task_events',
    task: 'old task',
    events: [
      {
        type: 'tool_call',
        name: 'Write',
        path: 'reports/old.md',
        content: '# old report',
      },
      {
        type: 'tool_result',
        content: writeSuccess('reports/old.md', '# old report'),
        is_error: false,
        tool_name: 'Write',
        path: 'reports/old.md',
      },
      {type: 'result', summary: 'done', success: true},
      {type: 'task_done', success: true},
    ],
  });
  assertNoReportTab(win, posted, 'task_events replay');
  win.close();
  console.log('  ok - replayed history does not auto-open report tabs');
}

// The replay-only report suppression must not survive into the live
// stream the replay's tail state becomes: a report the task writes
// AFTER its transcript was replayed opens like any live report, while
// a report inside the replayed transcript still opens nothing.
function testReplayDoesNotSuppressLiveReports() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  const t0 = Date.now() - 30000;
  const oldDoc = '<!DOCTYPE html><html><body>old</body></html>';
  send(win, {type: 'status', running: true, tabId, startTs: t0});
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-report',
    task: 'reporting task',
    task_id: '92',
    events: [
      {type: 'prompt', text: 'write reports', ts: t0},
      {
        type: 'tool_call',
        name: 'Write',
        path: 'reports/old.html',
        content: oldDoc,
        ts: t0 + 1000,
      },
      {
        type: 'tool_result',
        content: writeSuccess('reports/old.html', oldDoc),
        is_error: false,
        tool_name: 'Write',
        path: 'reports/old.html',
        ts: t0 + 2000,
      },
    ],
  });
  assertNoReportTab(win, posted, 'a replayed report inside a running task');
  const liveDoc = '<html><body>live</body></html>';
  writeReport(win, 'reports/live.html', liveDoc, {tabId});
  finishTask(win, {tabId});
  assert.deepStrictEqual(
    reportRequests(posted).map(r => r.path),
    ['reports/live.html'],
    'only the report written live is asked for',
  );
  answerReport(win, reportRequests(posted)[0], liveDoc);
  assertReportTabActive(win, 'the live report written after the replay');
  win.close();
  console.log('  ok - a replay does not suppress the live reports after it');
}

function testRegenerationReusesSameTab() {
  const {win, posted} = makePage({withMarked: true});
  const chatTab = win._testApi.getActiveTabId();
  // Two writes of one report within a task open it once, with the last
  // content (the pending reports are deduplicated by path).
  writeReport(win, 'reports/summary.md', '# v1', {tabId: chatTab});
  writeReport(win, 'reports/summary.md', '# v1 again', {tabId: chatTab});
  finishTask(win, {tabId: chatTab});
  assert.strictEqual(
    reportRequests(posted).length,
    1,
    'writing the same report twice must ask for it once',
  );
  answerReport(win, reportRequests(posted)[0], '# v1 again');
  let tabsFound = contentTabs(win);
  assert.strictEqual(
    tabsFound.length,
    1,
    'writing the same report twice must open it once',
  );
  const reportTabId = tabsFound[0].dataset.tabId;
  assert.ok(tabsFound[0].classList.contains('active'), 'the report tab opened');
  assert.ok(
    /<h1[^>]*>v1 again<\/h1>/.test(activeSrcdoc(win)),
    'the tab shows the last content written',
  );

  // A later task regenerates the report while its tab is still open: the
  // existing tab is brought forward with the new content, not doubled.
  clickTab(win, chatTab);
  writeReport(win, 'reports/summary.md', '# v2', {tabId: chatTab});
  finishTask(win, {tabId: chatTab});
  answerReport(win, reportRequests(posted)[1], '# v2');
  tabsFound = contentTabs(win);
  assert.strictEqual(
    tabsFound.length,
    1,
    'regenerating the same report must reuse its tab, not open a second',
  );
  assert.strictEqual(
    tabsFound[0].dataset.tabId,
    reportTabId,
    'the same tab, not a new one',
  );
  assert.ok(
    tabsFound[0].classList.contains('active'),
    'the reused report tab must be re-activated',
  );
  assert.ok(
    /<h1[^>]*>v2<\/h1>/.test(activeSrcdoc(win)),
    'the reused tab must show the regenerated content',
  );
  win.close();
  console.log('  ok - regenerated report reuses its existing tab');
}

function testReportsSegmentVariants() {
  const {win, posted} = makePage({withMarked: true});
  writeReport(win, 'out/reports/2026/deep.html', '<!DOCTYPE html><p>deep</p>');
  writeReport(win, 'Reports\\win.md', '# windows style');
  finishTask(win);
  const reqs = reportRequests(posted);
  assert.deepStrictEqual(
    reqs.map(r => r.path),
    ['out/reports/2026/deep.html', 'Reports\\win.md'],
    'the reports must be asked for in write order',
  );
  answerReport(win, reqs[0], '<!DOCTYPE html><p>deep</p>');
  answerReport(win, reqs[1], '# windows style');
  const tabsFound = contentTabs(win);
  assert.strictEqual(
    tabsFound.length,
    2,
    'nested reports/ dirs, backslash paths and case-insensitive ' +
      '"Reports" must all be recognized; got ' +
      tabsFound.length +
      ' tabs',
  );
  assert.ok(
    tabsFound[1].textContent.indexOf('win.md') >= 0 &&
      tabsFound[1].classList.contains('active'),
    'reports must open in write order with the last one active',
  );
  win.close();
  console.log('  ok - nested, backslash and case-variant reports paths open');
}

function testBackgroundTabReportOpensAtItsTaskDone() {
  const {win, posted} = makePage({withMarked: true});
  const firstTabId = win._testApi.getActiveTabId();
  // The first tab's task is running, so "+" leaves it open (an idle
  // chat left behind is retired).
  send(win, {type: 'status', running: true, tabId: firstTabId});
  const addBtn = win.document.querySelector('#new-chat-btn');
  assert.ok(addBtn, 'expected the add-tab button');
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const secondTabId = win._testApi.getActiveTabId();
  assert.notStrictEqual(
    secondTabId,
    firstTabId,
    'precondition: a second chat tab must now be active',
  );
  assert.ok(
    win._testApi.openTabs().some(t => t.id === firstTabId),
    'precondition: the running first chat stays open',
  );
  // The agent running in the (now background) first tab writes a report.
  writeReport(win, 'reports/bg.md', '# from background', {tabId: firstTabId});
  assertNoReportTab(win, posted, 'background report before its task_done');
  // A task finishing in ANOTHER tab must not open this tab's report.
  finishTask(win, {tabId: secondTabId});
  assertNoReportTab(win, posted, 'task_done of an unrelated tab');
  finishTask(win, {tabId: firstTabId});
  const req = reportRequests(posted)[0];
  assert.strictEqual(req.tabId, firstTabId, "asked for as the task's chat");
  answerReport(win, req, '# from background');
  const tab = assertReportTabActive(win, 'background-tab report');
  assert.ok(
    tab.textContent.indexOf('bg.md') >= 0,
    'the background-generated report tab must be titled after the file',
  );
  win.close();
  console.log('  ok - background-tab report opens at its own task_done');
}

// A sub-agent's reports are part of its top-level task.  When the
// sub-agent finishes (its own end, or `subagentDone` alone) they join
// that task's reports, each as it was written: its path, the folder it
// resolves in, when it was written.  Nothing is asked while the task
// runs; when it ends they are asked for under ITS tab, whose worktree
// the sub-agents wrote in, and they wait for that tab's merge.  The
// daemon may address a sub-agent's tab by another id (the replay of the
// parent's history row does): the reports go with the tab.
function testSubagentReportsJoinTheirTopTask() {
  const page = makePage({withMarked: true});
  sendWorkDir(page.win, '/ws');
  const {win, posted, parentId, subTabIds} = bootParallelRun(1, page);
  const subId = subTabIds[0];
  // The sub-agent runs a sub-agent of its own, in another folder.
  const deepId = spawnSubagent(page, subId, 'deep-task-1', 'deep 1', 0);
  send(win, {
    type: 'task_settings',
    tabId: deepId,
    settings: {work_dir: '/deep'},
  });

  writeReport(win, 'reports/own.md', '# own', {tabId: parentId, ts: 10});
  writeReport(win, 'reports/both.md', '# first', {tabId: parentId, ts: 11});
  writeReport(win, 'reports/sub.md', '# sub', {tabId: subId, ts: 20});
  // The task's file again, and a file of that name in another folder.
  writeReport(win, 'reports/both.md', '# second', {tabId: subId, ts: 21});
  writeReport(win, 'reports/both.md', '# deep', {tabId: deepId, ts: 30});

  // Same sub-agent, another tab id (one that differs from the open
  // tab's, or the page has nothing to move).
  const replayId = 'replay-' + subId;
  send(win, {
    type: 'openSubagentTab',
    tab_id: replayId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
  });

  // The nested sub-agent ends by its own event (the copy its tab gets
  // of the end the daemon sent the sub-agent's own tab), then its tab
  // closes; the first one's tab is closed by `subagentDone` alone.
  finishTask(win, {tabId: deepId, subagent: true, viewer: true});
  send(win, {type: 'subagentDone', tab_id: deepId});
  send(win, {type: 'subagentDone', tab_id: replayId});
  assertNoReportTab(win, posted, 'sub-agents done, their task running');
  assert.strictEqual(win._testApi.getActiveTabId(), parentId, 'nor a switch');

  finishTask(win, {tabId: parentId});
  const asked = reportRequests(posted);
  assert.deepStrictEqual(
    asked.map(r => [r.path, r.workDir, r.since, r.tabId]),
    [
      ['reports/own.md', '/ws', 10, parentId],
      ['reports/both.md', '/deep', 30, parentId],
      ['reports/sub.md', '/ws', 20, parentId],
      ['reports/both.md', '/ws', 21, parentId],
    ],
    "the task's reports and its sub-agents', asked for as the task's: " +
      'a later write of the same file stands for the earlier one',
  );
  // The task's worktree is not merged: all of them wait for it.
  asked.forEach(req => answerNotMerged(win, req));
  assert.strictEqual(storedReports(win)[parentId].length, 4, 'all pending');
  worktreeResult(win, parentId, 'merge');
  const again = reportRequests(posted).slice(4);
  assert.deepStrictEqual(
    again.map(r => r.path + '@' + r.workDir),
    asked.map(r => r.path + '@' + r.workDir),
    "the merge of the task's worktree asks for all of them again",
  );
  answerReport(win, again[2], '# sub', {path: '/ws/reports/sub.md'});
  const tab = assertReportTabActive(win, "a sub-agent's merged report");
  assert.ok(tab.textContent.indexOf('sub.md') >= 0, 'titled after the file');
  win.close();
  console.log("  ok - a sub-agent's reports open with its task's, after the merge");
}

// A sub-agent that ends after its task (a child the task did not wait
// for) hands its report to a task that is over: it is asked for at
// once.  A sub-agent whose task's tab is gone has nobody to hand it to,
// and an end the daemon flags `subagent` in a tab that is no sub-agent
// tab here (a dispatched child's chat, opened as a chat) names no task
// to hand it to either: those reports are dropped.
function testSubagentReportWithoutARunningTask() {
  const late = bootParallelRun(1, makePage({withMarked: true}));
  const lateSub = late.subTabIds[0];
  writeReport(late.win, 'reports/late.md', '# late', {tabId: lateSub});
  finishTask(late.win, {tabId: late.parentId});
  assertNoReportTab(late.win, late.posted, 'the task wrote no report');
  finishTask(late.win, {tabId: lateSub, subagent: true, viewer: true});
  const asked = reportRequests(late.posted);
  assert.deepStrictEqual(
    asked.map(r => [r.path, r.tabId]),
    [['reports/late.md', late.parentId]],
    "asked for when the sub-agent ends, as its task's",
  );
  answerReport(late.win, asked[0], '# late');
  assertReportTabActive(late.win, 'the report of a late sub-agent');
  late.win.close();

  const orphan = bootParallelRun(1, makePage({withMarked: true}));
  const {win, posted, parentId} = orphan;
  const subId = orphan.subTabIds[0];
  writeReport(win, 'reports/orphan.md', '# orphan', {tabId: subId});
  // The daemon refuses the task's tab (its registry is full) while
  // another chat is open: the tab goes, its running sub-agent's stays.
  win._testApi.createNewTab();
  const otherId = win._testApi.getActiveTabId();
  send(win, {type: 'openTabRejected', tabId: parentId});
  assert.deepStrictEqual(
    plain(win._testApi.openTabs().map(t => t.id)),
    [subId, otherId],
    'precondition: the sub-agent tab outlives the tab of its task',
  );
  send(win, {type: 'subagentDone', tab_id: subId});
  writeReport(win, 'reports/child.md', '# child', {tabId: otherId});
  finishTask(win, {tabId: otherId, subagent: true});
  finishTask(win, {tabId: otherId});
  assertNoReportTab(win, posted, 'reports nobody is left to show');
  assert.deepStrictEqual(storedReports(win), {}, 'none is kept either');
  win.close();
  console.log("  ok - a late sub-agent's report is asked at once, an orphan's dropped");
}

const WORKTREE = '/repo/.kiss-worktrees/kiss_wt-7/';

// The user leaves chat *tabId* with "+", which closes an idle chat (a
// chat's tab has no close button); another chat stays open.
function closeChat(win, tabId) {
  win._testApi.createNewTab();
  win._testApi.switchToTab(tabId);
  const addBtn = win.document.querySelector('#new-chat-btn');
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !win._testApi.openTabs().some(t => t.id === tabId),
    'precondition: the chat is closed',
  );
}

// Play the daemon for report request *req* while the worktree the
// report was written in is not merged: no file.
function answerNotMerged(win, req) {
  answerReport(win, req, null, {error: 'File not found: ' + req.path});
}

// A report is shown after the merge, never before.  Until the worktree
// a task wrote it in is merged the daemon answers with no file (`lost`
// below), and the daemon of a remote machine that predates that rule
// with the worktree's copy, inside `<repo>/.kiss-worktrees/kiss_wt-*`
// (`first`).  Either way the report stays pending and is asked for
// again once the tab's worktree is merged, when the daemon answers
// with the repository's copy.
function testWorktreeReportOpensAfterTheMerge() {
  const {win, posted} = makePage({withMarked: true});
  const tabId = win._testApi.getActiveTabId();
  writeReport(win, 'reports/wt.md', '# wt', {tabId});
  writeReport(win, 'reports/lost.md', '# lost', {tabId});
  // The agent's own commit_and_merge, mid-task: nothing is pending
  // yet, the reports are asked for when the task ends.
  worktreeResult(win, tabId, 'merge');
  assertNoReportTab(win, posted, 'a merge before the task ends');
  finishTask(win, {tabId});
  const [first, lost] = reportRequests(posted);
  answerReport(win, first, '# wt', {path: WORKTREE + 'reports/wt.md'});
  answerNotMerged(win, lost);
  assert.strictEqual(
    win.document.querySelector('.kiss-notification'),
    null,
    'a report that is not merged yet must not raise a toast',
  );
  // "Do nothing", a failed merge and another chat's merge leave this
  // worktree unmerged: the reports stay pending.
  worktreeResult(win, tabId, 'nothing', {kept: true});
  worktreeResult(win, tabId, 'merge', {success: false, retryable: true});
  worktreeResult(win, 'another-chat', 'merge');
  assertNoReportTab(win, posted, 'a report still in its worktree', 2);
  worktreeResult(win, tabId, 'merge');
  assertNoReportTab(win, posted, 'merged reports not read yet', 4);
  const [again, lostAgain] = reportRequests(posted).slice(2);
  assert.deepStrictEqual(
    [again.path, again.tabId, again.report, again.token === first.token],
    ['reports/wt.md', tabId, true, false],
    'the merge asks for the pending report again, under a new token',
  );
  assert.ok(
    !('since' in again),
    'an event without a time stamp tells nothing of when it was written',
  );
  answerReport(win, again, '# wt', {path: '/repo/reports/wt.md'});
  assertReportTabActive(win, 'the merged report');
  assert.strictEqual(
    registryContentTabs(win)[0].contentPath,
    '/repo/reports/wt.md',
    "the tab is the repository's file, as the daemon resolved it",
  );
  assert.deepStrictEqual(
    storedReports(win)[tabId].map(rep => rep.path),
    ['reports/lost.md'],
    'the report not shown yet is the one still pending',
  );
  answerReport(win, lostAgain, '# lost', {path: '/repo/reports/lost.md'});
  assert.deepStrictEqual(
    plain(registryContentTabs(win).map(t => t.contentPath)),
    ['/repo/reports/wt.md', '/repo/reports/lost.md'],
    'the report the daemon had no file for before the merge opens after it',
  );
  assert.deepStrictEqual(storedReports(win), {}, 'both are shown');
  worktreeResult(win, tabId, 'merge');
  assert.strictEqual(
    reportRequests(posted).length,
    4,
    'nothing is pending any more: a later merge asks for nothing',
  );
  win.close();
  console.log('  ok - a report in a worktree opens after the merge only');
}

// Only the reply to a report's latest request counts.  A reply that
// was on its way when the worktree was merged tells where the report
// was before the merge (the checkout's older copy, say), and one on
// its way when the worktree was discarded names a report that is gone.
function testMergeAndDiscardSettleRequestsInFlight() {
  const merged = makePage({withMarked: true});
  let tabId = merged.win._testApi.getActiveTabId();
  writeReport(merged.win, 'reports/wt.md', '# new', {tabId});
  finishTask(merged.win, {tabId});
  // The merge is reported before the daemon's first reply arrives.
  worktreeResult(merged.win, tabId, 'merge');
  const [early, afterMerge] = reportRequests(merged.posted);
  const inRepo = {path: '/repo/reports/wt.md'};
  answerReport(merged.win, early, '# old', inRepo);
  assertNoReportTab(merged.win, merged.posted, 'a reply out of date', 2);
  answerReport(merged.win, afterMerge, '# new', inRepo);
  assertReportTabActive(merged.win, 'the reply to the request the merge made');
  assert.ok(
    /<h1[^>]*>new<\/h1>/.test(activeSrcdoc(merged.win)),
    'the report shown is the merged one',
  );
  merged.win.close();

  const discarded = makePage({withMarked: true});
  tabId = discarded.win._testApi.getActiveTabId();
  writeReport(discarded.win, 'reports/wt.md', '# new', {tabId});
  finishTask(discarded.win, {tabId});
  worktreeResult(discarded.win, tabId, 'discard');
  answerReport(discarded.win, reportRequests(discarded.posted)[0], '# old', inRepo);
  assertNoReportTab(discarded.win, discarded.posted, 'a discarded report', 1);
  assert.deepStrictEqual(storedReports(discarded.win), {}, 'not kept either');
  discarded.win.close();
  console.log('  ok - a reply from before a merge or a discard opens nothing');
}

// The pending reports outlive the page and the connection: a reload, a
// lost reply or a merge reported while the page was away must not lose
// them.  They are asked for again with the first tab snapshot of every
// connection.
function testPendingReportsSurviveReloadAndReconnect() {
  const first = makePage({withMarked: true});
  const tabId = first.win._testApi.getActiveTabId();
  writeReport(first.win, 'reports/wt.md', '# wt', {tabId, ts: 1700000000000});
  finishTask(first.win, {tabId});
  // A tab snapshot asks for nothing that this connection asked for.
  sendTabsState(first.win, [{tabId}]);
  assert.strictEqual(reportRequests(first.posted).length, 1, 'asked once');
  const request = plain(reportRequests(first.posted)[0]);
  answerNotMerged(first.win, request);
  assert.deepStrictEqual(
    storedReports(first.win),
    {
      [tabId]: [
        {path: 'reports/wt.md', workDir: request.workDir, since: request.since},
      ],
    },
    'a pending report is kept in localStorage, without its request token',
  );

  // The page is reloaded.  The worktree was merged meanwhile.
  const {win, posted} = makePage({withMarked: true, reloadOf: first});
  first.win.close();
  assertNoReportTab(win, posted, 'a reloaded page, the daemon not reached');
  sendTabsState(win, [{tabId}]);
  sendTabsState(win, [{tabId}]);
  assert.deepStrictEqual(
    plain(reportRequests(posted)),
    [request],
    'the reloaded page asks for the pending report, once per connection',
  );
  // The connection drops before the reply: the next one asks again.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  sendTabsState(win, [{tabId}]);
  const [lostReply, again] = reportRequests(posted);
  assert.strictEqual(again.token, 'report:2', 'asked again, a new request');
  answerNotMerged(win, lostReply);
  answerReport(win, again, '# wt', {path: '/repo/reports/wt.md'});
  assertReportTabActive(win, 'the report merged while the page was away');
  assert.deepStrictEqual(storedReports(win), {}, 'shown: pending no more');
  win.close();

  // A chat that is gone when the page comes back takes its reports
  // along, and an item that is not a pending report's is left alone.
  const gone = storedItemName('gone-chat', '/ws', 'reports/x.md', null);
  const others = [
    'kiss-pending-report:5',
    'kiss-pending-report:{oops',
    'other',
  ];
  [gone].concat(others).forEach(name => {
    const page = makePage({storage: {[name]: '1700000000000'}});
    sendTabsState(page.win, [{tabId: page.win._testApi.getActiveTabId()}]);
    assertNoReportTab(page.win, page.posted, 'stored reports: ' + name);
    assert.strictEqual(
      page.win.localStorage.getItem(name),
      name === gone ? null : '1700000000000',
      'only the report of the chat that is gone is removed: ' + name,
    );
    page.win.close();
  });
  console.log('  ok - pending reports survive a reload and a lost connection');
}

// A page without localStorage (an opaque origin) keeps its pending
// reports for as long as it lives.
function testPendingReportsWithoutStorage() {
  const {win, posted} = makePage({withMarked: true, url: null});
  const tabId = win._testApi.getActiveTabId();
  writeReport(win, 'reports/wt.md', '# wt', {tabId});
  finishTask(win, {tabId});
  answerNotMerged(win, reportRequests(posted)[0]);
  worktreeResult(win, tabId, 'merge');
  answerReport(win, reportRequests(posted)[1], '# wt', {
    path: '/repo/reports/wt.md',
  });
  assertReportTabActive(win, 'a report on a page without storage');
  win.close();
  console.log('  ok - a page without localStorage still shows its reports');
}

// Every window of an origin keeps its pending reports in the same
// localStorage, each report in an item of its own for every window
// that stores it.  A window writes the items of the reports it holds,
// removes those of a report it shows or drops and leaves the rest:
// what another window stored since this one loaded outlives this
// window's tasks, replies and drops, also when both follow the same
// chat and one of them saw fewer of its reports or hears of them
// later.  (Two real windows that store at the same moment:
// test_report_opens_after_merge.py.)
function testWindowsKeepEachOthersStoredReports() {
  const a = makePage({withMarked: true});
  const b = makePage({withMarked: true, shareWith: a});
  const chats = ['a1', 'b1', 'c1'].map(tabId => ({tabId, chatId: tabId}));
  [a, b].forEach(page => {
    sendTabsState(page.win, chats);
    chats.forEach(({tabId}) => {
      send(page.win, {type: 'task_events', tabId, task: 'a task', events: []});
    });
  });
  const stored = () => {
    const reps = storedReports(a.win);
    Object.keys(reps).forEach(key => {
      reps[key] = reps[key].map(rep => rep.path);
    });
    return reps;
  };
  // A task ends in each window, in different chats.
  writeReport(a.win, 'reports/a.md', '# a', {tabId: 'a1', ts: 1});
  finishTask(a.win, {tabId: 'a1'});
  writeReport(b.win, 'reports/b.md', '# b', {tabId: 'b1', ts: 2});
  finishTask(b.win, {tabId: 'b1'});
  assert.deepStrictEqual(
    stored(),
    {a1: ['reports/a.md'], b1: ['reports/b.md']},
    'the second window keeps what the first one stored',
  );
  // Both windows follow chat c1; the second one came to it late and
  // saw the last report only.
  writeReport(a.win, 'reports/c-1.md', '# c1', {tabId: 'c1', ts: 3});
  writeReport(a.win, 'reports/c-2.md', '# c2', {tabId: 'c1', ts: 4});
  writeReport(b.win, 'reports/c-2.md', '# c2', {tabId: 'c1', ts: 4});
  finishTask(a.win, {tabId: 'c1'});
  finishTask(b.win, {tabId: 'c1'});
  assert.deepStrictEqual(stored().c1, ['reports/c-1.md', 'reports/c-2.md']);
  assert.strictEqual(
    reportItemNames(a.win, 'reports/c-2.md').length,
    2,
    'each window stores the report in an item of its own',
  );
  // A report shown in one window is pending no more; the others stay.
  answerReport(b.win, reportRequests(b.posted)[0], '# b');
  answerReport(a.win, reportRequests(a.posted)[2], '# c2');
  assert.deepStrictEqual(
    stored(),
    {a1: ['reports/a.md'], c1: ['reports/c-1.md']},
    'a window showing a report leaves the reports of the other one',
  );
  assert.deepStrictEqual(
    reportItemNames(a.win, 'reports/c-2.md').map(name => name.split(':')[0]),
    ['kiss-settled-report'],
    'the items of both windows are removed: one item says it is shown',
  );
  // The next task of chat c1 writes its last report again and ends.
  // The first window hears all of it before the second one hears of
  // its start: that one drops the report it holds, not the new one.
  send(a.win, {type: 'clear', tabId: 'c1'});
  writeReport(a.win, 'reports/c-2.md', '# c2 again', {tabId: 'c1', ts: 5});
  finishTask(a.win, {tabId: 'c1'});
  send(b.win, {type: 'clear', tabId: 'c1'});
  assert.deepStrictEqual(
    stored(),
    {a1: ['reports/a.md'], c1: ['reports/c-2.md']},
    'a window that hears of a task late leaves the report it has left',
  );
  // The second window hears of that task's report and end as well,
  // and only now of the first chat's: it stores both reports too.
  writeReport(b.win, 'reports/c-2.md', '# c2 again', {tabId: 'c1', ts: 5});
  finishTask(b.win, {tabId: 'c1'});
  writeReport(b.win, 'reports/a.md', '# a', {tabId: 'a1', ts: 1});
  finishTask(b.win, {tabId: 'a1'});
  assert.strictEqual(reportItemNames(a.win, 'reports/a.md').length, 2);
  // The first window is reloaded: it asks for what is still pending,
  // each report once, in the order they became pending (a report in
  // the first window that stored it), whatever order the browser
  // lists the items in (here the first report's first item last).
  const storage = a.win.localStorage;
  const first = Object.keys(storageSnapshot(a.win)).find(name => {
    return name.includes('reports/a.md');
  });
  const firstAt = storage.getItem(first);
  storage.removeItem(first);
  storage.setItem(first, firstAt);
  const reloaded = makePage({withMarked: true, reloadOf: a});
  sendTabsState(reloaded.win, chats);
  assert.deepStrictEqual(
    plain(
      reportRequests(reloaded.posted).map(req => {
        return [req.tabId, req.path, req.since];
      }),
    ),
    [
      ['a1', 'reports/a.md', 1],
      ['c1', 'reports/c-2.md', 5],
    ],
  );
  [a, b, reloaded].forEach(page => page.win.close());
  console.log('  ok - windows keep the reports each other stored');
}

// The item the page of an earlier version kept every pending report
// in, by tab, and two chats to have reports in it.
const OLD_REPORTS = 'kiss-pending-reports';
const OLD_CHATS = ['b1', 'a1'].map(tabId => ({tabId, chatId: tabId}));

// The paths of the reports *page* has asked the daemon for, once the
// daemon's tab snapshot (the two chats) is in.
function askedReports(page) {
  sendTabsState(page.win, OLD_CHATS);
  return plain(reportRequests(page.posted).map(req => req.path));
}

// A page takes the reports of an earlier version from the item that
// version kept them in, in its order and before the ones stored since,
// and a page that loads stores none: a window that read the item a
// while ago has nothing to write that brings back a report another
// window has shown or dropped since.  Such a report gets an item of
// its own that says so, which keeps it from every page that loads
// afterwards, whichever window stores the report later.  That item
// goes a month later, unless the old item still lists the report.
function testEarlierVersionsReportsAreReadInPlace() {
  const reports = {
    b1: [
      {path: 'reports/z.md', workDir: '/ws', since: 9},
      {path: 'reports/a.md', workDir: '/ws', since: 7},
    ],
    a1: [{path: 'reports/m.md', workDir: '/other', since: 8}],
  };
  // This version stored the first of them as well, whose place stays
  // the old item's, and one of its own.
  const own = storedItemName('b1', '/ws', 'reports/own.md', 3);
  const both = storedItemName('b1', '/ws', 'reports/z.md', 9);
  const storage = {
    [OLD_REPORTS]: JSON.stringify(reports),
    [both]: '1700000000002',
    [own]: '1700000000001',
  };
  const a = makePage({withMarked: true, storage});
  const b = makePage({withMarked: true, shareWith: a});
  [a, b].forEach(page => {
    assert.deepStrictEqual(
      askedReports(page),
      ['reports/z.md', 'reports/a.md', 'reports/own.md', 'reports/m.md'],
      "the old item's reports first, in its order, and each asked for once",
    );
  });
  assert.deepStrictEqual(storageSnapshot(a.win), storage, 'nothing written');
  // One window shows a report; the other one hears of the next task of
  // the other chat, which drops that chat's report.
  answerReport(a.win, reportRequests(a.posted)[0], '# z');
  send(b.win, {type: 'clear', tabId: 'a1'});
  const settled = [
    'kiss-settled-report:["b1","/ws","reports/z.md",9]',
    'kiss-settled-report:["a1","/other","reports/m.md",8]',
    'kiss-settled-report:["b1","/ws","reports/own.md",3]',
  ];
  assert.deepStrictEqual(
    Object.keys(storageSnapshot(a.win)).sort(),
    [OLD_REPORTS, own, settled[0], settled[1]].sort(),
    'each of the two has an item that says so',
  );
  assert.strictEqual(
    a.win.localStorage.getItem(OLD_REPORTS),
    storage[OLD_REPORTS],
    'the old item is as it was',
  );
  // Windows that were slow to hear of the task's end store a shown
  // report afterwards: one of this version in an item of its own, a
  // page of the earlier version in the old item, with a report of its
  // own.
  const late = storedItemName('b1', '/ws', 'reports/z.md', 9);
  a.win.localStorage.setItem(late, '1700000000009');
  reports.a1.push({path: 'reports/late.md', workDir: '/other', since: 11});
  a.win.localStorage.setItem(OLD_REPORTS, JSON.stringify(reports));
  const c = makePage({withMarked: true, shareWith: a});
  assert.deepStrictEqual(
    askedReports(c),
    ['reports/a.md', 'reports/own.md', 'reports/late.md'],
    'a page that loads now takes what is pending, and no settled report',
  );
  // So it is for a report the old item did not list when it was shown.
  answerReport(c.win, reportRequests(c.posted)[1], '# own');
  assert.strictEqual(
    a.win.localStorage.getItem(late),
    null,
    'the page that loaded removed what the slow window stored',
  );
  a.win.localStorage.setItem(
    storedItemName('b1', '/ws', 'reports/own.md', 3),
    '1700000000010',
  );
  reports.b1.push({path: 'reports/own.md', workDir: '/ws', since: 3});
  a.win.localStorage.setItem(OLD_REPORTS, JSON.stringify(reports));
  const d = makePage({withMarked: true, shareWith: a});
  assert.deepStrictEqual(askedReports(d), ['reports/a.md', 'reports/late.md']);
  // A month later the item of a report the old item does not list
  // goes, with what a slow window stored for it; the others stay.
  const gone = '["b1","/ws","reports/gone.md",2]';
  const monthAgo = String(Date.now() - 31 * 24 * 60 * 60 * 1000);
  a.win.localStorage.setItem(
    'kiss-pending-report:' + gone + randomUUID(),
    '1700000000011',
  );
  settled
    .concat('kiss-settled-report:' + gone)
    .forEach(name => a.win.localStorage.setItem(name, monthAgo));
  const e = makePage({withMarked: true, shareWith: a});
  assert.deepStrictEqual(askedReports(e), ['reports/a.md', 'reports/late.md']);
  assert.deepStrictEqual(
    Object.keys(storageSnapshot(e.win)).sort(),
    [OLD_REPORTS].concat(settled).sort(),
  );
  [d, e].forEach(page => page.win.close());
  // An old item the earlier version did not write tells of no report
  // (nor did it there) and is left alone.
  ['{oops', '{"a1":[null]}'].forEach(odd => {
    const page = makePage({storage: {[OLD_REPORTS]: odd, [own]: '5'}});
    assert.deepStrictEqual(askedReports(page), ['reports/own.md'], odd);
    assert.strictEqual(page.win.localStorage.getItem(OLD_REPORTS), odd);
    page.win.close();
  });
  [a, b, c].forEach(page => page.win.close());
  console.log('  ok - the reports of an earlier version are read in place');
}

// Storage that is full (the browser's quota) costs no report.  A page
// that loads writes nothing, so it has every report of the old item.
// One of them that is shown cannot get its item, which does not fit:
// the old item is written again without it, which does, and the
// report stays shown after a reload.
function testFullStorageKeepsEarlierVersionsReports() {
  const reports = {
    b1: [
      {path: 'reports/z.md', workDir: '/ws', since: 9},
      {path: 'reports/a.md', workDir: '/ws', since: 7},
      {path: 'reports/k.md', workDir: '/ws', since: 6},
    ],
    a1: [{path: 'reports/m.md', workDir: '/other', since: 8}],
  };
  const storage = {[OLD_REPORTS]: JSON.stringify(reports)};
  // jsdom's quota: 5,000,000 characters of names and values.
  const used = OLD_REPORTS.length + storage[OLD_REPORTS].length;
  storage.filler = 'x'.repeat(5000000 - used - 'filler'.length);
  const first = makePage({withMarked: true, storage});
  assert.deepStrictEqual(
    askedReports(first),
    ['reports/z.md', 'reports/a.md', 'reports/k.md', 'reports/m.md'],
    'the page that loads has every report of the old item',
  );
  assert.deepStrictEqual(storageSnapshot(first.win), storage, 'none is lost');
  assert.throws(
    () => first.win.localStorage.setItem('one-more', ''),
    {name: 'QuotaExceededError'},
    'precondition: no further item fits',
  );
  answerReport(first.win, reportRequests(first.posted)[0], '# z');
  assertReportTabActive(first.win, 'a report of the old item, storage full');
  reports.b1.shift();
  assert.strictEqual(
    first.win.localStorage.getItem(OLD_REPORTS),
    JSON.stringify(reports),
    'the old item lists the shown report no more',
  );
  const reloaded = makePage({withMarked: true, reloadOf: first});
  assert.deepStrictEqual(
    askedReports(reloaded),
    ['reports/a.md', 'reports/k.md', 'reports/m.md'],
    'the reloaded page asks for the reports that are not shown yet',
  );
  [first, reloaded].forEach(page => page.win.close());
  console.log('  ok - full storage loses no report of an earlier version');
}

// A report that is shown or dropped while storage is full stays so.
// The item that would say so does not fit, and nothing is removed to
// make room for it, which another window could take first: an item
// the report is stored in says so instead.  A window that was slow to
// hear of the task's end stores the report afterwards, in an item of
// its own; a page that loads then asks for neither report and removes
// those items.  What says so goes a month later.  (A window that
// stores a report between each two steps of another one:
// test_report_opens_after_merge.py.)
function testFullStorageKeepsSettledReportsSettled() {
  const a = makePage({withMarked: true});
  const b = makePage({withMarked: true, shareWith: a});
  const paths = ['reports/shown.md', 'reports/dropped.md'];
  [a, b].forEach(page => {
    sendTabsState(page.win, OLD_CHATS);
    send(page.win, {type: 'task_events', tabId: 'a1', task: 't', events: []});
    paths.forEach((path, ts) => {
      writeReport(page.win, path, '# report', {tabId: 'a1', ts: ts + 1});
    });
  });
  finishTask(a.win, {tabId: 'a1'});
  const storage = a.win.localStorage;
  const itemNames = () => Object.keys(storageSnapshot(a.win)).sort();
  // jsdom's quota: 5,000,000 characters of names and values.
  const used = Object.entries(storageSnapshot(a.win)).reduce(
    (sum, [name, value]) => sum + name.length + value.length,
    0,
  );
  storage.setItem('filler', 'x'.repeat(5000000 - used - 'filler'.length));
  assert.throws(
    () => storage.setItem('one-more', ''),
    {name: 'QuotaExceededError'},
    'precondition: no further item fits',
  );
  const names = itemNames();
  // The first report is shown; the chat's next task drops the other.
  answerReport(a.win, reportRequests(a.posted)[0], '# shown');
  assertReportTabActive(a.win, 'a report shown while storage is full');
  send(a.win, {type: 'clear', tabId: 'a1'});
  assert.deepStrictEqual(itemNames(), names, 'no item is removed or added');
  const settled = paths.map(path => reportItemNames(a.win, path)[0]);
  settled.forEach(item => {
    assert.match(storage.getItem(item), /^-\d{10}$/, 'it says when: ' + item);
  });
  // There is room again, and the second window hears of the task's
  // end: it stores both reports.
  storage.removeItem('filler');
  finishTask(b.win, {tabId: 'a1'});
  paths.forEach(path => {
    assert.strictEqual(reportItemNames(a.win, path).length, 2, path);
  });
  const c = makePage({withMarked: true, shareWith: a});
  assert.deepStrictEqual(askedReports(c), [], 'neither is pending again');
  assert.deepStrictEqual(
    itemNames(),
    names.filter(name => name !== 'filler'),
    'what the slow window stored is removed',
  );
  // A month later.
  const monthAgo = Math.floor(Date.now() / 1000) - 31 * 24 * 60 * 60;
  settled.forEach(item => storage.setItem(item, '-' + monthAgo));
  const d = makePage({withMarked: true, shareWith: a});
  assert.deepStrictEqual(askedReports(d), []);
  paths.forEach(path => {
    assert.deepStrictEqual(reportItemNames(a.win, path), [], path);
  });
  [a, b, c, d].forEach(page => page.win.close());
  console.log('  ok - a report settled while storage is full stays settled');
}

// A pending report goes with its worktree (a discard), with the tab's
// next task (`clear`) and with the tab itself (closed): a merge
// reported afterwards asks for nothing.
function testHeldReportIsDropped() {
  ['discard', 'clear', 'close'].forEach(how => {
    const {win, posted} = makePage({withMarked: true});
    const tabId = win._testApi.getActiveTabId();
    writeReport(win, 'reports/wt.md', '# wt', {tabId});
    finishTask(win, {tabId});
    answerNotMerged(win, reportRequests(posted)[0]);
    if (how === 'discard') worktreeResult(win, tabId, 'discard');
    if (how === 'clear') send(win, {type: 'clear', tabId});
    if (how === 'close') closeChat(win, tabId);
    worktreeResult(win, tabId, 'merge');
    assertNoReportTab(win, posted, 'a pending report after a ' + how, 1);
    assert.deepStrictEqual(storedReports(win), {}, how + ': not kept either');
    win.close();
  });
  console.log('  ok - a discard, a new task or a closed tab drops pending ones');
}

// The daemon's reply opens nothing when the tab's next task started
// while the daemon was reading (the report read or not, in a worktree
// or not: the merge of the new task must not ask for the old reports
// again), and when the chat whose task wrote it was closed meanwhile.
function testUnreadableAndOrphanedRepliesOpenNothing() {
  const {win, posted} = makePage({withMarked: true});
  const tabId = win._testApi.getActiveTabId();
  writeReport(win, 'reports/gone.md', '# gone', {tabId});
  writeReport(win, 'reports/stale.md', '# stale', {tabId});
  writeReport(win, 'reports/held.md', '# held', {tabId});
  finishTask(win, {tabId});
  const [gone, stale, held] = reportRequests(posted);
  answerReport(win, gone, null, {error: 'File not found: reports/gone.md'});
  assert.strictEqual(
    win.document.querySelector('.kiss-notification'),
    null,
    'a report that cannot be read must not raise a toast',
  );
  send(win, {type: 'clear', tabId});
  answerReport(win, stale, '# stale');
  answerReport(win, held, '# held', {
    path: '/repo/.kiss-worktrees/kiss_wt-1/reports/held.md',
  });
  send(win, {type: 'worktree_result', tabId, action: 'merge', success: true});
  assertNoReportTab(win, posted, 'an unreadable report and a new task', 3);
  writeReport(win, 'reports/late.md', '# late', {tabId});
  finishTask(win, {tabId});
  closeChat(win, tabId);
  answerReport(win, reportRequests(posted)[3], '# late');
  assertNoReportTab(win, posted, 'a closed chat', 4);
  win.close();
  console.log('  ok - an error reply, a new task or a closed chat opens none');
}

function testSwitchBackToChatRestoresOutput() {
  const {win, posted} = makePage({withMarked: true});
  send(win, {type: 'prompt', text: 'make a report'});
  writeReport(win, 'reports/summary.md', '# hi');
  finishTask(win);
  answerReport(win, reportRequests(posted)[0], '# hi');
  assertReportTabActive(win, 'switch-back');
  // The file screen's strip lists the report alone: the chat comes
  // back the way the Chats panel's pick brings it.
  clickTab(win, win._testApi.openTabs().find(t => !t.isContentTab).id);
  const out = win.document.getElementById('output');
  assert.notStrictEqual(
    out.style.display,
    'none',
    'going back to the chat must restore the chat surface',
  );
  const area = win.document.getElementById('content-tab-area');
  assert.strictEqual(
    area.style.display,
    'none',
    'the content area must hide when a chat tab is re-activated',
  );
  win.close();
  console.log('  ok - switching back to the chat tab restores the chat');
}

function main() {
  console.log('reportAutoOpen.test.js');
  testMdReportWaitsForTaskDone();
  testHtmlReportOpensVerbatim();
  testMarkdownFallbackWithoutMarked();
  testMarkedThrowFallsBack();
  testLaterToolWorkKeepsReportForTaskDone();
  testTaskDoneWithoutReportOpensNothing();
  testTaskDoneOpensReportOnlyOnce();
  testFailedWriteDoesNotOpen();
  testWriteFailureStringsDoNotOpen();
  testMismatchedResultDoesNotOpen();
  testPathlessWriteSuccessOpens();
  testCloseTabDiscardsQueuedReport();
  testClearDiscardsQueuedReport();
  testTraversalPathsDoNotOpen();
  testNonReportPathsDoNotOpen();
  testNonWriteToolsDoNotOpen();
  testInterveningToolCallClearsStash();
  testReplayDoesNotOpen();
  testReplayDoesNotSuppressLiveReports();
  testRegenerationReusesSameTab();
  testReportsSegmentVariants();
  testBackgroundTabReportOpensAtItsTaskDone();
  testSubagentReportsJoinTheirTopTask();
  testSubagentReportWithoutARunningTask();
  testWorktreeReportOpensAfterTheMerge();
  testMergeAndDiscardSettleRequestsInFlight();
  testPendingReportsSurviveReloadAndReconnect();
  testPendingReportsWithoutStorage();
  testWindowsKeepEachOthersStoredReports();
  testEarlierVersionsReportsAreReadInPlace();
  testFullStorageKeepsEarlierVersionsReports();
  testFullStorageKeepsSettledReportsSettled();
  testHeldReportIsDropped();
  testUnreadableAndOrphanedRepliesOpenNothing();
  testSwitchBackToChatRestoresOutput();
  console.log('reportAutoOpen.test.js: all tests passed');
}

main();
