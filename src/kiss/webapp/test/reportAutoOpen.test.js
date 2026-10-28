// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests: whenever the agent generates a report (a .md or
// .html file written by the agent into a `reports` folder), the chat
// page must open it as an HTML page in a content tab and switch to
// that tab -- but only AFTER the task finishes (task_done for the owning
// chat tab, or subagentDone for a subagent tab).  While the task is
// still running the report must NOT pop open.  Reports of tasks that
// end in task_error / task_interrupted / task_stopped still open (the
// task finished).  Markdown reports are converted to HTML first.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  readyTabId,
} = require('./simplify2_harness');
// Content tabs on the group strip (#tab-list).  The strip lists them on
// the file screen only: a report opened in the background (sub-agent
// done) is in the tab registry (registryContentTabs) but not here.
const {
  writeSuccess,
  writeReport,
  stripContentTabEls: contentTabs,
  contentTabs: registryContentTabs,
  reportFrames,
  visibleSrcdoc: activeSrcdoc,
  spawnSubagent,
} = require('./subagent_tab_helpers');

function brokenMarked(win) {
  win.marked = {
    parse: function () {
      throw new Error('marked exploded');
    },
  };
}

function makePage(opts) {
  opts = opts || {};
  return makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: opts.brokenMarked ? brokenMarked : undefined,
    extraScripts: opts.withMarked ? ['marked.min.js'] : [],
    sourceUrl: 'report-main.js',
  });
}

// The daemon emits task_done when the task finishes; only then may
// stashed report tabs open.
function finishTask(win, extra) {
  send(win, Object.assign({type: 'task_done', success: true}, extra || {}));
}

function assertNoReportTab(win, label) {
  assert.strictEqual(
    contentTabs(win).length,
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
  return tabsFound[0];
}

function testMdReportWaitsForTaskDone() {
  const {win} = makePage({withMarked: true});
  assert.ok(
    win.document.body.classList.contains('remote-chat'),
    'precondition: the web app marks its body with remote-chat',
  );
  writeReport(win, 'reports/summary.md', '# Big Title\n\nHello **world**');
  // Reproduces the bug: the report used to pop open right on the Write
  // tool_result, stealing focus while the task was still running.
  assertNoReportTab(win, 'md report before task_done');
  finishTask(win);
  const tab = assertReportTabActive(win, 'md report after task_done');
  assert.ok(
    (tab.textContent || '').indexOf('summary.md') >= 0,
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
  const {win} = makePage({withMarked: true});
  const htmlDoc = '<!DOCTYPE html><html><body><h2>Perf</h2></body></html>';
  writeReport(win, 'reports/perf.html', htmlDoc);
  assertNoReportTab(win, 'html report before task_done');
  finishTask(win);
  assertReportTabActive(win, 'html report');
  assert.strictEqual(
    activeSrcdoc(win),
    htmlDoc,
    '.html report content must be rendered verbatim in the iframe',
  );
  win.close();
  console.log('  ok - .html report opens verbatim at task_done');
}

function testMarkdownFallbackWithoutMarked() {
  const {win} = makePage();
  writeReport(win, 'reports/plain.md', '# Raw <stuff>');
  finishTask(win);
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
  const {win} = makePage({brokenMarked: true});
  writeReport(win, 'reports/crash.md', 'boom & bust');
  finishTask(win);
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
  const {win} = makePage({withMarked: true});
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
  assertNoReportTab(win, 'report while later tools run');
  finishTask(win);
  assertReportTabActive(win, 'report after later tool work');
  assert.ok(
    /<h1[^>]*>kept<\/h1>/.test(activeSrcdoc(win)),
    'the confirmed report must survive later tool calls',
  );
  win.close();
  console.log('  ok - later tool calls do not drop a confirmed report');
}

function testTaskDoneWithoutReportOpensNothing() {
  const {win} = makePage({withMarked: true});
  finishTask(win);
  assertNoReportTab(win, 'task_done without any report');
  win.close();
  console.log('  ok - task_done without a report opens nothing');
}

function testTaskDoneOpensReportOnlyOnce() {
  const {win} = makePage({withMarked: true});
  writeReport(win, 'reports/once.md', '# once');
  finishTask(win);
  assertReportTabActive(win, 'first task_done');
  // A second terminal event (e.g. replayed or duplicated) must not
  // re-open or duplicate the already-opened report.
  const before = contentTabs(win).length;
  finishTask(win);
  assert.strictEqual(
    contentTabs(win).length,
    before,
    'a duplicate task_done must not open the report again',
  );
  win.close();
  console.log('  ok - a report opens only once per task completion');
}

function testFailedWriteDoesNotOpen() {
  const {win} = makePage({withMarked: true});
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
  assertNoReportTab(win, 'failed Write followed by stray success');
  win.close();
  console.log('  ok - failed Write does not open a report tab');
}

function testWriteFailureStringsDoNotOpen() {
  // UsefulTools.Write returns "Error: ..." strings and DockerTools.Write
  // returns "[exit code: N] ..." strings without raising, so the
  // tool_result arrives with is_error:false even though the Write failed.
  // Only the canonical "Successfully wrote " result may open the report.
  const {win} = makePage({withMarked: true});
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
  assertNoReportTab(win, 'Write failure strings with is_error:false');
  win.close();
  console.log('  ok - Write failure strings with is_error:false do not open');
}

function testMismatchedResultDoesNotOpen() {
  const {win} = makePage({withMarked: true});
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
  assertNoReportTab(win, 'mismatched tool_name/path results');
  win.close();
  console.log('  ok - mismatched tool_name/path results do not open');
}

function testPathlessWriteSuccessOpens() {
  // json_printer's multi-block message route can omit tool_input (and
  // therefore path) on a tool_result; the path check applies only when
  // the field is present, so a path-less canonical Write success must
  // still open the stashed report at task_done.
  const {win} = makePage({withMarked: true});
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
  assertNoReportTab(win, 'path-less Write success before task_done');
  finishTask(win);
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
  const {win} = makePage({withMarked: true});
  const firstTabId = win._testApi.getActiveTabId();
  // The first tab confirms a report while the user works in another
  // chat (opened without leaving: "+" would retire the idle first one)...
  win._testApi.createNewTab();
  assert.notStrictEqual(win._testApi.getActiveTabId(), firstTabId);
  writeReport(win, 'reports/closed.md', '# never shown', {tabId: firstTabId});
  // ...then the user comes back to that tab and closes it (only the
  // chat on screen has a strip entry with a close button).
  win._testApi.switchToTab(firstTabId);
  const closeBtn = win.document.querySelector(
    '#tab-list .chat-tab[data-tab-id="' + firstTabId + '"] .chat-tab-close',
  );
  assert.ok(closeBtn, 'expected a close button on the first chat tab');
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !win._testApi.openTabs().some(t => t.id === firstTabId),
    'the first chat is closed',
  );
  finishTask(win, {tabId: firstTabId});
  assertNoReportTab(win, 'report of a closed tab');
  win.close();
  console.log('  ok - closing a chat tab discards its queued report');
}

function testClearDiscardsQueuedReport() {
  // `clear` marks the start of a NEW task in the tab; a report queued
  // by a previous task that never reached a terminal event is stale
  // and must not open at the new task's completion.
  const {win} = makePage({withMarked: true});
  writeReport(win, 'reports/stale.md', '# stale');
  send(win, {type: 'clear'});
  finishTask(win);
  assertNoReportTab(win, 'stale report across a clear');
  win.close();
  console.log('  ok - a clear (new task) discards the queued report');
}

function testTraversalPathsDoNotOpen() {
  const {win} = makePage({withMarked: true});
  writeReport(win, 'reports/../escape.md', '# escaped');
  writeReport(win, 'reports/sub/../../escape2.md', '# escaped again');
  finishTask(win);
  assertNoReportTab(win, 'lexical ".." traversal out of reports/');
  // Harmless "." segments must still be recognized as reports paths.
  writeReport(win, './reports/./ok.md', '# fine');
  finishTask(win);
  assertReportTabActive(win, 'dot-segment report');
  win.close();
  console.log('  ok - reports/../ paths do not open; ./ segments still do');
}

function testNonReportPathsDoNotOpen() {
  const {win} = makePage({withMarked: true});
  writeReport(win, 'notes/summary.md', '# not a report');
  writeReport(win, 'reports/data.txt', 'plain text');
  writeReport(win, 'reports.md', '# file named reports.md');
  writeReport(win, 'myreports/x.html', '<p>nope</p>');
  writeReport(win, 'reports', 'extensionless file named reports');
  finishTask(win);
  assertNoReportTab(win, 'non-report paths');
  win.close();
  console.log('  ok - md/html outside a reports folder do not open');
}

function testNonWriteToolsDoNotOpen() {
  const {win} = makePage({withMarked: true});
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
  assertNoReportTab(win, 'non-Write tools');
  win.close();
  console.log('  ok - non-Write tools touching reports/ do not open');
}

function testInterveningToolCallClearsStash() {
  const {win} = makePage({withMarked: true});
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
  assertNoReportTab(win, 'intervening tool_call');
  win.close();
  console.log('  ok - an intervening tool_call clears the pending report');
}

function testReplayDoesNotOpen() {
  const {win} = makePage({withMarked: true});
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
  assertNoReportTab(win, 'task_events replay');
  win.close();
  console.log('  ok - replayed history does not auto-open report tabs');
}

// The replay-only report suppression must not survive into the live
// stream the replay's tail state becomes: a report the task writes
// AFTER its transcript was replayed opens like any live report, while
// a report inside the replayed transcript still opens nothing.
function testReplayDoesNotSuppressLiveReports() {
  const {win} = makePage();
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
  assertNoReportTab(win, 'a replayed report inside a running task');
  writeReport(win, 'reports/live.html', '<html><body>live</body></html>', {
    tabId,
  });
  finishTask(win, {tabId});
  assertReportTabActive(win, 'the live report written after the replay');
  win.close();
  console.log('  ok - a replay does not suppress the live reports after it');
}

function testRegenerationReusesSameTab() {
  const {win} = makePage({withMarked: true});
  writeReport(win, 'reports/summary.md', '# v1');
  writeReport(win, 'reports/summary.md', '# v2');
  finishTask(win);
  const tabsFound = contentTabs(win);
  assert.strictEqual(
    tabsFound.length,
    1,
    'regenerating the same report must reuse its tab, not open a second',
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
  const {win} = makePage({withMarked: true});
  writeReport(win, 'out/reports/2026/deep.html', '<!DOCTYPE html><p>deep</p>');
  writeReport(win, 'Reports\\win.md', '# windows style');
  finishTask(win);
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
    (tabsFound[1].textContent || '').indexOf('win.md') >= 0 &&
      tabsFound[1].classList.contains('active'),
    'reports must open in write order with the last one active',
  );
  win.close();
  console.log('  ok - nested, backslash and case-variant reports paths open');
}

function testBackgroundTabReportOpensAtItsTaskDone() {
  const {win} = makePage({withMarked: true});
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
  assertNoReportTab(win, 'background report before its task_done');
  // A task finishing in ANOTHER tab must not open this tab's report.
  finishTask(win, {tabId: secondTabId});
  assertNoReportTab(win, 'task_done of an unrelated tab');
  finishTask(win, {tabId: firstTabId});
  const tab = assertReportTabActive(win, 'background-tab report');
  assert.ok(
    (tab.textContent || '').indexOf('bg.md') >= 0,
    'the background-generated report tab must be titled after the file',
  );
  win.close();
  console.log('  ok - background-tab report opens at its own task_done');
}

// A sub-agent finishing is not the user's task finishing: its report
// opening in the background at subagentDone is pinned by
// subagentReportNoTabSwitch.testSubagentReportDoesNotStealFocus.

// The daemon addresses one sub-agent's tab by several ids over its life
// (the live fan-out id, and the deterministic "<parentTab>__sub_<taskId>"
// id used when the parent's history row is replayed).  The page keeps
// the sub-agent on ONE tab by moving it onto the id in use, which must
// carry the report the sub-agent already wrote with it: the report is
// stashed per tab id and would otherwise be stranded under the old one.
function testRetaggedSubagentKeepsItsPendingReport() {
  const {win, posted} = makePage({withMarked: true});
  const parentId = readyTabId(posted);

  send(win, {type: 'status', running: true, tabId: parentId});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 1'])},
  });
  const subId = spawnSubagent(
    {win, posted},
    parentId,
    'sub-task-1',
    'sub 1',
    0,
  );

  writeReport(win, 'reports/sub.md', '# from retagged subagent', {
    tabId: subId,
  });
  assertNoReportTab(win, 'retagged subagent report before subagentDone');

  // The parent's history row is replayed: same sub-agent, new tab id.
  // The id must genuinely differ from the open tab's (which already is
  // the deterministic `<parent>__sub_<task>` id) or no re-addressing
  // happens and retagSubagentTab's report transfer is never exercised.
  const replayId = 'replay-' + subId;
  send(win, {
    type: 'openSubagentTab',
    tab_id: replayId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
  });
  assert.strictEqual(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab').length,
    1,
    'the re-announced sub-agent must keep exactly one tab',
  );

  send(win, {type: 'subagentDone', tab_id: replayId});
  const found = registryContentTabs(win);
  assert.strictEqual(
    found.length,
    1,
    'BUG -- the report written before the sub-agent tab was re-addressed ' +
      'was lost instead of opening when the sub-agent finished',
  );
  win._testApi.switchToTab(found[0].id);
  assert.ok(
    /<h1[^>]*>from retagged subagent<\/h1>/.test(activeSrcdoc(win)),
    'the retagged sub-agent report must render once the user opens it',
  );
  win.close();
  console.log('  ok - a re-addressed sub-agent keeps its pending report');
}

function testSwitchBackToChatRestoresOutput() {
  const {win} = makePage({withMarked: true});
  send(win, {type: 'prompt', text: 'make a report'});
  writeReport(win, 'reports/summary.md', '# hi');
  finishTask(win);
  assertReportTabActive(win, 'switch-back');
  const chatTab = Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab'),
  ).find(el => !el.classList.contains('content-tab'));
  chatTab.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const out = win.document.getElementById('output');
  assert.notStrictEqual(
    out.style.display,
    'none',
    'clicking the chat tab must restore the chat surface',
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
  testRetaggedSubagentKeepsItsPendingReport();
  testSwitchBackToChatRestoresOutput();
  console.log('reportAutoOpen.test.js: all tests passed');
}

main();
