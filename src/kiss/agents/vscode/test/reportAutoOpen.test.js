// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests: whenever the agent generates a report (a .md or
// .html file written by the agent into a `reports` folder), the chat
// webview must open it as an HTML page in a content tab and switch to
// that tab — but only AFTER the task finishes (task_done for the owning
// chat tab, or subagentDone for a subagent tab).  While the task is
// still running the report must NOT pop open.  Reports of tasks that
// end in task_error / task_interrupted / task_stopped are discarded.
// Markdown reports are converted to HTML first.  This applies to both
// the VS Code extension webview and the remote web app, which share
// media/main.js.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  opts = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  // The remote web app serves the same markup with a remote-chat body
  // class (web_server.py _build_html); opts.remote exercises that app.
  if (opts.remote) html = html.replace('<body', '<body class="remote-chat"');

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
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  if (opts.withMarked) {
    win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  }
  if (opts.brokenMarked) {
    win.marked = {
      parse: function () {
        throw new Error('marked exploded');
      },
    };
  }
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=report-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// Production tool_result events (json_printer.py _emit_tool_result) carry
// tool_name, is_error and — for tools whose input has a file_path — path;
// a successful Write's content is "Successfully wrote N characters to P".
function writeSuccess(filePath, content) {
  return (
    'Successfully wrote ' +
    String(content || '').length +
    ' characters to ' +
    filePath
  );
}

function writeReport(win, filePath, content, extra) {
  send(
    win,
    Object.assign(
      {type: 'tool_call', name: 'Write', path: filePath, content: content},
      extra || {},
    ),
  );
  send(
    win,
    Object.assign(
      {
        type: 'tool_result',
        content: writeSuccess(filePath, content),
        is_error: false,
        tool_name: 'Write',
        path: filePath,
      },
      extra || {},
    ),
  );
}

// The extension / web server emit task_done when the task finishes
// (src/types.ts); only then may stashed report tabs open.
function finishTask(win, extra) {
  send(win, Object.assign({type: 'task_done', success: true}, extra || {}));
}

function contentTabs(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.content-tab'),
  );
}

function reportFrames(win) {
  return Array.from(
    win.document.querySelectorAll(
      '#content-tab-area .content-tab-view iframe.content-html-frame',
    ),
  );
}

function activeSrcdoc(win) {
  const frames = reportFrames(win).filter(
    f => f.closest('.content-tab-view').style.display !== 'none',
  );
  assert.strictEqual(frames.length, 1, 'expected exactly one visible frame');
  return frames[0].getAttribute('srcdoc') || '';
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
      ': BUG — the agent generated a report but the chat webview did ' +
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
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview();
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
  const {win} = makeWebview({brokenMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
  finishTask(win);
  assertNoReportTab(win, 'task_done without any report');
  win.close();
  console.log('  ok - task_done without a report opens nothing');
}

function testTaskDoneOpensReportOnlyOnce() {
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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

function testTaskFailureStillOpensReport() {
  // A task that ends in task_error / task_interrupted / task_stopped
  // still FINISHED: a report it successfully wrote is a real artifact
  // and must open at that terminal event (parity with the pre-delay
  // behavior, where the report was already visible before the failure).
  ['task_error', 'task_interrupted', 'task_stopped'].forEach(terminal => {
    const {win} = makeWebview({withMarked: true});
    writeReport(win, 'reports/salvaged.md', '# salvaged');
    assertNoReportTab(win, terminal + ' before the terminal event');
    send(win, {type: terminal});
    assertReportTabActive(win, terminal);
    // A stray later task_done must not duplicate the opened report.
    const before = contentTabs(win).length;
    finishTask(win);
    assert.strictEqual(
      contentTabs(win).length,
      before,
      terminal + ': a later task_done must not reopen the report',
    );
    win.close();
  });
  console.log('  ok - error/interrupted/stopped tasks still open the report');
}

function testCloseTabDiscardsQueuedReport() {
  // Closing a chat tab means the user no longer cares about that
  // task's pending report; it must not pop open later and must not
  // leak in the queue.
  const {win} = makeWebview({withMarked: true});
  const firstTabEl = win.document.querySelector('#tab-list .chat-tab');
  const firstTabId = firstTabEl.dataset.tabId;
  const addBtn = win.document.querySelector('.chat-tab-add');
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  // The (now background) first tab confirms a report...
  writeReport(win, 'reports/closed.md', '# never shown', {tabId: firstTabId});
  // ...then the user closes that tab.
  const closeBtn = win.document.querySelector(
    '#tab-list .chat-tab[data-tab-id="' + firstTabId + '"] .chat-tab-close',
  );
  assert.ok(closeBtn, 'expected a close button on the first chat tab');
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  finishTask(win, {tabId: firstTabId});
  assertNoReportTab(win, 'report of a closed tab');
  win.close();
  console.log('  ok - closing a chat tab discards its queued report');
}

function testClearDiscardsQueuedReport() {
  // `clear` marks the start of a NEW task in the tab; a report queued
  // by a previous task that never reached a terminal event is stale
  // and must not open at the new task's completion.
  const {win} = makeWebview({withMarked: true});
  writeReport(win, 'reports/stale.md', '# stale');
  send(win, {type: 'clear'});
  finishTask(win);
  assertNoReportTab(win, 'stale report across a clear');
  win.close();
  console.log('  ok - a clear (new task) discards the queued report');
}

function testTraversalPathsDoNotOpen() {
  const {win} = makeWebview({withMarked: true});
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

function testRemoteWebAppMdReport() {
  const {win} = makeWebview({withMarked: true, remote: true});
  assert.ok(
    win.document.body.classList.contains('remote-chat'),
    'precondition: the remote web app marks its body with remote-chat',
  );
  writeReport(win, 'reports/summary.md', '# Remote Report\n\n**hi**');
  assertNoReportTab(win, 'remote report before task_done');
  finishTask(win);
  const tab = assertReportTabActive(win, 'remote web app md report');
  assert.ok(
    (tab.textContent || '').indexOf('summary.md') >= 0,
    'remote report tab must be titled after the report file',
  );
  const srcdoc = activeSrcdoc(win);
  assert.ok(
    /<h1[^>]*>Remote Report<\/h1>/.test(srcdoc),
    'remote web app must convert the markdown report to HTML',
  );
  win.close();
  console.log('  ok - remote web app opens md report at task_done');
}

function testNonReportPathsDoNotOpen() {
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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

function testRegenerationReusesSameTab() {
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
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
  const {win} = makeWebview({withMarked: true});
  const firstTabEl = win.document.querySelector('#tab-list .chat-tab');
  assert.ok(firstTabEl, 'expected an initial chat tab');
  const firstTabId = firstTabEl.dataset.tabId;
  const addBtn = win.document.querySelector('.chat-tab-add');
  assert.ok(addBtn, 'expected the add-tab button');
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const activeChat = win.document.querySelector('#tab-list .chat-tab.active');
  assert.notStrictEqual(
    activeChat.dataset.tabId,
    firstTabId,
    'precondition: a second chat tab must now be active',
  );
  // The agent running in the (now background) first tab writes a report.
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/bg.md',
    content: '# from background',
    tabId: firstTabId,
  });
  send(win, {
    type: 'tool_result',
    content: writeSuccess('reports/bg.md', '# from background'),
    is_error: false,
    tool_name: 'Write',
    path: 'reports/bg.md',
    tabId: firstTabId,
  });
  assertNoReportTab(win, 'background report before its task_done');
  // A task finishing in ANOTHER tab must not open this tab's report.
  finishTask(win, {tabId: activeChat.dataset.tabId});
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

// A sub-agent finishing is not the user's task finishing: the parent
// task keeps running, so the report it wrote must become reachable in
// the tab bar without pulling the user onto it - not even when the user
// was standing on the sub-agent tab that is now being closed.
function testSubagentDoneOpensReportInTheBackground() {
  const {win} = makeWebview({withMarked: true});
  const addBtn = win.document.querySelector('.chat-tab-add');
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const subTabId = win.document.querySelector('#tab-list .chat-tab.active')
    .dataset.tabId;
  writeReport(win, 'reports/sub.md', '# from subagent', {tabId: subTabId});
  assertNoReportTab(win, 'subagent report before subagentDone');
  send(win, {type: 'subagentDone', tab_id: subTabId});

  const found = contentTabs(win);
  assert.strictEqual(
    found.length,
    1,
    'subagent report: the report must open as a content tab',
  );
  assert.ok(
    !found[0].classList.contains('active'),
    'subagent report: the parent task is still running, so the report ' +
      'must stay in the background',
  );
  const out = win.document.getElementById('output');
  assert.notStrictEqual(
    out.style.display,
    'none',
    'subagent report: the chat surface must stay on screen',
  );

  found[0].dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    /<h1[^>]*>from subagent<\/h1>/.test(activeSrcdoc(win)),
    'the subagent report must render once the user opens it',
  );
  win.close();
  console.log('  ok - subagent report opens in the background on done');
}

// The daemon addresses one sub-agent's tab by several ids over its life
// (the live fan-out id, and the deterministic "<parentTab>__sub_<taskId>"
// id used when the parent's history row is replayed).  The webview keeps
// the sub-agent on ONE tab by moving it onto the id in use, which must
// carry the report the sub-agent already wrote with it: the report is
// stashed per tab id and would otherwise be stranded under the old one.
function testRetaggedSubagentKeepsItsPendingReport() {
  const {win, posted} = makeWebview({withMarked: true});
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'the webview must post ready with a tabId');
  const parentId = ready.tabId;

  send(win, {type: 'status', running: true, tabId: parentId});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 1'])},
  });
  send(win, {
    type: 'new_tab',
    task_id: 'sub-task-1',
    parent_tab_id: parentId,
    taskId: '',
  });
  const resume = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 'sub-task-1',
  );
  assert.ok(resume, 'the fan-out must resume its sub-agent in its own tab');
  const announce = {
    type: 'openSubagentTab',
    tab_id: resume.tabId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
  };
  send(win, announce);

  writeReport(win, 'reports/sub.md', '# from retagged subagent', {
    tabId: resume.tabId,
  });
  assertNoReportTab(win, 'retagged subagent report before subagentDone');

  // The parent's history row is replayed: same sub-agent, new tab id.
  const replayId = parentId + '__sub_sub-task-1';
  send(win, Object.assign({}, announce, {tab_id: replayId}));
  assert.strictEqual(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab').length,
    1,
    'the re-announced sub-agent must keep exactly one tab',
  );

  send(win, {type: 'subagentDone', tab_id: replayId});
  const found = contentTabs(win);
  assert.strictEqual(
    found.length,
    1,
    'BUG — the report written before the sub-agent tab was re-addressed ' +
      'was lost instead of opening when the sub-agent finished',
  );
  found[0].dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    /<h1[^>]*>from retagged subagent<\/h1>/.test(activeSrcdoc(win)),
    'the retagged sub-agent report must render once the user opens it',
  );
  win.close();
  console.log('  ok - a re-addressed sub-agent keeps its pending report');
}

function testSwitchBackToChatRestoresOutput() {
  const {win} = makeWebview({withMarked: true});
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
  testTaskFailureStillOpensReport();
  testCloseTabDiscardsQueuedReport();
  testClearDiscardsQueuedReport();
  testTraversalPathsDoNotOpen();
  testRemoteWebAppMdReport();
  testNonReportPathsDoNotOpen();
  testNonWriteToolsDoNotOpen();
  testInterveningToolCallClearsStash();
  testReplayDoesNotOpen();
  testRegenerationReusesSameTab();
  testReportsSegmentVariants();
  testBackgroundTabReportOpensAtItsTaskDone();
  testSubagentDoneOpensReportInTheBackground();
  testRetaggedSubagentKeepsItsPendingReport();
  testSwitchBackToChatRestoresOutput();
  console.log('reportAutoOpen.test.js: all tests passed');
}

main();
