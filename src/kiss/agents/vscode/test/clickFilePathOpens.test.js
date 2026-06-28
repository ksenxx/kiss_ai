// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end JSDOM tests for the "click filepath in chat → open file"
// feature: rendered chat content (prompts, results, streamed text,
// bash tool output, error tool_result text) must surface filepath
// tokens as clickable elements that post ``openFile`` to the
// extension.  The extension's existing ``openFile`` handler (see
// ``SorcarSidebarView.ts``) does the actual fs.existsSync gating and
// dispatches to the VS Code editor or the native viewer.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/clickFilePathOpens.test.js

'use strict';

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
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function flushTextStream(win) {
  // ``text_end`` flushes the streamed buffer through marked.parse +
  // kissSanitize so the linkifier runs.
  send(win, {type: 'text_end'});
}

function findLinks(win, p) {
  // Both the new ``kiss-filelink`` markers and the existing tool_call
  // ``.tp[data-path]`` hooks expose a ``data-path`` attribute that the
  // global click handler in ``main.js`` dispatches on.  The feature
  // covers all rendered free-text chat content (prompts, results,
  // streamed text, tool output, errors) — wherever a filepath token
  // appears in user-visible text it must surface a ``data-path``
  // element matching ``p``.
  return Array.from(
    win.document.querySelectorAll('#output [data-path]'),
  ).filter(el => el.dataset.path === p);
}

function clickFirstLink(win, p) {
  const el = findLinks(win, p)[0];
  assert.ok(el, 'expected a [data-path="' + p + '"] in #output');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

function testAbsoluteFilePathInPromptIsLinkified() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'See /tmp/foo.py for details.'});
  const links = findLinks(win, '/tmp/foo.py');
  assert.strictEqual(
    links.length,
    1,
    'absolute path /tmp/foo.py in prompt must be linkified',
  );
  win.close();
  console.log('  ok - absolute path in prompt is linkified');
}

function testRelativeFilePathLinkified() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'open ./src/main.js'});
  const links = findLinks(win, './src/main.js');
  assert.strictEqual(
    links.length,
    1,
    'relative path ./src/main.js must be linkified',
  );
  win.close();
  console.log('  ok - dot-relative path is linkified');
}

function testWorkspaceRelativeFilePathLinkified() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'open src/kiss/INJECTIONS.md'});
  const links = findLinks(win, 'src/kiss/INJECTIONS.md');
  assert.strictEqual(
    links.length,
    1,
    'workspace-relative path src/kiss/INJECTIONS.md must be linkified',
  );
  win.close();
  console.log('  ok - workspace-relative path is linkified');
}

function testHomeRelativeFilePathLinkified() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'open ~/work/foo/bar.txt now'});
  const links = findLinks(win, '~/work/foo/bar.txt');
  assert.strictEqual(
    links.length,
    1,
    'home-relative path ~/work/foo/bar.txt must be linkified',
  );
  win.close();
  console.log('  ok - home-relative path is linkified');
}

function testParentRelativeFilePathLinkified() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'see ../README.md please'});
  const links = findLinks(win, '../README.md');
  assert.strictEqual(
    links.length,
    1,
    'parent-relative path ../README.md must be linkified',
  );
  win.close();
  console.log('  ok - parent-relative path is linkified');
}

function testFilePathWithLineSuffixLinkified() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'crash at /tmp/foo.py:42 today'});
  const links = findLinks(win, '/tmp/foo.py:42');
  assert.strictEqual(
    links.length,
    1,
    'path with :42 suffix must be linkified',
  );
  win.close();
  console.log('  ok - path with :line suffix is linkified');
}

function testFilePathInResultSummaryLinkified() {
  const {win} = makeWebview();
  send(win, {
    type: 'result',
    summary: 'Wrote /var/log/x.log successfully',
    success: true,
  });
  const links = findLinks(win, '/var/log/x.log');
  assert.strictEqual(
    links.length,
    1,
    'path in result summary must be linkified',
  );
  win.close();
  console.log('  ok - path in result summary is linkified');
}

function testFilePathInResultTextLinkified() {
  const {win} = makeWebview();
  send(win, {
    type: 'result',
    text: 'Wrote /var/log/y.log successfully',
    success: true,
  });
  const links = findLinks(win, '/var/log/y.log');
  assert.strictEqual(
    links.length,
    1,
    'path in result text must be linkified',
  );
  win.close();
  console.log('  ok - path in result text is linkified');
}

function testFilePathInBashOutputLinkified() {
  const {win} = makeWebview();
  // Bash tool output flows in via streamed ``system_output`` events
  // into the panel created by the ``tool_call`` event.  The
  // ``tool_result`` for a Bash call is suppressed by the
  // ``hadBash && !is_error`` early-exit (the streamed bytes already
  // populated the panel), so we drive the bash path with
  // system_output events explicitly.
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls /tmp'});
  send(win, {type: 'system_output', text: 'found at /etc/hosts on line 3'});
  send(win, {type: 'tool_result', content: ''});
  const links = findLinks(win, '/etc/hosts');
  assert.strictEqual(
    links.length,
    1,
    'path in bash system_output stream must be linkified',
  );
  win.close();
  console.log('  ok - path in bash system_output is linkified');
}

function testFilePathInNonBashToolOutputLinkified() {
  const {win} = makeWebview();
  // For non-Bash tools, the success tool_result content is rendered
  // into a bash-panel (textContent).  Same linkifier must run.
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/x.txt'});
  send(win, {
    type: 'tool_result',
    content: 'opened at /opt/share/lib.so happily',
  });
  const links = findLinks(win, '/opt/share/lib.so');
  assert.strictEqual(
    links.length,
    1,
    'path in non-Bash tool_result content must be linkified',
  );
  win.close();
  console.log('  ok - path in non-Bash tool_result content is linkified');
}

function testFilePathInToolErrorContentLinkified() {
  const {win} = makeWebview();
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/x.txt'});
  send(win, {
    type: 'tool_result',
    content: 'no such file: /tmp/missing.py:7',
    is_error: true,
  });
  const links = findLinks(win, '/tmp/missing.py:7');
  assert.strictEqual(
    links.length,
    1,
    'path in tool_result error content must be linkified',
  );
  win.close();
  console.log('  ok - path in tool_result error content is linkified');
}

function testFilePathInStreamedTextLinkifiedOnFlush() {
  const {win} = makeWebview();
  send(win, {type: 'text_delta', text: 'Edit '});
  send(win, {type: 'text_delta', text: '/srv/app/main.py'});
  send(win, {type: 'text_delta', text: ' please'});
  // Before flush, no links yet (raw text node).
  flushTextStream(win);
  const links = findLinks(win, '/srv/app/main.py');
  assert.strictEqual(
    links.length,
    1,
    'streamed text path must be linkified on text_end flush',
  );
  win.close();
  console.log('  ok - streamed text path is linkified on flush');
}

function testFilePathInsideCodeBlockLinkified() {
  const {win} = makeWebview();
  send(win, {
    type: 'prompt',
    text: '```\nmv /tmp/a.txt /tmp/b.txt\n```',
  });
  const links = findLinks(win, '/tmp/a.txt');
  assert.strictEqual(
    links.length,
    1,
    'path inside code block must be linkified (code paths are exact)',
  );
  win.close();
  console.log('  ok - path in code block is linkified');
}

function testHttpUrlNotLinkifiedAsFilePath() {
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'see https://example.com/foo/bar.json'});
  // The URL becomes an <a href=…>; its path slice "/foo/bar.json" must
  // NOT be wrapped as a filepath (it is part of the URL).
  const links = findLinks(win, '/foo/bar.json');
  assert.strictEqual(
    links.length,
    0,
    'URL path component must NOT be linkified as a filepath',
  );
  win.close();
  console.log('  ok - URL path slice is not linkified');
}

function testBareFilenameNotLinkified() {
  // Bare filenames (no slash, just an extension) are too ambiguous to
  // detect reliably and would noise-up sentences like "version 1.0".
  // The feature explicitly requires at least one slash or a relative
  // directory component.
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'package.json was updated to 2.0'});
  const links = findLinks(win, 'package.json');
  assert.strictEqual(
    links.length,
    0,
    'bare filename "package.json" must NOT be linkified (no slash)',
  );
  win.close();
  console.log('  ok - bare filename is not linkified');
}

function testSentenceTrailingPunctuationNotIncluded() {
  const {win} = makeWebview();
  send(win, {
    type: 'prompt',
    text: 'fix /tmp/a.py, /tmp/b.py.',
  });
  // The trailing comma after a.py and period after b.py must not be
  // captured as part of the path.
  const a = findLinks(win, '/tmp/a.py');
  const b = findLinks(win, '/tmp/b.py');
  assert.strictEqual(a.length, 1, '/tmp/a.py must be linkified');
  assert.strictEqual(b.length, 1, '/tmp/b.py must be linkified');
  // And neither should include trailing punctuation.
  const aWithComma = findLinks(win, '/tmp/a.py,');
  const bWithPeriod = findLinks(win, '/tmp/b.py.');
  assert.strictEqual(
    aWithComma.length,
    0,
    'trailing comma must not be included in path',
  );
  assert.strictEqual(
    bWithPeriod.length,
    0,
    'trailing period must not be included in path',
  );
  win.close();
  console.log('  ok - trailing punctuation is stripped from paths');
}

function testClickPostsOpenFileWithCorrectPath() {
  const {win, posted} = makeWebview();
  send(win, {type: 'prompt', text: 'open /tmp/click.py'});
  clickFirstLink(win, '/tmp/click.py');
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(opens.length, 1, 'one openFile must be posted');
  assert.strictEqual(opens[0].path, '/tmp/click.py');
  assert.strictEqual(
    opens[0].line,
    undefined,
    'no line should be sent when path has no :line suffix',
  );
  win.close();
  console.log('  ok - click posts openFile with correct path');
}

function testClickPostsOpenFileWithLine() {
  const {win, posted} = makeWebview();
  send(win, {type: 'prompt', text: 'crash at /tmp/click2.py:99 see above'});
  clickFirstLink(win, '/tmp/click2.py:99');
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(opens.length, 1, 'one openFile must be posted');
  assert.strictEqual(opens[0].path, '/tmp/click2.py');
  assert.strictEqual(opens[0].line, 99, 'line 99 must be parsed and sent');
  win.close();
  console.log('  ok - click posts openFile with parsed :line');
}

function testFilePathInsideExistingAnchorNotDoubleLinkified() {
  // marked autolinks bare URLs; the linkifier must skip text nodes
  // already inside <a> so we never produce nested <a><span/></a>.
  const {win} = makeWebview();
  send(win, {
    type: 'prompt',
    text: '<a href="https://x.com/foo">https://x.com/foo</a>',
  });
  const links = findLinks(win, '/foo');
  assert.strictEqual(
    links.length,
    0,
    'no data-path span inside an existing anchor',
  );
  win.close();
  console.log('  ok - anchor contents are not double-linkified');
}

function testToolCallPathArgStillUsesExistingHook() {
  const {win, posted} = makeWebview();
  // The existing ``Read`` tool_call panel header lists ``path: <tp>``
  // with ``data-path`` already (pre-existing code).  Click still
  // posts openFile; the new feature must not break this.
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/old/style.py'});
  const links = findLinks(win, '/tmp/old/style.py');
  assert.ok(
    links.length >= 1,
    'tool_call path arg keeps its existing data-path hook',
  );
  links[0].dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const opens = posted.filter(m => m.type === 'openFile');
  assert.ok(
    opens.some(m => m.path === '/tmp/old/style.py'),
    'click on tool_call path arg still posts openFile',
  );
  win.close();
  console.log('  ok - tool_call path arg keeps existing data-path hook');
}

function testLinkifierIsIdempotent() {
  // Re-rendering must not nest data-path spans inside themselves.  The
  // simplest assertion: after a prompt event the only data-path
  // elements have no data-path descendant.
  const {win} = makeWebview();
  send(win, {type: 'prompt', text: 'open /tmp/once.py twice /tmp/once.py'});
  const links = Array.from(
    win.document.querySelectorAll('#output [data-path]'),
  );
  for (const a of links) {
    assert.strictEqual(
      a.querySelectorAll('[data-path]').length,
      0,
      'no data-path element nested inside another',
    );
  }
  win.close();
  console.log('  ok - linkifier is idempotent (no nested data-path)');
}

function runTests() {
  testAbsoluteFilePathInPromptIsLinkified();
  testRelativeFilePathLinkified();
  testWorkspaceRelativeFilePathLinkified();
  testHomeRelativeFilePathLinkified();
  testParentRelativeFilePathLinkified();
  testFilePathWithLineSuffixLinkified();
  testFilePathInResultSummaryLinkified();
  testFilePathInResultTextLinkified();
  testFilePathInBashOutputLinkified();
  testFilePathInNonBashToolOutputLinkified();
  testFilePathInToolErrorContentLinkified();
  testFilePathInStreamedTextLinkifiedOnFlush();
  testFilePathInsideCodeBlockLinkified();
  testHttpUrlNotLinkifiedAsFilePath();
  testBareFilenameNotLinkified();
  testSentenceTrailingPunctuationNotIncluded();
  testClickPostsOpenFileWithCorrectPath();
  testClickPostsOpenFileWithLine();
  testFilePathInsideExistingAnchorNotDoubleLinkified();
  testToolCallPathArgStillUsesExistingHook();
  testLinkifierIsIdempotent();
}

try {
  runTests();
  console.log('\n21 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
