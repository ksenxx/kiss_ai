// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  answerCheckPaths,
} = require('./simplify2_harness');
const {findLinks} = require('./file_link_helpers');

// Confirm every checkPaths existence query, like the server does for
// files that exist: these tests exercise the linkifier itself
// (clickFilePathExistsOnly.test.js covers the exists / does-not-exist
// split against the real filesystem).
function makePage() {
  const {win, posted} = makeChatPage({beforeScripts: answerCheckPaths});
  return {win, posted};
}

function flushTextStream(win) {
  send(win, {type: 'text_end'});
}

function clickFirstLink(win, p) {
  const el = findLinks(win, p)[0];
  assert.ok(el, 'expected a [data-path="' + p + '"] in #output');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function testAbsoluteFilePathInPromptIsLinkified() {
  const {win} = makePage();
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
  const {win} = makePage();
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
  const {win} = makePage();
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
  const {win} = makePage();
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
  const {win} = makePage();
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
  const {win} = makePage();
  send(win, {type: 'prompt', text: 'crash at /tmp/foo.py:42 today'});
  const links = findLinks(win, '/tmp/foo.py:42');
  assert.strictEqual(links.length, 1, 'path with :42 suffix must be linkified');
  win.close();
  console.log('  ok - path with :line suffix is linkified');
}

function testFilePathInResultSummaryLinkified() {
  const {win} = makePage();
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
  const {win} = makePage();
  send(win, {
    type: 'result',
    text: 'Wrote /var/log/y.log successfully',
    success: true,
  });
  const links = findLinks(win, '/var/log/y.log');
  assert.strictEqual(links.length, 1, 'path in result text must be linkified');
  win.close();
  console.log('  ok - path in result text is linkified');
}

function testFilePathInBashOutputLinkified() {
  const {win} = makePage();
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
  const {win} = makePage();
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
  const {win} = makePage();
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
  const {win} = makePage();
  send(win, {type: 'text_delta', text: 'Edit '});
  send(win, {type: 'text_delta', text: '/srv/app/main.py'});
  send(win, {type: 'text_delta', text: ' please'});
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
  const {win} = makePage();
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
  const {win} = makePage();
  send(win, {type: 'prompt', text: 'see https://example.com/foo/bar.json'});
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
  const {win} = makePage();
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
  const {win} = makePage();
  send(win, {
    type: 'prompt',
    text: 'fix /tmp/a.py, /tmp/b.py.',
  });
  const a = findLinks(win, '/tmp/a.py');
  const b = findLinks(win, '/tmp/b.py');
  assert.strictEqual(a.length, 1, '/tmp/a.py must be linkified');
  assert.strictEqual(b.length, 1, '/tmp/b.py must be linkified');
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
  const {win, posted} = makePage();
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
  const {win, posted} = makePage();
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
  const {win} = makePage();
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
  const {win, posted} = makePage();
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
  const {win} = makePage();
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
