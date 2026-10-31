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

// Every path shape (absolute, dot-relative, workspace-relative,
// home-relative, parent-relative) in a prompt becomes exactly one link.
const PROMPT_PATHS = [
  ['absolute', 'See /tmp/foo.py for details.', '/tmp/foo.py'],
  ['dot-relative', 'open ./src/main.js', './src/main.js'],
  ['workspace-relative', 'open src/kiss/INJECTIONS.md', 'src/kiss/INJECTIONS.md'],
  ['home-relative', 'open ~/work/foo/bar.txt now', '~/work/foo/bar.txt'],
  ['parent-relative', 'see ../README.md please', '../README.md'],
];

function testFilePathShapesInPromptAreLinkified() {
  for (const [shape, text, p] of PROMPT_PATHS) {
    const {win} = makePage();
    send(win, {type: 'prompt', text});
    assert.strictEqual(
      findLinks(win, p).length,
      1,
      `${shape} path ${p} in prompt must be linkified`,
    );
    win.close();
    console.log(`  ok - ${shape} path in prompt is linkified`);
  }
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

function testFilePathInsideExistingAnchorNotDoubleLinkified() {
  const {win} = makePage();
  // A result summary renders its HTML, so the anchor is a real element
  // (a prompt would show the same markup as escaped text).
  send(win, {
    type: 'result',
    summary:
      '<p><a href="https://x.com/foo">see /tmp/x/y.py</a> and /tmp/x/z.py</p>',
    success: true,
  });
  const out = win.document.getElementById('output');
  assert.ok(
    out.querySelector('a[href="https://x.com/foo"]'),
    'the anchor in the summary must render as an element',
  );
  assert.strictEqual(
    findLinks(win, '/tmp/x/z.py').length,
    1,
    'a path next to the anchor must still be linkified',
  );
  assert.strictEqual(
    out.querySelectorAll('a [data-path]').length,
    0,
    'no data-path span inside an existing anchor',
  );
  win.close();
  console.log('  ok - anchor contents are not double-linkified');
}

// A tool_call `path` arg being clickable is pinned against the real
// filesystem by clickFilePathExistsOnly.testToolCallExistingPathArgIsClickable.

// The linkifier being idempotent (no nested [data-path]) is pinned against
// the real filesystem by clickFilePathExistsOnly.testLinkifierIdempotentOnCandidatesAndMissing.

function runTests() {
  testFilePathShapesInPromptAreLinkified();
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
  testFilePathInsideExistingAnchorNotDoubleLinkified();
}

try {
  runTests();
  console.log('\nclickFilePathOpens.test.js: all tests passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
