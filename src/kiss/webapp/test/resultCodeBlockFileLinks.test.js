// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Regression test: file paths inside a result panel's <pre><code>
// block must stay clickable after highlight.js processes the block.
//
// Deferred highlighting (transcript replay, collapsed-panel expand)
// runs AFTER linkifyFilePaths() has inserted [data-path] spans, and
// hljs.highlightElement() rewrites the block's innerHTML from its
// text — which used to destroy the links, leaving every path in a
// replayed result's code block unclickable.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const harness = require('./simplify2_harness');
const {send, click: clickEl, answerCheckPaths} = harness;
const {
  findLinks,
  checkPathsOnRealFs,
  pendingCounts,
} = require('./file_link_helpers');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rescode-'));
const realFile = path.join(tmpDir, 'API.md');
fs.writeFileSync(realFile, 'hello\n');
const realFile2 = path.join(tmpDir, 'README.md');
fs.writeFileSync(realFile2, 'world\n');
const missingFile = path.join(tmpDir, 'missing.md');

// Loads the real chat page (chat.html + api.js + main.js) in jsdom
// WITH the real highlight.js, answering every checkPaths against the
// real filesystem exactly like the web server does.
function makePage() {
  return harness.makePage({
    extraScripts: ['highlight.min.js', 'marked.min.js'],
    beforeScripts: win => answerCheckPaths(win, checkPathsOnRealFs(tmpDir)),
  });
}

// Summary HTML shaped like a real finish() result: an `ls -l` style
// listing inside <pre><code> plus one inline mention in a <p>.
function summaryHtml() {
  return (
    '<h3>Listing with full paths</h3>' +
    '<pre><code>-rw-r--r--@  1 ksen  staff   45245  ' +
    realFile +
    '\n-rw-r--r--@  1 ksen  staff   36367  ' +
    realFile2 +
    '\n-rw-r--r--@  1 ksen  staff       0  ' +
    missingFile +
    '\n</code></pre>' +
    '<p>Also see ' +
    realFile +
    ' for details.</p>'
  );
}

function testLiveResultCodeBlockPathsClickable() {
  const {win, posted} = makePage();
  send(win, {type: 'result', summary: summaryHtml(), success: true});
  assert.strictEqual(
    findLinks(win, realFile).length,
    2,
    'live: path must be clickable in the code block AND the paragraph',
  );
  assert.strictEqual(
    findLinks(win, realFile2).length,
    1,
    'live: second code-block path must be clickable',
  );
  assert.strictEqual(
    findLinks(win, missingFile).length,
    0,
    'live: missing path must NOT be clickable',
  );
  const inCode = findLinks(win, realFile2)[0];
  assert.ok(inCode.closest('pre'), 'link must live inside the code block');
  clickEl(win, inCode);
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(opens.length, 1, 'code-block link must open on click');
  assert.strictEqual(opens[0].path, realFile2);
  win.close();
  console.log('  ok - live result code-block paths are clickable');
}

// Replay a finished task whose result is summaryHtml().
function replayResult(win) {
  send(win, {
    type: 'task_events',
    task: 'replayed task',
    extra: JSON.stringify({work_dir: tmpDir}),
    events: [
      {
        type: 'result',
        summary: summaryHtml(),
        success: true,
        total_tokens: 1,
        cost: '$0.01',
      },
    ],
  });
}

function testReplayedResultCodeBlockPathsClickable() {
  const {win, posted} = makePage();
  replayResult(win);
  assert.strictEqual(
    findLinks(win, realFile).length,
    2,
    'replay: highlight.js must not destroy code-block file links',
  );
  assert.strictEqual(
    findLinks(win, realFile2).length,
    1,
    'replay: second code-block path must stay clickable',
  );
  assert.strictEqual(
    findLinks(win, missingFile).length,
    0,
    'replay: missing path must NOT become clickable',
  );
  // Once every reply arrived, nothing is pending.
  assert.deepStrictEqual(
    pendingCounts(win),
    {spans: 0, checks: 0},
    'no file-link span or path check may stay pending after the replies',
  );
  assert.strictEqual(
    win.document.querySelectorAll('#output [data-path-candidate]').length,
    0,
    'no unresolved candidate spans may remain after replies arrive',
  );
  const inCode = findLinks(win, realFile2)[0];
  assert.ok(inCode.closest('pre'), 'link must live inside the code block');
  clickEl(win, inCode);
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(opens.length, 1, 'replayed link must open on click');
  assert.strictEqual(opens[0].path, realFile2);
  win.close();
  console.log('  ok - replayed result code-block paths are clickable');
}

// Re-highlighting a replayed code block destroys its candidate spans
// while their checkPaths replies are still out.  The destroyed spans
// must leave the pending registry then, not linger until a reply
// sweeps them: with no reply at all (this page never answers), the
// registry must hold exactly the candidate spans still on the page.
function testRehighlightDropsDestroyedSpansFromRegistry() {
  const {win} = harness.makePage({
    extraScripts: ['highlight.min.js', 'marked.min.js'],
  });
  replayResult(win);
  const live = win.document.querySelectorAll(
    '#output [data-path-candidate]',
  ).length;
  assert.ok(live > 0, 'the replay leaves candidate spans awaiting replies');
  assert.strictEqual(
    pendingCounts(win).spans,
    live,
    'BUG: spans destroyed by re-highlighting linger in the pending registry',
  );
  win.close();
  console.log('  ok - re-highlighting drops destroyed spans from the registry');
}

// Code blocks inside collapsed panels are highlighted lazily on expand
// (highlightPending): highlightSplitPathFileLinks.test.js's
// testCollapsedTextPanelExpandLinksWholePaths covers that path.

function runTests() {
  testLiveResultCodeBlockPathsClickable();
  testReplayedResultCodeBlockPathsClickable();
  testRehighlightDropsDestroyedSpansFromRegistry();
}

try {
  runTests();
  console.log('resultCodeBlockFileLinks.test.js: all tests passed');
} finally {
  fs.rmSync(tmpDir, {recursive: true, force: true});
}
