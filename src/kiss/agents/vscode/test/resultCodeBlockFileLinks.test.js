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
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rescode-'));
const realFile = path.join(tmpDir, 'API.md');
fs.writeFileSync(realFile, 'hello\n');
const realFile2 = path.join(tmpDir, 'README.md');
fs.writeFileSync(realFile2, 'world\n');
const missingFile = path.join(tmpDir, 'missing.md');

function checkPathsOnRealFs(msg) {
  const results = {};
  for (const p of msg.paths) {
    let abs = p;
    if (p.startsWith('~/')) {
      abs = path.join(os.homedir(), p.slice(2));
    } else if (!path.isAbsolute(p)) {
      abs = path.resolve(msg.workDir || tmpDir, p);
    }
    let ok = false;
    try {
      ok = fs.statSync(abs).isFile();
    } catch {
      ok = false;
    }
    results[p] = ok;
  }
  return results;
}

// Loads the real chat webview (chat.html + api.js + main.js) in jsdom
// WITH the real highlight.js, answering every checkPaths against the
// real filesystem exactly like the extension host / web server do.
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
      postMessage: msg => {
        posted.push(msg);
        if (msg.type === 'checkPaths') {
          send(win, {
            type: 'pathsExist',
            results: checkPathsOnRealFs(msg),
            workDir: msg.workDir,
            tabId: msg.tabId,
          });
        }
      },
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'highlight.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function findLinks(win, p) {
  return Array.from(
    win.document.querySelectorAll('#output [data-path]'),
  ).filter(el => el.dataset.path === p);
}

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
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
  const {win, posted} = makeWebview();
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

function testReplayedResultCodeBlockPathsClickable() {
  const {win, posted} = makeWebview();
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
  const inCode = findLinks(win, realFile2)[0];
  assert.ok(inCode.closest('pre'), 'link must live inside the code block');
  clickEl(win, inCode);
  const opens = posted.filter(m => m.type === 'openFile');
  assert.strictEqual(opens.length, 1, 'replayed link must open on click');
  assert.strictEqual(opens[0].path, realFile2);
  win.close();
  console.log('  ok - replayed result code-block paths are clickable');
}

function testCollapsedPanelExpandKeepsLinks() {
  // Code blocks inside collapsed panels are highlighted lazily on
  // expand (highlightPending); the links must survive that too.
  const {win} = makeWebview();
  send(win, {
    type: 'task_events',
    task: 'replayed task',
    extra: JSON.stringify({work_dir: tmpDir}),
    events: [
      {
        type: 'text_delta',
        text: 'listing:\n```\n-rw-r--r--  1 u  g  5  ' + realFile + '\n```\n',
      },
      {type: 'text_end'},
      {
        type: 'result',
        summary: summaryHtml(),
        success: true,
        total_tokens: 1,
        cost: '$0.01',
      },
    ],
  });
  const collapsed = Array.from(
    win.document.querySelectorAll('#output .collapsible.collapsed'),
  );
  assert.ok(collapsed.length >= 1, 'replay must collapse non-result panels');
  for (const panel of collapsed) {
    const header = panel.querySelector('.collapse-header');
    if (header) clickEl(win, header);
  }
  const links = findLinks(win, realFile);
  assert.ok(
    links.some(el => el.closest('.collapsible')),
    'expanding a collapsed panel must keep its code-block links',
  );
  win.close();
  console.log('  ok - expanding collapsed panels keeps code-block links');
}

function testNoStaleSpansLeakInRegistryAfterRehighlight() {
  // The spans destroyed by re-highlighting must not linger in the
  // pending registry: after replay + replies, no candidates remain.
  const {win} = makeWebview();
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
  assert.strictEqual(
    win.document.querySelectorAll('#output [data-path-candidate]').length,
    0,
    'no unresolved candidate spans may remain after replies arrive',
  );
  win.close();
  console.log('  ok - no stale candidate spans after re-highlighting');
}

function runTests() {
  testLiveResultCodeBlockPathsClickable();
  testReplayedResultCodeBlockPathsClickable();
  testCollapsedPanelExpandKeepsLinks();
  testNoStaleSpansLeakInRegistryAfterRehighlight();
}

try {
  runTests();
  console.log('resultCodeBlockFileLinks.test.js: all tests passed');
} finally {
  fs.rmSync(tmpDir, {recursive: true, force: true});
}
