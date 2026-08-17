// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests: when the user clicks a linkified .md/.markdown
// file path, the daemon answers with a `fileContent` event carrying the
// RAW markdown text; media/main.js must convert it to HTML
// (markdownReportToHtml) and render the result in a sandboxed iframe
// inside a content tab — exactly how an .html file renders — instead of
// showing markdown source in a code view.  Report markdown (isReport)
// arrives already converted by openReadyReportTabs and must NOT be
// converted twice.  This applies to both the VS Code extension webview
// and the remote web app, which share media/main.js.

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=mdlink-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function contentTabs(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.content-tab'),
  );
}

function htmlFrames(win) {
  return Array.from(
    win.document.querySelectorAll(
      '#content-tab-area .content-tab-view iframe.content-html-frame',
    ),
  );
}

function visibleSrcdoc(win) {
  const frames = htmlFrames(win).filter(
    f => f.closest('.content-tab-view').style.display !== 'none',
  );
  assert.strictEqual(frames.length, 1, 'expected exactly one visible frame');
  return frames[0].getAttribute('srcdoc') || '';
}

// 1. A clicked .md link (raw markdown fileContent, remote web app) is
//    converted to HTML and rendered in a sandboxed iframe content tab.
function testMdFileContentRendersHtml() {
  const {win} = makeWebview({remote: true, withMarked: true});
  send(win, {
    type: 'fileContent',
    path: '/repo/docs/notes.md',
    name: 'notes.md',
    content: '# Doc Title\n\nSome **bold** text.\n',
  });
  const tabsFound = contentTabs(win);
  assert.strictEqual(tabsFound.length, 1, 'a content tab must open');
  assert.ok(
    tabsFound[0].classList.contains('active'),
    'the md tab must become the active tab',
  );
  assert.ok(
    (tabsFound[0].textContent || '').indexOf('notes.md') >= 0,
    'tab must be titled after the file',
  );
  const frames = htmlFrames(win);
  assert.strictEqual(frames.length, 1, 'md must render in an iframe');
  assert.strictEqual(
    frames[0].getAttribute('sandbox'),
    'allow-scripts',
    'the iframe must stay sandboxed like the .html one',
  );
  const doc = visibleSrcdoc(win);
  assert.ok(
    /<h1[^>]*>Doc Title<\/h1>/.test(doc),
    'markdown heading must be converted to <h1>: ' + doc.slice(0, 200),
  );
  assert.ok(
    /<strong>bold<\/strong>/.test(doc),
    'markdown emphasis must be converted to <strong>',
  );
  assert.ok(
    doc.indexOf('# Doc Title') < 0,
    'raw markdown source must not leak into the rendered page',
  );
  assert.strictEqual(
    win.document.querySelectorAll('.content-monaco-holder').length,
    0,
    'no code view may be used for markdown',
  );
  console.log('  ok - .md fileContent renders converted HTML in an iframe');
}

// 2. .markdown converts too (vscode-webview flavour of the same page).
function testMarkdownExtensionAlsoConverts() {
  const {win} = makeWebview({withMarked: true});
  send(win, {
    type: 'fileContent',
    path: '/repo/docs/guide.markdown',
    name: 'guide.markdown',
    content: '## Section\n',
  });
  const doc = visibleSrcdoc(win);
  assert.ok(
    /<h2[^>]*>Section<\/h2>/.test(doc),
    '.markdown must be converted like .md',
  );
  console.log('  ok - .markdown fileContent converts too');
}

// 3. Without the marked library the page still renders — as escaped
//    preformatted text, never as a script-capable raw injection.
function testNoMarkedFallsBackToEscapedPre() {
  const {win} = makeWebview({remote: true});
  send(win, {
    type: 'fileContent',
    path: '/repo/docs/evil.md',
    name: 'evil.md',
    content: '# T\n<script>alert(1)</script>',
  });
  const doc = visibleSrcdoc(win);
  assert.ok(doc.indexOf('<pre>') >= 0, 'fallback must use <pre>');
  assert.ok(
    doc.indexOf('<script>alert(1)</script>') < 0 &&
      doc.indexOf('&lt;script&gt;') >= 0,
    'fallback must escape the markdown source',
  );
  console.log('  ok - missing marked falls back to escaped <pre>');
}

// 4. Report markdown (isReport) arrives ALREADY converted by
//    openReadyReportTabs — it must render as-is, not be converted again.
function testReportMarkdownNotDoubleConverted() {
  const {win} = makeWebview({remote: true, withMarked: true});
  const already = '<!DOCTYPE html><html><body><h1>done</h1></body></html>';
  send(win, {
    type: 'fileContent',
    path: '/repo/reports/summary.md',
    name: 'summary.md',
    content: already,
    isReport: true,
  });
  const doc = visibleSrcdoc(win);
  assert.strictEqual(
    doc,
    already,
    'isReport content is already HTML and must pass through untouched',
  );
  console.log('  ok - report markdown is not converted twice');
}

// 5. Clicking the same .md link again re-renders the SAME tab with the
//    freshly served (re-converted) content — no duplicate tab.
function testSecondClickReusesTabAndReconverts() {
  const {win} = makeWebview({remote: true, withMarked: true});
  send(win, {
    type: 'fileContent',
    path: '/repo/docs/notes.md',
    name: 'notes.md',
    content: '# First\n',
  });
  send(win, {
    type: 'fileContent',
    path: '/repo/docs/notes.md',
    name: 'notes.md',
    content: '# Second\n',
  });
  assert.strictEqual(contentTabs(win).length, 1, 'no duplicate tab');
  const doc = visibleSrcdoc(win);
  assert.ok(
    /<h1[^>]*>Second<\/h1>/.test(doc),
    'the reused tab must show freshly converted content',
  );
  console.log('  ok - second click reuses the tab and re-converts');
}

// 6. An empty markdown file still opens a (blank) rendered tab.
function testEmptyMarkdownRenders() {
  const {win} = makeWebview({remote: true, withMarked: true});
  send(win, {
    type: 'fileContent',
    path: '/repo/docs/empty.md',
    name: 'empty.md',
  });
  const doc = visibleSrcdoc(win);
  assert.ok(
    doc.indexOf('<!DOCTYPE html>') === 0,
    'empty markdown must still produce the HTML shell',
  );
  console.log('  ok - empty markdown renders a blank page');
}

// 7. Regression: .html content still renders untouched, .py still gets
//    the code view — the shared iframe helper changed neither path.
function testHtmlAndCodeRegressions() {
  const {win} = makeWebview({remote: true, withMarked: true});
  send(win, {
    type: 'fileContent',
    path: '/repo/page.html',
    name: 'page.html',
    content: '<!DOCTYPE html><html><body><h1>raw</h1></body></html>',
  });
  assert.ok(
    visibleSrcdoc(win).indexOf('<h1>raw</h1>') >= 0,
    '.html must render its content as-is',
  );
  send(win, {
    type: 'fileContent',
    path: '/repo/main.py',
    name: 'main.py',
    content: 'print("hi")\n',
  });
  assert.strictEqual(
    win.document.querySelectorAll('.content-monaco-holder').length,
    1,
    '.py must still use the code view',
  );
  assert.strictEqual(htmlFrames(win).length, 1, 'no iframe for .py');
  console.log('  ok - .html and code-view regressions hold');
}

testMdFileContentRendersHtml();
testMarkdownExtensionAlsoConverts();
testNoMarkedFallsBackToEscapedPre();
testReportMarkdownNotDoubleConverted();
testSecondClickReusesTabAndReconverts();
testEmptyMarkdownRenders();
testHtmlAndCodeRegressions();
console.log('mdLinkContentTab tests passed');
