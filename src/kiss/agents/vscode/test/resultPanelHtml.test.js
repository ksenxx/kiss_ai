// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end webview tests: the Result panel renders the finish() summary
// as HTML (the wire format is now always HTML) — NOT as Markdown.
//
// Reproduces the issue where createResultPanel() piped the summary through
// marked.parse(), so an HTML summary was mangled by the Markdown parser and
// Markdown syntax was styled even though finish() now guarantees HTML.
// Also covers splitMultiSessionSummary()'s new <h3> HTML session markers
// (with backward compatibility for old persisted Markdown markers).
//
// The same media/main.js is served to BOTH the VS Code webview and the
// remote webapp (src/kiss/server/web_server.py), so this covers both.

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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage() {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sendResult(win, summary) {
  send(win, {
    type: 'result',
    success: true,
    is_continue: false,
    summary,
    total_tokens: 20,
    cost: '$0.02',
  });
}

function resultBodies(win) {
  return Array.from(win.document.querySelectorAll('#output > .rc .rc-body'));
}

function testHtmlSummaryRendersAsHtml() {
  const win = makeWebview();
  sendResult(
    win,
    '<h2>Report</h2>\n<ul><li>alpha</li><li>beta</li></ul>\n' +
      '<pre><code>x = 1</code></pre>',
  );
  const bodies = resultBodies(win);
  assert.strictEqual(bodies.length, 1, 'expected one Result panel');
  const body = bodies[0];
  const h2 = body.querySelector('h2');
  assert.ok(h2, 'BUG: <h2> in the HTML summary must render as a heading');
  assert.strictEqual(h2.textContent, 'Report');
  const items = Array.from(body.querySelectorAll('li')).map(
    li => li.textContent,
  );
  assert.deepStrictEqual(items, ['alpha', 'beta']);
  assert.ok(body.querySelector('pre code'), 'code block must survive');
  assert.ok(
    !body.textContent.includes('<h2>'),
    'BUG: raw HTML tags must not appear as literal text',
  );
  win.close();
}

function testSummaryIsNotParsedAsMarkdown() {
  const win = makeWebview();
  // The wire format is HTML; Markdown syntax inside text must stay literal.
  sendResult(win, '<p>keep **stars** and _underscores_ literal</p>');
  const body = resultBodies(win)[0];
  assert.ok(
    !body.querySelector('strong') && !body.querySelector('em'),
    'BUG: summary must not be run through a Markdown parser',
  );
  assert.ok(
    body.textContent.includes('**stars**'),
    'literal Markdown syntax must survive HTML rendering; got: ' +
      body.textContent,
  );
  assert.ok(body.textContent.includes('_underscores_'));
  win.close();
}

function testHtmlSummaryIsSanitized() {
  const win = makeWebview();
  sendResult(
    win,
    '<p>safe</p><script>window.__pwned = 1;</script>' +
      '<img src="x" onerror="window.__pwned = 2;">',
  );
  const body = resultBodies(win)[0];
  assert.ok(!body.querySelector('script'), 'script tags must be stripped');
  const img = body.querySelector('img');
  if (img) {
    assert.ok(!img.hasAttribute('onerror'), 'on* handlers must be stripped');
  }
  assert.ok(!win.__pwned, 'sanitizer must prevent script execution');
  assert.ok(body.textContent.includes('safe'));
  win.close();
}

function testMultiSessionSplitWithHtmlMarkers() {
  const win = makeWebview();
  sendResult(
    win,
    '<h3>Previous Session 1</h3>\n<p>did A</p>\n\n---\n\n' +
      '<h3>Final Session</h3>\n<p>finished B</p>',
  );
  const panels = Array.from(win.document.querySelectorAll('#output > .rc'));
  const headings = panels.map(p => p.querySelector('.rc-h h3').textContent);
  assert.deepStrictEqual(
    headings,
    ['Previous Sessions', 'Result'],
    'BUG: HTML <h3> session markers must split into two panels; got ' +
      headings.join(' -> '),
  );
  assert.ok(panels[0].textContent.includes('did A'));
  assert.ok(!panels[0].textContent.includes('finished B'));
  assert.ok(panels[1].textContent.includes('finished B'));
  assert.ok(
    panels[1].querySelector('.rc-body p'),
    'final session HTML must render as elements',
  );
  win.close();
}

function testMultiSessionSplitExhaustionBannerWithHtmlMarkers() {
  const win = makeWebview();
  sendResult(
    win,
    '<h3>Previous Session 1</h3>\n<p>did A</p>\n\n---\n\n' +
      'Task failed after 2 sub-sessions',
  );
  const panels = Array.from(win.document.querySelectorAll('#output > .rc'));
  const headings = panels.map(p => p.querySelector('.rc-h h3').textContent);
  assert.deepStrictEqual(headings, ['Previous Sessions', 'Result']);
  assert.ok(
    panels[1].textContent.includes('Task failed after 2 sub-sessions'),
  );
  win.close();
}

function testMultiSessionSplitKeepsMarkdownBackCompat() {
  // Old persisted history events still carry Markdown '###' markers.
  const win = makeWebview();
  sendResult(
    win,
    '### Previous Session 1\ndid A\n\n---\n\n### Final Session\nfinished B',
  );
  const panels = Array.from(win.document.querySelectorAll('#output > .rc'));
  const headings = panels.map(p => p.querySelector('.rc-h h3').textContent);
  assert.deepStrictEqual(
    headings,
    ['Previous Sessions', 'Result'],
    'old Markdown session markers must still split; got ' +
      headings.join(' -> '),
  );
  win.close();
}

function main() {
  testHtmlSummaryRendersAsHtml();
  console.log('  ok - HTML summary renders as HTML elements');
  testSummaryIsNotParsedAsMarkdown();
  console.log('  ok - summary is not parsed as Markdown');
  testHtmlSummaryIsSanitized();
  console.log('  ok - HTML summary is sanitized');
  testMultiSessionSplitWithHtmlMarkers();
  console.log('  ok - multi-session split works with <h3> HTML markers');
  testMultiSessionSplitExhaustionBannerWithHtmlMarkers();
  console.log('  ok - exhaustion banner splits with <h3> HTML markers');
  testMultiSessionSplitKeepsMarkdownBackCompat();
  console.log('  ok - Markdown session markers remain back-compatible');
  console.log('resultPanelHtml.test.js: all tests passed');
}

main();
