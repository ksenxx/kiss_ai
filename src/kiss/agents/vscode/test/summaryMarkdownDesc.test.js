// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const MD_DESC =
  'Recap of the last steps:\n' +
  '- Read the **main** entry points\n' +
  '- Ran `uv run check` to lint\n' +
  '- Fixed the bug in the parser\n';

function makeWebview(withMarked) {
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
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  if (withMarked) {
    win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  }
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function summaryDesc(win) {
  const p = win.document.querySelector('#output .tc.tc-summary');
  assert.ok(p, 'a .tc-summary panel must render');
  const desc = p.querySelector(':scope > .tc-summary-desc');
  assert.ok(desc, '.tc-summary-desc must be a direct child of the panel');
  return desc;
}

function testMarkdownDescriptionRendersFormatted() {
  const win = makeWebview(true);
  send(win, {type: 'prompt', text: 'go'});
  send(win, {type: 'tool_call', name: 'summary', description: MD_DESC});
  const desc = summaryDesc(win);
  assert.ok(
    desc.classList.contains('md-body'),
    'desc must get the md-body class so markdown CSS applies',
  );
  const lis = desc.querySelectorAll('ul li');
  assert.strictEqual(lis.length, 3, 'the 3 bullets must render as <li>');
  const strong = desc.querySelector('strong');
  assert.ok(strong, '**main** must render as <strong>');
  assert.strictEqual(strong.textContent, 'main');
  const code = desc.querySelector('code');
  assert.ok(code, '`uv run check` must render as <code>');
  assert.strictEqual(code.textContent, 'uv run check');
  assert.strictEqual(
    desc.dataset.rawText,
    MD_DESC,
    'dataset.rawText must keep the RAW markdown for copy-as-markdown',
  );
  assert.ok(
    desc.textContent.includes('Fixed the bug in the parser'),
    'the full description content must be present',
  );
  win.close();
  console.log('  ok - markdown description renders formatted (ul/strong/code)');
}

function testDescriptionHtmlIsSanitized() {
  const win = makeWebview(true);
  send(win, {type: 'prompt', text: 'go'});
  send(win, {
    type: 'tool_call',
    name: 'summary',
    description:
      'bad <script>window.__pwned = 1;</script> ' +
      '<img src="x" onerror="window.__pwned = 2"> end',
  });
  const desc = summaryDesc(win);
  assert.strictEqual(
    desc.querySelector('script'),
    null,
    'script tags must be stripped by the sanitizer',
  );
  const img = desc.querySelector('img');
  if (img) {
    assert.strictEqual(
      img.getAttribute('onerror'),
      null,
      'onerror handlers must be stripped by the sanitizer',
    );
  }
  assert.strictEqual(win.__pwned, undefined, 'no injected code may run');
  win.close();
  console.log('  ok - summary description HTML is sanitized');
}

function testPlainTextFallbackWithoutMarked() {
  const win = makeWebview(false);
  send(win, {type: 'prompt', text: 'go'});
  send(win, {type: 'tool_call', name: 'summary', description: MD_DESC});
  const desc = summaryDesc(win);
  assert.strictEqual(
    desc.textContent,
    MD_DESC,
    'without marked the raw text must render untouched',
  );
  assert.strictEqual(desc.dataset.rawText, MD_DESC);
  win.close();
  console.log('  ok - plain-text fallback when marked is unavailable');
}

function testEmptyDescription() {
  const win = makeWebview(true);
  send(win, {type: 'prompt', text: 'go'});
  send(win, {type: 'tool_call', name: 'summary'});
  const desc = summaryDesc(win);
  assert.strictEqual(desc.textContent, '', 'empty description stays empty');
  assert.strictEqual(desc.dataset.rawText, '');
  win.close();
  console.log('  ok - empty description renders empty');
}

function testFencedCodeBlockRenders() {
  const win = makeWebview(true);
  send(win, {type: 'prompt', text: 'go'});
  send(win, {
    type: 'tool_call',
    name: 'summary',
    description: 'Steps:\n\n```python\nprint("hi")\n```\n',
  });
  const desc = summaryDesc(win);
  const pre = desc.querySelector('pre code');
  assert.ok(pre, 'fenced code blocks must render as <pre><code>');
  assert.ok(pre.textContent.includes('print("hi")'));
  win.close();
  console.log('  ok - fenced code block renders as <pre><code>');
}

function testPanelStillCollapsesWithMarkdown() {
  const win = makeWebview(true);
  send(win, {type: 'prompt', text: 'go'});
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/a.txt'});
  send(win, {type: 'tool_result', name: 'Read', content: 'data'});
  send(win, {type: 'tool_call', name: 'summary', description: MD_DESC});
  const p = win.document.querySelector('#output .tc.tc-summary');
  assert.ok(
    p.classList.contains('collapsed'),
    'summary panel must still auto-collapse',
  );
  const sub = p.querySelector(':scope > .summary-sub');
  assert.ok(sub, 'summary-sub nesting must still work');
  assert.strictEqual(sub.children.length, 1, 'the Read panel must nest');
  win.close();
  console.log('  ok - collapse + nesting behavior unchanged with markdown');
}

function testReplayRendersFormattedMarkdown() {
  const win = makeWebview(true);
  const events = [{type: 'prompt', text: 'replayed task'}];
  events.push({type: 'tool_call', name: 'Read', path: '/tmp/r0'});
  events.push({type: 'tool_result', name: 'Read', content: 'x0'});
  events.push({type: 'tool_call', name: 'summary', description: MD_DESC});
  events.push({
    type: 'tool_result',
    name: 'summary',
    content: 'Summary recorded.',
  });
  send(win, {
    type: 'task_events',
    task: 'replayed task',
    task_id: 42,
    events: events,
  });
  const desc = summaryDesc(win);
  assert.ok(
    desc.classList.contains('md-body'),
    'replayed summary desc must also render formatted markdown',
  );
  assert.strictEqual(desc.querySelectorAll('ul li').length, 3);
  assert.strictEqual(desc.dataset.rawText, MD_DESC);
  win.close();
  console.log('  ok - replay via task_events renders formatted markdown');
}

function testCopyReturnsRawMarkdown() {
  const win = makeWebview(true);
  send(win, {type: 'prompt', text: 'go'});
  send(win, {type: 'tool_call', name: 'summary', description: MD_DESC});
  const desc = summaryDesc(win);
  assert.ok(win.PanelCopy, 'PanelCopy must be available in the webview');
  assert.strictEqual(
    win.PanelCopy.getRawText(desc),
    MD_DESC,
    'copying the formatted desc must yield the RAW markdown',
  );
  win.close();
  console.log('  ok - PanelCopy.getRawText returns the raw markdown');
}

function runTests() {
  testMarkdownDescriptionRendersFormatted();
  testDescriptionHtmlIsSanitized();
  testPlainTextFallbackWithoutMarked();
  testEmptyDescription();
  testFencedCodeBlockRenders();
  testPanelStillCollapsesWithMarkdown();
  testReplayRendersFormattedMarkdown();
  testCopyReturnsRawMarkdown();
}

try {
  runTests();
  console.log('\n8 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
