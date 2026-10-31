// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send} = require('./simplify2_harness');

const MD_DESC =
  'Recap of the last steps:\n' +
  '- Read the **main** entry points\n' +
  '- Ran `uv run check` to lint\n' +
  '- Fixed the bug in the parser\n';

function makePage(withMarked) {
  return makeChatPage({extraScripts: withMarked ? ['marked.min.js'] : []}).win;
}

function summaryDesc(win) {
  const p = win.document.querySelector('#output .tc.tc-summary');
  assert.ok(p, 'a .tc-summary panel must render');
  const desc = p.querySelector(':scope > .tc-summary-desc');
  assert.ok(desc, '.tc-summary-desc must be a direct child of the panel');
  return desc;
}

function testMarkdownDescriptionRendersFormatted() {
  const win = makePage(true);
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
  const win = makePage(true);
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
  // A script inserted through innerHTML never runs and jsdom loads no
  // image, so "did the payload run" cannot fail here: the markup itself
  // must come out without any inline handler.
  assert.strictEqual(
    desc.querySelector('[onerror]'),
    null,
    'onerror handlers must be stripped by the sanitizer',
  );
  win.close();
  console.log('  ok - summary description HTML is sanitized');
}

function testPlainTextFallbackWithoutMarked() {
  const win = makePage(false);
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
  const win = makePage(true);
  send(win, {type: 'prompt', text: 'go'});
  send(win, {type: 'tool_call', name: 'summary'});
  const desc = summaryDesc(win);
  assert.strictEqual(desc.textContent, '', 'empty description stays empty');
  assert.strictEqual(desc.dataset.rawText, '');
  win.close();
  console.log('  ok - empty description renders empty');
}

function testFencedCodeBlockRenders() {
  const win = makePage(true);
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
  const win = makePage(true);
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
  assert.strictEqual(
    sub.children.length,
    2,
    'the prompt and the Read panel must nest',
  );
  win.close();
  console.log('  ok - collapse + nesting behavior unchanged with markdown');
}

function testReplayRendersFormattedMarkdown() {
  const win = makePage(true);
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
  const win = makePage(true);
  send(win, {type: 'prompt', text: 'go'});
  send(win, {type: 'tool_call', name: 'summary', description: MD_DESC});
  const desc = summaryDesc(win);
  assert.ok(win.PanelCopy, 'PanelCopy must be available in the page');
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
