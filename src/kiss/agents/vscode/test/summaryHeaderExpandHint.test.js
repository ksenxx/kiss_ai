// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: the "(click to expand)" affordance next to the
// ``summary`` label in the header of the agent's collapsed ``summary``
// digest panel.
//
// The summary panel auto-collapses when it renders (see
// ``summaryToolCollapse.test.js``), hiding the adopted event panels
// behind a header click.  Nothing used to tell the user the header is
// clickable, so the panel header must now read
//
//     summary (click to expand)
//
// with the hint (a) rendered in a dedicated ``.tc-summary-hint`` span
// directly after the tool name, (b) visible while the panel is
// collapsed, (c) hidden once the user expands the panel (the label
// would lie), (d) restored on re-collapse, (e) absent from every
// non-summary tool panel, (f) present on the history-replay
// (``task_events``) path exactly like the live path, and (g) excluded
// from the panel Copy button's clipboard payload and from the
// collapsed-header text preview.
//
// The tests exercise the real ``media/main.js`` + ``media/panelCopy.js``
// against the real ``media/chat.html`` (and, where computed styles
// matter, the real ``media/main.css``) in jsdom — the same harness as
// ``summaryToolCollapse.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/summaryHeaderExpandHint.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const HINT = '(click to expand)';

const DESC =
  'The agent read the panel-rendering code, wrote a failing test for ' +
  'the missing header affordance, and then implemented the hint span.';

/**
 * Build a jsdom window running the real chat webview (chat.html +
 * panelCopy.js + main.js).
 */
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
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  // The sourceURL pragma names this eval instance in V8 coverage
  // output so summaryHeaderExpandHint.coverage.js can locate it and
  // gate the summaryhint-coverage region of main.js at 100%.
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=summaryhint-main.js',
  );
  return win;
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Inject the real production stylesheet so computed styles resolve. */
function injectCss(win) {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const styleEl = win.document.createElement('style');
  styleEl.textContent = css;
  win.document.head.appendChild(styleEl);
}

/** Emit *n* distinct tool_call panels (Read path-arg panels). */
function sendToolPanels(win, n) {
  for (let i = 0; i < n; i++) {
    send(win, {type: 'tool_call', name: 'Read', path: '/tmp/f' + i + '.txt'});
    send(win, {type: 'tool_result', name: 'Read', content: 'data ' + i});
  }
}

function output(win) {
  return win.document.getElementById('output');
}

function summaryPanels(win) {
  return Array.from(output(win).querySelectorAll('.tc.tc-summary'));
}

function clickHeader(win, panel) {
  panel
    .querySelector(':scope > .tc-h')
    .dispatchEvent(
      new win.MouseEvent('click', {bubbles: true, cancelable: true}),
    );
}

// ── tests ──────────────────────────────────────────────────────────

function testHintRenderedNextToSummaryLabel() {
  const win = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  assert.ok(p.classList.contains('collapsed'), 'panel auto-collapses');
  const hdr = p.querySelector(':scope > .tc-h');
  const hint = hdr.querySelector(':scope > .tc-summary-hint');
  assert.ok(
    hint,
    'the header must contain a .tc-summary-hint span next to the label',
  );
  assert.strictEqual(
    (hint.textContent || '').trim(),
    HINT,
    'the hint must read exactly "(click to expand)"',
  );
  // The hint sits NEXT TO (after) the "summary" tool name: the
  // header's flattened text reads "summary (click to expand)"
  // (ignoring the chevron glyph addCollapse prepends).
  const flat = (hdr.textContent || '').replace(/\s+/g, ' ');
  assert.ok(
    /summary \(click to expand\)/.test(flat),
    'header text must read "summary (click to expand)" — got: ' + flat,
  );
  win.close();
  console.log('  ok - header renders "summary (click to expand)"');
}

function testHintHiddenWhenExpandedRestoredOnRecollapse() {
  const win = makeWebview();
  injectCss(win);
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 3);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const hint = p.querySelector(':scope > .tc-h > .tc-summary-hint');
  assert.ok(hint, 'hint span exists');
  assert.notStrictEqual(
    win.getComputedStyle(hint).getPropertyValue('display').trim(),
    'none',
    'the hint must be VISIBLE while the summary panel is collapsed',
  );
  clickHeader(win, p);
  assert.ok(!p.classList.contains('collapsed'), 'header click expands');
  assert.strictEqual(
    win.getComputedStyle(hint).getPropertyValue('display').trim(),
    'none',
    'the hint must HIDE once the panel is expanded (it would lie)',
  );
  clickHeader(win, p);
  assert.ok(p.classList.contains('collapsed'), 'second click re-collapses');
  assert.notStrictEqual(
    win.getComputedStyle(hint).getPropertyValue('display').trim(),
    'none',
    'the hint must come back when the panel re-collapses',
  );
  assert.strictEqual(
    p.querySelectorAll(':scope > .tc-h > .tc-summary-hint').length,
    1,
    'toggling must never duplicate the hint span',
  );
  win.close();
  console.log('  ok - hint hides on expand, returns on re-collapse');
}

function testNonSummaryToolPanelsHaveNoHint() {
  const win = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 2);
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    description: 'list',
  });
  send(win, {type: 'tool_result', name: 'Bash', content: 'f0.txt'});
  assert.strictEqual(
    output(win).querySelectorAll('.tc-summary-hint').length,
    0,
    'no non-summary tool panel may carry the expand hint',
  );
  win.close();
  console.log('  ok - non-summary tool panels carry no hint');
}

function testReplayPathRendersHint() {
  const win = makeWebview();
  const events = [{type: 'prompt', text: 'replayed task'}];
  for (let i = 0; i < 3; i++) {
    events.push({type: 'tool_call', name: 'Read', path: '/tmp/r' + i});
    events.push({type: 'tool_result', name: 'Read', content: 'x' + i});
  }
  events.push({type: 'tool_call', name: 'summary', description: DESC});
  events.push({
    type: 'tool_result',
    name: 'summary',
    content: 'Summary recorded.',
  });
  events.push({type: 'result', text: 'done', summary: 'done', success: true});
  send(win, {
    type: 'task_events',
    task: 'replayed task',
    task_id: 7,
    events: events,
  });
  const p = summaryPanels(win)[0];
  assert.ok(p, 'summary panel replays');
  const hint = p.querySelector(':scope > .tc-h > .tc-summary-hint');
  assert.ok(hint, 'the replayed summary panel header must carry the hint');
  assert.strictEqual((hint.textContent || '').trim(), HINT);
  win.close();
  console.log('  ok - history replay (task_events) renders the hint');
}

function testEverySummaryPanelGetsItsOwnHint() {
  const win = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: 'first'});
  send(win, {
    type: 'tool_result',
    name: 'summary',
    content: 'Summary recorded.',
  });
  sendToolPanels(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: 'second'});
  const panels = summaryPanels(win);
  assert.strictEqual(panels.length, 2);
  for (const p of panels) {
    const hints = p.querySelectorAll(':scope > .tc-h > .tc-summary-hint');
    assert.strictEqual(
      hints.length,
      1,
      'each summary panel header carries exactly one hint',
    );
    assert.strictEqual((hints[0].textContent || '').trim(), HINT);
  }
  win.close();
  console.log('  ok - every summary panel gets exactly one hint');
}

function testHintExcludedFromCopyPayload() {
  const win = makeWebview();
  send(win, {type: 'prompt', text: 'go'});
  sendToolPanels(win, 1);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const raw = win.PanelCopy.getRawText(p);
  assert.ok(
    raw.indexOf(HINT) === -1,
    'the Copy button payload must not contain the UI-only hint — got: ' + raw,
  );
  assert.ok(
    raw.indexOf(DESC) !== -1,
    'the Copy button payload must still contain the description',
  );
  win.close();
  console.log('  ok - hint never leaks into the Copy payload');
}

function testHintExcludedFromCollapsePreview() {
  const win = makeWebview();
  sendToolPanels(win, 2);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  // Toggle once each way so collapsePreview() recomputes.
  clickHeader(win, p);
  clickHeader(win, p);
  const prev = p.querySelector(':scope > .tc-h > .collapse-preview');
  assert.ok(prev, 'addCollapse installs the preview span');
  assert.strictEqual(
    (prev.textContent || '').trim(),
    '',
    'the summary header preview stays empty — the hint must not leak ' +
      'into it',
  );
  win.close();
  console.log('  ok - hint never leaks into the collapse preview');
}

function testHintStyledDimmerThanLabel() {
  // The hint is an affordance, not part of the tool name: the real
  // stylesheet must render it dimmer than the orange header label.
  const win = makeWebview();
  injectCss(win);
  sendToolPanels(win, 1);
  send(win, {type: 'tool_call', name: 'summary', description: DESC});
  const p = summaryPanels(win)[0];
  const hdr = p.querySelector(':scope > .tc-h');
  const hint = hdr.querySelector(':scope > .tc-summary-hint');
  const hdrColor = win.getComputedStyle(hdr).getPropertyValue('color');
  const hintColor = win.getComputedStyle(hint).getPropertyValue('color');
  assert.notStrictEqual(
    hintColor.trim(),
    hdrColor.trim(),
    'the hint must not inherit the header label color unchanged',
  );
  // The header uppercases its label (``.tc-h { text-transform:
  // uppercase }``) — the hint must escape that transform so the user
  // sees the literal "(click to expand)", not "(CLICK TO EXPAND)"
  // (adversarial-review finding).
  assert.strictEqual(
    win.getComputedStyle(hint).getPropertyValue('text-transform').trim(),
    'none',
    'the hint must not be uppercased by the header text-transform',
  );
  win.close();
  console.log('  ok - hint is styled distinctly from the header label');
}

function runTests() {
  testHintRenderedNextToSummaryLabel();
  testHintHiddenWhenExpandedRestoredOnRecollapse();
  testNonSummaryToolPanelsHaveNoHint();
  testReplayPathRendersHint();
  testEverySummaryPanelGetsItsOwnHint();
  testHintExcludedFromCopyPayload();
  testHintExcludedFromCollapsePreview();
  testHintStyledDimmerThanLabel();
}

try {
  runTests();
  console.log('\n8 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
