// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: the fixed task panel at the top of the chat webview
// exposes a labelled button — not a bare chevron icon — for collapsing
// and uncollapsing the chat panels of the visible task.
//
// Locked-in behaviour:
//   * the button (#task-panel-collapse-btn) lives inside #task-panel
//     and carries a visible text label (#task-panel-collapse-label);
//   * while panels are collapsed (the default) the label reads
//     "Uncollapse Chats" and aria-expanded is "false";
//   * clicking flips the state: label becomes "Collapse Chats",
//     aria-expanded becomes "true", and hidden panels are revealed;
//   * clicking again re-collapses: non-result panels get .chv-hidden
//     and the label returns to "Uncollapse Chats";
//   * the legacy icon-only #task-panel-chevron id is gone from both
//     the markup and the stylesheet, and main.css styles the new
//     button as a real button (cursor: pointer).
//
// Exercises the real ``media/chat.html`` + ``media/main.js`` in jsdom
// (same harness as bashHeaderCyan.test.js).  Run directly with node:
//
//     node src/kiss/agents/vscode/test/taskPanelCollapseButton.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the real chat webview (chat.html +
 * panelCopy.js + main.js), mirroring the production extension.
 */
function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  // Strip the production <script> tags — we eval the source files
  // ourselves below so they pick up the jsdom globals.
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

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function testButtonExistsWithDefaultUncollapseLabel() {
  const {win} = makeWebview();
  send(win, {type: 'setTaskText', text: 'demo task'});

  const btn = win.document.getElementById('task-panel-collapse-btn');
  assert.ok(btn, '#task-panel must contain #task-panel-collapse-btn');
  assert.strictEqual(
    btn.tagName,
    'BUTTON',
    'the collapse control must be a real <button>',
  );
  assert.ok(
    btn.closest('#task-panel'),
    'the collapse button must live inside the fixed #task-panel',
  );
  const label = win.document.getElementById('task-panel-collapse-label');
  assert.ok(label, 'button must carry a #task-panel-collapse-label span');
  assert.strictEqual(
    label.textContent,
    'Uncollapse Chats',
    'default (collapsed) label must read "Uncollapse Chats"',
  );
  assert.strictEqual(
    btn.getAttribute('aria-expanded'),
    'false',
    'default aria-expanded must be "false"',
  );
  // The legacy icon-only chevron id must be gone from the markup.
  assert.strictEqual(
    win.document.getElementById('task-panel-chevron'),
    null,
    'legacy #task-panel-chevron must no longer exist in chat.html',
  );
  win.close();
  console.log('  ok - button exists with default "Uncollapse Chats" label');
}

function testClickTogglesLabelAndPanelVisibility() {
  const {win} = makeWebview();
  send(win, {type: 'setTaskText', text: 'demo task'});
  // Render one collapsible tool panel and one result panel for the
  // (idle) current task — exactly what the extension streams.
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la',
    description: 'list files',
  });
  send(win, {type: 'result', content: 'all done'});

  const btn = win.document.getElementById('task-panel-collapse-btn');
  const label = win.document.getElementById('task-panel-collapse-label');
  assert.ok(btn && label, 'collapse button + label must exist');

  const toolPanel = win.document.querySelector('#output .tc.collapsible');
  assert.ok(toolPanel, 'expected a collapsible tool_call panel');
  const resultPanel = win.document.querySelector('#output .rc');
  assert.ok(resultPanel, 'expected a result panel');
  // Panels start out collapsed (default chevron state = collapsed).
  assert.ok(
    toolPanel.classList.contains('chv-hidden'),
    'tool panel must start chevron-hidden in the default collapsed state',
  );

  // Click 1: uncollapse — label flips to the collapse action and no
  // panel may remain chevron-hidden.
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    label.textContent,
    'Collapse Chats',
    'after uncollapsing, the label must offer "Collapse Chats"',
  );
  assert.strictEqual(btn.getAttribute('aria-expanded'), 'true');
  assert.ok(
    btn.classList.contains('expanded'),
    'button must carry .expanded so the icon rotates',
  );
  assert.ok(
    !toolPanel.classList.contains('chv-hidden'),
    'tool panel must be revealed after uncollapsing',
  );

  // Click 2: collapse again — label flips back and every non-result
  // panel of the idle task is hidden while the result stays visible.
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    label.textContent,
    'Uncollapse Chats',
    'after collapsing, the label must offer "Uncollapse Chats"',
  );
  assert.strictEqual(btn.getAttribute('aria-expanded'), 'false');
  assert.ok(!btn.classList.contains('expanded'));
  assert.ok(
    toolPanel.classList.contains('chv-hidden'),
    'collapsing must hide the tool panel again via .chv-hidden',
  );
  assert.ok(
    !resultPanel.classList.contains('chv-hidden'),
    'result panels must never be chevron-hidden',
  );
  win.close();
  console.log('  ok - click toggles label, aria state and panel visibility');
}

function testCssStylesButtonAndDropsLegacyChevron() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  assert.ok(
    css.includes('#task-panel-collapse-btn'),
    'main.css must style #task-panel-collapse-btn',
  );
  assert.ok(
    !css.includes('#task-panel-chevron'),
    'main.css must not keep dead #task-panel-chevron rules',
  );
  // The button block must at least behave like a button (pointer) and
  // keep the rotating-icon affordance for the expanded state.
  const blockRe = /#task-panel-collapse-btn[^{,]*\{([^}]*)\}/g;
  let m;
  let pointer = false;
  while ((m = blockRe.exec(css)) !== null) {
    if (/cursor\s*:\s*pointer/.test(m[1])) pointer = true;
  }
  assert.ok(pointer, '#task-panel-collapse-btn must set cursor: pointer');
  assert.ok(
    /#task-panel-collapse-btn\.expanded\s+svg/.test(css),
    'main.css must rotate the button icon in the .expanded state',
  );
  console.log('  ok - main.css styles the button and drops the old chevron');
}

function runTests() {
  testButtonExistsWithDefaultUncollapseLabel();
  testClickTogglesLabelAndPanelVisibility();
  testCssStylesButtonAndDropsLegacyChevron();
}

try {
  runTests();
  console.log('\n3 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
