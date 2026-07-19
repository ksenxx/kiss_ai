// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: hovering over the task text (#task-panel-text) in
// the fixed task panel of the chat webview shows a tooltip containing
// the ENTIRE task text, rendered at the SAME font size as the task
// text in the panel.
//
// Locked-in behaviour:
//   * setTaskText stamps the full trimmed task text into the
//     data-tooltip attribute of #task-panel-text (and removes the
//     attribute when the task text is cleared);
//   * the tab-restore path (restoreTab) keeps data-tooltip in sync
//     when switching tabs — no stale tooltip from another tab;
//   * hovering #task-panel-text pops the shared #custom-tooltip with
//     the entire task text and the .task-panel-tooltip class, whose
//     main.css rule pins font-size to var(--vscode-editor-font-size)
//     — the exact declaration #task-panel itself uses, so the tooltip
//     font size always equals the task text font size (both in the
//     VS Code webview and on the remote web app, which serves the
//     same main.css);
//   * hovering any OTHER [data-tooltip] element drops the
//     .task-panel-tooltip class so ordinary tooltips keep --fs-sm;
//   * mouseout hides the tooltip again.
//
// Exercises the real ``media/chat.html`` + ``media/main.js`` in jsdom
// (same harness as taskPanelCollapseButton.test.js).  Run directly:
//
//     node src/kiss/agents/vscode/test/taskPanelTooltip.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// A long multi-line task: the tooltip must carry ALL of it.
const LONG_TASK =
  'Refactor the payment pipeline to support multi-currency ' +
  'settlement.\nStep 1: normalize every ledger entry to minor ' +
  'units.\nStep 2: add an FX-rate snapshot table keyed by ' +
  '(currency, day).\nStep 3: migrate historical rows in batches of ' +
  '10k with checkpoints.\n' +
  'x'.repeat(600) +
  '\nFinally run the full reconciliation suite and attach the report.';

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

/** Real-time sleep — the tooltip uses a 400 ms show delay. */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** Fire a bubbling mouse event of *type* on *el*. */
function mouse(win, el, type) {
  el.dispatchEvent(new win.MouseEvent(type, {bubbles: true}));
}

// ---------------------------------------------------------------------------
// 1. setTaskText stamps (and clears) the full task text as data-tooltip.
// ---------------------------------------------------------------------------
function testSetTaskTextStampsDataTooltip() {
  const {win} = makeWebview();
  send(win, {type: 'setTaskText', text: '  ' + LONG_TASK + '\n'});

  const txt = win.document.getElementById('task-panel-text');
  assert.ok(txt, '#task-panel-text must exist');
  assert.strictEqual(
    txt.getAttribute('data-tooltip'),
    LONG_TASK,
    'data-tooltip must carry the ENTIRE trimmed task text',
  );
  assert.strictEqual(
    txt.getAttribute('data-tooltip'),
    txt.textContent,
    'tooltip text must equal the task text shown in the panel',
  );

  // Clearing the task must also clear the tooltip.
  send(win, {type: 'setTaskText', text: ''});
  assert.strictEqual(
    txt.getAttribute('data-tooltip'),
    null,
    'clearing the task text must remove data-tooltip',
  );
  win.close();
  console.log('  ok - setTaskText stamps/clears data-tooltip');
}

// ---------------------------------------------------------------------------
// 2. Hover shows the entire task text in the shared custom tooltip,
//    with the .task-panel-tooltip font-size marker; mouseout hides it.
// ---------------------------------------------------------------------------
async function testHoverShowsFullTaskTooltip() {
  const {win} = makeWebview();
  send(win, {type: 'setTaskText', text: LONG_TASK});

  const txt = win.document.getElementById('task-panel-text');
  const tip = win.document.getElementById('custom-tooltip');
  assert.ok(tip, 'the shared #custom-tooltip element must exist');
  assert.ok(!tip.classList.contains('visible'), 'tooltip must start hidden');

  mouse(win, txt, 'mouseover');
  await sleep(500); // show delay is 400 ms
  assert.ok(
    tip.classList.contains('visible'),
    'hovering the task text must show the tooltip',
  );
  assert.strictEqual(
    tip.textContent,
    LONG_TASK,
    'the tooltip must contain the ENTIRE task text',
  );
  assert.ok(
    tip.classList.contains('task-panel-tooltip'),
    'the task-text tooltip must carry .task-panel-tooltip so main.css ' +
      'renders it at the task text font size',
  );

  mouse(win, txt, 'mouseout');
  assert.ok(
    !tip.classList.contains('visible'),
    'mouseout must hide the tooltip',
  );
  win.close();
  console.log('  ok - hover shows entire task text; mouseout hides');
}

// ---------------------------------------------------------------------------
// 3. Hovering an ordinary [data-tooltip] control (the task drawer
//    button) keeps its own text and DROPS the .task-panel-tooltip
//    class, so normal tooltips keep the small --fs-sm font.
// ---------------------------------------------------------------------------
async function testOtherTooltipsKeepSmallFont() {
  const {win} = makeWebview();
  send(win, {type: 'setTaskText', text: LONG_TASK});

  const txt = win.document.getElementById('task-panel-text');
  const tip = win.document.getElementById('custom-tooltip');

  // First hover the task text so the class is set...
  mouse(win, txt, 'mouseover');
  await sleep(500);
  assert.ok(tip.classList.contains('task-panel-tooltip'));
  mouse(win, txt, 'mouseout');

  // ...then hover the drawer button: class must be removed.
  const drawerBtn = win.document.getElementById('task-panel-drawer-btn');
  assert.ok(
    drawerBtn.getAttribute('data-tooltip'),
    'drawer button must keep its own data-tooltip',
  );
  mouse(win, drawerBtn, 'mouseover');
  await sleep(500);
  assert.ok(
    tip.classList.contains('visible'),
    'drawer button tooltip must still work',
  );
  assert.strictEqual(
    tip.textContent,
    drawerBtn.getAttribute('data-tooltip'),
    'drawer button keeps its own tooltip text',
  );
  assert.ok(
    !tip.classList.contains('task-panel-tooltip'),
    'non-task tooltips must NOT carry .task-panel-tooltip',
  );
  win.close();
  console.log('  ok - other tooltips keep the small font class-free');
}

// ---------------------------------------------------------------------------
// 4. Tab switch (restoreTab path) keeps data-tooltip in sync: each
//    tab restores ITS OWN task tooltip (sub tab → its description,
//    fresh "+" tab → none, parent → the full task again).
// ---------------------------------------------------------------------------
function testTabRestoreKeepsTooltipInSync() {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const parentId = ready.tabId;

  send(win, {type: 'setTaskText', text: LONG_TASK, tabId: parentId});
  const txt = win.document.getElementById('task-panel-text');
  assert.strictEqual(txt.getAttribute('data-tooltip'), LONG_TASK);

  // Spawn a sub-agent tab (backend replay sequence: new_tab →
  // resumeSession → openSubagentTab) — it opens in the BACKGROUND.
  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'sub-task-1',
    parent_tab_id: parentId,
    taskId: '',
  });
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === 'sub-task-1');
  assert.ok(resume, 'new_tab must make the webview post resumeSession');
  send(win, {
    type: 'openSubagentTab',
    tab_id: resume.tabId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
  });

  // Sub tabs open in the background: click the sub tab in the tab
  // bar to switch to it (saveCurrentTab + restoreTab round trip).
  // Its panel restores its OWN task ('sub 1') — never the parent's.
  const subEl = win.document.querySelector('#tab-list .chat-tab.subagent-tab');
  assert.ok(subEl, 'sub-agent tab element must exist in the tab bar');
  subEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    txt.textContent,
    'sub 1',
    'sub tab restores its own task text',
  );
  assert.strictEqual(
    txt.getAttribute('data-tooltip'),
    'sub 1',
    'sub tab tooltip must be ITS task, not the stale parent task',
  );

  // Open a brand-new empty chat tab (the "+" button): its task panel
  // restores to nothing, covering the tooltip-removal branch.
  const addBtn = win.document.querySelector('#tab-bar .chat-tab-add');
  assert.ok(addBtn, 'the "+" new-chat tab button must exist');
  addBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    txt.textContent.trim(),
    '',
    'a fresh chat tab must have an empty task panel',
  );
  assert.strictEqual(
    txt.getAttribute('data-tooltip'),
    null,
    'a fresh chat tab must not keep any stale tooltip',
  );

  // Switch back to the parent tab via its tab-bar button: restoreTab
  // must restore BOTH the text and the tooltip.
  const tabEls = win.document.querySelectorAll('#tab-list .chat-tab');
  const parentEl = Array.from(tabEls).find(el => {
    return !el.classList.contains('subagent-tab');
  });
  assert.ok(parentEl, 'parent tab element must exist in the tab bar');
  parentEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  assert.strictEqual(
    txt.textContent,
    LONG_TASK,
    'switching back must restore the task text',
  );
  assert.strictEqual(
    txt.getAttribute('data-tooltip'),
    LONG_TASK,
    'switching back must restore the full-task tooltip',
  );
  win.close();
  console.log('  ok - tab restore keeps data-tooltip in sync');
}

// ---------------------------------------------------------------------------
// 5. main.css pins the task tooltip font size to the task panel's own
//    declaration — var(--vscode-editor-font-size) — for BOTH the
//    extension webview and the remote web app (same stylesheet).
// ---------------------------------------------------------------------------
function testCssPinsTooltipFontSizeToTaskPanel() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

  const panel = /#task-panel\s*\{([^}]*)\}/.exec(css);
  assert.ok(panel, '#task-panel rule missing from main.css');
  assert.ok(
    /font-size\s*:\s*var\(--vscode-editor-font-size\)/.test(panel[1]),
    '#task-panel must size its text with var(--vscode-editor-font-size)',
  );

  const tipRule = /#custom-tooltip\.task-panel-tooltip\s*\{([^}]*)\}/.exec(css);
  assert.ok(tipRule, 'main.css must style #custom-tooltip.task-panel-tooltip');
  assert.ok(
    /font-size\s*:\s*var\(--vscode-editor-font-size\)/.test(tipRule[1]),
    'the task tooltip must use the SAME font-size declaration as ' +
      '#task-panel: var(--vscode-editor-font-size)',
  );
  console.log('  ok - main.css pins tooltip font size to the task panel');
}

async function runTests() {
  testSetTaskTextStampsDataTooltip();
  await testHoverShowsFullTaskTooltip();
  await testOtherTooltipsKeepSmallFont();
  testTabRestoreKeepsTooltipInSync();
  testCssPinsTooltipFontSizeToTaskPanel();
}

runTests()
  .then(() => {
    console.log('\n5 passed, 0 failed');
    process.exit(0);
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
