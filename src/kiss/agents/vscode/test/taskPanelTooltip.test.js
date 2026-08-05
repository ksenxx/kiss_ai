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

const LONG_TASK =
  'Refactor the payment pipeline to support multi-currency ' +
  'settlement.\nStep 1: normalize every ledger entry to minor ' +
  'units.\nStep 2: add an FX-rate snapshot table keyed by ' +
  '(currency, day).\nStep 3: migrate historical rows in batches of ' +
  '10k with checkpoints.\n' +
  'x'.repeat(600) +
  '\nFinally run the full reconciliation suite and attach the report.';

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
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function mouse(win, el, type) {
  el.dispatchEvent(new win.MouseEvent(type, {bubbles: true}));
}

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

  send(win, {type: 'setTaskText', text: ''});
  assert.strictEqual(
    txt.getAttribute('data-tooltip'),
    null,
    'clearing the task text must remove data-tooltip',
  );
  win.close();
  console.log('  ok - setTaskText stamps/clears data-tooltip');
}

async function testHoverShowsFullTaskTooltip() {
  const {win} = makeWebview();
  send(win, {type: 'setTaskText', text: LONG_TASK});

  const txt = win.document.getElementById('task-panel-text');
  const tip = win.document.getElementById('custom-tooltip');
  assert.ok(tip, 'the shared #custom-tooltip element must exist');
  assert.ok(!tip.classList.contains('visible'), 'tooltip must start hidden');

  mouse(win, txt, 'mouseover');
  await sleep(500);
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

async function testOtherTooltipsKeepSmallFont() {
  const {win} = makeWebview();
  send(win, {type: 'setTaskText', text: LONG_TASK});

  const txt = win.document.getElementById('task-panel-text');
  const tip = win.document.getElementById('custom-tooltip');

  mouse(win, txt, 'mouseover');
  await sleep(500);
  assert.ok(tip.classList.contains('task-panel-tooltip'));
  mouse(win, txt, 'mouseout');

  const drawerBtn = win.document.getElementById('task-panel-drawer-btn');
  assert.strictEqual(
    drawerBtn.getAttribute('data-tooltip'),
    null,
    'drawer button must NOT have a data-tooltip',
  );
  mouse(win, drawerBtn, 'mouseover');
  await sleep(500);
  assert.ok(
    !tip.classList.contains('visible'),
    'hovering the drawer button must not show a tooltip',
  );

  const modelBtn = win.document.getElementById('model-btn');
  assert.ok(
    modelBtn.getAttribute('data-tooltip'),
    'model button must keep its own data-tooltip',
  );
  mouse(win, modelBtn, 'mouseover');
  await sleep(500);
  assert.ok(
    tip.classList.contains('visible'),
    'model button tooltip must still work',
  );
  assert.strictEqual(
    tip.textContent,
    modelBtn.getAttribute('data-tooltip'),
    'model button keeps its own tooltip text',
  );
  assert.ok(
    !tip.classList.contains('task-panel-tooltip'),
    'non-task tooltips must NOT carry .task-panel-tooltip',
  );
  win.close();
  console.log('  ok - other tooltips keep the small font class-free');
}

function testTabRestoreKeepsTooltipInSync() {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const parentId = ready.tabId;

  send(win, {type: 'setTaskText', text: LONG_TASK, tabId: parentId});
  const txt = win.document.getElementById('task-panel-text');
  assert.strictEqual(txt.getAttribute('data-tooltip'), LONG_TASK);

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
