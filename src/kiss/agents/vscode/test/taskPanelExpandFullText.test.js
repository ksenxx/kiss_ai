// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: the pinned task panel (#task-panel) in the chat
// webview after the removal of the "Collapse/Uncollapse Chats" button.
//
// Features under test:
//
//   1. The "Collapse/Uncollapse Chats" button (#task-panel-collapse-btn
//      + #task-panel-collapse-label) is GONE — from the DOM of both the
//      VS Code extension webview and the remote web app.
//   2. Clicking the "Expand task panel" toggle on the collapsed task
//      drawer expands the panel so it shows the ENTIRE task text: the
//      text wraps (pre-wrap, no ellipsis) and its height limit is
//      viewport-relative (max-height in vh <= 100vh) so the panel
//      grows with the task text while always remaining within the
//      chat webview — overlong tasks scroll inside the panel
//      (overflow-y auto) instead of pushing the chat off screen.
//      The old fixed three-line clamp
//      (calc(var(--vscode-editor-font-size) * 1.6 * 3)) is gone.
//   3. The task-panel-driven collapse pass (applyChevronState) still
//      works without the button: at task end it tucks finished
//      panels away (.chv-hidden) while exempting result panels,
//      ``summary`` digests, panels adopted inside a summary, and
//      panels of the running task — and it still closes a hidden
//      run_parallel fan-out's sub-agent tabs.
//
// This drives the production chat.html + panelCopy.js + main.js in
// jsdom with the REAL main.css (and remote-codex.css for remote mode)
// attached — jsdom resolves the stylesheet cascade for
// getComputedStyle, so the clamping assertions exercise the real CSS.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/taskPanelExpandFullText.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// A task text that is far taller than the old three-line clamp.
const LONG_TASK = Array.from(
  {length: 12},
  (_, i) =>
    `step ${i + 1}: a long requirement line that wraps and continues ` +
    'with plenty of detail about what the agent must do',
).join('\n');

/**
 * Build a jsdom window running the production chat webview with the
 * real stylesheets attached.
 *
 * @param {object} [opts]
 * @param {boolean} [opts.remote=false] add class="remote-chat" to body
 *     and layer remote-codex.css over main.css (the web app cascade).
 * @returns {{win: object, posted: Array}}
 */
function makeWebview(opts) {
  const {remote = false} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) html = html.replace('<body', '<body class="remote-chat"');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  // Attach the REAL stylesheet cascade so getComputedStyle resolves
  // the task-panel rules exactly as a browser would.
  const style = win.document.createElement('style');
  style.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(style);
  if (remote) {
    const remoteStyle = win.document.createElement('style');
    remoteStyle.textContent = fs.readFileSync(
      path.join(MEDIA, 'remote-codex.css'),
      'utf8',
    );
    win.document.head.appendChild(remoteStyle);
  }

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
  // The sourceURL pragma names this eval instance in V8 coverage
  // output so taskPanelExpandFullText.coverage.js can locate main.js
  // and enforce 100% line coverage of the chevron-coverage region.
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=taskpanel-main.js',
  );
  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Click element *id* like a user would. */
function click(win, id) {
  const el = win.document.getElementById(id);
  assert.ok(el, `element #${id} must exist`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** getComputedStyle shorthand. */
function cs(win, id) {
  return win.getComputedStyle(win.document.getElementById(id));
}

/** Show the pinned task panel with *task*, as the daemon replay does. */
function showTaskPanel(win, posted, task) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  send(win, {
    type: 'task_events',
    events: [],
    task: task,
    tabId: ready.tabId,
    chat_id: 'chat-taskpanel',
  });
  assert.ok(
    win.document.getElementById('task-panel').classList.contains('visible'),
    'task panel must be visible after a task replay',
  );
  return ready.tabId;
}

/**
 * Assert the expanded task panel shows the ENTIRE task text while
 * remaining within the chat webview: wrapped text, no ellipsis, a
 * viewport-relative height bound (vh) and inner scrolling as the
 * within-the-webview fallback for overlong tasks.
 */
function assertFullTextPanel(win, why) {
  const textCs = cs(win, 'task-panel-text');
  assert.strictEqual(
    textCs.whiteSpace,
    'pre-wrap',
    `task text must wrap so every line is shown (${why})`,
  );
  assert.notStrictEqual(
    textCs.textOverflow,
    'ellipsis',
    `the expanded task text must not be ellipsized (${why})`,
  );
  assert.strictEqual(
    textCs.overflowY,
    'auto',
    'an overlong task must scroll INSIDE the panel so the panel ' +
      `never grows beyond the webview (${why})`,
  );
  const maxHeight = textCs.maxHeight;
  const m = /^(\d+(?:\.\d+)?)vh$/.exec(maxHeight);
  assert.ok(
    m,
    'the expanded task panel height bound must be viewport-relative ' +
      `(vh) so the panel shows the whole task yet stays within the ` +
      `chat webview — got "${maxHeight}" (${why})`,
  );
  assert.ok(
    parseFloat(m[1]) > 0 && parseFloat(m[1]) <= 100,
    `the vh bound must keep the panel within the webview (${why})`,
  );
  assert.ok(
    !maxHeight.includes('calc('),
    `the old fixed three-line clamp must be gone (${why})`,
  );
}

// ── 1. The Collapse/Uncollapse Chats button is gone ─────────────────
function testCollapseChatsButtonGone(remote) {
  const {win, posted} = makeWebview({remote});
  showTaskPanel(win, posted, LONG_TASK);
  const d = win.document;
  assert.strictEqual(
    d.getElementById('task-panel-collapse-btn'),
    null,
    `#task-panel-collapse-btn must not exist (remote=${remote})`,
  );
  assert.strictEqual(
    d.getElementById('task-panel-collapse-label'),
    null,
    `#task-panel-collapse-label must not exist (remote=${remote})`,
  );
  assert.ok(
    !/(Uncollapse Chats|Collapse Chats)/.test(
      d.getElementById('task-panel').textContent,
    ),
    `no Collapse/Uncollapse Chats text may remain (remote=${remote})`,
  );
  // The two surviving task-panel controls are untouched.
  assert.ok(
    d.getElementById('task-panel-drawer-btn'),
    'the drawer toggle must survive the removal',
  );
  assert.ok(
    d.getElementById('task-panel-copy'),
    'the copy-task button must survive the removal',
  );
  win.close();
}

// ── 2. "Expand task panel" reveals the ENTIRE task text ─────────────
function testExpandTaskPanelShowsEntireTask(remote) {
  const {win, posted} = makeWebview({remote});
  showTaskPanel(win, posted, LONG_TASK);
  const d = win.document;
  const panel = d.getElementById('task-panel');
  const btn = d.getElementById('task-panel-drawer-btn');

  // Collapse the drawer: the slim bar clamps the task to one line and
  // the toggle now offers "Expand task panel".
  click(win, 'task-panel-drawer-btn');
  assert.ok(
    panel.classList.contains('drawer-collapsed'),
    'clicking the toggle must collapse the task drawer',
  );
  assert.strictEqual(
    btn.getAttribute('aria-label'),
    'Expand task panel',
    'the collapsed drawer toggle must offer "Expand task panel"',
  );
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'nowrap',
    'collapsed task drawer must clamp the task text to one line',
  );

  // Click "Expand task panel": the panel grows to show the WHOLE task.
  click(win, 'task-panel-drawer-btn');
  assert.ok(
    !panel.classList.contains('drawer-collapsed'),
    '"Expand task panel" must expand the drawer',
  );
  assert.strictEqual(
    btn.getAttribute('aria-label'),
    'Collapse task panel',
    'the expanded drawer toggle must offer "Collapse task panel"',
  );
  assert.strictEqual(
    d.getElementById('task-panel-text').textContent,
    LONG_TASK,
    'the expanded panel must contain the entire task text',
  );
  assertFullTextPanel(win, `after Expand task panel, remote=${remote}`);
  win.close();
}

// ── 3. The default (expanded) panel already shows the full task ─────
function testDefaultExpandedPanelShowsEntireTask() {
  const {win, posted} = makeWebview();
  showTaskPanel(win, posted, LONG_TASK);
  assert.strictEqual(
    win.document.getElementById('task-panel-text').textContent,
    LONG_TASK,
    'the default expanded panel must hold the entire task text',
  );
  assertFullTextPanel(win, 'default expanded state');
  win.close();
}

// ── 4. The chevron collapse pass still works without the button ─────
function testChevronPassWorksWithoutButton() {
  const {win, posted} = makeWebview();
  const d = win.document;
  const ready = posted.find(m => m.type === 'ready');
  const parentId = ready.tabId;
  win._demoApi.hideWelcome();
  send(win, {type: 'status', running: true, tabId: parentId, startTs: 1});
  send(win, {type: 'setTaskText', text: 'live task', tabId: parentId});

  // Plain tool panels + a summary digest that adopts them.
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: parentId});
  send(win, {
    type: 'tool_result',
    name: 'Bash',
    content: 'file1\nfile2',
    tabId: parentId,
  });
  send(win, {
    type: 'tool_call',
    name: 'summary',
    description: 'digest of the run so far',
    tabId: parentId,
  });
  send(win, {type: 'tool_call', name: 'Read', path: '/tmp/a', tabId: parentId});
  // A run_parallel fan-out with one live sub-agent tab.
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 1'])},
  });
  send(win, {
    type: 'new_tab',
    task_id: 'sub-task-1',
    parent_tab_id: parentId,
    taskId: '',
  });
  const resume = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 'sub-task-1',
  );
  assert.ok(resume, 'the spawned sub-agent must open its own tab');
  send(win, {
    type: 'openSubagentTab',
    tab_id: resume.tabId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 'sub-task-1',
    taskIndex: 0,
  });
  send(win, {
    type: 'tool_result',
    name: 'run_parallel',
    content: 'done',
    tabId: parentId,
  });

  const O = d.getElementById('output');
  const rpPanel = O.querySelector('.tc-run-parallel');
  assert.ok(rpPanel, 'run_parallel panel must render');
  const summaryPanel = O.querySelector('.tc-summary');
  assert.ok(summaryPanel, 'summary panel must render');
  const adopted = summaryPanel.querySelector('.summary-sub .collapsible');
  assert.ok(adopted, 'the summary must adopt the earlier panels');

  // While RUNNING, the streaming consistency pass leaves the running
  // task's panels visible (no chv-hidden).
  assert.ok(
    !rpPanel.classList.contains('chv-hidden'),
    'running-task panels must not be tucked away',
  );

  // Task end: the collapse pass tucks finished panels away.
  send(win, {
    type: 'result',
    tabId: parentId,
    summary: 'done',
    success: true,
  });
  send(win, {type: 'status', running: false, tabId: parentId});
  // A trailing post-task event (the daemon streams usage_info after
  // the result) re-runs the consistency pass with isRunning=false.
  send(win, {type: 'usage_info', tabId: parentId});

  const rc = O.querySelector('.rc');
  assert.ok(rc, 'the result panel must render');
  assert.ok(
    !rc.classList.contains('chv-hidden'),
    'the result panel must stay visible',
  );
  assert.ok(
    !summaryPanel.classList.contains('chv-hidden') &&
      summaryPanel.classList.contains('collapsed'),
    'the summary digest must stay visible in its collapsed state',
  );
  assert.ok(
    !adopted.classList.contains('chv-hidden'),
    'panels adopted inside the summary must not be chv-hidden',
  );
  const readPanel = Array.from(O.querySelectorAll('.tc')).find(p =>
    (p.textContent || '').includes('/tmp/a'),
  );
  assert.ok(readPanel, 'the Read tool panel must exist');
  assert.ok(
    readPanel.classList.contains('chv-hidden'),
    'plain finished panels must be tucked away',
  );
  assert.ok(
    rpPanel.classList.contains('chv-hidden') &&
      rpPanel.classList.contains('collapsed'),
    'the finished run_parallel panel must be hidden AND collapsed',
  );
  assert.strictEqual(
    d.querySelectorAll('.tab.subagent-tab, .tab[data-subagent="1"]').length +
      Array.from(d.querySelectorAll('#tab-bar .tab, .tabs .tab')).filter(t =>
        (t.textContent || '').includes('sub 1'),
      ).length,
    0,
    'hiding the run_parallel panel must close its sub-agent tabs',
  );

  // Scrolling up to an ADJACENT (previously completed) task replays it
  // into an .adjacent-task container; the pass targets only THAT
  // task's panels and leaves the other tasks' panels untouched.
  send(win, {
    type: 'adjacent_task_events',
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Older task'},
      {type: 'tool_call', name: 'Bash', command: 'echo old'},
      {type: 'tool_result', name: 'Bash', content: 'old'},
    ],
  });
  const adjacent = O.querySelector('.adjacent-task[data-task="Older task"]');
  assert.ok(adjacent, 'the adjacent task container must render');
  const adjPanel = adjacent.querySelector('.collapsible:not(.rc)');
  assert.ok(adjPanel, 'the adjacent task must replay its tool panel');
  assert.ok(
    adjPanel.classList.contains('chv-hidden'),
    "the adjacent task's finished panels must be tucked away too",
  );
  win.close();
}

function runTests() {
  const tests = [
    () => testCollapseChatsButtonGone(false),
    () => testCollapseChatsButtonGone(true),
    () => testExpandTaskPanelShowsEntireTask(false),
    () => testExpandTaskPanelShowsEntireTask(true),
    testDefaultExpandedPanelShowsEntireTask,
    testChevronPassWorksWithoutButton,
  ];
  const names = [
    'testCollapseChatsButtonGone(vscode)',
    'testCollapseChatsButtonGone(remote)',
    'testExpandTaskPanelShowsEntireTask(vscode)',
    'testExpandTaskPanelShowsEntireTask(remote)',
    'testDefaultExpandedPanelShowsEntireTask',
    'testChevronPassWorksWithoutButton',
  ];
  for (let i = 0; i < tests.length; i++) {
    tests[i]();
    console.log('PASS', names[i]);
  }
}

try {
  runTests();
  console.log('\nAll tests passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
