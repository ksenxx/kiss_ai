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

const LONG_TASK = Array.from(
  {length: 12},
  (_, i) =>
    `step ${i + 1}: a long requirement line that wraps and continues ` +
    'with plenty of detail about what the agent must do',
).join('\n');

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=taskpanel-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(win, id) {
  const el = win.document.getElementById(id);
  assert.ok(el, `element #${id} must exist`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function cs(win, selector) {
  const el = win.document.querySelector(selector);
  assert.ok(el, `element ${selector} must exist`);
  return win.getComputedStyle(el);
}

/** Whether *el* and every ancestor up to #output is displayed. */
function isDisplayed(win, el) {
  for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
    if (win.getComputedStyle(n).getPropertyValue('display').trim() === 'none')
      return false;
    if (n.id === 'output') break;
  }
  return true;
}

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
  const panel = win.document.querySelector('#output .task-panel');
  assert.ok(panel, 'the transcript must open with the task panel');
  return ready.tabId;
}

function assertFullTextPanel(win, why) {
  const textCs = cs(win, '#output .task-panel .task-panel-text');
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

function testTaskPanelOpensTheTranscript(remote) {
  const {win, posted} = makeWebview({remote});
  showTaskPanel(win, posted, LONG_TASK);
  const d = win.document;
  const output = d.getElementById('output');
  const panel = output.querySelector('.task-panel');
  assert.strictEqual(
    output.firstElementChild,
    panel,
    `the task panel is the first panel of the transcript (remote=${remote})`,
  );
  assert.ok(
    panel.classList.contains('collapsible') &&
      !panel.classList.contains('collapsed'),
    `the task panel is a regular, open event panel (remote=${remote})`,
  );
  assert.strictEqual(
    panel.querySelector('.task-panel-h').textContent.includes('Task'),
    true,
    'the panel header names it as the task',
  );
  assert.strictEqual(
    panel.querySelector('.task-panel-text').textContent,
    LONG_TASK,
    'the panel must contain the entire task text',
  );
  assert.ok(
    panel.querySelector(':scope > .panel-copy-btn'),
    'the task panel carries the copy button every event panel has',
  );
  assert.strictEqual(
    d.getElementById('task-panel'),
    null,
    `no fixed task panel remains above the transcript (remote=${remote})`,
  );
  assertFullTextPanel(win, `transcript task panel, remote=${remote}`);
  win.close();
}

// A click on the header folds the panel like any other event panel:
// the text is hidden behind a one-line preview, and the whole task is
// one click away again.
function testTaskPanelFoldsLikeAnyPanel() {
  const {win, posted} = makeWebview();
  showTaskPanel(win, posted, LONG_TASK);
  const panel = win.document.querySelector('#output .task-panel');
  const header = panel.querySelector('.collapse-header');
  header.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(panel.classList.contains('collapsed'), 'a header click folds');
  assert.strictEqual(
    cs(win, '#output .task-panel .task-panel-text').display,
    'none',
    'the folded panel hides the task text',
  );
  assert.ok(
    panel.querySelector('.collapse-preview').textContent.startsWith('step 1'),
    'the folded panel previews the task text',
  );
  header.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(!panel.classList.contains('collapsed'), 'a second click opens');
  assert.strictEqual(
    panel.querySelector('.task-panel-text').textContent,
    LONG_TASK,
    'expanding must hold the entire task text',
  );
  assertFullTextPanel(win, 'expanded again');
  win.close();
}

function testChevronPassWorksWithoutButton() {
  const {win, posted} = makeWebview();
  const d = win.document;
  const ready = posted.find(m => m.type === 'ready');
  const parentId = ready.tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'status', running: true, tabId: parentId, startTs: 1});
  send(win, {type: 'setTaskText', text: 'live task', tabId: parentId});

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

  const readPanel = Array.from(O.querySelectorAll('.tc')).find(p =>
    (p.textContent || '').includes('/tmp/a'),
  );
  assert.ok(readPanel, 'the Read tool panel must exist');
  assert.ok(
    !rpPanel.classList.contains('collapsed'),
    'precondition: the fan-out panel is open while the task runs',
  );
  // A tool-call panel starts folded; the user opens the adopted one.
  assert.ok(adopted.classList.contains('collapsed'), 'adopted: folded at birth');
  adopted.querySelector(':scope > .collapse-header').click();
  assert.ok(!adopted.classList.contains('collapsed'), 'the user opened it');

  send(win, {
    type: 'result',
    tabId: parentId,
    summary: 'done',
    success: true,
  });
  send(win, {type: 'status', running: false, tabId: parentId});
  send(win, {type: 'usage_info', tabId: parentId});

  const rc = O.querySelector('.rc');
  assert.ok(rc, 'the result panel must render');
  assert.ok(
    !rc.classList.contains('collapsed'),
    'the result panel must stay open',
  );
  // The end of the task folds every event panel into one collapsed
  // Trajectory panel; the result stays outside it.
  const traj = O.querySelector(':scope > .trajectory');
  assert.ok(
    traj && traj.classList.contains('collapsed'),
    'the end folds the panels into a collapsed Trajectory panel',
  );
  assert.ok(
    traj.contains(summaryPanel) &&
      traj.contains(readPanel) &&
      traj.contains(rpPanel) &&
      !traj.contains(rc),
    'the summary, Read and fan-out panels sit inside the Trajectory',
  );
  assert.ok(
    summaryPanel.classList.contains('collapsed'),
    'the summary digest must fold',
  );
  assert.ok(
    !adopted.classList.contains('collapsed'),
    'panels adopted inside the summary are left as they are',
  );
  assert.ok(
    readPanel.classList.contains('collapsed') && !isDisplayed(win, readPanel),
    'a plain finished panel folds and hides behind the Trajectory',
  );
  assert.ok(
    rpPanel.classList.contains('collapsed') && !isDisplayed(win, rpPanel),
    'the finished run_parallel panel folds behind the Trajectory too',
  );
  traj.querySelector('.collapse-header').dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(
    isDisplayed(win, readPanel) && isDisplayed(win, rpPanel),
    'opening the Trajectory shows the folded panels, one click away',
  );
  assert.strictEqual(
    d.querySelectorAll('.tab.subagent-tab, .tab[data-subagent="1"]').length +
      Array.from(d.querySelectorAll('#tab-bar .tab, .tabs .tab')).filter(t =>
        (t.textContent || '').includes('sub 1'),
      ).length,
    0,
    'folding the run_parallel panel must close its sub-agent tabs',
  );

  // The user opens a folded panel: a later chevron pass (any trailing
  // event) leaves the panel the user pinned open.
  readPanel.querySelector('.collapse-header').dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.ok(
    !readPanel.classList.contains('collapsed') &&
      readPanel.classList.contains('user-pinned'),
    'the header click opens and pins the panel',
  );
  send(win, {type: 'usage_info', tabId: parentId});
  assert.ok(
    !readPanel.classList.contains('collapsed'),
    'the chevron pass must not re-fold a panel the user opened',
  );

  // A panel that lands after the end (a late tool call) stands outside
  // the Trajectory: the chevron pass that follows every rendered event
  // folds it to its digest, and it stays on screen.
  send(win, {type: 'tool_call', name: 'Bash', command: 'late', tabId: parentId});
  const late = O.querySelector(':scope > .tc-bash');
  assert.ok(late, 'the late panel renders outside the Trajectory');
  assert.ok(
    late.classList.contains('collapsed') && isDisplayed(win, late),
    'the chevron pass folds a late finished panel but keeps it on screen',
  );
  // A late fan-out is born OPEN (its open panel is what keeps the
  // sub-agent tabs up); with the task over, the chevron pass folds it.
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['late sub'])},
  });
  const lateRp = O.querySelector(':scope > .tc-run-parallel');
  assert.ok(lateRp, 'the late fan-out renders outside the Trajectory');
  assert.ok(
    lateRp.classList.contains('collapsed') && isDisplayed(win, lateRp),
    'the chevron pass folds a late fan-out once the task is over',
  );

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
  const adjTraj = adjacent.querySelector(':scope > .trajectory');
  assert.ok(
    adjTraj && adjTraj.classList.contains('collapsed'),
    "the adjacent task's replay folds its panels into its own Trajectory",
  );
  const adjPanel = adjTraj.querySelector('.trajectory-sub .collapsible');
  assert.ok(adjPanel, 'the adjacent task must replay its tool panel');
  assert.ok(
    adjPanel.classList.contains('collapsed') && !isDisplayed(win, adjPanel),
    "the adjacent task's finished panels fold behind its Trajectory",
  );
  win.close();
}

// A task that finishes ON SCREEN folds its panels into the Trajectory
// panel; the chevron pass a trailing event triggers leaves the panels
// inside the Trajectory exactly as the fold left them.
function testTrajectoryPanelsSkipChevronPass() {
  const {win, posted} = makeWebview();
  const d = win.document;
  const ready = posted.find(m => m.type === 'ready');
  const parentId = ready.tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'status', running: true, tabId: parentId, startTs: 1});
  send(win, {type: 'setTaskText', text: 'live task', tabId: parentId});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: parentId});
  send(win, {type: 'tool_result', name: 'Bash', content: 'f', tabId: parentId});
  send(win, {type: 'result', tabId: parentId, summary: 'done', success: true});
  send(win, {type: 'task_done', tabId: parentId});
  send(win, {type: 'status', running: false, tabId: parentId});

  const traj = d.querySelector('#output > .trajectory');
  assert.ok(traj, 'the finish folded the panels into a Trajectory panel');
  const inner = Array.from(traj.querySelectorAll('.trajectory-sub .collapsible'));
  assert.ok(inner.length > 0, 'the Trajectory holds the event panels');
  // The user opens the Trajectory and one panel inside it.
  traj.querySelector('.collapse-header').dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  inner[0].classList.remove('collapsed');
  const before = inner.map(p => p.classList.contains('collapsed'));

  send(win, {type: 'usage_info', tabId: parentId});
  inner.forEach((p, i) => {
    assert.strictEqual(
      p.classList.contains('collapsed'),
      before[i],
      'the chevron pass must not touch Trajectory panel #' + i,
    );
  });
  assert.ok(
    !traj.classList.contains('collapsed'),
    'the Trajectory the user opened stays open',
  );
  win.close();
}

function runTests() {
  const tests = [
    () => testTaskPanelOpensTheTranscript(false),
    () => testTaskPanelOpensTheTranscript(true),
    testTaskPanelFoldsLikeAnyPanel,
    testChevronPassWorksWithoutButton,
    testTrajectoryPanelsSkipChevronPass,
  ];
  const names = [
    'testTaskPanelOpensTheTranscript(vscode)',
    'testTaskPanelOpensTheTranscript(remote)',
    'testTaskPanelFoldsLikeAnyPanel',
    'testChevronPassWorksWithoutButton',
    'testTrajectoryPanelsSkipChevronPass',
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
