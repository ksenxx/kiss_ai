// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the per-id copy buttons in the History sidebar's
// task-row ids line.
//
// Requirement driven by this test:
//
//   The ids line of every History row (``.running-item-ids``) renders
//   a small copy-to-clipboard button IMMEDIATELY AFTER the chat id and
//   IMMEDIATELY AFTER the task id.  Clicking the chat-id button copies
//   JUST the raw chat id string to the system clipboard; clicking the
//   task-id button copies JUST the raw task id string.
//
//   * The buttons carry the classes ``ids-copy-btn ids-copy-chat`` and
//     ``ids-copy-btn ids-copy-task`` respectively.
//   * A row with no chat id renders no chat-copy button; a row with no
//     task id renders no task-copy button; a row with neither renders
//     no buttons at all (and, when the parent id is also missing, no
//     ids span at all — pre-existing behaviour).
//   * Clicking a copy button must NOT bubble into the row's own click
//     handler: no new tab is created and no ``resumeSession`` message
//     is posted to the extension host.
//   * After a successful copy the button icon flips to a check mark
//     and the button gains the ``copied`` class for visual feedback,
//     then reverts after 1.5 s.
//   * When ``navigator.clipboard.writeText`` is unavailable the button
//     falls back to the textarea + ``document.execCommand('copy')``
//     path (same fallback as every other copy button in the webview).
//   * The buttons contribute NO text of their own (icon-only), so the
//     ids line's ``textContent`` stays exactly
//     ``chat <id> • task <id> [• parent <id>]`` — the format asserted
//     by historyTaskIds.test.js.
//
// This test drives the production ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom,
// exactly like ``historyTaskIds.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyIdsCopyButtons.test.js

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

  // Record every clipboard write made through the async clipboard API.
  const clipboardWrites = [];
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: text => {
        clipboardWrites.push(String(text));
        return Promise.resolve();
      },
    },
  });

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

  return {win, posted, clipboardWrites};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

const WS = '/Users/koushik/work/repo';

function baseRow(overrides) {
  return Object.assign(
    {
      has_events: false,
      failed: false,
      is_running: false,
      tokens: 100,
      cost: 0.01,
      steps: 1,
      is_favorite: false,
      work_dir: WS,
      model: 'gpt-5',
      is_worktree: true,
      is_parallel: false,
      auto_commit_mode: true,
      startTs: 1_700_000_000_000,
      endTs: 1_700_000_005_000,
    },
    overrides,
  );
}

const SESSIONS_FIXTURE = [
  baseRow({
    id: 'chat-a',
    task_id: 'task-a',
    parent_task_id: 'parent-a',
    title: 'copy row A — chat + task + parent',
    timestamp: 1_700_000_000,
    preview: 'copy row A — chat + task + parent',
  }),
  baseRow({
    id: 'chat-b',
    task_id: 'task-b',
    title: 'copy row B — chat + task',
    timestamp: 1_700_000_100,
    preview: 'copy row B — chat + task',
  }),
  baseRow({
    id: 'chat-c',
    task_id: null,
    title: 'copy row C — chat only',
    timestamp: 1_700_000_200,
    preview: 'copy row C — chat only',
  }),
  baseRow({
    id: '',
    task_id: 'task-d',
    title: 'copy row D — task only',
    timestamp: 1_700_000_300,
    preview: 'copy row D — task only',
  }),
  baseRow({
    id: '',
    task_id: null,
    parent_task_id: 'parent-e',
    title: 'copy row E — parent only',
    timestamp: 1_700_000_400,
    preview: 'copy row E — parent only',
  }),
];

function rowsByTitle(win) {
  const list = win.document.getElementById('history-list');
  const rows = list.querySelectorAll('.sidebar-item');
  const map = {};
  rows.forEach(r => {
    const t = r.querySelector('.sidebar-item-text');
    if (!t) return;
    map[t.textContent] = r;
  });
  return map;
}

function disableWorkspaceFilter(win) {
  send(win, {
    type: 'configData',
    config: {work_dir: ''},
    apiKeys: {},
  });
  const ws = win.document.getElementById('hf-workspace');
  if (ws && ws.checked) {
    ws.checked = false;
    ws.dispatchEvent(new win.Event('change', {bubbles: true}));
  }
}

function loadHistory(ctx) {
  disableWorkspaceFilter(ctx.win);
  send(ctx.win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});
  return rowsByTitle(ctx.win);
}

function click(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

function tabCount(win) {
  return win.document.querySelectorAll('#tab-bar .chat-tab').length;
}

function testButtonsRenderPerId() {
  const ctx = makeWebview();
  const rows = loadHistory(ctx);
  const {win} = ctx;

  const a = rows['copy row A — chat + task + parent'];
  const b = rows['copy row B — chat + task'];
  const c = rows['copy row C — chat only'];
  const d = rows['copy row D — task only'];
  const e = rows['copy row E — parent only'];
  assert.ok(a && b && c && d && e, 'all five fixture rows must render');

  // Rows A and B: both buttons present, chat button before task button.
  for (const [label, row] of [
    ['A', a],
    ['B', b],
  ]) {
    const ids = row.querySelector('.running-item-ids');
    assert.ok(ids, `row ${label} must render the ids line`);
    const chatBtn = ids.querySelector('button.ids-copy-btn.ids-copy-chat');
    const taskBtn = ids.querySelector('button.ids-copy-btn.ids-copy-task');
    assert.ok(chatBtn, `row ${label} must render a chat-id copy button`);
    assert.ok(taskBtn, `row ${label} must render a task-id copy button`);
    assert.ok(
      chatBtn.compareDocumentPosition(taskBtn) &
        win.Node.DOCUMENT_POSITION_FOLLOWING,
      `row ${label}: the chat copy button must precede the task copy button`,
    );
    // Icon-only buttons: they must not add any text to the ids line.
    assert.strictEqual(
      chatBtn.textContent,
      '',
      `row ${label}: chat copy button must be icon-only`,
    );
    assert.strictEqual(
      taskBtn.textContent,
      '',
      `row ${label}: task copy button must be icon-only`,
    );
    assert.ok(
      chatBtn.querySelector('svg'),
      `row ${label}: chat copy button must contain an SVG icon`,
    );
    assert.ok(
      taskBtn.querySelector('svg'),
      `row ${label}: task copy button must contain an SVG icon`,
    );
  }

  // The ids-line text format asserted by historyTaskIds.test.js must
  // be preserved exactly even with the buttons injected.
  assert.strictEqual(
    a.querySelector('.running-item-ids').textContent,
    'chat chat-a • task task-a • parent parent-a',
    'row A ids text must be unchanged by the copy buttons',
  );
  assert.strictEqual(
    b.querySelector('.running-item-ids').textContent,
    'chat chat-b • task task-b',
    'row B ids text must be unchanged by the copy buttons',
  );

  // Row C: chat only → chat button, no task button.
  const cIds = c.querySelector('.running-item-ids');
  assert.ok(
    cIds.querySelector('.ids-copy-chat'),
    'row C (chat only) must render a chat-id copy button',
  );
  assert.strictEqual(
    cIds.querySelector('.ids-copy-task'),
    null,
    'row C (chat only) must NOT render a task-id copy button',
  );

  // Row D: task only → task button, no chat button.
  const dIds = d.querySelector('.running-item-ids');
  assert.strictEqual(
    dIds.querySelector('.ids-copy-chat'),
    null,
    'row D (task only) must NOT render a chat-id copy button',
  );
  assert.ok(
    dIds.querySelector('.ids-copy-task'),
    'row D (task only) must render a task-id copy button',
  );

  // Row E: parent only → no copy buttons at all.
  const eIds = e.querySelector('.running-item-ids');
  assert.ok(eIds, 'row E must still render the ids line (parent id)');
  assert.strictEqual(
    eIds.querySelectorAll('.ids-copy-btn').length,
    0,
    'row E (parent only) must render no copy buttons',
  );

  ctx.win.close();
  console.log('  ok - copy buttons render exactly for the ids that exist');
}

async function testChatCopyButtonCopiesChatId() {
  const ctx = makeWebview();
  const rows = loadHistory(ctx);
  const {win, clipboardWrites, posted} = ctx;

  const a = rows['copy row A — chat + task + parent'];
  const chatBtn = a.querySelector('.ids-copy-chat');
  assert.ok(chatBtn, 'row A chat copy button must exist');

  const tabsBefore = tabCount(win);
  const postedBefore = posted.length;

  click(win, chatBtn);
  await new Promise(r => setTimeout(r, 0));

  assert.deepStrictEqual(
    clipboardWrites,
    ['chat-a'],
    'clicking the chat copy button must copy exactly the raw chat id; ' +
      `clipboard got: ${JSON.stringify(clipboardWrites)}`,
  );

  // The click must not bubble into the row handler: no new tab, no
  // resumeSession message.
  assert.strictEqual(
    tabCount(win),
    tabsBefore,
    'clicking the chat copy button must not open a new tab',
  );
  const newMsgs = posted.slice(postedBefore);
  assert.ok(
    !newMsgs.some(m => m && m.type === 'resumeSession'),
    'clicking the chat copy button must not post resumeSession',
  );

  // Visual feedback: check icon + ``copied`` class.
  assert.ok(
    chatBtn.classList.contains('copied'),
    'chat copy button must gain the "copied" class after a copy',
  );
  assert.ok(
    chatBtn.querySelector('svg polyline'),
    'chat copy button must show the check-mark icon after a copy',
  );

  win.close();
  console.log('  ok - chat copy button copies the raw chat id, no row click');
}

async function testTaskCopyButtonCopiesTaskId() {
  const ctx = makeWebview();
  const rows = loadHistory(ctx);
  const {win, clipboardWrites, posted} = ctx;

  const b = rows['copy row B — chat + task'];
  const taskBtn = b.querySelector('.ids-copy-task');
  assert.ok(taskBtn, 'row B task copy button must exist');

  const tabsBefore = tabCount(win);
  const postedBefore = posted.length;

  click(win, taskBtn);
  await new Promise(r => setTimeout(r, 0));

  assert.deepStrictEqual(
    clipboardWrites,
    ['task-b'],
    'clicking the task copy button must copy exactly the raw task id; ' +
      `clipboard got: ${JSON.stringify(clipboardWrites)}`,
  );

  assert.strictEqual(
    tabCount(win),
    tabsBefore,
    'clicking the task copy button must not open a new tab',
  );
  const newMsgs = posted.slice(postedBefore);
  assert.ok(
    !newMsgs.some(m => m && m.type === 'resumeSession'),
    'clicking the task copy button must not post resumeSession',
  );

  assert.ok(
    taskBtn.classList.contains('copied'),
    'task copy button must gain the "copied" class after a copy',
  );

  win.close();
  console.log('  ok - task copy button copies the raw task id, no row click');
}

async function testNumericTaskIdCopiesAsString() {
  // task_id often arrives as a NUMBER from the backend (SQLite integer
  // primary key).  The button must copy its decimal string form.
  const ctx = makeWebview();
  disableWorkspaceFilter(ctx.win);
  send(ctx.win, {
    type: 'history',
    sessions: [
      baseRow({
        id: 'chat-n',
        task_id: 12345,
        title: 'copy row N — numeric task id',
        timestamp: 1_700_000_600,
        preview: 'copy row N — numeric task id',
      }),
    ],
    offset: 0,
  });
  const rows = rowsByTitle(ctx.win);
  const n = rows['copy row N — numeric task id'];
  assert.ok(n, 'numeric-task-id row must render');
  const taskBtn = n.querySelector('.ids-copy-task');
  assert.ok(taskBtn, 'numeric-task-id row must render a task copy button');

  click(ctx.win, taskBtn);
  await new Promise(r => setTimeout(r, 0));

  assert.deepStrictEqual(
    ctx.clipboardWrites,
    ['12345'],
    'a numeric task_id must be copied as its decimal string',
  );

  ctx.win.close();
  console.log('  ok - numeric task ids are copied as strings');
}

async function testFallbackCopyPathWithoutClipboardApi() {
  const ctx = makeWebview();
  const {win} = ctx;

  // Remove the async clipboard API to force the textarea +
  // execCommand('copy') fallback.
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: undefined,
  });
  const execCalls = [];
  win.document.execCommand = function (cmd) {
    execCalls.push(cmd);
    return true;
  };

  const rows = loadHistory(ctx);
  const a = rows['copy row A — chat + task + parent'];
  const chatBtn = a.querySelector('.ids-copy-chat');
  assert.ok(chatBtn, 'row A chat copy button must exist');

  click(win, chatBtn);
  await new Promise(r => setTimeout(r, 0));

  assert.deepStrictEqual(
    execCalls,
    ['copy'],
    'without navigator.clipboard the button must fall back to ' +
      "document.execCommand('copy')",
  );
  assert.ok(
    chatBtn.classList.contains('copied'),
    'fallback copy must still flash the "copied" feedback',
  );

  win.close();
  console.log('  ok - execCommand fallback used when clipboard API missing');
}

async function testRejectedClipboardFallsBackToExecCommand() {
  // When ``navigator.clipboard.writeText`` REJECTS (e.g. the webview
  // lost focus or the permission was denied) the button must fall
  // back to the textarea + ``document.execCommand('copy')`` path so
  // the id is still copied.
  const ctx = makeWebview();
  const {win} = ctx;

  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: () => Promise.reject(new Error('denied')),
    },
  });
  const execCalls = [];
  win.document.execCommand = function (cmd) {
    execCalls.push(cmd);
    return true;
  };

  const rows = loadHistory(ctx);
  const a = rows['copy row A — chat + task + parent'];
  const chatBtn = a.querySelector('.ids-copy-chat');
  assert.ok(chatBtn, 'row A chat copy button must exist');

  click(win, chatBtn);
  // Two ticks: one for the rejected promise, one for the rejection
  // handler's fallback + flash.
  await new Promise(r => setTimeout(r, 0));
  await new Promise(r => setTimeout(r, 0));

  assert.deepStrictEqual(
    execCalls,
    ['copy'],
    'a rejected clipboard.writeText must fall back to ' +
      "document.execCommand('copy')",
  );
  assert.ok(
    chatBtn.classList.contains('copied'),
    'the fallback after a rejected clipboard write must still flash ' +
      'the "copied" feedback',
  );

  win.close();
  console.log('  ok - rejected clipboard write falls back to execCommand');
}

async function testRepeatedClicksRestartFlashTimer() {
  // Two quick clicks must keep the "copied" feedback visible for a
  // full 1.5 s after the SECOND click (the first click's timer must
  // be cleared, not allowed to end the feedback early).
  const ctx = makeWebview();
  const rows = loadHistory(ctx);
  const {win} = ctx;

  const a = rows['copy row A — chat + task + parent'];
  const chatBtn = a.querySelector('.ids-copy-chat');
  assert.ok(chatBtn, 'row A chat copy button must exist');

  click(win, chatBtn);
  await new Promise(r => setTimeout(r, 0));
  assert.ok(
    chatBtn.classList.contains('copied'),
    'first click must flash the "copied" feedback',
  );

  // Second click ~1.2 s after the first: the first timer (due at
  // 1.5 s) must be cancelled so the feedback survives past it.
  await new Promise(r => setTimeout(r, 1200));
  click(win, chatBtn);
  await new Promise(r => setTimeout(r, 0));

  // 0.5 s after the second click (1.7 s after the first): had the
  // first timer not been cleared it would already have removed the
  // class; with the fix the feedback is still visible.
  await new Promise(r => setTimeout(r, 500));
  assert.ok(
    chatBtn.classList.contains('copied'),
    "the first click's stale timer must not end the second " +
      'click\'s "copied" feedback early',
  );

  // And it still reverts eventually (1.5 s after the second click).
  await new Promise(r => setTimeout(r, 1200));
  assert.ok(
    !chatBtn.classList.contains('copied'),
    'the "copied" feedback must still revert after the second timer',
  );

  win.close();
  console.log('  ok - repeated clicks restart the flash timer');
}

function testCopyButtonCss() {
  // jsdom never loads the external ``main.css`` stylesheet, so read
  // the CSS file directly and assert a dedicated rule exists for the
  // ids copy buttons (so they render as small inline icon buttons and
  // are not styled as full-size native buttons).
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const re = /\.ids-copy-btn\s*\{([^}]*)\}/;
  const m = re.exec(css);
  assert.ok(m, 'main.css must define a .ids-copy-btn rule');
  const body = m[1];
  assert.match(
    body,
    /cursor\s*:\s*pointer/,
    '.ids-copy-btn must use cursor: pointer',
  );
  assert.match(
    body,
    /background\s*:\s*none/,
    '.ids-copy-btn must have no background (icon-only button)',
  );
  console.log('  ok - main.css styles the ids copy buttons');
}

async function main() {
  testButtonsRenderPerId();
  await testChatCopyButtonCopiesChatId();
  await testTaskCopyButtonCopiesTaskId();
  await testNumericTaskIdCopiesAsString();
  await testFallbackCopyPathWithoutClipboardApi();
  await testRejectedClipboardFallsBackToExecCommand();
  await testRepeatedClicksRestartFlashTimer();
  testCopyButtonCss();
  console.log('historyIdsCopyButtons.test.js: all assertions passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
