// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeHarnessPage,
  send,
  loadMainCss,
  click,
  sendEmptyWorkDir,
  rowsByTitle,
  stubClipboard,
} = require('./simplify2_harness');

function makePage() {
  let clipboardWrites;
  // Every timer the page arms is recorded (and still runs) so a test can
  // play the clock forward by hand instead of sleeping through it.
  const timeouts = [];
  const {win, posted} = makeHarnessPage({
    beforeScripts(win) {
      clipboardWrites = stubClipboard(win).writes;
      const realSetTimeout = win.setTimeout.bind(win);
      const realClearTimeout = win.clearTimeout.bind(win);
      win.setTimeout = function (fn, ms, ...args) {
        const timer = {
          fn,
          ms,
          id: realSetTimeout(fn, ms, ...args),
          cleared: false,
        };
        timeouts.push(timer);
        return timer.id;
      };
      win.clearTimeout = function (id) {
        for (const t of timeouts) if (t.id === id) t.cleared = true;
        realClearTimeout(id);
      };
    },
  });
  return {win, posted, clipboardWrites, timeouts};
}

/** Let the clock reach *timer*'s due time: it runs unless it was cleared. */
function fireTimeout(win, timer) {
  if (timer.cleared) return;
  win.clearTimeout(timer.id);
  timer.fn();
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
      is_parallel: false,
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

function loadHistory(ctx) {
  sendEmptyWorkDir(ctx.win);
  send(ctx.win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});
  return rowsByTitle(ctx.win);
}

// Every open tab: the strip renders only the chat on screen, so a DOM
// count of `.chat-tab` is 1 whatever the number of open chats.
function tabCount(win) {
  return win._testApi.openTabs().length;
}

function testButtonsRenderPerId() {
  const ctx = makePage();
  const rows = loadHistory(ctx);
  const {win} = ctx;

  const a = rows['copy row A — chat + task + parent'];
  const b = rows['copy row B — chat + task'];
  const c = rows['copy row C — chat only'];
  const d = rows['copy row D — task only'];
  const e = rows['copy row E — parent only'];
  assert.ok(a && b && c && d && e, 'all five fixture rows must render');

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
  const ctx = makePage();
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
  const ctx = makePage();
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
  const ctx = makePage();
  sendEmptyWorkDir(ctx.win);
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
  const ctx = makePage();
  const {win} = ctx;

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
  const ctx = makePage();
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
  const ctx = makePage();
  const rows = loadHistory(ctx);
  const {win, timeouts} = ctx;

  const a = rows['copy row A — chat + task + parent'];
  const chatBtn = a.querySelector('.ids-copy-chat');
  assert.ok(chatBtn, 'row A chat copy button must exist');

  let armed = timeouts.length;
  click(win, chatBtn);
  await new Promise(r => setTimeout(r, 0));
  assert.ok(
    chatBtn.classList.contains('copied'),
    'first click must flash the "copied" feedback',
  );
  const firstFlash = timeouts.slice(armed).filter(t => t.ms === 1500);
  assert.strictEqual(firstFlash.length, 1, 'the first click arms one revert');

  // A second click 1.2 s later, then the clock reaches the first
  // click's 1.5 s mark and, later, the second click's.
  armed = timeouts.length;
  click(win, chatBtn);
  await new Promise(r => setTimeout(r, 0));
  const secondFlash = timeouts.slice(armed).filter(t => t.ms === 1500);
  assert.strictEqual(secondFlash.length, 1, 'the second click arms one revert');

  fireTimeout(win, firstFlash[0]);
  assert.ok(
    chatBtn.classList.contains('copied'),
    "the first click's stale timer must not end the second " +
      'click\'s "copied" feedback early',
  );

  fireTimeout(win, secondFlash[0]);
  assert.ok(
    !chatBtn.classList.contains('copied'),
    'the "copied" feedback must still revert after the second timer',
  );

  win.close();
  console.log('  ok - repeated clicks restart the flash timer');
}

function testCopyButtonCss() {
  const ctx = makePage();
  loadMainCss(ctx.win);
  const rows = loadHistory(ctx);
  const btn =
    rows['copy row A — chat + task + parent'].querySelector('.ids-copy-btn');
  const style = ctx.win.getComputedStyle(btn);
  assert.strictEqual(
    style.cursor,
    'pointer',
    'an ids copy button is clickable',
  );
  assert.strictEqual(
    style.backgroundImage,
    'none',
    'an ids copy button has no background (icon-only button)',
  );
  assert.strictEqual(
    style.backgroundColor,
    'rgba(0, 0, 0, 0)',
    'an ids copy button has no background colour (icon-only button)',
  );
  ctx.win.close();
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
