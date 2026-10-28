// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  sleep,
  recordTimers,
  fireTimers,
  stubClipboard,
} = require('./simplify2_harness');

// The 1500 ms flash-revert timers (wireCopyButton in panelCopy.js) are
// recorded instead of waited for and fired by hand, so no wall-clock
// sleep decides whether the first click's timer lands before or after
// the second click.

const SESSION = {
  id: 'chatABC',
  task_id: 42,
  title: 'my task',
  timestamp: 1_700_000_000,
  preview: 'my task preview',
  has_events: false,
  failed: false,
  is_running: false,
  tokens: 0,
  cost: 0,
  steps: 0,
  is_favorite: false,
  work_dir: '',
  startTs: 1_700_000_000_000,
  endTs: 1_700_000_100_000,
};

async function run() {
  const {win} = makePage();
  const doc = win.document;
  const timers = recordTimers(win, 1500);

  const {writes: copied} = stubClipboard(win);

  send(win, {type: 'history', sessions: [SESSION], offset: 0});

  const rowBtn = doc.querySelector('.sidebar-item-copy');
  assert.ok(rowBtn, 'history row renders a .sidebar-item-copy button');
  const idleSvg = rowBtn.innerHTML;
  rowBtn.click();
  await sleep(0);
  assert.deepStrictEqual(copied, ['my task preview'], 'row copies preview');
  assert.ok(rowBtn.classList.contains('copied'), 'row button flashes');
  assert.notStrictEqual(rowBtn.innerHTML, idleSvg, 'icon swaps to check');

  const chatBtn = doc.querySelector('.ids-copy-btn.ids-copy-chat');
  const taskBtn = doc.querySelector('.ids-copy-btn.ids-copy-task');
  assert.ok(chatBtn, 'ids line renders a chat-id copy button');
  assert.ok(taskBtn, 'ids line renders a task-id copy button');
  assert.strictEqual(chatBtn.dataset.tooltip, 'Copy chat id');
  assert.strictEqual(taskBtn.dataset.tooltip, 'Copy task id');
  chatBtn.click();
  taskBtn.click();
  await sleep(0);
  assert.deepStrictEqual(
    copied.slice(1),
    ['chatABC', '42'],
    'id buttons copy the raw id strings',
  );
  assert.ok(chatBtn.classList.contains('copied'), 'chat button flashes');
  assert.ok(taskBtn.classList.contains('copied'), 'task button flashes');
  assert.strictEqual(timers.length, 3, 'one flash timer per button');

  const firstClick = timers.slice();
  rowBtn.click();
  chatBtn.click();
  await sleep(0);
  // audit0903: every copy button restarts its flash timer on re-click,
  // so the first click's timers of the re-clicked buttons are cleared
  // and firing them (as the clock would at 1.5 s) cuts no second flash
  // short; only the untouched task button reverts.
  fireTimers(firstClick);
  assert.ok(
    rowBtn.classList.contains('copied'),
    'row second click resets its timer',
  );
  assert.ok(
    chatBtn.classList.contains('copied'),
    'id second click resets its timer',
  );
  assert.ok(
    !taskBtn.classList.contains('copied'),
    'untouched task flash reverts',
  );

  fireTimers(timers.slice(3));
  assert.ok(!rowBtn.classList.contains('copied'), 'reset row timer reverts');
  assert.ok(!chatBtn.classList.contains('copied'), 'reset id timer reverts');
  assert.strictEqual(rowBtn.innerHTML, idleSvg, 'row icon reverts to copy svg');

  win.close();
  console.log('  ok - history copy buttons copy and flash identically');
}

run()
  .then(() => {
    console.log('simplify2_copy_buttons.test.js: all tests passed');
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
