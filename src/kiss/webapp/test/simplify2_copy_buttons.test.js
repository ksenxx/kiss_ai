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
  rowsByTitle,
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
  // A daemon before 2026-10-31 sent the text a second time as
  // `preview`; the row reads `title` first.
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

// Two more tasks, each a row with a copy button of its own.
const SECOND = {
  ...SESSION,
  id: 'chatDEF',
  task_id: 43,
  title: 'second task',
  preview: 'second task',
};
const THIRD = {
  ...SESSION,
  id: 'chatGHI',
  task_id: 44,
  title: 'third task',
  preview: 'third task',
};

async function run() {
  const {win} = makePage();
  const doc = win.document;
  const timers = recordTimers(win, 1500);

  const {writes: copied} = stubClipboard(win);

  send(win, {type: 'history', sessions: [SESSION, SECOND, THIRD], offset: 0});

  assert.strictEqual(
    doc.querySelectorAll('#history-list .sidebar-item-copy').length,
    3,
    'every history row renders one .sidebar-item-copy button',
  );
  const rows = rowsByTitle(win);
  const rowBtn = rows['my task'].querySelector('.sidebar-item-copy');
  assert.ok(rowBtn, 'history row renders a .sidebar-item-copy button');
  const idleSvg = rowBtn.innerHTML;
  rowBtn.click();
  await sleep(0);
  assert.deepStrictEqual(copied, ['my task'], 'row copies its title');
  assert.ok(rowBtn.classList.contains('copied'), 'row button flashes');
  assert.notStrictEqual(rowBtn.innerHTML, idleSvg, 'icon swaps to check');

  const secondBtn = rows['second task'].querySelector('.sidebar-item-copy');
  const thirdBtn = rows['third task'].querySelector('.sidebar-item-copy');
  secondBtn.click();
  thirdBtn.click();
  await sleep(0);
  assert.deepStrictEqual(
    copied.slice(1),
    ['second task', 'third task'],
    'each row button copies the title of its own row',
  );
  assert.ok(secondBtn.classList.contains('copied'), 'second button flashes');
  assert.ok(thirdBtn.classList.contains('copied'), 'third button flashes');
  assert.strictEqual(timers.length, 3, 'one flash timer per button');

  const firstClick = timers.slice();
  rowBtn.click();
  secondBtn.click();
  await sleep(0);
  // audit0903: every copy button restarts its flash timer on re-click,
  // so the first click's timers of the re-clicked buttons are cleared
  // and firing them (as the clock would at 1.5 s) cuts no second flash
  // short; only the untouched third button reverts.
  fireTimers(firstClick);
  assert.ok(
    rowBtn.classList.contains('copied'),
    'row second click resets its timer',
  );
  assert.ok(
    secondBtn.classList.contains('copied'),
    'second row second click resets its timer',
  );
  assert.ok(
    !thirdBtn.classList.contains('copied'),
    'untouched third flash reverts',
  );

  fireTimers(timers.slice(3));
  assert.ok(!rowBtn.classList.contains('copied'), 'reset row timer reverts');
  assert.ok(
    !secondBtn.classList.contains('copied'),
    'reset second row timer reverts',
  );
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
