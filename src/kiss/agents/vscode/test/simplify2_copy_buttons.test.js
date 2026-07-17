// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Regression tests for the History-sidebar copy buttons in
// media/main.js: the per-row task copy button (makeSidebarCopyButton)
// and the tiny per-id copy buttons on the ids line (makeIdCopyButton).
// Locks in the clipboard payloads and the icon-swap "copied" feedback
// so the duplicated builders can share one wiring helper without any
// visible change.
//
//     node src/kiss/agents/vscode/test/simplify2_copy_buttons.test.js
'use strict';

const assert = require('assert');
const {makeWebview, send, sleep} = require('./simplify2_harness');

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
  const {win} = makeWebview();
  const doc = win.document;

  // Recording async clipboard, installed like a real secure context.
  const copied = [];
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText(text) {
        copied.push(text);
        return Promise.resolve();
      },
    },
  });

  send(win, {type: 'history', sessions: [SESSION], offset: 0});

  // --- Per-row task copy button copies the row's preview text. ---
  const rowBtn = doc.querySelector('.sidebar-item-copy');
  assert.ok(rowBtn, 'history row renders a .sidebar-item-copy button');
  const idleSvg = rowBtn.innerHTML;
  rowBtn.click();
  await sleep(0);
  assert.deepStrictEqual(copied, ['my task preview'], 'row copies preview');
  assert.ok(rowBtn.classList.contains('copied'), 'row button flashes');
  assert.notStrictEqual(rowBtn.innerHTML, idleSvg, 'icon swaps to check');

  // --- Per-id copy buttons copy the raw chat / task ids. ---
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

  // --- Preserve each builder's historical rapid-click timer semantics. ---
  // A row-button second click does NOT cancel the first timer, so the first
  // click still reverts it at t=1.5 s.  ID buttons do reset their timer and
  // therefore remain flashed for 1.5 s after the second click.
  await sleep(1_000);
  rowBtn.click();
  chatBtn.click();
  await sleep(0);
  await sleep(600);
  assert.ok(!rowBtn.classList.contains('copied'), 'row first timer still reverts');
  assert.ok(chatBtn.classList.contains('copied'), 'id second click resets its timer');
  assert.ok(!taskBtn.classList.contains('copied'), 'untouched task flash reverts');
  assert.strictEqual(rowBtn.innerHTML, idleSvg, 'row icon reverts to copy svg');

  await sleep(1_000);
  assert.ok(!chatBtn.classList.contains('copied'), 'reset id timer eventually reverts');

  win.close();
  console.log('  ok - history copy buttons copy and flash identically');
}

run().then(() => {
  console.log('simplify2_copy_buttons.test.js: all tests passed');
});
