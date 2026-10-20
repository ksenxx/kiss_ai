// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A sidebar row's copy button (history rows, id chips, the remote-URL
// bar) is wired by the same PanelCopy.wireCopyButton as a panel's copy
// button: when both clipboard paths fail it shows the same visible
// failure state (a cross, `copy-failed`, a "Copy failed" title) instead
// of doing nothing, and a success flashes the check mark and reverts.
// Runs with: node test/sidebarCopyButtonFailure.test.js

const assert = require('assert');
const {makePage: makeHarnessPage, send, sleep} = require('./simplify2_harness');

function makePage() {
  const {win, posted} = makeHarnessPage({bodyClass: 'remote-chat'});
  win.document.getElementById('sidebar').classList.add('open');
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  const req = posted.filter(m => m && m.type === 'getHistory').pop();
  send(win, {
    type: 'history',
    offset: 0,
    generation: req ? req.generation : undefined,
    sessions: [
      {
        id: 'chat-1',
        task_id: 1,
        title: 'task one',
        timestamp: 1_700_000_000,
        preview: 'task one',
        has_events: true,
        failed: false,
        is_running: false,
        awaiting_answer: false,
        tokens: 1,
        cost: 0,
        steps: 1,
        is_favorite: false,
        work_dir: '',
        startTs: 1_700_000_000_000,
        endTs: 0,
      },
    ],
  });
  const btn = win.document.querySelector('#history-list .sidebar-item-copy');
  assert.ok(btn, 'setup: the history row has a copy button');
  return {win, btn};
}

function click(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

async function testFailureIsVisible() {
  const {win, btn} = makePage();
  // jsdom: no navigator.clipboard, execCommand undefined -> both fail.
  assert.strictEqual(win.navigator.clipboard, undefined);
  click(win, btn);
  await sleep(5);
  assert.ok(btn.classList.contains('copy-failed'), btn.className);
  assert.ok(!btn.classList.contains('copied'));
  assert.ok(btn.querySelector('line'), 'a cross replaces the copy icon');
  assert.ok(/^Copy failed/.test(btn.title), btn.title);
  win.close();
  console.log('  ok - a failed sidebar copy shows the failure state');
}

async function testSuccessFlashesAndReverts() {
  const {win, btn} = makePage();
  const copied = [];
  win.document.execCommand = cmd => {
    if (cmd !== 'copy') return false;
    // The fallback's hidden textarea is appended after the composer's.
    const ta = Array.from(win.document.querySelectorAll('textarea')).pop();
    copied.push(ta ? ta.value : null);
    return true;
  };
  const idle = btn.innerHTML;
  assert.strictEqual(btn.hasAttribute('title'), false, 'no native title');
  click(win, btn);
  await sleep(5);
  assert.deepStrictEqual(copied, ['task one'], 'the row text was copied');
  assert.ok(btn.classList.contains('copied'));
  assert.ok(!btn.classList.contains('copy-failed'));
  assert.ok(btn.querySelector('polyline'), 'the check mark confirms');
  assert.strictEqual(btn.title, 'Copied');
  await sleep(1600);
  assert.strictEqual(btn.innerHTML, idle, 'the copy icon is back');
  assert.ok(!btn.classList.contains('copied'));
  assert.strictEqual(btn.hasAttribute('title'), false, 'no title again');
  win.close();
  console.log('  ok - a successful sidebar copy flashes and reverts');
}

async function main() {
  await testFailureIsVisible();
  await testSuccessFlashesAndReverts();
  console.log('sidebarCopyButtonFailure: all passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
