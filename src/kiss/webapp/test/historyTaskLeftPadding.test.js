// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {loadMainCss, makePage, send} = require('./simplify2_harness');

// A history row reserves a left column for its status icon: the row's
// padding-left is 19px (a 13px icon column plus a 6px gap to the text)
// and the icon sits 5px in, so it never overlaps the task text.  The
// values are read off the rendered row, not the stylesheet source.

function renderRunningRow(win) {
  loadMainCss(win);
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  send(win, {
    type: 'history',
    offset: 0,
    sessions: [
      {
        id: 'chat-1',
        task_id: 1,
        title: 'running task',
        preview: 'running task',
        timestamp: 1_700_000_000,
        has_events: false,
        failed: false,
        is_running: true,
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
  const row = win.document.querySelector('#history-list .running-item');
  assert.ok(row, 'the history list renders a .running-item row');
  const dot = row.querySelector(':scope > .sidebar-item-running');
  assert.ok(dot, 'the running row carries its status icon as a direct child');
  return {row, dot};
}

function testRunningItemPaddingLeftHalved() {
  const {win} = makePage();
  const {row} = renderRunningRow(win);
  assert.strictEqual(
    win.getComputedStyle(row).paddingLeft,
    '19px',
    'a history row keeps a 19px left column (13px icon + 6px gap)',
  );
  win.close();
  console.log('  ok - .running-item padding-left is 19px (13px dot column + 6px gap)');
}

function testStatusIndicatorLeftHalved() {
  const {win} = makePage();
  const {dot} = renderRunningRow(win);
  assert.strictEqual(
    win.getComputedStyle(dot).left,
    '5px',
    'the status icon sits 5px into the row (halved from 10px)',
  );
  win.close();
  console.log('  ok - status dot left offset is 5px (halved from 10px)');
}

function testDotFitsInsidePadding() {
  const {win} = makePage();
  const {row, dot} = renderRunningRow(win);
  const padPx = Number.parseInt(win.getComputedStyle(row).paddingLeft, 10);
  const dotStyle = win.getComputedStyle(dot);
  const leftPx = Number.parseInt(dotStyle.left, 10);
  const widthPx = Number.parseInt(dotStyle.width, 10);
  assert.ok(
    Number.isFinite(padPx) && Number.isFinite(widthPx) && Number.isFinite(leftPx),
    'padding-left / dot width / dot left must all be integer px values',
  );
  assert.ok(
    leftPx + widthPx <= padPx,
    `status dot (left=${leftPx}px + width=${widthPx}px = ${
      leftPx + widthPx
    }px) must fit inside the .running-item padding column (${padPx}px) ` +
      'so it never overlaps text content',
  );
  win.close();
  console.log(
    `  ok - dot (${leftPx}px + ${widthPx}px) fits inside ${padPx}px padding column`,
  );
}

function main() {
  testRunningItemPaddingLeftHalved();
  testStatusIndicatorLeftHalved();
  testDotFitsInsidePadding();
  console.log('historyTaskLeftPadding.test.js: all assertions passed.');
}

main();
