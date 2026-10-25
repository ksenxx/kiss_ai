// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {loadMainCss, makePage, send} = require('./simplify2_harness');
const {makeRow} = require('./history_helpers');

// A history row reserves a left column for its status icon: the row's
// padding-left is 19px (a 13px icon column plus a 6px gap to the text)
// and the icon sits 5px in, so it never overlaps the task text.  The
// values are read off the rendered row, not the stylesheet source.

function testStatusIconFitsInsideRowPadding() {
  const {win} = makePage();
  loadMainCss(win);
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  send(win, {
    type: 'history',
    offset: 0,
    sessions: [
      makeRow({task_id: 1, title: 'running task', is_running: true, endTs: 0}),
    ],
  });
  const row = win.document.querySelector('#history-list .running-item');
  assert.ok(row, 'the history list renders a .running-item row');
  const dot = row.querySelector(':scope > .sidebar-item-running');
  assert.ok(dot, 'the running row carries its status icon as a direct child');

  const rowStyle = win.getComputedStyle(row);
  const dotStyle = win.getComputedStyle(dot);
  assert.strictEqual(
    rowStyle.paddingLeft,
    '19px',
    'a history row keeps a 19px left column (13px icon + 6px gap)',
  );
  assert.strictEqual(
    dotStyle.left,
    '5px',
    'the status icon sits 5px into the row (halved from 10px)',
  );
  const padPx = Number.parseInt(rowStyle.paddingLeft, 10);
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
    `  ok - dot (${leftPx}px + ${widthPx}px) fits inside the ${padPx}px padding column`,
  );
}

testStatusIconFitsInsideRowPadding();
console.log('historyTaskLeftPadding.test.js: all assertions passed.');
