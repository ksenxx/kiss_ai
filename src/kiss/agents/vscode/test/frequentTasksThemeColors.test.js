// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The frequent-tasks list paints with the theme's colours alone on
// every surface: renderFrequentTasks stamps no inline background or
// text colour on a row (the old per-task pastel tint, a hue hashed from
// the task text with near-black text, is gone), so the rows are the
// same neutral .sidebar-item panels the history list shows, in the VS
// Code webview and on the remote page (remote-codex.css) alike.

'use strict';

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness.js');

function frequentRows(win) {
  return Array.from(
    win.document.querySelectorAll('#frequent-list .frequent-item'),
  );
}

function loadFrequent(win) {
  send(win, {
    type: 'frequentTasks',
    tasks: [
      {task: 'run the tests', count: 4},
      {task: 'lint everything', count: 2},
    ],
  });
}

function testWebviewRowsHaveNoInlineColours() {
  const {win} = makeWebview();
  loadFrequent(win);
  const rows = frequentRows(win);
  assert.strictEqual(rows.length, 2, 'two frequent rows rendered');
  for (const row of rows) {
    assert.strictEqual(
      row.style.backgroundColor,
      '',
      `webview row must not carry the old pastel tint; got ${row.style.backgroundColor}`,
    );
    assert.strictEqual(
      row.style.color,
      '',
      'webview row must not carry the old near-black inline text colour',
    );
    assert.ok(
      row.classList.contains('sidebar-item'),
      'the row is a plain themed .sidebar-item panel',
    );
  }
  console.log('PASS webview frequent rows carry no inline colours');
}

function testRemoteRowsHaveNoInlineColours() {
  const {win} = makeWebview({
    beforeScripts: w => w.document.body.classList.add('remote-chat'),
  });
  loadFrequent(win);
  const rows = frequentRows(win);
  assert.strictEqual(rows.length, 2, 'two frequent rows rendered');
  for (const row of rows) {
    assert.strictEqual(
      row.style.backgroundColor,
      '',
      'remote row must not carry an inline background (theme colours only)',
    );
    assert.strictEqual(
      row.style.color,
      '',
      'remote row must not carry an inline text colour',
    );
    assert.ok(
      row.querySelector('.sidebar-item-text'),
      'the row still renders its task text',
    );
  }
  console.log('PASS remote frequent rows carry no inline colours');
}

function main() {
  testWebviewRowsHaveNoInlineColours();
  testRemoteRowsHaveNoInlineColours();
  console.log('All frequentTasksThemeColors tests passed');
}

main();
