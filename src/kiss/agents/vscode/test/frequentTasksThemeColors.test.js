// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The frequent-tasks list's per-task pastel tint (a hue hashed from the
// task text, with near-black text) is a VS Code webview cue only.  The
// remote page paints with VS Code's Dark Modern / Light Modern theme
// colours alone (remote-codex.css), so renderFrequentTasks must leave
// the rows' inline colours empty there.

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

function testWebviewRowsKeepThePastelTint() {
  const {win} = makeWebview();
  loadFrequent(win);
  const rows = frequentRows(win);
  assert.strictEqual(rows.length, 2, 'two frequent rows rendered');
  for (const row of rows) {
    // jsdom serialises the hashed hsl() tint as rgb().
    assert.match(
      row.style.backgroundColor,
      /^rgb\(/,
      `webview row keeps its hashed tint; got ${row.style.backgroundColor}`,
    );
    assert.strictEqual(
      row.style.color,
      'rgb(26, 26, 26)',
      'webview row keeps the near-black text over the pastel',
    );
  }
  console.log('PASS webview frequent rows keep the pastel tint');
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
  testWebviewRowsKeepThePastelTint();
  testRemoteRowsHaveNoInlineColours();
  console.log('All frequentTasksThemeColors tests passed');
}

main();
