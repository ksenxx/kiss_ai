// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end test: every suggestion the daemon's `recommendations`
// reply paints in the RECOMMENDATIONS subpanel sits in its own bordered
// box (1px solid, rounded), while the neighbouring Machines rows stay
// borderless.
// The page is booted with the real chat.html, main.css and main.js;
// jsdom cannot resolve var() colours, so the test checks the border's
// style and width (literal values) and the rounded corners.

'use strict';

const assert = require('assert');
const {
  makePage: makeSharedPage,
  send,
  stubRaf,
  fakeIntervals,
  fakeMatchMedia,
  loadRemoteCss,
} = require('./simplify2_harness');

const TASKS = [
  'Add a changelog entry for /recommendations',
  'Write an end-to-end test for the Spend heatmap',
  'Fix the flaky terminal tab test',
];

// The daemon's `machines` row shape (sidebar_panels.py).
const MACHINES = [
  {
    name: 'laptop',
    address: 'https://10.0.0.5:8787',
    local: true,
    status: 'reachable',
  },
];

/** Boot a chat page with the given body attributes and the real CSS. */
function makePage() {
  const {win} = makeSharedPage({
    extraScripts: ['marked.min.js'],
    beforeScripts: win => {
      stubRaf(win);
      fakeIntervals(win);
      win.ResizeObserver = class {
        observe() {}
        disconnect() {}
      };
      fakeMatchMedia(win, true);
      loadRemoteCss(win);
    },
  });
  return win;
}

function checkSurface() {
  const win = makePage();
  send(win, {type: 'recommendations', generatedAt: Date.now(), tasks: TASKS});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  const rows = Array.from(
    win.document.querySelectorAll(
      '#meta-recommendations-list .recommendation-row',
    ),
  );
  assert.strictEqual(rows.length, TASKS.length, 'one row per suggestion');
  for (const row of rows) {
    const cs = win.getComputedStyle(row);
    assert.strictEqual(cs.borderTopStyle, 'solid', 'top border');
    assert.strictEqual(cs.borderRightStyle, 'solid', 'right border');
    assert.strictEqual(cs.borderBottomStyle, 'solid', 'bottom border');
    assert.strictEqual(cs.borderLeftStyle, 'solid', 'left border');
    assert.strictEqual(cs.borderTopWidth, '1px', '1px border');
    assert.strictEqual(cs.borderLeftWidth, '1px', '1px border');
    assert.notStrictEqual(cs.borderRadius, '0px', 'rounded corners');
    assert.notStrictEqual(cs.borderRadius, '', 'rounded corners');
  }
  const machine = win.document.querySelector(
    '#meta-machines-list .machine-row',
  );
  assert.ok(machine, 'a machine row to compare against');
  assert.ok(
    !machine.classList.contains('unreachable'),
    'setup: the row is the reachable machine a user sees',
  );
  const machineStyle = win.getComputedStyle(machine);
  assert.strictEqual(
    machineStyle.borderTopStyle,
    'none',
    'the border is specific to recommendation rows',
  );
  // An empty reply leaves nothing bordered behind.
  send(win, {type: 'recommendations', generatedAt: 0, tasks: []});
  assert.strictEqual(
    win.document.querySelectorAll(
      '#meta-recommendations-list .recommendation-row',
    ).length,
    0,
    'an empty reply clears the rows',
  );
  win.close();
}

checkSurface();
console.log('all recommendationRowBorder tests passed');
