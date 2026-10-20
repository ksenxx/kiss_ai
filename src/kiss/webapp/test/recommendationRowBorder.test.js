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
const fs = require('fs');
const path = require('path');
const {
  makePage: makeSharedPage,
  send,
  stubRaf,
  fakeMatchMedia,
  loadMainCss,
  MEDIA,
} = require('./simplify2_harness');

const TASKS = [
  'Add a changelog entry for /recommendations',
  'Write an end-to-end test for the Spend heatmap',
  'Fix the flaky terminal tab test',
];

const MACHINES = [
  {name: 'laptop', url: 'https://10.0.0.5:8787', local: true, reachable: true},
];

/** Boot a chat page with the given body attributes and the real CSS. */
function makePage() {
  const {win} = makeSharedPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts: win => {
      stubRaf(win);
      win.setInterval = function () {
        return 1;
      };
      win.clearInterval = function () {};
      win.ResizeObserver = class {
        observe() {}
        disconnect() {}
      };
      fakeMatchMedia(win, true);
      loadMainCss(win);
      const remoteStyle = win.document.createElement('style');
      remoteStyle.textContent = fs.readFileSync(
        path.join(MEDIA, 'remote-codex.css'),
        'utf8',
      );
      win.document.head.appendChild(remoteStyle);
    },
  });
  return win;
}

function checkSurface() {
  const label = 'remote web app';
  const win = makePage();
  send(win, {type: 'recommendations', generatedAt: Date.now(), tasks: TASKS});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  const rows = Array.from(
    win.document.querySelectorAll(
      '#meta-recommendations-list .recommendation-row',
    ),
  );
  assert.strictEqual(
    rows.length,
    TASKS.length,
    `${label}: one row per suggestion`,
  );
  for (const row of rows) {
    const cs = win.getComputedStyle(row);
    assert.strictEqual(cs.borderTopStyle, 'solid', `${label}: top border`);
    assert.strictEqual(cs.borderRightStyle, 'solid', `${label}: right border`);
    assert.strictEqual(
      cs.borderBottomStyle,
      'solid',
      `${label}: bottom border`,
    );
    assert.strictEqual(cs.borderLeftStyle, 'solid', `${label}: left border`);
    assert.strictEqual(cs.borderTopWidth, '1px', `${label}: 1px border`);
    assert.strictEqual(cs.borderLeftWidth, '1px', `${label}: 1px border`);
    assert.notStrictEqual(cs.borderRadius, '0px', `${label}: rounded corners`);
    assert.notStrictEqual(cs.borderRadius, '', `${label}: rounded corners`);
  }
  const machine = win.document.querySelector(
    '#meta-machines-list .machine-row',
  );
  assert.ok(machine, `${label}: a machine row to compare against`);
  const machineStyle = win.getComputedStyle(machine);
  assert.strictEqual(
    machineStyle.borderTopStyle,
    'none',
    `${label}: the border is specific to recommendation rows`,
  );
  // An empty reply leaves nothing bordered behind.
  send(win, {type: 'recommendations', generatedAt: 0, tasks: []});
  assert.strictEqual(
    win.document.querySelectorAll(
      '#meta-recommendations-list .recommendation-row',
    ).length,
    0,
    `${label}: an empty reply clears the rows`,
  );
  win.close();
}

checkSurface();
console.log('all recommendationRowBorder tests passed');
