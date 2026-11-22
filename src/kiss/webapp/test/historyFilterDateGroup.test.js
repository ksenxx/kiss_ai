// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The Date submenu's From and To controls each keep their label, date
// box and calendar button on one line: the trio shares one parent that
// the stylesheet lays out as a single non-wrapping flex row of the
// submenu.

const assert = require('assert');
const {loadMainCss, makePage} = require('./simplify2_harness');

function assertTrioNeverWraps(win, which) {
  const doc = win.document;
  const label = doc.querySelector(`label[for="hf-${which}"]`);
  const input = doc.getElementById(`hf-${which}`);
  const button = doc.getElementById(`hf-${which}-btn`);
  assert.ok(label && input && button, `the "${which}" label, box and button exist`);
  const group = label.parentElement;
  assert.strictEqual(
    input.parentElement,
    group,
    `the "${which}" label and box share one parent`,
  );
  assert.strictEqual(
    button.parentElement,
    group,
    `the "${which}" box and button share one parent`,
  );
  const style = win.getComputedStyle(group);
  assert.ok(
    style.flexWrap === 'nowrap' || style.whiteSpace === 'nowrap',
    `the "${which}" trio never splits across lines; ` +
      `got flex-wrap=${style.flexWrap} white-space=${style.whiteSpace}`,
  );
  assert.strictEqual(
    style.display,
    'flex',
    `the "${which}" trio is one row of the Date submenu`,
  );
  assert.strictEqual(
    group.parentElement,
    doc.querySelector('.fg-submenu[data-fg="date"]'),
    `the "${which}" trio is in the Date submenu`,
  );
}

function testFromAndToTriosStayOnOneLine() {
  const {win} = makePage();
  loadMainCss(win);
  assertTrioNeverWraps(win, 'from');
  assertTrioNeverWraps(win, 'to');
  win.close();
  console.log('  ok - the From and To label/box/button trios stay on one line');
}

testFromAndToTriosStayOnOneLine();
console.log('historyFilterDateGroup.test.js: all assertions passed.');
