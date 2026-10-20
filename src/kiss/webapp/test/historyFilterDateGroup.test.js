// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {loadMainCss, makePage} = require('./simplify2_harness');

function assertGroupedTrio(win, which) {
  const doc = win.document;
  const inputId = `hf-${which}`;
  const btnId = `hf-${which}-btn`;
  const label = doc.querySelector(`label[for="${inputId}"]`);
  const input = doc.getElementById(inputId);
  const button = doc.getElementById(btnId);

  assert.ok(label, `"${which}" label[for="${inputId}"] must exist`);
  assert.ok(input, `"${which}" #${inputId} input must exist`);
  assert.ok(button, `"${which}" #${btnId} button must exist`);

  assert.strictEqual(
    label.parentElement,
    input.parentElement,
    `"${which}" label and input must share the same direct parent`,
  );
  assert.strictEqual(
    input.parentElement,
    button.parentElement,
    `"${which}" input and button must share the same direct parent`,
  );

  const group = label.parentElement;
  assert.ok(group, `"${which}" trio must have a direct parent element`);

  assert.ok(
    !group.classList.contains('history-filter-bar'),
    `"${which}" trio must be wrapped in a grouping element ` +
      'distinct from .history-filter-bar (so the three pieces ' +
      'cannot wrap onto separate lines individually)',
  );

  assert.ok(
    group.classList.contains('history-filter-date-group'),
    `"${which}" trio's wrapper must have class ` +
      `"history-filter-date-group" (got: "${group.className}")`,
  );

  const rangeWrap = group.parentElement;
  assert.ok(
    rangeWrap && rangeWrap.classList.contains('history-filter-daterange'),
    `"${which}" date group must be a direct child of ` +
      '.history-filter-daterange',
  );
  const bar = rangeWrap.parentElement;
  assert.ok(
    bar && bar.classList.contains('history-filter-bar'),
    '.history-filter-daterange must be a direct child of ' +
      '.history-filter-bar',
  );

  const kids = Array.from(group.children);
  const idx = el => kids.indexOf(el);
  assert.ok(
    idx(label) < idx(input) && idx(input) < idx(button),
    `"${which}" group children must be ordered: label, input, ` +
      `button (got: [${kids.map(k => k.tagName + (k.id ? '#' + k.id : '')).join(', ')}])`,
  );

  assert.strictEqual(
    kids.length,
    3,
    `"${which}" date group must contain exactly 3 children ` +
      `(label, input, button); got ${kids.length}`,
  );
}

function testFromAndToGroupsAreAtomic() {
  const {win} = makePage();
  assertGroupedTrio(win, 'from');
  assertGroupedTrio(win, 'to');
  win.close();
  console.log(
    '  ok - From and To label/input/button trios share a single ' +
      '.history-filter-date-group parent inside .history-filter-bar',
  );
}

function testDateGroupCssPreventsInternalWrap() {
  const {win} = makePage();
  loadMainCss(win);
  const group = win.document.querySelector(
    'label[for="hf-from"]',
  ).parentElement;
  assert.ok(
    group.classList.contains('history-filter-date-group'),
    'the From trio sits in a .history-filter-date-group',
  );
  const style = win.getComputedStyle(group);
  assert.ok(
    style.flexWrap === 'nowrap' || style.whiteSpace === 'nowrap',
    'the date group never splits its label/input/button across lines; ' +
      `got flex-wrap=${style.flexWrap} white-space=${style.whiteSpace}`,
  );
  assert.ok(
    style.display === 'inline-flex' || style.display === 'inline-block',
    'the date group behaves as a single flex item of .history-filter-bar; ' +
      `got display=${style.display}`,
  );
  win.close();
  console.log(
    '  ok - .history-filter-date-group CSS keeps label/input/button ' +
      'on the same line',
  );
}

function main() {
  testFromAndToGroupsAreAtomic();
  testDateGroupCssPreventsInternalWrap();
  console.log('historyFilterDateGroup.test.js: all assertions passed.');
}

main();
