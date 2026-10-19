// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, process, console */

// The history date picker's cells are civil dates (a year, a month and
// a day of the month), not instants.  Formatting one by way of a local
// `new Date(y, m, d)` normalizes a civil day the zone skipped: Samoa
// jumped from 2011-12-29 straight to 2011-12-31, so under
// TZ=Pacific/Apia `new Date(2011, 11, 30)` is December 31 and the cell
// labelled "30" would commit `2011-12-31`.  The cell must format its
// own components.

// Node reads TZ when a Date is first formatted; set it before any Date
// work and prove the zone really skips the day the test relies on.
process.env.TZ = 'Pacific/Apia';

const assert = require('assert');
const {makePage} = require('./simplify2_harness');

assert.strictEqual(
  new Date(2011, 11, 30).getDate(),
  31,
  'precondition: Pacific/Apia skipped 2011-12-30',
);

const {win} = makePage();
const doc = win.document;
const input = doc.getElementById('hf-from');
const btn = doc.getElementById('hf-from-btn');
assert.ok(input && btn, 'the history From date field and its picker button');
input.value = '2011-12-01';
btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

const pop = doc.getElementById('kiss-datepicker-pop');
assert.ok(pop, 'the picker opens');
assert.strictEqual(
  pop.querySelector('.dp-title').textContent,
  'December 2011',
  'the picker shows the month of the field',
);
const cells = Array.from(pop.querySelectorAll('.dp-day[data-date]'));
assert.strictEqual(cells.length, 31, 'December has 31 cells');
for (const cell of cells) {
  assert.strictEqual(
    cell.dataset.date,
    '2011-12-' + (cell.textContent.length === 1 ? '0' : '') + cell.textContent,
    'cell ' + cell.textContent + ' commits the date it shows',
  );
}
const day30 = cells.find(c => c.textContent === '30');
assert.strictEqual(day30.dataset.date, '2011-12-30');
day30.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
assert.strictEqual(
  input.value,
  '2011-12-30',
  'clicking the cell labelled 30 selects December 30, not 31',
);
assert.strictEqual(
  doc.getElementById('kiss-datepicker-pop'),
  null,
  'the pick closes the picker',
);
win.close();
console.log('  ok - a calendar cell commits the civil date it shows');
