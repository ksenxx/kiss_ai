// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the Task Progress
// subpanel's time-remaining code in media/main.js (the region fenced
// by `// tasketa-coverage:start` / `// tasketa-coverage:end`) when
// running the functional jsdom suite in test/taskProgressEta.test.js.
// A line counts as covered only when every byte of its code ran, so an
// untaken `return`, `||` operand or `?:` arm on a line that otherwise
// ran fails the gate too: full line coverage here is full branch
// coverage.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'taskProgressEta.test.js',
  mark: 'tasketa',
  label: 'task progress time remaining',
  everyByte: true,
  targets: [{file: 'main.js', sourceUrl: 'tasketa-main.js', regions: 1}],
});
