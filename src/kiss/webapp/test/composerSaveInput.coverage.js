// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the composer's Save
// button code in media/main.js (the region fenced by
// `// savedinput-coverage:start` / `// savedinput-coverage:end`) when
// running the jsdom suite in test/composerSaveInput.test.js.  A line
// counts as covered only when every byte of its code ran, so an untaken
// `return`, `||` operand or `?:` arm on a line that otherwise ran fails
// the gate too: full line coverage here is full branch coverage.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'composerSaveInput.test.js',
  mark: 'savedinput',
  label: 'composer save button',
  everyByte: true,
  targets: [{file: 'main.js', sourceUrl: 'savedinput-main.js', regions: 1}],
});
