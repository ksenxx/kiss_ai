// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the right sidebar's
// Skills subpanel code in media/main.js (the region fenced by
// `// skillspanel-coverage:start` / `// skillspanel-coverage:end`) when
// running the functional jsdom suite in test/skillsPanel.test.js.  A
// line counts as covered only when every byte of its code ran, so an
// untaken `return`, `||` operand or `?:` arm on a line that otherwise
// ran fails the gate too: full line coverage here is full branch
// coverage.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'skillsPanel.test.js',
  mark: 'skillspanel',
  label: 'skills panel',
  everyByte: true,
  targets: [{file: 'main.js', sourceUrl: 'skillspanel-main.js', regions: 1}],
});
