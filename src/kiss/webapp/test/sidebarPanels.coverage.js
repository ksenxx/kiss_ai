// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the right sidebar's
// Schedule, Apps and Spend subpanel code in media/main.js (the region fenced
// by `// sidebarpanels-coverage:start` / `// sidebarpanels-coverage:end`)
// when running the functional jsdom suite in test/sidebarPanels.test.js.
// That suite drives every branch -- every surface, empty and filled
// replies, each badge and time format, both launch paths, the poll
// timer and the refresh buttons -- so full line coverage of the fenced
// region is full branch coverage.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'sidebarPanels.test.js',
  mark: 'sidebarpanels',
  label: 'sidebar panels',
  targets: [
    {file: 'main.js', sourceUrl: 'sidebarpanels-main.js'},
  ],
});
