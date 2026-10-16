// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the tab status-dot
// code in media/main.js (the regions fenced by
// `// faildot-coverage:start` / `// faildot-coverage:end`) when
// running the functional jsdom suite in
// test/audit0902_main_tab_fail_dot.test.js.  That suite drives
// markTabDone with and without a terminal verdict, for the visible and
// a background tab, and task_events with and without a tab, so full
// line coverage of the fenced regions is full branch coverage of the
// fix.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'audit0902_main_tab_fail_dot.test.js',
  mark: 'faildot',
  label: 'tab fail-dot',
  targets: [
    {file: 'main.js', sourceUrl: 'audit0902-faildot-main.js', minRegions: 2},
  ],
});
