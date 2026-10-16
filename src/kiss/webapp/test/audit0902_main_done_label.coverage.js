// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the task_done label
// code in media/main.js (the region fenced by
// `// donelabel-coverage:start` / `// donelabel-coverage:end`) when
// running the functional jsdom suite in
// test/audit0902_main_done_label.test.js.  That suite covers
// the stamped span, the fallback for the visible tab, for a background
// tab, for an unknown tab id and for a tab with no recorded start, so
// full line coverage of the fenced region is full branch coverage.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'audit0902_main_done_label.test.js',
  mark: 'donelabel',
  label: 'task_done label',
  targets: [
    {file: 'main.js', sourceUrl: 'audit0902-donelabel-main.js'},
  ],
});
