// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the chevron collapse-pass code
// fenced by `// chevron-coverage:start` / `:end` in media/*.js when
// running test/taskPanelExpandFullText.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'taskPanelExpandFullText.test.js',
  mark: 'chevron',
  label: 'chevron collapse-pass',
  targets: [
    {file: 'main.js', sourceUrl: 'taskpanel-main.js'},
  ],
});
