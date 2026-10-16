// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the panel-timestamp code
// fenced by `// panelts-coverage:start` / `:end` in media/*.js when
// running test/panelEventTimestamp.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'panelEventTimestamp.test.js',
  mark: 'panelts',
  label: 'panel-timestamp',
  targets: [
    {file: 'panelCopy.js', sourceUrl: 'panelts-panelCopy.js'},
    {file: 'main.js', sourceUrl: 'panelts-main.js'},
  ],
});
