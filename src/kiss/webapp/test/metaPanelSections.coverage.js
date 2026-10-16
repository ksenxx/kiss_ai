// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the task-info panel
// SECTIONS code in media/main.js (the region fenced by
// `// metasections-coverage:start` / `// metasections-coverage:end`)
// when running the functional jsdom suite in
// test/metaPanelSections.test.js.  That suite drives every branch --
// collapse and expand, stored and unstored state, shown / hidden /
// static / draggable separators, keyboard, double-click and pointer
// resizing, and a window without localStorage -- so full line
// coverage of the fenced region is full branch coverage.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'metaPanelSections.test.js',
  mark: 'metasections',
  label: 'task-info sections',
  targets: [
    {file: 'main.js', sourceUrl: 'metasections-main.js'},
  ],
});
