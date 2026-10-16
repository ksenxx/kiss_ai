// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the visible-task panel code
// fenced by `// visibletask-coverage:start` / `:end` in media/*.js when
// running test/visibleTaskLabel.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'visibleTaskLabel.test.js',
  mark: 'visibletask',
  label: 'visible-task panel',
  targets: [
    {file: 'main.js', sourceUrl: 'visibletask-main.js'},
  ],
});
