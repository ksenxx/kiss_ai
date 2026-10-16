// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate for the run-start layout feature of media/main.js: every
// line between the `runstart-coverage:start` / `:end` marks (the status
// handler's fresh-start test, showChatForRun) must be executed by
// runStartLayout.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'runStartLayout.test.js',
  mark: 'runstart',
  label: 'run-start feature',
  targets: [
    {file: 'main.js', sourceUrl: 'runstart-main.js', minRegions: 3},
  ],
});
