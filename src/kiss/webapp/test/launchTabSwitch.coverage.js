// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the launch-switch code
// fenced by `// launchswitch-coverage:start` / `:end` in media/*.js when
// running test/launchTabSwitch.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'launchTabSwitch.test.js',
  mark: 'launchswitch',
  label: 'launch-switch',
  targets: [
    {file: 'main.js', sourceUrl: 'launch-main.js'},
  ],
});
