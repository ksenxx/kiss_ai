// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the drawer feature code
// fenced by `// drawer-coverage:start` / `:end` in media/*.js when
// running test/drawerPanels.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'drawerPanels.test.js',
  mark: 'drawer',
  label: 'drawer feature',
  targets: [
    {file: 'main.js', sourceUrl: 'drawer-main.js', minRegions: 2},
  ],
});
