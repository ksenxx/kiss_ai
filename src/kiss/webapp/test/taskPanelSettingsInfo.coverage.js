// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the task-settings info code
// fenced by `// taskinfo-coverage:start` / `:end` in media/*.js when
// running test/taskPanelSettingsInfo.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'taskPanelSettingsInfo.test.js',
  mark: 'taskinfo',
  label: 'task-settings info',
  targets: [
    {file: 'main.js', sourceUrl: 'taskinfo-main.js', minRegions: 3},
  ],
});
