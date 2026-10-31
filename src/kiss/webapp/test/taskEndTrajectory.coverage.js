// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate for the task-end Trajectory fold in media/main.js: runs
// taskEndTrajectory.test.js under V8 coverage and requires every line
// of the `trajectory-coverage` regions (foldTrajectory, foldTabTrajectory
// and their call sites in the terminal-event, status and replay paths)
// to have executed.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'taskEndTrajectory.test.js',
  mark: 'trajectory',
  label: 'trajectory fold',
  targets: [
    {file: 'main.js', sourceUrl: 'trajectory-main.js', regions: 4},
  ],
});
