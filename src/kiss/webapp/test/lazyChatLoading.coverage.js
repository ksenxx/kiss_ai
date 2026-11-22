// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate for the lazily loaded transcript panels in
// media/main.js: runs lazyChatLoading.test.js under V8 coverage and
// requires every line of the `lazypanel-coverage` regions (the panels
// a lazily loaded transcript left with the daemon, their loading and
// their share export, and the events that tell whether a lost prompt
// ran) to have executed.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'lazyChatLoading.test.js',
  mark: 'lazypanel',
  label: 'lazy chat loading',
  targets: [
    {file: 'main.js', sourceUrl: 'lazypanel-main.js', regions: 2},
  ],
});
