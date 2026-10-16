// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the summary-hint code
// fenced by `// summaryhint-coverage:start` / `:end` in media/*.js when
// running test/summaryHeaderExpandHint.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'summaryHeaderExpandHint.test.js',
  mark: 'summaryhint',
  label: 'summary-hint',
  targets: [
    {file: 'main.js', sourceUrl: 'summaryhint-main.js'},
  ],
});
