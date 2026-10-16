// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the coalesced
// stream-tail sweep code in media/main.js (the regions fenced by
// `// streamtail-coverage:start` / `// streamtail-coverage:end`) when
// running the functional jsdom suite in test/streamTailCoalesce.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'streamTailCoalesce.test.js',
  mark: 'streamtail',
  label: 'stream-tail sweep',
  targets: [
    {file: 'main.js', sourceUrl: 'streamtail-main.js', minRegions: 6},
  ],
});
