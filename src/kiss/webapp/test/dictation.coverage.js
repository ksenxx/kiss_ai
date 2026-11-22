// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the dictation code in
// media/voice.js (the region fenced by `// dictation-coverage:start` /
// `// dictation-coverage:end`) when running the functional jsdom suite in
// test/dictation.test.js.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'dictation.test.js',
  mark: 'dictation',
  label: 'dictation',
  targets: [{file: 'voice.js', sourceUrl: 'dictation-voice.js', regions: 1}],
});
