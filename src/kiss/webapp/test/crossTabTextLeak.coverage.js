// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the cross-tab text leak
// guard code in BOTH media/main.js and media/voice.js (the regions fenced by
// `// tableak-coverage:start` / `// tableak-coverage:end`) when running the
// functional jsdom suite in test/crossTabTextLeak.test.js.
//
// The guard code is the whole point of the fix, so an unexecuted guard line
// is an untested claim. Both files are scanned because the leak is fixed in
// both: main.js decides what may touch a shared surface, voice.js decides
// which conversation a transcript belongs to.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'crossTabTextLeak.test.js',
  mark: 'tableak',
  label: 'cross-tab leak guard',
  targets: [
    {file: 'main.js', sourceUrl: 'tableak-main.js', regions: 22},
    {file: 'voice.js', sourceUrl: 'tableak-voice.js', regions: 3},
  ],
});
