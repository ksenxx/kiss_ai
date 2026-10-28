// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the opened-chat landing
// code in media/main.js (the regions fenced by
// `// landing-coverage:start` / `// landing-coverage:end`:
// landOpenedTranscript and its call in restoreTab) when running the
// functional jsdom suite in test/openedTaskLanding.test.js.  That suite
// opens a finished chat (replay and tab switch, the latter with a
// neighbouring task spliced in above the own one), a running chat
// (replay and tab switch) and a tab with no transcript, so full line
// coverage of the fenced regions is full branch coverage.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'openedTaskLanding.test.js',
  mark: 'landing',
  label: 'opened-chat landing',
  targets: [{file: 'main.js', sourceUrl: 'opened-task-landing-main.js'}],
});
