// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the victory-sound code
// in media/main.js (the regions fenced by `// victory-coverage:start` /
// `// victory-coverage:end`: the Settings checkbox, playVictorySound,
// isUsersTaskTab and the task_done hook) when running the functional
// jsdom suite in test/victorySound.test.js.  That suite covers a visible,
// a background, a sub-agent, an unknown and an unnamed tab, the copy of
// an end sent to a tab viewing the task, the muted
// and the storage-less page, a missing Audio constructor or clip URL,
// a play() that rejects, returns nothing or throws, and a player whose
// clip fails to load once and twice, so full line coverage of the
// fenced regions is full branch coverage.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'victorySound.test.js',
  mark: 'victory',
  label: 'victory sound',
  targets: [{file: 'main.js', sourceUrl: 'victory-main.js', regions: 2}],
});
