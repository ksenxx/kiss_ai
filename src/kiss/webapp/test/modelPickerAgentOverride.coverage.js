// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the model-picker override code
// in media/main.js (the regions fenced by `// modelpick-coverage:start` /
// `// modelpick-coverage:end`) when running the functional jsdom suite in
// test/modelPickerAgentOverride.test.js.
//
// That code is the whole point of the feature -- it decides whether the picker
// shows the user's own model or the one a running agent switched itself to --
// so an unexecuted line there is an untested claim about what the user sees.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'modelPickerAgentOverride.test.js',
  mark: 'modelpick',
  label: 'model picker override',
  targets: [
    {file: 'main.js', sourceUrl: 'modelpick-main.js', regions: 4},
  ],
});
