// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the code in media/main.js
// that points the model picker and the open settings sheet at the machine
// the chat header's drop-down shows (the regions fenced by
// `// machinescope-coverage:start` / `// machinescope-coverage:end`) when
// running the functional jsdom suite in test/machineScopedPicker.test.js.
//
// That code decides whose models the picker lists, whose default a pick
// changes and whose settings the sheet saves, so an unexecuted line there
// is an untested claim about which machine the user is changing.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'machineScopedPicker.test.js',
  mark: 'machinescope',
  label: 'machine-scoped picker and settings',
  targets: [
    {file: 'main.js', sourceUrl: 'machinescope-main.js', regions: 3},
  ],
});
