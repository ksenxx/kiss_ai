// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires full coverage of the code in media/main.js
// that labels a task that is over with the time it took (the regions
// fenced by `// donelabel-coverage:start` / `// donelabel-coverage:end`)
// when running the functional jsdom suite in
// test/audit0902_main_done_label.test.js.  Every non-whitespace byte of
// every fenced line must have run (`everyByte`), so an untaken arm of a
// conditional or operand of `&&` / `||` fails the gate: the suite covers
// the stamped span and its fallbacks, a replayed history row with and
// without an end, an end that comes as a bare status, and the Time row
// of the tab's own task, of a neighbouring one and of a chat behind a
// file view.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'audit0902_main_done_label.test.js',
  mark: 'donelabel',
  label: 'task_done label',
  everyByte: true,
  targets: [
    {file: 'main.js', sourceUrl: 'audit0902-donelabel-main.js'},
  ],
});
