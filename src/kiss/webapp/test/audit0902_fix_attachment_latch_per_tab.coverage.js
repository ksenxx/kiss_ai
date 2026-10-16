// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate for the per-tab attachment latch in media/main.js (the
// region fenced by `// attachlatch-coverage:start` / `:end` in
// sendMessage()) when running
// test/audit0902_fix_attachment_latch_per_tab.test.js.
//
// Stricter than a line gate: every non-whitespace byte of every code line
// in the region must have executed.  V8 block coverage assigns a zero
// count to an untaken `return;`, `finally` body or `||` right operand
// even when it shares its line with executed code, so an unexercised
// branch fails the gate.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'audit0902_fix_attachment_latch_per_tab.test.js',
  mark: 'attachlatch',
  label: 'attachment-latch',
  everyByte: true,
  targets: [
    {file: 'main.js', sourceUrl: 'audit0902-attachlatch-main.js'},
  ],
});
