// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the retryable
// worktree-bar code in media/main.js (the regions fenced by
// `// retrybar-coverage:start` / `// retrybar-coverage:end`) when
// running the functional jsdom suite in
// test/audit0902_main_retryable_bar.test.js.  Both branches of
// each `retryable` decision (re-arm vs dismiss, foreground and
// background) are exercised by that suite, so full line coverage of
// the fenced regions is full branch coverage of the fix.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'audit0902_main_retryable_bar.test.js',
  mark: 'retrybar',
  label: 'retryable worktree-bar',
  targets: [
    {file: 'main.js', sourceUrl: 'audit0902-retrybar-main.js', minRegions: 3},
  ],
});
