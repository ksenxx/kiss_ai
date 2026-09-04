// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of consumeTipsFirstRun()
// in out/SorcarTab.js (the region fenced by `// audit0903-coverage:start`
// / `// audit0903-coverage:end`) when running the functional suite in
// test/audit0903_tips_first_run.test.js.  The suite's child processes
// inherit NODE_V8_COVERAGE, so the racing claims themselves are what
// covers the win and lose paths.

'use strict';

const fs = require('fs');
const path = require('path');
const {runGate} = require('./audit0903_voice_intent.coverage.js');

const TARGET = path.join(__dirname, '..', 'out', 'SorcarTab.js');
if (!fs.existsSync(TARGET)) {
  console.log(`SKIP: ${TARGET} missing — run \`npm run compile\``);
  process.exit(0);
}
runGate(
  'audit0903_tips_first_run.coverage',
  [path.join(__dirname, 'audit0903_tips_first_run.test.js')],
  [TARGET],
);
