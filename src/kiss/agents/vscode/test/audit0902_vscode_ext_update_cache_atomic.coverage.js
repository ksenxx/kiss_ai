// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of writeCache() in
// src/UpdateChecker.js (the region fenced by `// audit0902-coverage:start`
// / `// audit0902-coverage:end`) when running the functional suite in
// test/audit0902_vscode_ext_update_cache_atomic.test.js.

'use strict';

const path = require('path');
const {runGate} = require('./audit0902_vscode_ext_voice_lifecycle.coverage.js');

runGate(
  'audit0902_vscode_ext_update_cache_atomic.coverage',
  [path.join(__dirname, 'audit0902_vscode_ext_update_cache_atomic.test.js')],
  [path.join(__dirname, '..', 'src', 'UpdateChecker.js')],
);
