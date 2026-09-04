// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of saveKissConfig() and
// ensureRemotePassword() in out/DependencyInstaller.js (the regions fenced
// by `// audit0902-coverage:start` / `// audit0902-coverage:end`) when
// running the functional suite in
// test/audit0902_vscode_ext2_remote_password_single_writer.test.js.

/* global require, process, console, __dirname */

'use strict';

const fs = require('fs');
const path = require('path');
const {runGate} = require('./audit0902_vscode_ext_voice_lifecycle.coverage.js');

const TARGET = path.join(__dirname, '..', 'out', 'DependencyInstaller.js');
if (!fs.existsSync(TARGET)) {
  console.log(`SKIP: ${TARGET} missing — run \`npm run compile\``);
  process.exit(0);
}

runGate(
  'audit0902_vscode_ext2_remote_password_single_writer.coverage',
  [
    path.join(
      __dirname,
      'audit0902_vscode_ext2_remote_password_single_writer.test.js',
    ),
  ],
  [TARGET],
);
