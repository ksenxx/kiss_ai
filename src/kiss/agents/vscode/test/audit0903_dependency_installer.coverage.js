// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of every audit0903 change
// in out/DependencyInstaller.js — the regions fenced by
// `// audit0903-coverage:start` / `// audit0903-coverage:end`:
//
//   - resolveSymlinkTargetSync() + writeFileAtomicSync() and their two
//     POSIX callers (installCliScript / writeShellRc), exercised by
//     audit0903_cli_shellrc_atomic_write.test.js (dangling symlink,
//     link chain, link loop, EFBIG write failure, rename failure,
//     concurrent hammers);
//   - the remote-password prompt lock protocol (readPromptOutcome,
//     checkPromptHolder, acquireRemotePasswordPromptLock,
//     ensureRemotePassword, ensureRemotePasswordLocked), exercised by
//     audit0903_remote_password_single_prompt.test.js (normal
//     contention, dead-holder takeover, deliberate skip, under-lock
//     re-check, live holder never age-evicted, unreadable locks,
//     outcome-write failure).
//
// Both suites run real child processes; NODE_V8_COVERAGE is inherited,
// so the workers' own executions are what covers the cross-process
// branches.  Branches that no real test can time are OUTSIDE the
// fences, each with an inline "(fence split)" comment in the source:
// the win32 sorcar.cmd caller (needs Windows) and none currently in the
// prompt-lock code.

/* global require, process, console, __dirname */

'use strict';

const fs = require('fs');
const path = require('path');
const {runGate} = require('./audit0903_voice_intent.coverage.js');

const TARGET = path.join(__dirname, '..', 'out', 'DependencyInstaller.js');
if (!fs.existsSync(TARGET)) {
  console.log(`SKIP: ${TARGET} missing — run \`npm run compile\``);
  process.exit(0);
}

runGate(
  'audit0903_dependency_installer.coverage',
  [
    path.join(__dirname, 'audit0903_cli_shellrc_atomic_write.test.js'),
    path.join(__dirname, 'audit0903_remote_password_single_prompt.test.js'),
  ],
  [TARGET],
);
