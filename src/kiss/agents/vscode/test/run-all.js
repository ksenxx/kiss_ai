// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Runs every test in this directory. The list is discovered from disk rather
// than hand-maintained in package.json, which silently drifted and left 32
// suites unrun. Files are executed in sorted order so a failure is always
// reproducible, and each one runs in its own node process so a suite cannot
// leak globals, timers or listeners into the next.

/* global require, __dirname, console, process */

'use strict';

const fs = require('fs');
const path = require('path');
const {spawnSync} = require('child_process');

const TEST_DIR = __dirname;

function testFiles() {
  return fs
    .readdirSync(TEST_DIR)
    .filter(f => f.endsWith('.test.js') || f.endsWith('.coverage.js'))
    .sort();
}

function main() {
  const files = testFiles();
  if (files.length === 0) {
    console.error('no test files found in ' + TEST_DIR);
    process.exit(1);
  }
  const failed = [];
  files.forEach((file, i) => {
    console.log(`\n[${i + 1}/${files.length}] ${file}`);
    const res = spawnSync(process.execPath, [path.join(TEST_DIR, file)], {
      stdio: 'inherit',
      cwd: path.dirname(TEST_DIR),
    });
    if (res.status !== 0) failed.push(file);
  });
  console.log(
    `\n${files.length - failed.length}/${files.length} suites passed`,
  );
  if (failed.length > 0) {
    console.error('FAILED SUITES:');
    failed.forEach(f => console.error('  ' + f));
    process.exit(1);
  }
}

main();
