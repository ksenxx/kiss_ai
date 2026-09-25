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

// V8 flags for every suite process.  Node 24+ enables the Maglev optimizing
// compiler and background ("concurrent") Sparkplug compilation.  Either can
// leave a V8 worker thread parked waiting for a main-thread GC at the moment
// `process.exit()` joins the worker pool, and the join then never returns
// (nodejs/node#64274, fix PR #66171 unmerged as of Node 25.8.1).  One suite
// in ~700 runs hung this way after printing its final "passed" line.  The
// flags must be passed on the command line: NODE_OPTIONS rejects V8 flags,
// and `v8.setFlagsFromString` inside the suite leaves code compiled before
// the call eligible for the background compilers.
const V8_FLAGS = ['--no-maglev', '--no-concurrent-sparkplug'];

// Upper bound on one suite's wall-clock time.  The slowest suite
// (editorTitleGitCommit.test.js) takes about two minutes and the next one
// about half a minute; this exists so that any hang (in a suite, or in the
// runtime at exit) is reported as a failure naming the suite instead of
// stalling `npm test` until someone kills it.
const SUITE_TIMEOUT_MS = 10 * 60 * 1000;

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
    const res = spawnSync(
      process.execPath,
      [...V8_FLAGS, path.join(TEST_DIR, file)],
      {
        stdio: 'inherit',
        cwd: path.dirname(TEST_DIR),
        timeout: SUITE_TIMEOUT_MS,
        killSignal: 'SIGKILL',
      },
    );
    if (res.error && res.error.code === 'ETIMEDOUT') {
      console.error(
        `${file}: no exit after ${SUITE_TIMEOUT_MS / 1000}s, killed`,
      );
    }
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
