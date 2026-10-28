// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Runs every test in this directory. The list is discovered from disk rather
// than hand-maintained in package.json, which silently drifted and left 32
// suites unrun. Each suite runs in its own node process so a suite cannot
// leak globals, timers or listeners into the next; the suites share no
// port, file or global state, so a pool of them runs at once (one per CPU,
// capped) and `npm test` takes about the longest suite instead of the sum
// of all of them (~25 s instead of ~700 s on an 80-core machine). Suites
// are started in sorted order and each one's output is printed in one
// piece when it ends, so the log stays grouped by suite. `--serial` runs
// them one at a time, for debugging.

/* global require, __dirname, console, process, setTimeout, clearTimeout */

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawn} = require('child_process');

const TEST_DIR = __dirname;

// V8 flags for every suite process.  Node 24+ enables the Maglev optimizing
// compiler and background ("concurrent") Sparkplug compilation.  Either can
// leave a V8 worker thread parked waiting for a main-thread GC at the moment
// `process.exit()` joins the worker pool, and the join then never returns
// (nodejs/node#64274, fix PR #66171 unmerged as of Node 25.8.1).  One suite
// in ~700 runs hung this way after printing its final "passed" line.  The
// flags must be passed on the command line: NODE_OPTIONS rejects V8 flags,
// and `v8.setFlagsFromString` inside the suite leaves code compiled before
// the call eligible for the background compilers.  Spelled out here rather
// than required from coverage_gate.js: conc1018_m_runall_grandchild copies
// this file alone into a scratch directory and runs it there.
const V8_FLAGS = ['--no-maglev', '--no-concurrent-sparkplug'];

// Upper bound on one suite's wall-clock time, so that a hang (in a suite,
// or in the runtime at exit) is reported as a failure naming the suite
// instead of stalling `npm test` until someone kills it.  Measured with
// every suite timed under a 32-wide pool on an 80-core box (2026-10):
// the slowest, remotePasswordBypass.coverage.js, took 23 s and the next
// four 9-16 s, so two minutes is five times the slowest suite's real time
// and a hung suite costs the run two minutes instead of ten.
const SUITE_TIMEOUT_MS = 2 * 60 * 1000;

// Suites running at once: one per CPU, capped (each is a node + jsdom
// process of a few hundred MB, and the *.coverage.js gates spawn a child
// suite of their own).
const MAX_JOBS = 32;

// Every *.test.js and *.coverage.js suite.  A *.coverage.js gate runs its
// *.test.js as a child (under NODE_V8_COVERAGE) and fails when that child
// fails, so a suite with a gate is run through the gate only.
function testFiles() {
  const files = fs.readdirSync(TEST_DIR);
  return files
    .filter(f => f.endsWith('.test.js') || f.endsWith('.coverage.js'))
    .filter(
      f =>
        !(
          f.endsWith('.test.js') &&
          files.includes(f.replace(/\.test\.js$/, '.coverage.js'))
        ),
    )
    .sort();
}

function poolSize() {
  if (process.argv.includes('--serial')) return 1;
  return Math.max(1, Math.min(os.cpus().length, MAX_JOBS));
}

// The suite processes not yet reported, each the leader of its own
// process group (spawned `detached`), so that killing a suite also kills
// whatever it spawned.
const children = new Set();

// Set by SIGINT/SIGTERM: no further suite is started.
let stopping = false;

/** SIGKILL `child`'s whole process group; a group already gone is fine. */
function killGroup(child) {
  try {
    process.kill(-child.pid, 'SIGKILL');
  } catch {
    // ESRCH: the group exited before the kill.
  }
}

/** Whether the suite process has exited (been reaped), whether or not
 * its stdio pipes have closed yet. */
function exited(child) {
  return child.exitCode !== null || child.signalCode !== null;
}

/**
 * Run one suite in its own node process and call `done(file, passed)`
 * when it ends.  Its stdout and stderr are buffered and replayed to
 * this process's streams in one piece, under a header, so the output
 * of suites running at once does not interleave.  Normally the suite is
 * reported when its pipes close (`close`), so that no trailing output is
 * lost.  After SUITE_TIMEOUT_MS the suite's process group is SIGKILLed
 * and the suite is reported as failed, with the output so far, as soon
 * as its process has exited, which may be long before the timeout: a
 * process the suite started can inherit its pipes and outlive it (even
 * the kill, when it left the suite's process group), so `close` is not
 * waited for and the runner's ends of the pipes are dropped.
 */
function runSuite(file, index, total, done) {
  const child = spawn(
    process.execPath,
    [...V8_FLAGS, path.join(TEST_DIR, file)],
    {
      stdio: ['ignore', 'pipe', 'pipe'],
      cwd: path.dirname(TEST_DIR),
      detached: true,
    },
  );
  children.add(child);
  const output = [];
  child.stdout.on('data', chunk =>
    output.push({stream: process.stdout, chunk}),
  );
  child.stderr.on('data', chunk =>
    output.push({stream: process.stderr, chunk}),
  );
  let timedOut = false;
  let exitedBeforeTimeout = false;
  let reported = false;
  const timer = setTimeout(onTimeout, SUITE_TIMEOUT_MS);
  function onTimeout() {
    timedOut = true;
    exitedBeforeTimeout = exited(child);
    killGroup(child);
    if (exitedBeforeTimeout) report(child.exitCode);
  }
  function report(code) {
    if (reported || stopping) return;
    reported = true;
    clearTimeout(timer);
    children.delete(child);
    child.stdout.destroy();
    child.stderr.destroy();
    console.log(`\n[${index + 1}/${total}] ${file}`);
    for (const {stream, chunk} of output) stream.write(chunk);
    if (timedOut) {
      const seconds = SUITE_TIMEOUT_MS / 1000;
      console.error(
        exitedBeforeTimeout
          ? `${file}: exited, but a process it started still held its ` +
              `output ${seconds}s later, killed`
          : `${file}: no exit after ${seconds}s, killed`,
      );
    }
    done(file, code === 0 && !timedOut);
  }
  child.on('close', report);
  child.on('exit', code => {
    if (timedOut) report(code);
  });
  // The process could not be started (EAGAIN, EMFILE, ...): a failure of
  // this suite, not of the runner.
  child.on('error', err => {
    output.push({
      stream: process.stderr,
      chunk: Buffer.from(`${file}: ${err}\n`),
    });
    report(1);
  });
}

/**
 * SIGINT/SIGTERM: start no more suites, SIGKILL every unreported suite's
 * process group (the suites are in their own process groups, so the
 * terminal's Ctrl-C does not reach them) and exit with 128 + signal once
 * every suite process has exited, so that none is handed to our parent
 * as a zombie.  A SIGKILLed child exits promptly, so this is bounded.
 */
function killAllAndExit(signal) {
  stopping = true;
  let alive = 0;
  function onSuiteExit() {
    alive -= 1;
    if (alive === 0) process.exit(128 + os.constants.signals[signal]);
  }
  for (const child of children) {
    killGroup(child);
    if (exited(child)) continue;
    alive += 1;
    child.once('exit', onSuiteExit);
  }
  if (alive === 0) process.exit(128 + os.constants.signals[signal]);
}

function main() {
  const files = testFiles();
  if (files.length === 0) {
    console.error('no test files found in ' + TEST_DIR);
    process.exit(1);
  }
  const jobs = poolSize();
  const failed = [];
  let next = 0;
  let running = 0;

  function suiteDone(file, passed) {
    if (!passed) failed.push(file);
    running -= 1;
    if (next < files.length) startSuites();
    else if (running === 0) report();
  }

  function startSuites() {
    while (!stopping && running < jobs && next < files.length) {
      running += 1;
      runSuite(files[next], next, files.length, suiteDone);
      next += 1;
    }
  }

  function report() {
    console.log(
      `\n${files.length - failed.length}/${files.length} suites passed`,
    );
    if (failed.length > 0) {
      console.error('FAILED SUITES:');
      failed.sort().forEach(f => console.error('  ' + f));
      process.exit(1);
    }
  }

  console.log(`running ${files.length} suites, ${jobs} at a time`);
  process.on('SIGINT', killAllAndExit);
  process.on('SIGTERM', killAllAndExit);
  // A runner bug must not leave the detached suites running as orphans.
  process.on('uncaughtException', err => {
    console.error(err);
    killAllAndExit('SIGTERM');
  });
  startSuites();
}

main();
