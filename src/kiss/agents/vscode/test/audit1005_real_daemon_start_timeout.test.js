// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end regression test for test/_realDaemon.js: when the real
// daemon never reports READY, startRealDaemon must reject AND kill the
// whole process tree it spawned.
//
// Before the fix the start timer only rejected.  No caller can stop a
// daemon it never received (clickFilePathBridge.test.js and friends
// hold `daemon === null` in their `.finally`), the orphan sat in
// `sys.stdin.readline()` on a pipe the test process still held, and the
// child's open stdio pipes kept the Node event loop (and so the whole
// suite) alive forever.  Killing only the direct child is not enough
// either: `uv run` keeps python as a grandchild, and a python stuck
// before READY would survive uv's death still holding our stdout pipe.
//
// The "daemon" here is a shell script standing in for the `uv` binary:
// it ignores `run python _real_daemon.py <dir>`, backgrounds a sleeper
// that inherits the stdout pipe (the "python"), records the sleeper's
// pid and waits, never printing READY.  This is the only way to reach
// the timeout branch (the real daemon cannot be told to hang).  Skipped
// on win32 where a shebang script is not spawnable.

/* global require, __dirname, console, process, setTimeout */

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {startRealDaemon} = require(path.join(__dirname, '_realDaemon.js'));

if (process.platform === 'win32') {
  console.log('audit1005_real_daemon_start_timeout.test.js: skipped on win32');
  process.exit(0);
}

const START_TIMEOUT_MS = 1000;

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitForNoChildProcess(timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!process.getActiveResourcesInfo().includes('ProcessWrap')) {
      return true;
    }
    await sleep(20);
  }
  return false;
}

// The one-letter process state ('R', 'S', 'Z', ...) or '' when the pid is
// gone.  `process.kill(pid, 0)` alone is not enough: it succeeds for a
// zombie, and when this test runs under a subreaper (containers, some CI
// runners) the killed sleeper stays a zombie until its reaper collects it.
function processState(pid) {
  if (process.platform === 'linux') {
    try {
      const status = fs.readFileSync(`/proc/${pid}/status`, 'utf8');
      const m = /^State:\s*(\S)/m.exec(status);
      return m ? m[1] : '';
    } catch {
      return '';
    }
  }
  const {status, stdout} = require('child_process').spawnSync(
    'ps',
    ['-o', 'stat=', '-p', String(pid)],
    {encoding: 'utf8'},
  );
  return status === 0 ? stdout.trim().charAt(0) : '';
}

// True while the process still runs; a terminated-but-unreaped zombie
// counts as dead because it can no longer hold our stdout pipe open.
function alive(pid) {
  try {
    process.kill(pid, 0);
  } catch (err) {
    if (err.code === 'ESRCH') return false;
  }
  const state = processState(pid);
  return state !== '' && state !== 'Z';
}

function recordedPid(pidFile) {
  if (!fs.existsSync(pidFile)) return 0;
  return Number(fs.readFileSync(pidFile, 'utf8').trim()) || 0;
}

async function runTests() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit1005-'));
  const pidFile = path.join(dir, 'pid');
  const fakeUv = path.join(dir, 'uv');
  fs.writeFileSync(
    fakeUv,
    `#!/bin/sh\nsleep 300 &\necho $! > "${pidFile}"\nwait\n`,
    {mode: 0o755},
  );

  try {
    let rejected = null;
    try {
      await startRealDaemon(fakeUv, dir, process.env, START_TIMEOUT_MS);
    } catch (err) {
      rejected = err;
    }
    assert.ok(rejected, 'startRealDaemon resolved although READY never came');
    assert.match(rejected.message, /did not start in time/);

    assert.ok(
      await waitForNoChildProcess(5000),
      'the never-ready daemon child is still a live handle of this process',
    );
    const pid = recordedPid(pidFile);
    assert.ok(pid > 0, 'the fake daemon did not record its sleeper pid');
    const deadline = Date.now() + 5000;
    while (alive(pid) && Date.now() < deadline) await sleep(20);
    assert.ok(
      !alive(pid),
      `the never-ready daemon's grandchild (pid ${pid}) is still running`,
    );
  } finally {
    // On failure the sleeper would otherwise hold this process's stdout
    // pipe open for its full 300 s.
    const pid = recordedPid(pidFile);
    if (pid > 0 && alive(pid)) process.kill(pid, 'SIGKILL');
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

runTests()
  .then(() => {
    console.log(
      'audit1005_real_daemon_start_timeout.test.js: all tests passed',
    );
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exitCode = 1;
  });
