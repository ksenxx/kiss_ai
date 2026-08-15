// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the cross-process guard around a kiss-web restart.
//
// Two VS Code windows reopened together both see a stale fingerprint and
// both probe the daemon as "dead" -- it has not finished binding yet.
// With no lock anywhere in DependencyInstaller, the second window ran
// killProcessOnPort(8787) and SIGTERMed the daemon the first had just
// started, WHILE it was booting: it had not yet accepted a UDS
// connection, so daemonHasActiveTasks() could not report the in-flight
// work that decideRestart() exists to protect.
//
// The lock is exercised here with REAL child processes against a REAL
// lock file under a temp HOME, and the real (compiled) restart entry
// point is shown to honour it.
//
// Deliberate limit: this test never lets restartKissWebDaemon get PAST
// the lock. Beyond it the function probes port 8787, runs
// killProcessOnPort(8787) and bootstraps a real LaunchAgent/systemd
// unit -- against the developer's own machine and the production daemon.
// So the winner's path is left to the existing macLaunchdRestart /
// daemonHealth suites, and what is proven here is the property the fix
// adds: exactly one process may be inside that path at a time.

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OUT = path.join(__dirname, '..', 'out', 'DependencyInstaller.js');
if (!fs.existsSync(OUT)) {
  console.log('SKIP: out/DependencyInstaller.js missing — run `npm run compile`');
  process.exit(0);
}
if (process.platform === 'win32') {
  console.log('SKIP: the daemon restart path is POSIX-only');
  process.exit(0);
}

// DependencyInstaller imports the VS Code API, which only exists inside
// the editor; everything this test touches is plain fs/child_process.
const Module = require('module');
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = {
  window: {showInputBox: () => Promise.resolve(undefined)},
  workspace: {workspaceFolders: undefined},
  ProgressLocation: {Notification: 15},
};

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-lock-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});

const installer = require(OUT);
const lockFile = path.join(tmpHome, '.kiss', '.kiss-web.restart.lock');

// A child window: takes the lock, holds it, and says whether it won.
const CHILD = `
const path = require('path');
const Module = require('module');
const stubPath = path.join(process.env.KISS_TEST_DIR, '_vscode-stub.js');
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return stubPath;
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = {
  window: {showInputBox: () => Promise.resolve(undefined)},
  workspace: {workspaceFolders: undefined},
  ProgressLocation: {Notification: 15},
};
const {acquireDaemonRestartLock} = require(process.env.KISS_TEST_MODULE);
const release = acquireDaemonRestartLock();
process.stdout.write(release ? 'WON' : 'LOST');
if (release) {
  // Hold it for a moment, the way a real restart would.
  setTimeout(() => release(), 400);
}
`;

function spawnWindow() {
  return new Promise(resolve => {
    const child = spawn(process.execPath, ['-e', CHILD], {
      stdio: ['ignore', 'pipe', 'inherit'],
      env: {
        ...process.env,
        HOME: tmpHome,
        USERPROFILE: tmpHome,
        KISS_TEST_DIR: __dirname,
        KISS_TEST_MODULE: OUT,
      },
    });
    let out = '';
    child.stdout.on('data', d => {
      out += d.toString();
    });
    child.on('close', () => resolve(out.trim()));
  });
}

async function testOnlyOneWindowMayRestart() {
  const results = await Promise.all([
    spawnWindow(),
    spawnWindow(),
    spawnWindow(),
    spawnWindow(),
  ]);
  const won = results.filter(r => r === 'WON');
  assert.strictEqual(
    won.length,
    1,
    `exactly one of four windows may restart the daemon, got ${JSON.stringify(results)}`,
  );
  assert.ok(
    !fs.existsSync(lockFile),
    'the winner must release the lock when it is done',
  );
  console.log('  ok - only one of four concurrent windows takes the lock');
}

function testLockIsReusableAfterRelease() {
  const first = installer.acquireDaemonRestartLock();
  assert.ok(first, 'the lock must be free to begin with');
  assert.strictEqual(
    installer.acquireDaemonRestartLock(),
    null,
    'a second acquire while held must be refused',
  );
  first();
  const second = installer.acquireDaemonRestartLock();
  assert.ok(second, 'the lock must be free again after release');
  second();
  console.log('  ok - the lock is released and reusable');
}

function testStaleLockIsBroken() {
  fs.writeFileSync(lockFile, '999999\n');
  const old = Date.now() - 10 * 60 * 1000;
  fs.utimesSync(lockFile, new Date(old), new Date(old));
  const release = installer.acquireDaemonRestartLock();
  assert.ok(
    release,
    'a lock left behind by a window that died mid-restart must be broken, ' +
      'or no window can ever restart the daemon again',
  );
  release();
  console.log('  ok - a stale lock is broken');
}

// The real entry point must honour the lock: with it held, it returns
// without probing, killing or bootstrapping anything.
async function testRestartHonoursTheLock() {
  const project = path.join(tmpHome, 'kiss_project');
  const binDir = path.join(project, '.venv', 'bin');
  fs.mkdirSync(binDir, {recursive: true});
  fs.writeFileSync(path.join(binDir, 'kiss-web'), '#!/bin/sh\nexit 0\n');
  fs.chmodSync(path.join(binDir, 'kiss-web'), 0o755);

  const held = installer.acquireDaemonRestartLock();
  assert.ok(held, 'the test must be able to take the lock');
  const startedAt = Date.now();
  await installer.restartKissWebDaemon(project, tmpHome);
  const elapsed = Date.now() - startedAt;
  held();

  // The guarded path opens with a 1.5s health probe and a 1.5s UDS
  // probe; returning promptly is the observable proof it was skipped.
  assert.ok(
    elapsed < 1000,
    `restartKissWebDaemon must return immediately while another window ` +
      `holds the lock (took ${elapsed}ms — it probed the daemon)`,
  );
  console.log('  ok - restartKissWebDaemon defers to the lock holder');
}

// A missing kiss-web binary must not even take the lock, or a window
// with a broken checkout would block every healthy one.
async function testMissingBinaryDoesNotTakeTheLock() {
  await installer.restartKissWebDaemon(
    path.join(tmpHome, 'no-such-project'),
    tmpHome,
  );
  assert.ok(
    !fs.existsSync(lockFile),
    'a window with no kiss-web binary must not hold the restart lock',
  );
  console.log('  ok - a missing binary does not take the lock');
}

(async () => {
  try {
    testLockIsReusableAfterRelease();
    testStaleLockIsBroken();
    await testOnlyOneWindowMayRestart();
    await testRestartHonoursTheLock();
    await testMissingBinaryDoesNotTakeTheLock();
    console.log('daemonRestartLock.test.js: all tests passed');
  } finally {
    fs.rmSync(tmpHome, {recursive: true, force: true});
  }
})().catch(err => {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
});

