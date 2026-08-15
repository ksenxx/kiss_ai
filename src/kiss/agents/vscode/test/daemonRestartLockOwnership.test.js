// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of WHO owns the cross-process kiss-web restart lock,
// driven with REAL child processes against a REAL lock file under a
// temp HOME.
//
// daemonRestartLock.test.js proves that exactly one window may enter the
// restart path. That is only half the guarantee. The other half is that
// the window inside it stays inside it:
//
//   * The lock used to be declared stale from its mtime alone. The
//     startup verifier is allowed to run for 180s, and a suspended
//     laptop adds however long it was asleep, so a window still
//     legitimately restarting could have its lock broken by the next
//     window to open -- the exact "one window kills the daemon another
//     just started" race the lock exists to stop.
//
//   * The release closure used to unlink the path unconditionally. So
//     once a lock HAD been broken, the original owner's eventual
//     release deleted the NEW owner's lock, and a third window walked
//     straight in beside the second.
//
// The counterweight is that a genuinely abandoned lock must still be
// broken, and promptly: a window that dies mid-restart must not wedge
// every later one.
//
// Like the sibling suite, nothing here is ever allowed PAST the lock:
// beyond it restartKissWebDaemon() probes port 8787, kills whatever is
// listening and bootstraps a real LaunchAgent -- against the
// developer's own machine and the production daemon.

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OUT = path.join(__dirname, '..', 'out', 'DependencyInstaller.js');
if (!fs.existsSync(OUT)) {
  console.log(
    'SKIP: out/DependencyInstaller.js missing — run `npm run compile`',
  );
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-lockown-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});

const installer = require(OUT);
const lockFile = path.join(tmpHome, '.kiss', '.kiss-web.restart.lock');

const STUB_PRELUDE = `
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
`;

// A window that takes the lock and stays in the restart path until it is
// told to finish -- the way a slow launchd restart plus a 180s startup
// verification does.
const CHILD_HOLD =
  STUB_PRELUDE +
  `
const release = acquireDaemonRestartLock();
process.stdout.write(release ? 'WON\\n' : 'LOST\\n');
if (!release) process.exit(0);
process.stdin.on('data', () => {
  release();
  process.stdout.write('RELEASED\\n');
  process.exit(0);
});
process.stdin.resume();
`;

// A window that dies mid-restart: it never releases.
const CHILD_DIE =
  STUB_PRELUDE +
  `
const release = acquireDaemonRestartLock();
process.stdout.write(release ? 'WON\\n' : 'LOST\\n');
process.exit(release ? 0 : 1);
`;

function spawnWindow(script, keepOpen) {
  const child = spawn(process.execPath, ['-e', script], {
    stdio: [keepOpen ? 'pipe' : 'ignore', 'pipe', 'inherit'],
    env: {
      ...process.env,
      HOME: tmpHome,
      USERPROFILE: tmpHome,
      KISS_TEST_DIR: __dirname,
      KISS_TEST_MODULE: OUT,
    },
  });
  // The window logs to stdout as well, so only its verdict is kept.
  const VERDICTS = ['WON', 'LOST', 'RELEASED'];
  const lines = [];
  let buf = '';
  child.stdout.on('data', d => {
    buf += d.toString();
    const parts = buf.split('\n');
    buf = parts.pop();
    for (const l of parts) {
      if (VERDICTS.includes(l.trim())) lines.push(l.trim());
    }
  });
  const exited = new Promise(r => child.on('exit', r));
  return {
    child,
    lines,
    exited,
    firstLine: () => waitFor(() => lines[0], 'the window never answered'),
    finish: () => {
      child.stdin.write('go\n');
      return exited;
    },
  };
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitFor(predicate, message, timeoutMs = 5000) {
  const start = Date.now();
  for (;;) {
    const v = predicate();
    if (v) return v;
    if (Date.now() - start >= timeoutMs) throw new Error(message);
    await sleep(10);
  }
}

/** Make the lock look as old as `ms`, without touching who owns it. */
function backdateLock(ms) {
  const when = new Date(Date.now() - ms);
  fs.utimesSync(lockFile, when, when);
}

// A window that is still legitimately restarting keeps the lock however
// long that takes: age alone says nothing about whether it is finished.
async function testALiveOwnerIsNotEvicted() {
  const holder = spawnWindow(CHILD_HOLD, true);
  assert.strictEqual(await holder.firstLine(), 'WON');

  // The verifier alone may run for 180s, and a suspended laptop adds
  // whatever it slept for on top.
  backdateLock(5 * 60 * 1000);

  assert.strictEqual(
    installer.acquireDaemonRestartLock(),
    null,
    'a window still inside the restart path must keep the lock however ' +
      'old it looks — breaking it is how one window ends up SIGTERMing ' +
      'the daemon another just started',
  );

  await holder.finish();
  const after = installer.acquireDaemonRestartLock();
  assert.ok(after, 'the lock must be free once the owner leaves the path');
  after();
  console.log('  ok - a live owner is not evicted however old its lock looks');
}

// Once a lock HAS legitimately changed hands, the previous owner must
// not delete the new one's lock on its way out.
async function testAnOldOwnerDoesNotDeleteTheNewOwnersLock() {
  const mine = installer.acquireDaemonRestartLock();
  assert.ok(mine, 'the lock must be free to begin with');

  // A window wedged for this long is not coming back, so the next one is
  // allowed in; that is exactly the case the stale rule is for.
  backdateLock(20 * 60 * 1000);

  const holder = spawnWindow(CHILD_HOLD, true);
  assert.strictEqual(
    await holder.firstLine(),
    'WON',
    'a lock held far past the ceiling must be breakable, or a wedged ' +
      'window blocks every restart for ever',
  );

  // The first owner finally finishes and lets go.
  mine();

  const gatecrasher = spawnWindow(CHILD_DIE, false);
  assert.strictEqual(
    await gatecrasher.firstLine(),
    'LOST',
    'the previous owner\u2019s release must not delete the lock the new ' +
      'owner is holding — that lets a third window into the restart ' +
      'path beside the second',
  );
  await gatecrasher.exited;

  await holder.finish();
  const after = installer.acquireDaemonRestartLock();
  assert.ok(after, 'the lock must be free once the real owner releases it');
  after();
  console.log('  ok - an old owner does not delete the new owner\u2019s lock');
}

// The counterweight: a window that died mid-restart must not wedge the
// next one, and must not make it wait out the stale timeout either.
async function testADeadOwnersLockIsBrokenAtOnce() {
  const corpse = spawnWindow(CHILD_DIE, false);
  assert.strictEqual(await corpse.firstLine(), 'WON');
  await corpse.exited;
  assert.ok(
    fs.existsSync(lockFile),
    'the test needs a lock left behind by a window that never released it',
  );

  const startedAt = Date.now();
  const release = installer.acquireDaemonRestartLock();
  assert.ok(
    release,
    'a lock whose owner is gone must be broken — otherwise one crash ' +
      'stops the daemon ever being restarted again',
  );
  assert.ok(
    Date.now() - startedAt < 1000,
    'and broken at once: the owner is provably dead, so there is nothing ' +
      'to wait for',
  );
  release();
  assert.ok(!fs.existsSync(lockFile), 'the new owner must release cleanly');
  console.log('  ok - a dead owner\u2019s lock is broken at once');
}

// Control: a fresh lock held by a live window is still refused, and
// releasing it still frees it.
async function testAFreshLiveLockIsStillRefused() {
  const holder = spawnWindow(CHILD_HOLD, true);
  assert.strictEqual(await holder.firstLine(), 'WON');
  assert.strictEqual(
    installer.acquireDaemonRestartLock(),
    null,
    'a lock held right now must be refused',
  );
  await holder.finish();
  assert.ok(
    !fs.existsSync(lockFile),
    'the owner must delete its own lock when it releases',
  );
  const release = installer.acquireDaemonRestartLock();
  assert.ok(release, 'and the next window may then have it');
  release();
  console.log('  ok - a fresh lock held by a live window is refused');
}

(async () => {
  try {
    await testAFreshLiveLockIsStillRefused();
    await testALiveOwnerIsNotEvicted();
    await testADeadOwnersLockIsBrokenAtOnce();
    await testAnOldOwnerDoesNotDeleteTheNewOwnersLock();
    console.log('daemonRestartLockOwnership.test.js: all tests passed');
  } finally {
    fs.rmSync(tmpHome, {recursive: true, force: true});
  }
})().catch(err => {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
});
