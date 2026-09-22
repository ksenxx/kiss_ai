// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: a kiss-web restart that is deferred because
// tasks are in flight must be retried once they finish, and must leave a
// durable "restart pending" record so later activations do not take the
// "nothing to do" fast path.
//
// The bug (2026-09-22): install.sh copied new code into the extension's
// kiss_project while a task was running.  On the next activation
// restartKissWebDaemon() saw the active task and deferred, but the
// .extension-updated marker had already been consumed, so every later
// activation logged "All dependencies satisfied and daemon running —
// nothing to do" and the daemon kept running the old code for 12 hours.
//
// This test runs the REAL compiled restart entry point in a child process
// against a fake HOME, a fake `systemctl`, and a stand-in daemon whose
// activeTasksQuery answer is driven by a "busy" file:
//   1. busy + fingerprint mismatch -> restart deferred, pending file written,
//      no systemctl restart;
//   2. busy file removed -> the in-process retry timer restarts the daemon,
//      records the fingerprint and clears the pending file;
//   3. busy again + fingerprint now matching -> deferred again, but no
//      pending file (nothing to retry).
//
// Safety: a fake `lsof` reports no PIDs, so killProcessOnPort() never
// signals the developer's real daemon, and the fake systemctl never touches
// real units.

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
if (process.platform !== 'linux') {
  console.log('SKIP: the systemd restart path is Linux-only');
  process.exit(0);
}

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-pending-'));
const tmpHome = path.join(tmpRoot, 'home');
const fakeBin = path.join(tmpRoot, 'bin');
const fakeProj = path.join(tmpRoot, 'proj');
const fakeWork = path.join(tmpRoot, 'work');
const kissDir = path.join(tmpHome, '.kiss');
const systemctlLog = path.join(tmpRoot, 'systemctl.log');
const busyFile = path.join(tmpRoot, 'busy');
const helperPidFile = path.join(kissDir, 'helper.pid');
const resultFile = path.join(tmpRoot, 'result.json');
const pendingFile = path.join(kissDir, '.kiss-web.restart-pending');
const fingerprintFile = path.join(kissDir, '.kiss-web.fingerprint');
for (const d of [
  tmpHome,
  fakeBin,
  fakeWork,
  kissDir,
  path.join(fakeProj, '.venv', 'bin'),
  path.join(fakeProj, 'src', 'kiss'),
]) {
  fs.mkdirSync(d, {recursive: true});
}

const kissWebBin = path.join(fakeProj, '.venv', 'bin', 'kiss-web');
fs.writeFileSync(kissWebBin, '#!/bin/sh\nexit 0\n');
fs.chmodSync(kissWebBin, 0o755);
fs.writeFileSync(path.join(fakeProj, 'src', 'kiss', 'x.py'), '# new code\n');
fs.writeFileSync(busyFile, '1\n');

// Stand-in daemon: answers activeTasksQuery with count 1 while the busy
// file exists, 0 otherwise; holds 127.0.0.1:8787 when nothing else does so
// the health probe and verifyDaemonStartup() see a live daemon.
const udsHelper = path.join(tmpRoot, 'uds-helper.js');
fs.writeFileSync(
  udsHelper,
  `
'use strict';
const net = require('net');
const fs = require('fs');
const path = require('path');
const kissDir = path.join(process.env.HOME, '.kiss');
const sock = path.join(kissDir, 'sorcar.sock');
try { fs.unlinkSync(sock); } catch {}
const srv = net.createServer(c => {
  c.setEncoding('utf-8');
  let buf = '';
  c.on('data', d => {
    buf += d;
    let nl;
    while ((nl = buf.indexOf('\\n')) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      try {
        const msg = JSON.parse(line);
        if (msg.type === 'activeTasksQuery') {
          const count = fs.existsSync(${JSON.stringify(busyFile)}) ? 1 : 0;
          c.write(JSON.stringify(
            {type: 'activeTasksResponse', count, tabs: count ? ['tab'] : []}) + '\\n');
        }
      } catch {}
    }
  });
  c.on('error', () => {});
});
srv.listen(sock, () => {
  fs.writeFileSync(path.join(kissDir, 'helper.pid'), String(process.pid));
});
const tcp = net.createServer(() => {});
tcp.on('error', () => {});
tcp.listen(8787, '127.0.0.1');
setTimeout(() => process.exit(0), 60000);
`,
);

// Fake systemctl: log every call, never touch real units.  The stand-in
// daemon is already up, so `restart --no-block` has nothing to start.
const fakeSystemctl = path.join(fakeBin, 'systemctl');
fs.writeFileSync(
  fakeSystemctl,
  `#!/bin/sh
echo "$@" >> "${systemctlLog}"
exit 0
`,
);
fs.chmodSync(fakeSystemctl, 0o755);
for (const name of ['lsof', 'loginctl']) {
  fs.writeFileSync(path.join(fakeBin, name), '#!/bin/sh\nexit 0\n');
  fs.chmodSync(path.join(fakeBin, name), 0o755);
}

const CHILD = `
'use strict';
const fs = require('fs');
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
const installer = require(process.env.KISS_TEST_MODULE);
const pending = process.env.KISS_TEST_PENDING;
const busy = process.env.KISS_TEST_BUSY;
const systemctlLog = process.env.KISS_TEST_SYSTEMCTL_LOG;
const result = {};
const restartCalls = () => {
  try {
    return fs.readFileSync(systemctlLog, 'utf-8').split('\\n')
      .filter(l => l.includes('restart')).length;
  } catch { return 0; }
};
const waitFor = (pred, ms) => new Promise((resolve, reject) => {
  const started = Date.now();
  const t = setInterval(() => {
    if (pred()) { clearInterval(t); resolve(); }
    else if (Date.now() - started > ms) { clearInterval(t); reject(new Error('timeout')); }
  }, 100);
});
(async () => {
  await installer.restartKissWebDaemon(process.env.KISS_TEST_PROJ, process.env.KISS_TEST_WORK);
  result.pendingAfterDeferral = fs.existsSync(pending)
    ? fs.readFileSync(pending, 'utf-8').trim() : null;
  result.restartsAfterDeferral = restartCalls();
  fs.unlinkSync(busy);
  // Another window holds the restart lock (live pid): the first retry must
  // be skipped without restarting, and the timer must be re-armed.
  const lock = process.env.KISS_TEST_LOCK;
  fs.writeFileSync(lock, JSON.stringify({pid: process.pid, token: 'other-window'}));
  await new Promise(r => setTimeout(r, 2 * Number(process.env.KISS_RESTART_RETRY_MS)));
  result.restartsWhileLocked = restartCalls();
  result.pendingWhileLocked = fs.existsSync(pending);
  fs.unlinkSync(lock);
  // The retry timer (KISS_RESTART_RETRY_MS) must restart the daemon and
  // clear the pending file without any further call from us.
  await waitFor(() => !fs.existsSync(pending), 30000);
  result.restartsAfterRetry = restartCalls();
  result.fingerprintAfterRetry = fs.existsSync(process.env.KISS_TEST_FP);
  fs.writeFileSync(busy, '1\\n');
  await installer.restartKissWebDaemon(process.env.KISS_TEST_PROJ, process.env.KISS_TEST_WORK);
  result.pendingWhenUnchanged = fs.existsSync(pending);
  result.restartsWhenUnchanged = restartCalls();
  fs.writeFileSync(process.env.KISS_TEST_RESULT, JSON.stringify(result));
  process.stdout.write('DONE');
})().catch(err => {
  fs.writeFileSync(process.env.KISS_TEST_RESULT, JSON.stringify(result));
  process.stdout.write('FAIL:' + (err && err.stack));
  process.exitCode = 1;
});
`;

function cleanup() {
  try {
    const pid = parseInt(fs.readFileSync(helperPidFile, 'utf-8'), 10);
    if (pid > 0) process.kill(pid, 'SIGKILL');
  } catch {}
  try {
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  } catch {}
}

function startHelper() {
  const helper = spawn(process.execPath, [udsHelper], {
    stdio: 'ignore',
    detached: true,
    env: {...process.env, HOME: tmpHome},
  });
  helper.unref();
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const t = setInterval(() => {
      if (fs.existsSync(helperPidFile)) {
        clearInterval(t);
        resolve();
      } else if (Date.now() - started > 10000) {
        clearInterval(t);
        reject(new Error('stand-in daemon did not start'));
      }
    }, 50);
  });
}

async function main() {
  await startHelper();
  const out = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['-e', CHILD], {
      stdio: ['ignore', 'pipe', 'inherit'],
      env: {
        ...process.env,
        HOME: tmpHome,
        USERPROFILE: tmpHome,
        // Pin the daemon state root explicitly: userAssets.ts gives these
        // precedence over $HOME, so an inherited value would point the
        // installer at the developer's real socket and markers.
        KISS_HOME: kissDir,
        KISS_SORCAR_SOCK: path.join(kissDir, 'sorcar.sock'),
        PATH: `${fakeBin}:${process.env.PATH}`,
        KISS_RESTART_RETRY_MS: '1500',
        KISS_TEST_LOCK: path.join(kissDir, '.kiss-web.restart.lock'),
        KISS_TEST_DIR: __dirname,
        KISS_TEST_MODULE: OUT,
        KISS_TEST_PROJ: fakeProj,
        KISS_TEST_WORK: fakeWork,
        KISS_TEST_PENDING: pendingFile,
        KISS_TEST_FP: fingerprintFile,
        KISS_TEST_BUSY: busyFile,
        KISS_TEST_SYSTEMCTL_LOG: systemctlLog,
        KISS_TEST_RESULT: resultFile,
      },
    });
    let buf = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error('child did not finish within 90s'));
    }, 90000);
    child.stdout.on('data', d => {
      buf += d.toString();
    });
    child.on('close', () => {
      clearTimeout(timer);
      resolve(buf.trim());
    });
  });
  assert.ok(out.endsWith('DONE'), `restart entry point failed: ${out}`);
  const result = JSON.parse(fs.readFileSync(resultFile, 'utf-8'));

  assert.strictEqual(
    result.pendingAfterDeferral,
    'active-tasks',
    'a deferred restart must record why it is pending',
  );
  assert.strictEqual(
    result.restartsAfterDeferral,
    0,
    'the daemon must not be restarted while tasks are active',
  );
  console.log('  ok - deferred restart writes .kiss-web.restart-pending');

  assert.strictEqual(
    result.restartsWhileLocked,
    0,
    'a retry must not restart while another window holds the restart lock',
  );
  assert.strictEqual(
    result.pendingWhileLocked,
    true,
    'a lock-skipped retry must keep the pending record',
  );
  console.log('  ok - lock contention keeps the restart pending');

  assert.ok(
    result.restartsAfterRetry >= 1,
    'the retry timer never restarted the daemon after the tasks finished',
  );
  assert.ok(
    result.fingerprintAfterRetry,
    'fingerprint not recorded by the retry',
  );
  console.log(
    '  ok - retry timer restarts kiss-web once tasks finish and clears the record',
  );

  assert.strictEqual(
    result.pendingWhenUnchanged,
    false,
    'a deferral with a matching fingerprint must not leave a pending record',
  );
  assert.strictEqual(
    result.restartsWhenUnchanged,
    result.restartsAfterRetry,
    'no extra restart when the fingerprint is unchanged',
  );
  console.log('  ok - matching fingerprint leaves nothing pending');

  const installLog = fs.readFileSync(
    path.join(kissDir, 'install.log'),
    'utf-8',
  );
  assert.ok(
    installLog.includes(
      'kiss-web restart pending (active-tasks) — retrying in 1.5s',
    ),
    'install.log must announce the pending restart and retry delay',
  );
  console.log('  ok - install.log records the pending restart');
}

main()
  .then(() => {
    cleanup();
    console.log('daemonRestartPendingRetry: all assertions passed');
  })
  .catch(err => {
    cleanup();
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
