// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: a pending kiss-web update must not be
// deferred forever behind a daemon that wrongly reports a busy tab.
//
// The incident (2026-09-22, ~/.kiss/kiss-web-stderr.log): a daemon
// started before the stale-merge-claim fixes kept one closed tab
// "busy" for hours.  Every main-tree prompt ("cron list") was refused
// with "A worktree merge is in progress", and because the daemon
// answered activeTasksQuery with count=1, decideRestart() deferred the
// restart that would have loaded the fixed code — every minute, for
// hours, silently.
//
// The fix offers the user a way out: when an update is deferred on
// active tasks, a warning notification explains what it is waiting for
// and offers "Restart now".  Choosing it re-runs the restart with
// force=true, which restarts despite the reported tasks.
//
// This test runs the REAL compiled restart entry point in a child
// process against a fake HOME.  A stand-in "wedged" daemon answers
// activeTasksQuery with count=1 and never changes its mind.  The vscode
// stub records the notification and answers "Restart now".  A fake
// `systemctl restart --no-block` starts a healthy stand-in daemon
// (count=0), like systemd would.  The child reports when the pending
// record is gone, which only the completed forced restart clears.
//
// Three scenarios, each from a clean slate:
//   plain      the click restarts the daemon exactly once;
//   contended  the click lands while a live window holds the restart
//              lock — it must be kept and carried out once the lock is
//              released, not dropped;
//   stale      the click arrives after the update was already applied
//              (pending record gone) — nothing may be restarted.
//
// Safety: the child runs with a fake `lsof` that reports no PIDs, so
// killProcessOnPort() never signals the developer's real daemon, and
// the fake systemctl never touches real units.

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

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-forced-'));
const tmpHome = path.join(tmpRoot, 'home');
const fakeBin = path.join(tmpRoot, 'bin');
const fakeProj = path.join(tmpRoot, 'proj');
const fakeWork = path.join(tmpRoot, 'work');
const kissDir = path.join(tmpHome, '.kiss');
const systemctlLog = path.join(tmpRoot, 'systemctl.log');
const notificationLog = path.join(tmpRoot, 'notifications.log');
const helperPidFile = path.join(kissDir, 'helper.pid');
const pendingFile = path.join(kissDir, '.kiss-web.restart-pending');
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

// Stand-in daemon.  With KISS_FAKE_ACTIVE=1 it is the wedged daemon:
// one phantom busy tab, forever.  Without it, it is the healthy daemon
// "systemd" starts after the forced restart.  Either way it holds
// 127.0.0.1:8787 open when nothing else does, so the health probe says
// "alive" and verifyDaemonStartup() returns promptly.
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
const active = process.env.KISS_FAKE_ACTIVE === '1';
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
          c.write(JSON.stringify(active
            ? {type: 'activeTasksResponse', count: 1,
               tabs: ['337de134(task=ce7c419d)']}
            : {type: 'activeTasksResponse', count: 0, tabs: []}) + '\\n');
        }
      } catch {}
    }
  });
  c.on('error', () => {});
});
srv.listen(sock, () => {
  fs.appendFileSync(path.join(kissDir, 'helper.pid'), process.pid + '\\n');
  if (active) fs.writeFileSync(path.join(kissDir, 'wedged.ready'), '1');
});
const tcp = net.createServer(() => {});
tcp.on('error', () => {});
tcp.listen(8787, '127.0.0.1');
setTimeout(() => process.exit(0), 60000);
`,
);

const fakeSystemctl = path.join(fakeBin, 'systemctl');
fs.writeFileSync(
  fakeSystemctl,
  `#!/bin/sh
echo "$@" >> "${systemctlLog}"
case " $* " in
  *" restart "*" --no-block "*|*" --no-block "*" restart "*)
    KISS_FAKE_ACTIVE=0 nohup "${process.execPath}" "${udsHelper}" >/dev/null 2>&1 &
    ;;
esac
exit 0
`,
);
fs.chmodSync(fakeSystemctl, 0o755);
for (const name of ['lsof', 'loginctl']) {
  const p = path.join(fakeBin, name);
  fs.writeFileSync(p, '#!/bin/sh\nexit 0\n');
  fs.chmodSync(p, 0o755);
}

// The child: run the real, UNFORCED entry point against the wedged
// daemon.  It must defer, record the pending restart and raise the
// notification; answering "Restart now" must then complete a restart
// on its own.  The pending record disappearing is the proof.
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
const pending = process.env.KISS_TEST_PENDING;
const lockFile = process.env.KISS_TEST_LOCK;
const scenario = process.env.KISS_TEST_SCENARIO;
global.__kissVscodeStub = {
  window: {
    showInputBox: () => Promise.resolve(undefined),
    showWarningMessage: (message, _options, ...actions) => {
      fs.appendFileSync(process.env.KISS_TEST_NOTIFY,
        JSON.stringify({message, actions}) + '\\n');
      if (scenario === 'contended') {
        // Another live window is mid-restart when the user clicks: the
        // lock names a live pid, so it is honoured, not broken.
        fs.writeFileSync(lockFile,
          JSON.stringify({pid: process.pid, token: 'held-by-a-live-window'}));
        setTimeout(() => { try { fs.unlinkSync(lockFile); } catch {} }, 5000);
      } else if (scenario === 'stale') {
        // The user clicks a notification that outlived its deferral:
        // a retry (or another window) already applied the update.
        fs.unlinkSync(pending);
      }
      return Promise.resolve('Restart now');
    },
  },
  workspace: {workspaceFolders: undefined},
  ProgressLocation: {Notification: 15},
};
const installer = require(process.env.KISS_TEST_MODULE);
async function main() {
  await installer.restartKissWebDaemon(
    process.env.KISS_TEST_PROJ, process.env.KISS_TEST_WORK);
  if (scenario === 'stale') {
    // Nothing observable is supposed to happen.  The click lands while
    // the unforced attempt still holds the lock, so the forced attempt
    // that reads the missing pending record is the SECOND one, two
    // seconds later; give a wrongly forced restart ample time beyond
    // that to show up in the systemctl log.
    await new Promise(r => setTimeout(r, 8000));
    process.stdout.write('DONE');
    return;
  }
  if (!fs.existsSync(pending)) {
    process.stdout.write('FAIL: the unforced restart did not defer');
    process.exitCode = 1;
    return;
  }
  const deadline = Date.now() + 60000;
  while (fs.existsSync(pending)) {
    if (Date.now() > deadline) {
      process.stdout.write('FAIL: the forced restart never cleared the pending record');
      process.exitCode = 1;
      return;
    }
    await new Promise(r => setTimeout(r, 200));
  }
  process.stdout.write('DONE');
}
main().catch(err => {
  process.stdout.write('FAIL:' + (err && err.message));
  process.exitCode = 1;
});
`;

function cleanup() {
  try {
    for (const line of fs.readFileSync(helperPidFile, 'utf-8').split('\n')) {
      const pid = parseInt(line, 10);
      if (pid > 0) {
        try {
          process.kill(pid, 'SIGKILL');
        } catch {}
      }
    }
  } catch {}
  try {
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  } catch {}
}

function startWedgedDaemon() {
  const child = spawn(process.execPath, [udsHelper], {
    stdio: 'ignore',
    detached: true,
    env: {...process.env, HOME: tmpHome, KISS_FAKE_ACTIVE: '1'},
  });
  child.unref();
  const ready = path.join(kissDir, 'wedged.ready');
  const deadline = Date.now() + 10000;
  return new Promise((resolve, reject) => {
    const poll = () => {
      if (fs.existsSync(ready)) return resolve();
      if (Date.now() > deadline) {
        return reject(new Error('the wedged stand-in daemon never came up'));
      }
      setTimeout(poll, 50);
    };
    poll();
  });
}

function runChild(scenario) {
  return new Promise((resolve, reject) => {
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
        KISS_TEST_DIR: __dirname,
        KISS_TEST_MODULE: OUT,
        KISS_TEST_PROJ: fakeProj,
        KISS_TEST_WORK: fakeWork,
        KISS_TEST_NOTIFY: notificationLog,
        KISS_TEST_PENDING: pendingFile,
        KISS_TEST_LOCK: path.join(kissDir, '.kiss-web.restart.lock'),
        KISS_TEST_SCENARIO: scenario,
      },
    });
    let buf = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error('the child did not finish within 120s'));
    }, 120000);
    child.stdout.on('data', d => {
      buf += d.toString();
    });
    child.on('close', () => {
      clearTimeout(timer);
      resolve(buf.trim());
    });
  });
}

function readLines(file) {
  try {
    return fs.readFileSync(file, 'utf-8').trim().split('\n').filter(Boolean);
  } catch {
    return [];
  }
}

/**
 * Run one scenario from a clean slate: a fresh wedged stand-in daemon,
 * empty notification and systemctl logs, and no recorded fingerprint
 * (a previous scenario's forced restart recorded one, and a matching
 * fingerprint means there is no update to offer a restart for).
 */
async function runScenario(scenario) {
  for (const f of [
    notificationLog,
    systemctlLog,
    path.join(kissDir, '.kiss-web.fingerprint'),
    path.join(kissDir, 'wedged.ready'),
  ]) {
    try {
      fs.unlinkSync(f);
    } catch {}
  }
  await startWedgedDaemon();
  const out = await runChild(scenario);
  assert.ok(out.endsWith('DONE'), `[${scenario}] restart flow failed: ${out}`);
  return {
    out,
    notifications: readLines(notificationLog).map(line => JSON.parse(line)),
    restartCalls: readLines(systemctlLog).filter(c => c.includes('restart')),
  };
}

async function main() {
  // 1. The incident: the user clicks "Restart now" and the daemon is
  //    restarted despite the phantom busy tab.
  const plain = await runScenario('plain');
  console.log('  ok - the forced restart cleared the pending record');
  assert.strictEqual(
    plain.notifications.length,
    1,
    `expected exactly one notification, got ${JSON.stringify(plain.notifications)}`,
  );
  const [notice] = plain.notifications;
  assert.ok(
    /waiting for 1 running task/.test(notice.message),
    `the notification must say what the update waits for: ${notice.message}`,
  );
  assert.deepStrictEqual(notice.actions, ['Restart now', 'Keep waiting']);
  console.log(
    '  ok - one notification offered "Restart now" for 1 reported task',
  );
  assert.strictEqual(
    plain.restartCalls.length,
    1,
    `exactly one systemctl restart expected (the forced one), got: ${plain.restartCalls}`,
  );
  console.log(
    '  ok - the daemon was restarted exactly once, by the forced path',
  );
  assert.ok(
    /forced-by-user/.test(plain.out),
    'the installer log must record that the restart was forced by the user',
  );
  console.log('  ok - the restart was logged as forced-by-user');

  // 2. The click lands while a live window holds the restart lock: the
  //    forced attempt is refused, kept, and carried out once the lock
  //    is released — not dropped.
  const contended = await runScenario('contended');
  assert.ok(
    /another window is restarting kiss-web/.test(contended.out),
    'the held lock must have refused at least one forced attempt',
  );
  assert.strictEqual(
    contended.restartCalls.length,
    1,
    `the forced restart must still happen exactly once, got: ${contended.restartCalls}`,
  );
  assert.ok(/forced-by-user/.test(contended.out));
  console.log(
    '  ok - a click during lock contention is kept until the lock is free',
  );

  // 3. The click arrives after the update was already applied (the
  //    pending record is gone): nothing is restarted.
  const stale = await runScenario('stale');
  assert.strictEqual(
    stale.notifications.length,
    1,
    'the stale scenario must still have offered the restart once',
  );
  assert.deepStrictEqual(
    stale.restartCalls,
    [],
    `a stale click must not restart anything, got: ${stale.restartCalls}`,
  );
  assert.ok(
    /kiss-web update already applied — ignoring the forced restart/.test(
      stale.out,
    ),
    'the ignored click must be logged',
  );
  console.log('  ok - a click on an outdated notification restarts nothing');
}

main()
  .then(() => {
    cleanup();
    console.log('daemonRestartForcedOnStaleBusyTab: all assertions passed');
  })
  .catch(err => {
    cleanup();
    console.error(`FAIL: ${err && err.stack}`);
    process.exit(1);
  });
