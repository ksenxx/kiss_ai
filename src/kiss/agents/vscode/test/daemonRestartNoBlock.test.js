// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: a slow daemon shutdown must not fork a
// SECOND kiss-web behind systemd's back.
//
// The bug: `systemctl --user restart kiss-web` BLOCKS until the old
// daemon finishes shutting down, which can take longer than the 10s
// execSync timeout (tunnel cleanup; the daemon's SIGTERM failsafe
// allows 30s).  The ETIMEDOUT was misread as "systemd failed" and the
// direct-spawn fallback launched a duplicate daemon WHILE systemd's
// restart job was still in flight.  The duplicate grabbed port 8787,
// systemd's own instance crash-looped against it every RestartSec, and
// the user saw the "KISS Sorcar Server is starting ..." screen over
// and over.
//
// The fix queues the job with `--no-block` instead.  This test runs
// the REAL compiled restart entry point in a child process against a
// fake HOME and a fake `systemctl` that simulates exactly that slow
// shutdown: a blocking `restart` sleeps far past the execSync timeout,
// while `restart --no-block` returns immediately and starts a stand-in
// daemon (UDS responder) the way systemd would.  A regressed build
// times out and direct-spawns; the fixed build finishes fast, never
// direct-spawns, and passes `--no-block`.
//
// Safety: the child runs with a fake `lsof` that reports no PIDs, so
// killProcessOnPort() never signals the developer's real daemon, and
// the fake systemctl never touches real units.

const assert = require('assert');
const {spawn, execFileSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OUT = path.join(__dirname, '..', 'out', 'DependencyInstaller.js');
if (!fs.existsSync(OUT)) {
  console.log('SKIP: out/DependencyInstaller.js missing — run `npm run compile`');
  process.exit(0);
}
if (process.platform !== 'linux') {
  console.log('SKIP: the systemd restart path is Linux-only');
  process.exit(0);
}

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-noblock-'));
const tmpHome = path.join(tmpRoot, 'home');
const fakeBin = path.join(tmpRoot, 'bin');
const fakeProj = path.join(tmpRoot, 'proj');
const fakeWork = path.join(tmpRoot, 'work');
const kissDir = path.join(tmpHome, '.kiss');
const systemctlLog = path.join(tmpRoot, 'systemctl.log');
const directSpawnMarker = path.join(tmpRoot, 'direct-spawn.marker');
const helperPidFile = path.join(kissDir, 'helper.pid');
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

// The fake kiss-web binary: if the direct-spawn fallback ever runs it,
// it leaves a marker behind.
const kissWebBin = path.join(fakeProj, '.venv', 'bin', 'kiss-web');
fs.writeFileSync(
  kissWebBin,
  `#!/bin/sh\ntouch "${directSpawnMarker}"\nexit 0\n`,
);
fs.chmodSync(kissWebBin, 0o755);

// Stand-in daemon that "systemd" starts: answers activeTasksQuery on
// the UDS and holds 127.0.0.1:8787 open when nothing else does, so
// verifyDaemonStartup() sees a healthy daemon and returns promptly.
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
          c.write(JSON.stringify(
            {type: 'activeTasksResponse', count: 0, tabs: []}) + '\\n');
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
setTimeout(() => process.exit(0), 30000);
`,
);

// Fake systemctl: `restart --no-block` returns at once and starts the
// stand-in daemon (like systemd finishing the job asynchronously); a
// BLOCKING `restart` simulates the >10s shutdown that caused the bug.
const fakeSystemctl = path.join(fakeBin, 'systemctl');
fs.writeFileSync(
  fakeSystemctl,
  `#!/bin/sh
echo "$@" >> "${systemctlLog}"
case " $* " in
  *" daemon-reload "*) exit 0 ;;
esac
case " $* " in
  *" restart "*)
    case " $* " in
      *" --no-block "*)
        nohup "${process.execPath}" "${udsHelper}" >/dev/null 2>&1 &
        exit 0
        ;;
    esac
    sleep 20
    exit 0
    ;;
esac
exit 0
`,
);
fs.chmodSync(fakeSystemctl, 0o755);

// Fake lsof: report no listeners so killProcessOnPort() is inert and
// the developer's real daemon is never signalled. Fake loginctl: no-op.
const fakeLsof = path.join(fakeBin, 'lsof');
fs.writeFileSync(fakeLsof, '#!/bin/sh\nexit 0\n');
fs.chmodSync(fakeLsof, 0o755);
const fakeLoginctl = path.join(fakeBin, 'loginctl');
fs.writeFileSync(fakeLoginctl, '#!/bin/sh\nexit 0\n');
fs.chmodSync(fakeLoginctl, 0o755);

const CHILD = `
'use strict';
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
installer
  .restartKissWebDaemon(process.env.KISS_TEST_PROJ, process.env.KISS_TEST_WORK)
  .then(() => process.stdout.write('DONE'))
  .catch(err => {
    process.stdout.write('FAIL:' + (err && err.message));
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

async function main() {
  const startedAt = Date.now();
  const out = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['-e', CHILD], {
      stdio: ['ignore', 'pipe', 'inherit'],
      env: {
        ...process.env,
        HOME: tmpHome,
        USERPROFILE: tmpHome,
        PATH: `${fakeBin}:${process.env.PATH}`,
        KISS_TEST_DIR: __dirname,
        KISS_TEST_MODULE: OUT,
        KISS_TEST_PROJ: fakeProj,
        KISS_TEST_WORK: fakeWork,
      },
    });
    let buf = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error(
        'restartKissWebDaemon did not finish within 90s — a blocking ' +
        'systemctl restart (or the direct-spawn fallback) is back',
      ));
    }, 90000);
    child.stdout.on('data', d => {
      buf += d.toString();
    });
    child.on('close', () => {
      clearTimeout(timer);
      resolve(buf.trim());
    });
  });

  // The installer's log() also writes to stdout in the stub
  // environment, so only the tail is the child's own verdict.
  assert.ok(out.endsWith('DONE'), `restart entry point failed: ${out}`);

  const calls = fs
    .readFileSync(systemctlLog, 'utf-8')
    .trim()
    .split('\n');
  const restartCalls = calls.filter(c => c.includes('restart'));
  assert.ok(restartCalls.length >= 1, 'systemctl restart was never invoked');
  for (const call of restartCalls) {
    assert.ok(
      call.includes('--no-block'),
      `systemctl restart must use --no-block, got: ${call}`,
    );
  }
  console.log('  ok - systemctl restart is queued with --no-block');

  assert.ok(
    !fs.existsSync(directSpawnMarker),
    'the direct-spawn fallback ran even though systemd accepted the job',
  );
  console.log('  ok - no duplicate daemon is spawned behind systemd');

  const elapsed = Date.now() - startedAt;
  assert.ok(
    elapsed < 60000,
    `restart took ${elapsed}ms — it must not block on the old daemon`,
  );
  console.log(`  ok - restart entry point returned in ${elapsed}ms`);
}

main()
  .then(() => {
    cleanup();
    console.log('daemonRestartNoBlock: all assertions passed');
  })
  .catch(err => {
    cleanup();
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
