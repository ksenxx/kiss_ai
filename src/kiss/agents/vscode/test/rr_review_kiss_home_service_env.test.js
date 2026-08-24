// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Review finding 4: the extension probes sorcarSockPath()
// ($KISS_SORCAR_SOCK ?? $KISS_HOME/sorcar.sock), but the generated
// systemd unit / launchd plist exported only PATH. A KISS_HOME set only
// in the VS Code process was invisible to the service daemon, which
// bound ~/.kiss/sorcar.sock while the extension polled
// $KISS_HOME/sorcar.sock — a 180s poll plus restart loop of a healthy
// daemon.
//
// This test drives the REAL compiled restartKissWebDaemon() in a child
// process against a fake HOME and a fake `systemctl` (like
// daemonRestartNoBlock.test.js) and asserts the generated
// kiss-web.service:
//   - contains Environment=KISS_HOME=... when KISS_HOME is set, and
//   - omits KISS_HOME entirely when it is not.
// KISS_SORCAR_SOCK is a client-side override the daemon does not read,
// so it must never appear in the unit.

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
if (process.platform !== 'linux') {
  console.log('SKIP: the systemd unit-generation path is Linux-only');
  process.exit(0);
}

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rrkh-'));
const tmpHome = path.join(tmpRoot, 'home');
const customKissHome = path.join(tmpRoot, 'custom-kiss-home');
const fakeBin = path.join(tmpRoot, 'bin');
const fakeProj = path.join(tmpRoot, 'proj');
const fakeWork = path.join(tmpRoot, 'work');
const directSpawnMarker = path.join(tmpRoot, 'direct-spawn.marker');
const helperPidFile = path.join(tmpRoot, 'helper.pid');
const serviceFile = path.join(
  tmpHome,
  '.config',
  'systemd',
  'user',
  'kiss-web.service',
);
for (const d of [
  tmpHome,
  customKissHome,
  fakeBin,
  fakeWork,
  path.join(tmpHome, '.kiss'),
  path.join(fakeProj, '.venv', 'bin'),
  path.join(fakeProj, 'src', 'kiss'),
]) {
  fs.mkdirSync(d, {recursive: true});
}

// Fake kiss-web binary: the direct-spawn fallback leaves a marker.
const kissWebBin = path.join(fakeProj, '.venv', 'bin', 'kiss-web');
fs.writeFileSync(
  kissWebBin,
  `#!/bin/sh\ntouch "${directSpawnMarker}"\nexit 0\n`,
);
fs.chmodSync(kissWebBin, 0o755);

// Stand-in daemon "systemd" starts: binds the UDS under $KISS_HOME
// (falling back to ~/.kiss) exactly like the real daemon, and holds
// port 8787 so verifyDaemonStartup() returns promptly.
const udsHelper = path.join(tmpRoot, 'uds-helper.js');
fs.writeFileSync(
  udsHelper,
  `
'use strict';
const net = require('net');
const fs = require('fs');
const path = require('path');
const kissDir =
  process.env.KISS_HOME || path.join(process.env.HOME, '.kiss');
fs.mkdirSync(kissDir, {recursive: true});
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
  fs.writeFileSync(${JSON.stringify(helperPidFile)}, String(process.pid));
});
const tcp = net.createServer(() => {});
tcp.on('error', () => {});
tcp.listen(8787, '127.0.0.1');
setTimeout(() => process.exit(0), 30000);
`,
);

// Fake systemctl: `restart --no-block` starts the stand-in daemon with
// the caller's environment (systemd would apply the unit's own
// Environment= lines; passing the caller's KISS_HOME through models a
// unit that propagates it — the assertions below are on the unit TEXT,
// which is what the real systemd would honor).
const fakeSystemctl = path.join(fakeBin, 'systemctl');
fs.writeFileSync(
  fakeSystemctl,
  `#!/bin/sh
case " $* " in
  *" restart "*)
    nohup "${process.execPath}" "${udsHelper}" >/dev/null 2>&1 &
    ;;
esac
exit 0
`,
);
fs.chmodSync(fakeSystemctl, 0o755);

// Inert lsof (killProcessOnPort never signals a real daemon), no-op
// loginctl.
fs.writeFileSync(path.join(fakeBin, 'lsof'), '#!/bin/sh\nexit 0\n');
fs.chmodSync(path.join(fakeBin, 'lsof'), 0o755);
fs.writeFileSync(path.join(fakeBin, 'loginctl'), '#!/bin/sh\nexit 0\n');
fs.chmodSync(path.join(fakeBin, 'loginctl'), 0o755);

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

function killHelper() {
  try {
    const pid = parseInt(fs.readFileSync(helperPidFile, 'utf-8'), 10);
    if (pid > 0) process.kill(pid, 'SIGKILL');
  } catch {}
  try {
    fs.unlinkSync(helperPidFile);
  } catch {}
}

function runRestart(extraEnv) {
  const env = {
    ...process.env,
    HOME: tmpHome,
    USERPROFILE: tmpHome,
    PATH: `${fakeBin}:${process.env.PATH}`,
    KISS_TEST_DIR: __dirname,
    KISS_TEST_MODULE: OUT,
    KISS_TEST_PROJ: fakeProj,
    KISS_TEST_WORK: fakeWork,
    ...extraEnv,
  };
  delete env.KISS_SORCAR_SOCK;
  if (!('KISS_HOME' in extraEnv)) delete env.KISS_HOME;
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['-e', CHILD], {
      stdio: ['ignore', 'pipe', 'inherit'],
      env,
    });
    let buf = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error('restartKissWebDaemon did not finish within 90s'));
    }, 90000);
    child.stdout.on('data', d => {
      buf += d.toString();
    });
    child.on('close', () => {
      clearTimeout(timer);
      resolve(buf.trim());
    });
  });
}

async function main() {
  // Run 1: KISS_HOME set only in the extension-host process.
  let out = await runRestart({KISS_HOME: customKissHome});
  assert.ok(out.endsWith('DONE'), `restart with KISS_HOME failed: ${out}`);
  let unit = fs.readFileSync(serviceFile, 'utf-8');
  assert.ok(
    unit.includes(`Environment=KISS_HOME=${customKissHome}`),
    `generated unit must export KISS_HOME to the daemon; got:\n${unit}`,
  );
  assert.ok(
    !unit.includes('KISS_SORCAR_SOCK'),
    'KISS_SORCAR_SOCK is client-side only and must not reach the unit',
  );
  console.log('  ok - unit exports KISS_HOME when the extension host has it');
  killHelper();

  // Run 2: no KISS_HOME — the unit must be regenerated without it.
  out = await runRestart({});
  assert.ok(out.endsWith('DONE'), `restart without KISS_HOME failed: ${out}`);
  unit = fs.readFileSync(serviceFile, 'utf-8');
  assert.ok(
    !unit.includes('KISS_HOME'),
    `unit must omit KISS_HOME when the env var is unset; got:\n${unit}`,
  );
  console.log('  ok - unit omits KISS_HOME when the env var is unset');

  assert.ok(
    !fs.existsSync(directSpawnMarker),
    'direct-spawn fallback ran despite the fake systemd succeeding',
  );
  console.log('rr_review_kiss_home_service_env: all assertions passed');
}

main()
  .then(() => {
    killHelper();
    fs.rmSync(tmpRoot, {recursive: true, force: true});
    process.exit(0);
  })
  .catch(err => {
    killHelper();
    try {
      fs.rmSync(tmpRoot, {recursive: true, force: true});
    } catch {}
    console.error(err);
    process.exit(1);
  });
