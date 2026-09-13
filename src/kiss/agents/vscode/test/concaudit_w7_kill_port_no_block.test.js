// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: waiting for the old daemon to die must not
// freeze the extension host.
//
// The bug: killProcessOnPort() SIGTERMed the listener on 8787 and then
// polled `lsof` up to six times with a SYNCHRONOUS 500ms sleep
// (Atomics.wait) between polls.  Inside the async restart path that is
// a 3s+ stall of the whole extension-host event loop -- every other
// extension's timers, the daemon client's socket events and the
// webview's messages all stop -- while the daemon shuts down.
//
// This test runs the REAL compiled restart entry point in a child
// process against a fake HOME, a fake `lsof` that reports a stand-in
// "daemon" (a node process that ignores SIGTERM, so the poll loop runs
// to its SIGKILL escalation) and the fake `systemctl` used by the
// no-block test.  A 25ms heartbeat timer in the child measures the
// longest event-loop stall; a regressed build stalls for the whole
// 3s poll loop.  The fake `lsof`, `systemctl` and `loginctl` are slow
// (300ms) on purpose: the fixed build runs all of them asynchronously
// (pidsOnPort and the systemd calls are awaited), so the loop never
// stalls for anything near one such run.
//
// Safety: the fake lsof only ever names OUR stand-in process, so the
// developer's real daemon is never signalled.

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

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-killport-'));
const tmpHome = path.join(tmpRoot, 'home');
const fakeBin = path.join(tmpRoot, 'bin');
const fakeProj = path.join(tmpRoot, 'proj');
const fakeWork = path.join(tmpRoot, 'work');
const kissDir = path.join(tmpHome, '.kiss');
const helperPidFile = path.join(kissDir, 'helper.pid');
const victimPidFile = path.join(tmpRoot, 'victim.pid');
const lsofLog = path.join(tmpRoot, 'lsof.log');
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

// The stand-in "old daemon": ignores SIGTERM so killProcessOnPort() has
// to run its whole poll loop and escalate to SIGKILL.  It reports
// readiness on stdout only AFTER the handler is installed: the code under
// test must not be started before that, or a SIGTERM delivered to a
// still-booting node (default disposition) would kill it outright, the
// poll loop would end early and the ">= 7 lsof calls" assertion below
// would fail spuriously.
const victim = spawn(
  process.execPath,
  [
    '-e',
    'process.on("SIGTERM", () => {}); setInterval(() => {}, 1000); ' +
      'process.stdout.write("READY\\n");',
  ],
  {stdio: ['ignore', 'pipe', 'ignore']},
);
fs.writeFileSync(victimPidFile, String(victim.pid));
const victimReady = new Promise((resolve, reject) => {
  const timer = setTimeout(
    () => reject(new Error('victim never reported readiness')),
    10000,
  );
  victim.stdout.on('data', d => {
    if (d.toString().includes('READY')) {
      clearTimeout(timer);
      resolve();
    }
  });
  victim.on('error', err => {
    clearTimeout(timer);
    reject(err);
  });
  victim.on('exit', code => {
    clearTimeout(timer);
    reject(new Error(`victim exited early (code ${code})`));
  });
});

// Fake lsof: name the victim while it is alive, nothing afterwards.  It
// is deliberately SLOW (300ms): the poll loop runs it up to 8 times, and
// a synchronous lsof would freeze the event loop for those 300ms on
// every poll -- the heartbeat below catches exactly that.
const LSOF_DELAY_S = '0.3';
const fakeLsof = path.join(fakeBin, 'lsof');
fs.writeFileSync(
  fakeLsof,
  `#!/bin/sh
echo "$@" >> "${lsofLog}"
sleep ${LSOF_DELAY_S}
pid=$(cat "${victimPidFile}")
if kill -0 "$pid" 2>/dev/null; then echo "$pid"; fi
exit 0
`,
);
fs.chmodSync(fakeLsof, 0o755);

// Stand-in daemon that "systemd" starts after the kill: answers
// activeTasksQuery on the UDS and holds 127.0.0.1:8787 so
// verifyDaemonStartup() returns promptly.
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

// Fake systemctl / loginctl: slow like a busy systemd DBus round-trip.
// The restart path runs them three times; synchronous calls would stall
// the loop for the whole of each.
const fakeSystemctl = path.join(fakeBin, 'systemctl');
fs.writeFileSync(
  fakeSystemctl,
  `#!/bin/sh
sleep ${LSOF_DELAY_S}
case " $* " in
  *" restart "*)
    nohup "${process.execPath}" "${udsHelper}" >/dev/null 2>&1 &
    ;;
esac
exit 0
`,
);
fs.chmodSync(fakeSystemctl, 0o755);
const fakeLoginctl = path.join(fakeBin, 'loginctl');
fs.writeFileSync(fakeLoginctl, `#!/bin/sh\nsleep ${LSOF_DELAY_S}\nexit 0\n`);
fs.chmodSync(fakeLoginctl, 0o755);

// The child: runs the real restart while a heartbeat timer records the
// longest gap between ticks -- the longest the event loop was blocked.
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
let last = Date.now();
let maxStallMs = 0;
const beat = setInterval(() => {
  const now = Date.now();
  if (now - last > maxStallMs) maxStallMs = now - last;
  last = now;
}, 25);
const installer = require(process.env.KISS_TEST_MODULE);
installer
  .restartKissWebDaemon(process.env.KISS_TEST_PROJ, process.env.KISS_TEST_WORK)
  .then(() => {
    clearInterval(beat);
    process.stdout.write('\\nMAXSTALL=' + maxStallMs + '\\nDONE');
  })
  .catch(err => {
    clearInterval(beat);
    process.stdout.write('FAIL:' + (err && err.message));
    process.exitCode = 1;
  });
`;

function cleanup() {
  try {
    victim.kill('SIGKILL');
  } catch {}
  try {
    const pid = parseInt(fs.readFileSync(helperPidFile, 'utf-8'), 10);
    if (pid > 0) process.kill(pid, 'SIGKILL');
  } catch {}
  try {
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  } catch {}
}

async function main() {
  await victimReady;
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

  assert.ok(out.endsWith('DONE'), `restart entry point failed: ${out}`);

  // The poll loop really ran against a SIGTERM-ignoring listener: lsof
  // was consulted repeatedly and the victim ended up SIGKILLed.
  const lsofCalls = fs.readFileSync(lsofLog, 'utf-8').trim().split('\n');
  assert.ok(
    lsofCalls.length >= 7,
    `expected the full poll loop (>= 7 lsof calls), got ${lsofCalls.length}`,
  );
  const victimGone = await new Promise(resolve => {
    const deadline = Date.now() + 5000;
    const poll = () => {
      try {
        process.kill(victim.pid, 0);
      } catch {
        resolve(true);
        return;
      }
      if (Date.now() >= deadline) {
        resolve(false);
        return;
      }
      setTimeout(poll, 50);
    };
    poll();
  });
  assert.ok(victimGone, 'the SIGTERM-ignoring listener was never SIGKILLed');
  console.log('  ok - the stale listener was polled and then SIGKILLed');

  const m = /MAXSTALL=(\d+)/.exec(out);
  assert.ok(m, `child did not report its heartbeat: ${out}`);
  const maxStallMs = parseInt(m[1], 10);
  // Each fake lsof/systemctl/loginctl takes 300ms: any synchronous exec
  // of one stalls the loop by at least that much, so the bound sits
  // well below it.
  assert.ok(
    maxStallMs < 250,
    `event loop stalled for ${maxStallMs}ms while waiting for the old ` +
      'daemon to exit — the poll loop is blocking the extension host again',
  );
  console.log(
    `  ok - longest event-loop stall during restart: ${maxStallMs}ms`,
  );
}

main()
  .then(() => {
    cleanup();
    console.log('concaudit_w7_kill_port_no_block: all assertions passed');
  })
  .catch(err => {
    cleanup();
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
