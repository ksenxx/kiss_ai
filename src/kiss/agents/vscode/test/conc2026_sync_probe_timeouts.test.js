// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: the synchronous `which` probes must not
// wedge the extension host on a stalled child.
//
// The bug: kissPaths.findUvPath() ran `execSync('which uv')` and
// voiceAckPlayer's player probe ran `spawnSync('which', [player])` with
// NO timeout.  Both run on the extension host's event loop (activation,
// resolveDefaultModel, voiceWake.start, the webview's voiceAck message),
// so a `which` that stalls — a PATH entry on a dead network mount —
// froze the entire extension host for as long as the child lived.
// DependencyInstaller's own SYNC_PROBE_TIMEOUT_MS comment states the
// policy these two probes violated.
//
// The fix gives both probes a 5s ceiling with killSignal SIGKILL —
// Node's default timeout kill is SIGTERM, which a stalled probe can
// ignore, keeping spawnSync blocked past its timeout — and findUvPath
// uses execFileSync (no shell): execSync's shell layer meant a timeout
// killed only the shell and ORPHANED the underlying probe.
//
// A real 5s wait per probe is not acceptable in the suite, so the
// compiled modules are copied to a temp dir with the two constants
// lowered (each copy asserts its substitution matched, so a rename
// fails loudly).  The probes then run in a CHILD process against a PATH
// whose only `which` IGNORES SIGTERM, records its PID, and sleeps far
// past the watchdog; the parent's watchdog is the proof: before the fix
// the child wedged until the sleep ended, now it returns within the
// lowered ceilings AND no recorded probe process survives (no orphans).

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OUT_DIR = path.join(__dirname, '..', 'out');
if (!fs.existsSync(path.join(OUT_DIR, 'kissPaths.js'))) {
  console.log('SKIP: out/kissPaths.js missing — run `npm run compile`');
  process.exit(0);
}
if (process.platform === 'win32') {
  console.log('SKIP: POSIX PATH shims only');
  process.exit(0);
}

const SHORT_TIMEOUT_MS = 400;
// findUvPath: 1 probe; ackPlayerCommand: afplay (darwin only) + 3
// fallback players — every probe may take the full lowered ceiling.
const WATCHDOG_MS = 10_000;

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-syncprobe-'));
const outCopy = path.join(tmpRoot, 'out');
const fakeBin = path.join(tmpRoot, 'bin');
const fakeHome = path.join(tmpRoot, 'home');
for (const d of [outCopy, fakeBin, fakeHome]) {
  fs.mkdirSync(d, {recursive: true});
}

for (const f of fs.readdirSync(OUT_DIR)) {
  if (!f.endsWith('.js')) continue;
  let src = fs.readFileSync(path.join(OUT_DIR, f), 'utf-8');
  if (f === 'kissPaths.js') {
    const needle = 'const WHICH_TIMEOUT_MS = 5_000;';
    assert.ok(
      src.includes(needle),
      'kissPaths.js no longer defines WHICH_TIMEOUT_MS as expected — ' +
        'did the `which uv` probe lose its timeout?',
    );
    src = src.replace(needle, `const WHICH_TIMEOUT_MS = ${SHORT_TIMEOUT_MS};`);
  }
  if (f === 'voiceAckPlayer.js') {
    const needle = 'const PROBE_TIMEOUT_MS = 5_000;';
    assert.ok(
      src.includes(needle),
      'voiceAckPlayer.js no longer defines PROBE_TIMEOUT_MS as expected — ' +
        'did the player probe lose its timeout?',
    );
    src = src.replace(needle, `const PROBE_TIMEOUT_MS = ${SHORT_TIMEOUT_MS};`);
  }
  fs.writeFileSync(path.join(outCopy, f), src);
}

// The only `which` (and player binaries) on the child's PATH: a single
// process (exec, so no descendants to orphan) that IGNORES SIGTERM,
// records its PID, and sleeps far past the watchdog.  Only a timed-out
// probe killed with an unignorable signal lets the child exit — the
// default SIGTERM kill left spawnSync blocked on exactly such a probe.
const pidDir = path.join(tmpRoot, 'pids');
fs.mkdirSync(pidDir, {recursive: true});
const probeBody =
  'process.on("SIGTERM", () => {});' +
  'require("fs").writeFileSync(' +
  `require("path").join(${JSON.stringify(pidDir)}, String(process.pid)),` +
  '"");' +
  'setTimeout(() => {}, 120000);';
fs.writeFileSync(
  path.join(fakeBin, 'which'),
  `#!/bin/sh\nexec ${JSON.stringify(process.execPath)} -e ${JSON.stringify(
    probeBody,
  )}\n`,
);
fs.chmodSync(path.join(fakeBin, 'which'), 0o755);

// The probe body, run in a child so a wedged probe cannot wedge the
// suite: the watchdog kills it and the test fails with a clear message.
const childScript = path.join(tmpRoot, 'probe.js');
fs.writeFileSync(
  childScript,
  `
'use strict';
process.env.PATH = ${JSON.stringify(fakeBin)};
process.env.HOME = ${JSON.stringify(fakeHome)};
delete process.env.USERPROFILE;
delete process.env.KISS_SORCAR_PLAY_CMD;
const Module = require('module');
const orig = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') {
    return require.resolve(${JSON.stringify(
      path.join(__dirname, '_vscode-stub.js'),
    )});
  }
  return orig.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = {};
const {findUvPath} = require(${JSON.stringify(
    path.join(outCopy, 'kissPaths.js'),
  )});
const {ackPlayerCommand} = require(${JSON.stringify(
    path.join(outCopy, 'voiceAckPlayer.js'),
  )});
const t0 = Date.now();
const uv = findUvPath();
const uvMs = Date.now() - t0;
const t1 = Date.now();
const player = ackPlayerCommand(process.env);
const playerMs = Date.now() - t1;
console.log(JSON.stringify({uv, uvMs, player, playerMs}));
`,
);

function runChild() {
  return new Promise(resolve => {
    const child = spawn(process.execPath, [childScript], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    let timedOut = false;
    const watchdog = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, WATCHDOG_MS);
    child.stdout.on('data', d => {
      stdout += d.toString();
    });
    child.stderr.on('data', d => {
      stderr += d.toString();
    });
    child.on('close', code => {
      clearTimeout(watchdog);
      resolve({code, stdout, stderr, timedOut});
    });
  });
}

async function main() {
  try {
    const r = await runChild();
    assert.ok(
      !r.timedOut,
      'the probes wedged past the watchdog: a stalled `which` still ' +
        'blocks findUvPath()/ackPlayerCommand() (and with them the ' +
        'extension host) with no timeout',
    );
    assert.strictEqual(r.code, 0, `probe child failed:\n${r.stderr}`);
    const res = JSON.parse(r.stdout.trim());
    // The stalled `which` is the ONLY binary on the child's PATH and the
    // fake HOME holds no uv, so both lookups must come back empty —
    // each within its lowered ceiling instead of the 120s sleep.
    // (findUvPath also checks two fixed absolute locations; on a machine
    // that really has uv there the probe legitimately finds it fast and
    // the null assertion does not apply.)
    const systemUv = ['/usr/local/bin/uv', '/opt/homebrew/bin/uv'].find(p =>
      fs.existsSync(p),
    );
    assert.strictEqual(
      res.uv,
      systemUv ?? null,
      'findUvPath reports uv missing (or the fixed system location)',
    );
    assert.strictEqual(res.player, null, 'ackPlayerCommand finds no player');
    assert.ok(
      res.uvMs < SHORT_TIMEOUT_MS + 2_000,
      `findUvPath returned in ${res.uvMs}ms (expected ~${SHORT_TIMEOUT_MS}ms)`,
    );
    // Every timed-out probe recorded its PID before ignoring SIGTERM.
    // With killSignal SIGKILL (and no execSync shell layer to absorb the
    // kill) none may survive the probes' return: an execSync-style
    // timeout killed only the shell and left the real probe orphaned.
    const pids = fs
      .readdirSync(pidDir)
      .map(name => parseInt(name, 10))
      .filter(pid => Number.isInteger(pid) && pid > 1);
    assert.ok(pids.length >= 2, `probes ran (${pids.length} PIDs recorded)`);
    const deadline = Date.now() + 3_000;
    for (const pid of pids) {
      for (;;) {
        let alive = true;
        try {
          process.kill(pid, 0);
        } catch {
          alive = false;
        }
        if (!alive) break;
        assert.ok(
          Date.now() < deadline,
          `probe PID ${pid} survived its timeout: the SIGTERM-ignoring ` +
            'which was orphaned instead of being SIGKILLed',
        );
        await new Promise(r => setTimeout(r, 50));
      }
    }
  } finally {
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  }
  console.log('PASS conc2026_sync_probe_timeouts');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
