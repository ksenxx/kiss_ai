// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for ``macLaunchd.js`` — the drain-aware LaunchAgent
// restart that fixes the "kiss-web launch takes a lot of time after an
// update" bug.
//
// Bug being locked in
// -------------------
// ``restartKissWebDaemon`` used to run ``launchctl bootout →
// bootstrap → kickstart -k`` back to back.  When the old job instance
// was still draining (the kiss-web daemon takes ~5 s to unwind after
// SIGTERM), launchd accepted the ``bootstrap`` but the still-pending
// ``bootout`` removed the WHOLE service — including the fresh
// registration — the moment the old instance exited.  Unified-log
// evidence from the 2026-07-19 14:30:58Z update:
//
//   07:31:03.301 launchd: removing service: com.kiss.web-server
//   07:31:13.539 launchd: service inactive: com.kiss.web-server
//   07:31:23.561 launchd: Successfully spawned kiss-web[90306]
//                because inefficient
//
// Nothing was loaded for 15 s (until ``verifyDaemonStartup`` re-issued
// the restart) and the respawn was then throttled ~10 more seconds —
// ~26.4 s from SIGTERM to "Server starting" on EVERY restart.
//
// These tests use the REAL launchd (a throwaway LaunchAgent with a
// unique test label) to first REPRODUCE the race — the naive sequence
// leaves the service unloaded with no new instance — and then prove
// ``restartLaunchAgent`` brings a fresh instance up promptly.  Edge
// branches (fail-closed stuck drain, inconclusive probes from a
// hanging or missing launchctl, bootstrap retry/fallback, kickstart
// failure) are driven through REAL controlled ``launchctl``-shaped
// executables run as normal child processes — no project code is
// mocked.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/macLaunchdRestart.test.js

'use strict';

const assert = require('assert');
const {execFileSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {restartLaunchAgent, probeService} = require('../src/macLaunchd');

const IS_DARWIN = process.platform === 'darwin';

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function test(name, fn) {
  await fn();
  console.log(`  ok - ${name}`);
}

/** Lines currently in the start-marker file (one line per payload
 * process start). */
function markerLines(markerFile) {
  try {
    return fs
      .readFileSync(markerFile, 'utf-8')
      .split('\n')
      .filter(l => l.trim().length > 0);
  } catch {
    return [];
  }
}

/** Poll until the marker file has at least ``n`` lines. */
async function waitForStarts(markerFile, n, timeoutMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (markerLines(markerFile).length >= n) return true;
    await sleep(200);
  }
  return markerLines(markerFile).length >= n;
}

function tryLaunchctl(args) {
  try {
    execFileSync('launchctl', args, {stdio: 'ignore', timeout: 5000});
    return true;
  } catch {
    return false;
  }
}

/** The OLD, racy restart sequence (verbatim behaviour of the previous
 * ``reissueRestart``): bootout, bootstrap immediately, ``load -w``
 * fallback, kickstart — with no drain wait. */
function naiveRestartSequence(serviceTarget, domainTarget, plistFile) {
  tryLaunchctl(['bootout', serviceTarget]);
  if (!tryLaunchctl(['bootstrap', domainTarget, plistFile])) {
    tryLaunchctl(['load', '-w', plistFile]);
  }
  tryLaunchctl(['kickstart', '-k', serviceTarget]);
}

// ---------------------------------------------------------------------------
// Part A — REAL launchd: reproduce the drain race, verify the fix.
// ---------------------------------------------------------------------------

async function realLaunchdTests() {
  const uid = execFileSync('id', ['-u'], {encoding: 'utf-8'}).trim();
  const label = `com.kiss.test.maclaunchd.${process.pid}.${Date.now()}`;
  const serviceTarget = `gui/${uid}/${label}`;
  const domainTarget = `gui/${uid}`;
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-maclaunchd-'));
  const markerFile = path.join(tmp, 'starts.txt');
  const payload = path.join(tmp, 'payload.sh');

  // A long-running payload that records every start and — like the
  // kiss-web daemon — takes ~3 s to drain after SIGTERM.
  fs.writeFileSync(
    payload,
    '#!/bin/sh\n' +
      `echo "$$" >> "${markerFile}"\n` +
      "trap 'sleep 3; exit 0' TERM\n" +
      'while :; do sleep 1; done\n',
  );
  fs.chmodSync(payload, 0o755);

  const plistFile = path.join(tmp, `${label}.plist`);
  fs.writeFileSync(
    plistFile,
    `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>${payload}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>ThrottleInterval</key>
    <integer>1</integer>
</dict>
</plist>`,
  );

  try {
    // Load the agent and wait for the first payload instance.
    execFileSync('launchctl', ['bootstrap', domainTarget, plistFile], {
      stdio: 'ignore',
      timeout: 5000,
    });
    assert.ok(
      await waitForStarts(markerFile, 1, 10_000),
      'initial LaunchAgent instance never started',
    );

    await test(
      'REPRODUCTION: naive bootout→bootstrap→kickstart during drain ' +
        'leaves the service unloaded and spawns nothing',
      async () => {
        const baseline = markerLines(markerFile).length;
        naiveRestartSequence(serviceTarget, domainTarget, plistFile);
        // Old instance drains ~3 s; give launchd ample time to spawn a
        // replacement if the bootstrap registration had survived
        // (RunAtLoad + ThrottleInterval=1 spawn within ~2 s).
        await sleep(6_000);
        assert.strictEqual(
          markerLines(markerFile).length,
          baseline,
          'no new instance must start — the pending bootout removes ' +
            'the mid-drain bootstrap registration together with the ' +
            'service',
        );
        assert.strictEqual(
          await probeService('launchctl', serviceTarget),
          'absent',
          'service must be fully removed after the drain',
        );
      },
    );

    await test(
      'fix: restartLaunchAgent from the unloaded state spawns promptly',
      async () => {
        const baseline = markerLines(markerFile).length;
        const t0 = Date.now();
        const res = await restartLaunchAgent({
          serviceTarget,
          domainTarget,
          plistFile,
          pollIntervalMs: 100,
        });
        assert.strictEqual(res.drained, true);
        assert.strictEqual(res.bootstrapped, true);
        assert.strictEqual(res.registered, true);
        assert.ok(
          await waitForStarts(markerFile, baseline + 1, 8_000),
          'new instance must start after restartLaunchAgent',
        );
        assert.ok(
          Date.now() - t0 < 12_000,
          `restart must beat the old 15s+10s penalty (took ${Date.now() - t0}ms)`,
        );
        // A plain kickstart (no -k) must NOT kill the fresh instance
        // and force a second start.
        await sleep(1_500);
        assert.strictEqual(
          markerLines(markerFile).length,
          baseline + 1,
          'exactly one new instance must start (kickstart without -k ' +
            'must not kill the freshly bootstrapped daemon)',
        );
      },
    );

    await test(
      'fix: restartLaunchAgent while the old instance is draining ' +
        'waits out the drain and still spawns a fresh instance fast',
      async () => {
        const baseline = markerLines(markerFile).length;
        const t0 = Date.now();
        const res = await restartLaunchAgent({
          serviceTarget,
          domainTarget,
          plistFile,
          pollIntervalMs: 100,
        });
        assert.strictEqual(res.drained, true);
        assert.strictEqual(res.bootstrapped, true);
        assert.ok(
          res.drainedMs >= 1_000,
          `drain wait must cover the ~3s SIGTERM drain (drainedMs=${res.drainedMs})`,
        );
        assert.ok(
          await waitForStarts(markerFile, baseline + 1, 8_000),
          'a fresh instance must start after the drain-aware restart',
        );
        assert.ok(
          Date.now() - t0 < 12_000,
          `restart must beat the old 15s+10s penalty (took ${Date.now() - t0}ms)`,
        );
      },
    );

    await test(
      'defaults: restartLaunchAgent on an absent service with a ' +
        'missing plist reports bootstrap failure without throwing',
      async () => {
        const res = await restartLaunchAgent({
          serviceTarget: `gui/${uid}/com.kiss.test.absent.${process.pid}`,
          domainTarget,
          plistFile: path.join(tmp, 'does-not-exist.plist'),
          pollIntervalMs: 50,
        });
        assert.strictEqual(res.drained, true);
        assert.strictEqual(res.bootstrapped, false);
        // ``registered`` is NOT asserted here: the real ``launchctl
        // load`` notoriously exits 0 even when it loads nothing
        // (which is exactly why bootstrap is the primary path).
        assert.strictEqual(res.kickstarted, false);
      },
    );
  } finally {
    // Robust teardown: bootout (retried) and poll until launchd
    // positively no longer knows the service before deleting the
    // payload/plist files it references.
    let gone = false;
    for (let i = 0; i < 40 && !gone; i++) {
      tryLaunchctl(['bootout', serviceTarget]);
      gone = (await probeService('launchctl', serviceTarget)) === 'absent';
      if (!gone) await sleep(500);
    }
    assert.ok(gone, `test LaunchAgent ${serviceTarget} was not cleaned up`);
    fs.rmSync(tmp, {recursive: true, force: true});
  }
}

// ---------------------------------------------------------------------------
// Part B — controlled launchctl-shaped executables: edge branches.
// ---------------------------------------------------------------------------

/** Write a real executable that behaves like launchctl, driven by
 * state files in its own directory:
 *   - ``print``      exits 0 while ``present`` exists, hangs (past the
 *                    5s command timeout) while ``hang`` exists, else
 *                    exits 113 ("Could not find service");
 *   - ``bootout``    removes ``present`` only if ``bootout_removes``
 *                    exists (a stuck drain otherwise);
 *   - ``bootstrap``  succeeds from the Nth call (``bootstrap_ok_after``);
 *   - ``load``       fails while ``load_fail`` exists;
 *   - ``kickstart``  fails while ``kick_fail`` exists.
 */
function writeControlledLaunchctl(dir) {
  const bin = path.join(dir, 'launchctl');
  fs.writeFileSync(
    bin,
    `#!/bin/sh
dir="$(cd "$(dirname "$0")" && pwd)"
case "$1" in
  print)
    [ -f "$dir/hang" ] && sleep 30
    [ -f "$dir/present" ] && exit 0 || exit 113 ;;
  bootout) [ -f "$dir/bootout_removes" ] && rm -f "$dir/present"; exit 0 ;;
  bootstrap)
    n=0
    [ -f "$dir/count" ] && n=$(cat "$dir/count")
    n=$((n + 1))
    echo "$n" > "$dir/count"
    ok=999
    [ -f "$dir/bootstrap_ok_after" ] && ok=$(cat "$dir/bootstrap_ok_after")
    [ "$n" -ge "$ok" ] && exit 0 || exit 1 ;;
  load) [ -f "$dir/load_fail" ] && exit 1 || exit 0 ;;
  kickstart) [ -f "$dir/kick_fail" ] && exit 1 || exit 0 ;;
esac
exit 1
`,
  );
  fs.chmodSync(bin, 0o755);
  return bin;
}

async function controlledBinaryTests() {
  await test(
    'fail closed: a service that never drains gets NO bootstrap — ' +
      'only a kickstart of the surviving registration',
    async () => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-fakectl-'));
      try {
        const bin = writeControlledLaunchctl(dir);
        fs.writeFileSync(path.join(dir, 'present'), '');
        const logs = [];
        const res = await restartLaunchAgent({
          serviceTarget: 'gui/501/com.kiss.test.stuck',
          domainTarget: 'gui/501',
          plistFile: path.join(dir, 'x.plist'),
          launchctlPath: bin,
          log: m => logs.push(m),
          drainTimeoutMs: 400,
          pollIntervalMs: 100,
        });
        assert.strictEqual(res.drained, false);
        assert.ok(res.drainedMs >= 400);
        assert.strictEqual(res.bootstrapAttempts, 0);
        assert.strictEqual(res.bootstrapped, false);
        assert.strictEqual(res.registered, true);
        assert.strictEqual(res.kickstarted, true);
        assert.strictEqual(
          fs.existsSync(path.join(dir, 'count')),
          false,
          'bootstrap must never run while the service is present',
        );
        assert.ok(
          logs.some(m => m.includes('bootstrap refused (fail closed)')),
          `expected a fail-closed log, got: ${JSON.stringify(logs)}`,
        );
      } finally {
        fs.rmSync(dir, {recursive: true, force: true});
      }
    },
  );

  await test(
    'fail closed: an inconclusive probe (missing launchctl binary) ' +
      'never authorizes a bootstrap',
    async () => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-fakectl-'));
      try {
        const logs = [];
        const res = await restartLaunchAgent({
          serviceTarget: 'gui/501/com.kiss.test.enoent',
          domainTarget: 'gui/501',
          plistFile: path.join(dir, 'x.plist'),
          launchctlPath: path.join(dir, 'no-such-launchctl'),
          log: m => logs.push(m),
          drainTimeoutMs: 300,
          pollIntervalMs: 100,
        });
        assert.strictEqual(res.drained, false);
        assert.strictEqual(res.bootstrapAttempts, 0);
        assert.strictEqual(res.bootstrapped, false);
        assert.strictEqual(res.registered, false);
        assert.strictEqual(res.kickstarted, false);
        assert.ok(
          logs.some(m => m.includes('still unknown')),
          `expected an unknown-state log, got: ${JSON.stringify(logs)}`,
        );
      } finally {
        fs.rmSync(dir, {recursive: true, force: true});
      }
    },
  );

  await test(
    'fail closed: a hanging launchctl print (timeout, inconclusive) ' +
      'is never treated as drained',
    async () => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-fakectl-'));
      try {
        const bin = writeControlledLaunchctl(dir);
        fs.writeFileSync(path.join(dir, 'hang'), '');
        const state = await probeService(bin, 'gui/501/com.kiss.test.hang');
        assert.strictEqual(
          state,
          'unknown',
          'a timed-out print must be inconclusive, not "absent"',
        );
      } finally {
        fs.rmSync(dir, {recursive: true, force: true});
      }
    },
  );

  await test('bootstrap retries until launchd stops reporting EEXIST', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-fakectl-'));
    try {
      const bin = writeControlledLaunchctl(dir);
      fs.writeFileSync(path.join(dir, 'present'), '');
      fs.writeFileSync(path.join(dir, 'bootout_removes'), '');
      fs.writeFileSync(path.join(dir, 'bootstrap_ok_after'), '3');
      const res = await restartLaunchAgent({
        serviceTarget: 'gui/501/com.kiss.test.retry',
        domainTarget: 'gui/501',
        plistFile: path.join(dir, 'x.plist'),
        launchctlPath: bin,
        drainTimeoutMs: 1_000,
        pollIntervalMs: 50,
        bootstrapAttempts: 4,
      });
      assert.strictEqual(res.drained, true);
      assert.strictEqual(res.bootstrapped, true);
      assert.strictEqual(res.registered, true);
      assert.strictEqual(res.bootstrapAttempts, 3);
      assert.strictEqual(res.kickstarted, true);
    } finally {
      fs.rmSync(dir, {recursive: true, force: true});
    }
  });

  await test(
    'bootstrap exhausted: load -w fallback result is reported ' +
      'truthfully (registered) alongside a failed kickstart',
    async () => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-fakectl-'));
      try {
        const bin = writeControlledLaunchctl(dir);
        fs.writeFileSync(path.join(dir, 'kick_fail'), '');
        const logs = [];
        const res = await restartLaunchAgent({
          serviceTarget: 'gui/501/com.kiss.test.fallback',
          domainTarget: 'gui/501',
          plistFile: path.join(dir, 'x.plist'),
          launchctlPath: bin,
          log: m => logs.push(m),
          drainTimeoutMs: 200,
          pollIntervalMs: 50,
          bootstrapAttempts: 2,
        });
        assert.strictEqual(res.drained, true);
        assert.strictEqual(res.bootstrapped, false);
        assert.strictEqual(res.bootstrapAttempts, 2);
        assert.strictEqual(res.registered, true); // load -w succeeded
        assert.strictEqual(res.kickstarted, false);
        assert.ok(
          logs.some(m => m.includes('load -w fallback succeeded')),
          `expected a fallback log, got: ${JSON.stringify(logs)}`,
        );
      } finally {
        fs.rmSync(dir, {recursive: true, force: true});
      }
    },
  );

  await test(
    'bootstrap failure with load -w also failing reports registered=false',
    async () => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-fakectl-'));
      try {
        const bin = writeControlledLaunchctl(dir);
        fs.writeFileSync(path.join(dir, 'load_fail'), '');
        const logs = [];
        const res = await restartLaunchAgent({
          serviceTarget: 'gui/501/com.kiss.test.allfail',
          domainTarget: 'gui/501',
          plistFile: path.join(dir, 'x.plist'),
          launchctlPath: bin,
          log: m => logs.push(m),
          drainTimeoutMs: 200,
          pollIntervalMs: 50,
          bootstrapAttempts: 1,
        });
        assert.strictEqual(res.drained, true);
        assert.strictEqual(res.bootstrapped, false);
        assert.strictEqual(res.registered, false);
        assert.ok(
          logs.some(m => m.includes('load -w fallback failed')),
          `expected a fallback-failed log, got: ${JSON.stringify(logs)}`,
        );
      } finally {
        fs.rmSync(dir, {recursive: true, force: true});
      }
    },
  );
}

async function main() {
  if (process.platform === 'win32') {
    console.log('  skip - macLaunchdRestart tests (win32)');
    return;
  }
  await controlledBinaryTests();
  if (!IS_DARWIN) {
    console.log('  skip - real-launchd tests (not macOS)');
    return;
  }
  await realLaunchdTests();
  console.log('macLaunchdRestart tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
