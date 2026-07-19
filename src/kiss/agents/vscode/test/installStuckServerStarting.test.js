// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the "KISS Sorcar Server is starting ..."
// hang after running ``./install.sh``.
//
// Bug being locked in
// -------------------
// ``install.sh`` builds + installs the VSIX and writes
// ``~/.kiss/.extension-updated``; the extension reloads and
// ``ensureDependencies()`` → ``restartKissWebDaemon()`` restarts the
// kiss-web daemon.  The old restart flow killed the daemon, ran
// ``launchctl bootout/bootstrap/kickstart`` with ``stdio:'ignore'``,
// logged "restarted" and wrote the fingerprint file — WITHOUT ever
// verifying a new daemon came up.  Two production races then left the
// daemon down with no recovery path, so the chat webview hung forever
// on the "KISS Sorcar Server is starting ..." overlay
// (``AgentClient``'s 500 ms reconnect loop keeps failing and the
// ``daemonStatus: {connected: true}`` event never fires):
//
//   1. bootout → bootstrap race: ``launchctl bootstrap`` fails with
//      EEXIST while the old job instance is still draining, the
//      ``load -w`` fallback silently no-ops, ``kickstart`` kicks
//      nothing — NOTHING is loaded and the daemon never respawns.
//
//   2. venv-wipe race (install.log 2026-07-19T04:50:10Z, 2 m 16 s
//      outage): a concurrent ``code --install-extension --force``
//      deletes the extension dir including ``.venv/bin/kiss-web``
//      right after the restart decision; launchd's exec fails
//      silently on every KeepAlive tick until something re-runs
//      ``uv sync``.
//
// The fix is ``verifyDaemonStartup`` (``daemonRestartVerify.js``):
// after the initial bootstrap the restart flow now polls the REAL
// daemon state (TCP probe + UDS socket file) and re-issues the restart
// while the port is provably dead and the binary exists — and the
// caller only logs success / writes the fingerprint on a verified
// startup.
//
// The tests below exercise the REAL ``daemonRestartVerify.js`` (no
// mocks of project code) against real TCP listeners, real Unix-domain
// sockets, and a real on-disk fake ``.venv/bin/kiss-web`` tree.  The
// ``restart`` callbacks perform exactly what launchd would: spawn (or
// fail to spawn) real in-process listeners.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/installStuckServerStarting.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

const {verifyDaemonStartup} = require('../src/daemonRestartVerify');
const {probeDaemonHealth} = require('../src/daemonHealth');

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-install-stuck-'));

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  FAIL - ${name}: ${err && err.message}`);
  }
}

/** Pick a free TCP port by binding an ephemeral listener and closing it. */
function freeTcpPort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
  });
}

/**
 * Start the UDS half of a "kiss-web daemon" stand-in: a real UDS
 * listener at ``sockPath`` speaking the daemon's newline-delimited
 * JSON protocol — it answers ``activeTasksQuery`` with an
 * ``activeTasksResponse`` carrying ``activeCount`` (matching
 * ``RemoteAccessServer._uds_handler``), which is what the verifier's
 * ``daemonHasActiveTasks`` round-trip requires.
 */
function startUdsHalf(sockPath, activeCount) {
  return new Promise((resolve, reject) => {
    try {
      fs.rmSync(sockPath, {force: true});
    } catch {
      /* ignore */
    }
    const uds = net.createServer(conn => {
      conn.setEncoding('utf-8');
      let buf = '';
      conn.on('data', chunk => {
        buf += chunk;
        let nl = buf.indexOf('\n');
        while (nl >= 0) {
          const line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          nl = buf.indexOf('\n');
          if (!line.trim()) continue;
          let msg;
          try {
            msg = JSON.parse(line);
          } catch {
            continue;
          }
          if (msg && msg.type === 'activeTasksQuery') {
            conn.write(
              JSON.stringify({
                type: 'activeTasksResponse',
                count: activeCount || 0,
                tabs: [],
              }) + '\n',
            );
          }
        }
      });
      conn.on('error', () => {/* client went away — ignore */});
    });
    uds.once('error', reject);
    uds.listen(sockPath, () => {
      resolve({
        close: async () => {
          await new Promise(res => uds.close(() => res()));
          try {
            fs.rmSync(sockPath, {force: true});
          } catch {
            /* ignore */
          }
        },
      });
    });
  });
}

/** Start the TCP (WSS) half of the stand-in daemon on ``port``. */
function startTcpHalf(port) {
  return new Promise((resolve, reject) => {
    const tcp = net.createServer(() => {/* accept and idle */});
    tcp.once('error', reject);
    tcp.listen(port, '127.0.0.1', () => {
      resolve({
        close: () => new Promise(res => tcp.close(() => res())),
      });
    });
  });
}

/**
 * Start a full "kiss-web daemon" stand-in: TCP listener on ``port``
 * (the WSS side probed by ``probeDaemonHealth``) plus the
 * protocol-speaking UDS listener at ``sockPath`` (the side
 * ``AgentClient`` / ``daemonHasActiveTasks`` connects to).
 */
async function startFakeDaemon(port, sockPath) {
  const tcp = await startTcpHalf(port);
  const uds = await startUdsHalf(sockPath, 0);
  return {
    close: async () => {
      await tcp.close();
      await uds.close();
    },
  };
}

/** Create a real fake ``.venv/bin/kiss-web`` file and return its path. */
function makeKissWebBin(name) {
  const bin = path.join(tmpRoot, name, '.venv', 'bin', 'kiss-web');
  fs.mkdirSync(path.dirname(bin), {recursive: true});
  fs.writeFileSync(bin, '#!/bin/sh\nexit 0\n', {mode: 0o755});
  return bin;
}

async function main() {
  // -----------------------------------------------------------------
  // 1. REPRODUCTION — the pre-fix behaviour.  The restart flow claimed
  //    success while no daemon was listening and there was no recovery
  //    path (restart: null ≙ the old fire-and-forget bootstrap).  The
  //    verifier must report the truth: the daemon is NOT up — this is
  //    exactly the state the user was stranded in behind the
  //    "KISS Sorcar Server is starting ..." overlay.
  // -----------------------------------------------------------------
  await test(
    'reproduction: dead daemon + no recovery == stuck overlay (timeout)',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('repro-timeout');
      const sockPath = path.join(tmpRoot, 'repro-timeout.sock');
      // Pre-fix reality check: the port is provably dead and the UDS
      // file is absent immediately after the "successful" bootstrap.
      assert.strictEqual(await probeDaemonHealth(port, 500), 'dead');
      assert.strictEqual(fs.existsSync(sockPath), false);
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: null,
        timeoutMs: 400,
        pollIntervalMs: 25,
        probeTimeoutMs: 200,
      });
      assert.strictEqual(res.ok, false);
      assert.strictEqual(res.reason, 'timeout');
      assert.strictEqual(res.restarts, 0);
      assert.strictEqual(res.binaryVanished, false);
      assert.ok(res.waitedMs >= 400, `waitedMs=${res.waitedMs}`);
    },
  );

  // -----------------------------------------------------------------
  // 2. Bootstrap race healed: the first re-restart attempt throws
  //    (``launchctl bootstrap`` → EEXIST while the old job drains),
  //    the second brings up a REAL daemon.  The verifier must survive
  //    the throw, retry, and report success.
  // -----------------------------------------------------------------
  await test(
    'bootstrap EEXIST race: verifier retries through a throwing restart',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('bootstrap-race');
      const sockPath = path.join(tmpRoot, 'bootstrap-race.sock');
      let daemon = null;
      let calls = 0;
      let spawned = false;
      const logs = [];
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: () => {
          calls += 1;
          if (calls === 1) {
            throw new Error('Bootstrap failed: 5: Input/output error');
          }
          // launchd finally accepts the bootstrap: real daemon comes up.
          // (Idempotent like launchd itself — a second kick while the
          // daemon is already up must not double-bind the port.)
          if (!spawned) {
            spawned = true;
            startFakeDaemon(port, sockPath).then(d => {
              daemon = d;
            });
          }
        },
        log: msg => logs.push(msg),
        timeoutMs: 10_000,
        pollIntervalMs: 25,
        restartEveryMs: 50,
        probeTimeoutMs: 200,
      });
      assert.strictEqual(res.ok, true);
      assert.strictEqual(res.reason, 'alive');
      assert.ok(res.restarts >= 2, `restarts=${res.restarts}`);
      assert.strictEqual(res.binaryVanished, false);
      assert.ok(
        logs.some(m => m.includes('re-restart attempt failed')),
        'the throwing restart must be logged, not fatal',
      );
      assert.ok(daemon, 'real daemon must have been started');
      await daemon.close();
    },
  );

  // -----------------------------------------------------------------
  // 3. Venv-wipe race healed: the kiss-web binary has been deleted by
  //    a concurrent ``code --install-extension --force``.  While it is
  //    missing the verifier must NOT invoke restart (launchd cannot
  //    exec it anyway).  Once the "reinstall" puts the binary back,
  //    the next eligible poll re-restarts and the daemon comes up.
  // -----------------------------------------------------------------
  await test(
    'venv-wipe race: waits for binary to reappear, then restarts to alive',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('venv-wipe');
      const sockPath = path.join(tmpRoot, 'venv-wipe.sock');
      fs.rmSync(bin); // the concurrent reinstall wiped the venv
      let daemon = null;
      let restartsWhileBinMissing = 0;
      let spawned = false;
      // The "reinstall" finishes 250 ms in: binary reappears.
      const reinstall = setTimeout(() => {
        fs.mkdirSync(path.dirname(bin), {recursive: true});
        fs.writeFileSync(bin, '#!/bin/sh\nexit 0\n', {mode: 0o755});
      }, 250);
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: () => {
          if (!fs.existsSync(bin)) restartsWhileBinMissing += 1;
          if (!spawned) {
            spawned = true;
            startFakeDaemon(port, sockPath).then(d => {
              daemon = d;
            });
          }
        },
        timeoutMs: 10_000,
        pollIntervalMs: 25,
        restartEveryMs: 50,
        probeTimeoutMs: 200,
      });
      clearTimeout(reinstall);
      assert.strictEqual(res.ok, true);
      assert.strictEqual(res.reason, 'alive');
      assert.strictEqual(res.binaryVanished, true);
      assert.strictEqual(
        restartsWhileBinMissing,
        0,
        'restart must never fire while the binary is missing',
      );
      assert.ok(res.restarts >= 1, `restarts=${res.restarts}`);
      assert.ok(res.waitedMs >= 250, `waitedMs=${res.waitedMs}`);
      assert.ok(daemon, 'real daemon must have been started');
      await daemon.close();
    },
  );

  // -----------------------------------------------------------------
  // 4. Binary never reappears: budget exhausts with the binary absent
  //    → the caller must learn the venv is gone (reason
  //    'binary-missing'), not a generic timeout.
  // -----------------------------------------------------------------
  await test(
    'binary missing at deadline reports binary-missing and zero restarts',
    async () => {
      const port = await freeTcpPort();
      const bin = path.join(tmpRoot, 'never-reinstalled', '.venv', 'bin',
        'kiss-web');
      const sockPath = path.join(tmpRoot, 'never-reinstalled.sock');
      let calls = 0;
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: () => {
          calls += 1;
        },
        timeoutMs: 300,
        pollIntervalMs: 25,
        restartEveryMs: 0,
        probeTimeoutMs: 200,
      });
      assert.strictEqual(res.ok, false);
      assert.strictEqual(res.reason, 'binary-missing');
      assert.strictEqual(res.binaryVanished, true);
      assert.strictEqual(res.restarts, 0);
      assert.strictEqual(calls, 0);
    },
  );

  // -----------------------------------------------------------------
  // 5. Update-button hang variant: daemon alive on TCP but its UDS
  //    socket file was deleted out from under it.  The extension
  //    cannot reach it → NOT ok; reason must be 'sock-missing'.  The
  //    'dead'-gated restart must never fire against the alive daemon.
  // -----------------------------------------------------------------
  await test(
    'alive TCP + missing UDS file reports sock-missing without restarts',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('sock-missing');
      const sockPath = path.join(tmpRoot, 'sock-missing.sock');
      const daemon = await startFakeDaemon(port, sockPath);
      fs.rmSync(sockPath); // install.sh's `rm -f ~/.kiss/sorcar.sock` race
      let calls = 0;
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: () => {
          calls += 1;
        },
        timeoutMs: 300,
        pollIntervalMs: 25,
        restartEveryMs: 0,
        probeTimeoutMs: 200,
      });
      assert.strictEqual(res.ok, false);
      assert.strictEqual(res.reason, 'sock-missing');
      assert.strictEqual(res.restarts, 0);
      assert.strictEqual(calls, 0, 'must not bounce an alive daemon');
      await daemon.close();
    },
  );

  // -----------------------------------------------------------------
  // 6. Happy path: daemon already up when verification starts (the
  //    common healthy restart) → immediate success, no restarts.
  // -----------------------------------------------------------------
  await test('healthy restart verifies immediately without restarts',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('happy');
      const sockPath = path.join(tmpRoot, 'happy.sock');
      const daemon = await startFakeDaemon(port, sockPath);
      let calls = 0;
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: () => {
          calls += 1;
        },
        timeoutMs: 5_000,
        pollIntervalMs: 25,
        restartEveryMs: 0,
        probeTimeoutMs: 200,
      });
      assert.strictEqual(res.ok, true);
      assert.strictEqual(res.reason, 'alive');
      assert.strictEqual(res.restarts, 0);
      assert.strictEqual(res.binaryVanished, false);
      assert.strictEqual(calls, 0);
      assert.ok(res.waitedMs < 5_000);
      await daemon.close();
    });

  // -----------------------------------------------------------------
  // 7. Slow launchd respawn (the routine 42-136 s outage, scaled
  //    down): nothing to restart-fix — the daemon simply appears late.
  //    The verifier must keep polling past several intervals and
  //    return success the moment both listeners are up, WITHOUT
  //    re-restarting before ``restartEveryMs`` elapses.
  // -----------------------------------------------------------------
  await test('slow respawn: polls until the daemon appears, then succeeds',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('slow');
      const sockPath = path.join(tmpRoot, 'slow.sock');
      let daemon = null;
      const spawnTimer = setTimeout(() => {
        startFakeDaemon(port, sockPath).then(d => {
          daemon = d;
        });
      }, 300);
      let calls = 0;
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: () => {
          calls += 1;
        },
        timeoutMs: 10_000,
        pollIntervalMs: 25,
        restartEveryMs: 60_000, // first re-restart would only fire at 60 s
        probeTimeoutMs: 200,
      });
      clearTimeout(spawnTimer);
      assert.strictEqual(res.ok, true);
      assert.strictEqual(res.reason, 'alive');
      assert.strictEqual(
        calls,
        0,
        'a merely-slow daemon must not be re-bounced before restartEveryMs',
      );
      assert.ok(res.waitedMs >= 300, `waitedMs=${res.waitedMs}`);
      await daemon.close();
    });

  // -----------------------------------------------------------------
  // 8. Default option values: calling with only the required fields
  //    exercises every default branch (timeout/poll/restart/probe
  //    defaults, no log, no restart) against a real live daemon so it
  //    returns immediately.
  // -----------------------------------------------------------------
  await test('defaults: minimal options against a live daemon', async () => {
    const port = await freeTcpPort();
    const sockPath = path.join(tmpRoot, 'defaults.sock');
    const bin = makeKissWebBin('defaults');
    const daemon = await startFakeDaemon(port, sockPath);
    const res = await verifyDaemonStartup({binPath: bin, sockPath, port});
    assert.strictEqual(res.ok, true);
    assert.strictEqual(res.reason, 'alive');
    assert.strictEqual(res.restarts, 0);
    await daemon.close();
  });

  // -----------------------------------------------------------------
  // 9. Mid-boot protection (gpt-5.6-sol review, CRITICAL): kiss-web
  //    binds its UDS listener during startup independently of the WSS
  //    port, so a booting daemon can be ANSWERING on the UDS (and may
  //    already run user tasks) while the TCP probe still says 'dead'.
  //    The verifier must NEVER re-restart such a daemon — a successful
  //    UDS round-trip vetoes the restart even long past restartEveryMs.
  // -----------------------------------------------------------------
  await test('mid-boot daemon (UDS answering, TCP dead) is never bounced',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('mid-boot');
      const sockPath = path.join(tmpRoot, 'mid-boot.sock');
      const uds = await startUdsHalf(sockPath, 0); // UDS up first
      let tcp = null;
      const tcpTimer = setTimeout(() => {
        startTcpHalf(port).then(t => {
          tcp = t;
        });
      }, 300); // WSS side binds only 300 ms later
      let calls = 0;
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: () => {
          calls += 1;
        },
        timeoutMs: 10_000,
        pollIntervalMs: 25,
        restartEveryMs: 0, // eligible on EVERY poll — must still not fire
        probeTimeoutMs: 200,
      });
      clearTimeout(tcpTimer);
      assert.strictEqual(res.ok, true);
      assert.strictEqual(res.reason, 'alive');
      assert.strictEqual(res.restarts, 0);
      assert.strictEqual(
        calls,
        0,
        'a UDS-answering (mid-boot) daemon must never be re-restarted',
      );
      await uds.close();
      if (tcp) await tcp.close();
    });

  // -----------------------------------------------------------------
  // 10. Active-tasks veto: a daemon reachable only over the UDS that
  //     reports in-flight tasks must never be bounced even though the
  //     TCP probe is 'dead' for the whole budget — the verifier waits
  //     and reports an honest timeout instead of aborting the tasks.
  // -----------------------------------------------------------------
  await test('UDS daemon with active tasks + dead TCP: no restart, timeout',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('active-veto');
      const sockPath = path.join(tmpRoot, 'active-veto.sock');
      const uds = await startUdsHalf(sockPath, 2); // 2 in-flight tasks
      let calls = 0;
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: () => {
          calls += 1;
        },
        timeoutMs: 400,
        pollIntervalMs: 25,
        restartEveryMs: 0,
        probeTimeoutMs: 200,
      });
      assert.strictEqual(res.ok, false);
      assert.strictEqual(res.reason, 'timeout');
      assert.strictEqual(res.restarts, 0);
      assert.strictEqual(calls, 0, 'must not abort a busy daemon');
      await uds.close();
    });

  // -----------------------------------------------------------------
  // 11. Stale socket file (SIGKILLed old daemon left its UDS file
  //     behind, nothing listening): must NOT fake success — the
  //     verifier must classify it 'sock-missing'-unreachable and
  //     re-restart to a real daemon.
  // -----------------------------------------------------------------
  await test('stale UDS file does not fake success; restart still fires',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('stale-sock');
      const sockPath = path.join(tmpRoot, 'stale-sock.sock');
      // A dead daemon's leftover: bind a UDS then close the listener,
      // leaving the socket file on disk with nothing behind it.
      const dead = await startUdsHalf(sockPath, 0);
      await new Promise(res => setTimeout(res, 10));
      // close() removes the file; recreate a non-listening socket file
      // by binding+closing without cleanup:
      await dead.close();
      fs.writeFileSync(sockPath, ''); // plain file at the socket path
      let daemon = null;
      let spawned = false;
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: () => {
          if (!spawned) {
            spawned = true;
            startFakeDaemon(port, sockPath).then(d => {
              daemon = d;
            });
          }
        },
        timeoutMs: 10_000,
        pollIntervalMs: 25,
        restartEveryMs: 50,
        probeTimeoutMs: 200,
      });
      assert.strictEqual(res.ok, true);
      assert.strictEqual(res.reason, 'alive');
      assert.ok(res.restarts >= 1, `restarts=${res.restarts}`);
      assert.ok(daemon, 'real daemon must have been started');
      await daemon.close();
    });

  const total = passed + failures.length;
  console.log(`\n${passed}/${total} passed`);
  if (failures.length > 0) {
    for (const f of failures) {
      console.error(`\nFAILED: ${f.name}`);
      console.error(f.err && f.err.stack ? f.err.stack : f.err);
    }
    process.exitCode = 1;
  }
  fs.rmSync(tmpRoot, {recursive: true, force: true});
}

main().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
