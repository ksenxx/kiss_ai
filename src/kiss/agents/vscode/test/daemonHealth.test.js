// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the kiss-web daemon health probes used by the
// extension's ``restartKissWebDaemon`` to decide whether to SIGTERM the
// running daemon on (re-)activation.
//
// Reproduces the bug from the report:
//
//   - The OLD guard used ``execSync('lsof -i :8787 -t', {timeout: 2000})``
//     wrapped in a bare ``catch { return false; }``.  Under heavy load
//     ``lsof`` could time out, the catch returned ``daemonAlive=false``,
//     and the installer SIGTERMed a perfectly healthy daemon that was
//     mid-task — aborting a multi-hour agent run.
//
//   - The restart logic had no awareness of the daemon's in-flight
//     ``active_tasks`` (clearly visible in the SIGTERM log line that
//     printed them).  Even a legitimate code change should not
//     interrupt a running task.
//
// The tests below exercise the real ``daemonHealth.js`` helpers (no
// mocks, no fakes) against real TCP and Unix-domain socket servers
// spun up in-process, and assert the ``decideRestart`` decision table
// for every interesting combination.
//
// Run directly with ``node`` — no VS Code extension host required:
//
//     node src/kiss/agents/vscode/test/daemonHealth.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

const {
  probeDaemonHealth,
  daemonHasActiveTasks,
  decideRestart,
} = require('../src/daemonHealth');

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

/**
 * Spin up a TCP server bound to localhost on a kernel-chosen port.
 * Returns ``{port, close}``.  ``close()`` is idempotent.
 */
function listenTcp() {
  return new Promise((resolve, reject) => {
    const server = net.createServer(() => {/* accept and idle */});
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({
        port: typeof addr === 'object' && addr ? addr.port : 0,
        close: () => new Promise(res => server.close(() => res())),
      });
    });
  });
}

/**
 * Allocate a free TCP port (used so ``probeDaemonHealth`` can be
 * tested against a port that is GUARANTEED to refuse connections).
 */
function pickClosedPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      server.close(() => resolve(port));
    });
  });
}

/**
 * Spin up a Unix-domain stream server that speaks the real kiss-web
 * UDS protocol for ``activeTasksQuery``.  ``response`` is the JSON
 * object to send back; ``null`` makes the server hang silently
 * (timeout case); ``'gibberish'`` makes it write invalid JSON.
 */
function listenUds(sockPath, response) {
  return new Promise((resolve, reject) => {
    try {
      if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
    } catch {
      /* ignore */
    }
    const server = net.createServer(sock => {
      let buf = '';
      sock.setEncoding('utf-8');
      sock.on('data', chunk => {
        buf += chunk;
        const nl = buf.indexOf('\n');
        if (nl < 0) return;
        const line = buf.slice(0, nl);
        let cmd;
        try {
          cmd = JSON.parse(line);
        } catch {
          return;
        }
        if (cmd && cmd.type === 'activeTasksQuery') {
          if (response === null) return; // hang to trigger client timeout
          if (response === 'gibberish') {
            sock.write('not json\n');
            return;
          }
          sock.write(JSON.stringify(response) + '\n');
        }
      });
      sock.on('error', () => {/* ignore */});
    });
    server.once('error', reject);
    server.listen(sockPath, () => {
      resolve({
        close: () => new Promise(res => {
          server.close(() => {
            try {
              if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
            } catch {/* ignore */}
            res();
          });
        }),
      });
    });
  });
}

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-daemonhealth-'));

(async () => {
  // -------------------------------------------------------------------
  // probeDaemonHealth
  // -------------------------------------------------------------------

  await test('probeDaemonHealth: returns "alive" when a TCP server accepts the connection', async () => {
    const {port, close} = await listenTcp();
    try {
      const health = await probeDaemonHealth(port, 1500);
      assert.strictEqual(health, 'alive');
    } finally {
      await close();
    }
  });

  await test('probeDaemonHealth: returns "dead" when nothing listens on the port (ECONNREFUSED)', async () => {
    const port = await pickClosedPort();
    const health = await probeDaemonHealth(port, 1500);
    assert.strictEqual(health, 'dead');
  });

  await test('probeDaemonHealth: returns a definite value within the timeout (no hang)', async () => {
    // Regression: the OLD lsof-based probe could hang the whole
    // restart flow for the full ``timeout`` AND mis-report the result.
    // The new probe must resolve well under the timeout when there is
    // no listener — we give it 1500 ms and assert it returns under
    // 750 ms.
    const port = await pickClosedPort();
    const t0 = Date.now();
    await probeDaemonHealth(port, 1500);
    assert.ok(Date.now() - t0 < 750,
      `probe took ${Date.now() - t0}ms — too slow`);
  });

  // -------------------------------------------------------------------
  // daemonHasActiveTasks
  // -------------------------------------------------------------------

  await test('daemonHasActiveTasks: returns {ok:false, reason:"sock-missing"} when the socket file does not exist', async () => {
    const sockPath = path.join(tmpRoot, 'missing.sock');
    const res = await daemonHasActiveTasks(sockPath, 500);
    assert.strictEqual(res.ok, false);
    assert.strictEqual(res.reason, 'sock-missing');
  });

  await test('daemonHasActiveTasks: parses count=2 and the tabs list from a real UDS server', async () => {
    const sockPath = path.join(tmpRoot, 'busy.sock');
    const tabs = [
      'ad4ecb65-2878-4c2c-9736-3bb9be18814a(task=74)',
      'beadbabe-1111-2222-3333-444455556666(task=99)',
    ];
    const server = await listenUds(sockPath, {
      type: 'activeTasksResponse', count: 2, tabs,
    });
    try {
      const res = await daemonHasActiveTasks(sockPath, 1500);
      assert.strictEqual(res.ok, true);
      assert.strictEqual(res.count, 2);
      assert.deepStrictEqual(res.tabs, tabs);
    } finally {
      await server.close();
    }
  });

  await test('daemonHasActiveTasks: parses count=0 correctly (idle daemon)', async () => {
    const sockPath = path.join(tmpRoot, 'idle.sock');
    const server = await listenUds(sockPath, {
      type: 'activeTasksResponse', count: 0, tabs: [],
    });
    try {
      const res = await daemonHasActiveTasks(sockPath, 1500);
      assert.strictEqual(res.ok, true);
      assert.strictEqual(res.count, 0);
      assert.deepStrictEqual(res.tabs, []);
    } finally {
      await server.close();
    }
  });

  await test('daemonHasActiveTasks: returns {ok:false, reason:"timeout"} when the server never replies', async () => {
    const sockPath = path.join(tmpRoot, 'silent.sock');
    const server = await listenUds(sockPath, null);
    try {
      const res = await daemonHasActiveTasks(sockPath, 200);
      assert.strictEqual(res.ok, false);
      assert.strictEqual(res.reason, 'timeout');
    } finally {
      await server.close();
    }
  });

  await test('daemonHasActiveTasks: skips non-JSON broadcast noise and times out instead of mis-reporting', async () => {
    // Non-JSON noise can appear on the broadcast channel (future
    // protocol extensions, partial flushes, etc.).  Misclassifying it
    // as a fatal ``parse-failed`` would abort install.sh on a
    // perfectly healthy daemon — the same family of bug as the
    // ``Unknown command: activeTasksQuery`` regression.  The probe
    // must drain the bad line and keep waiting.
    const sockPath = path.join(tmpRoot, 'gibberish.sock');
    const server = await listenUds(sockPath, 'gibberish');
    try {
      const res = await daemonHasActiveTasks(sockPath, 200);
      assert.strictEqual(res.ok, false);
      assert.strictEqual(res.reason, 'timeout');
    } finally {
      await server.close();
    }
  });

  await test('daemonHasActiveTasks: skips broadcast lines that are not the awaited response (times out instead of mis-reporting)', async () => {
    // RemoteAccessServer._uds_handler registers every client as a
    // broadcast destination, so unrelated event lines can arrive
    // before the response.  We must NOT treat the first stray line
    // as a fatal "unexpected-type" — that mis-classification caused
    // the OLD-daemon install.sh abort.  Instead we skip and keep
    // waiting; if nothing else ever comes, the outer timeout fires.
    const sockPath = path.join(tmpRoot, 'wrong.sock');
    const server = await listenUds(sockPath, {type: 'something-else'});
    try {
      const res = await daemonHasActiveTasks(sockPath, 200);
      assert.strictEqual(res.ok, false);
      assert.strictEqual(res.reason, 'timeout');
    } finally {
      await server.close();
    }
  });

  await test('daemonHasActiveTasks: tolerates a stray broadcast line that precedes the real activeTasksResponse', async () => {
    // RemoteAccessServer broadcasts unrelated events to every UDS
    // client.  A race can put one of those events on the wire BEFORE
    // our response; the probe must drain it and keep waiting rather
    // than abort with "unexpected-type" — that misclassification was
    // the root cause of the install.sh failure even when the daemon
    // DID know about activeTasksQuery.
    const sockPath = path.join(tmpRoot, 'prefixed.sock');
    const server = await new Promise((resolve, reject) => {
      try {
        if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
      } catch {/* ignore */}
      const srv = net.createServer(s => {
        let inBuf = '';
        s.setEncoding('utf-8');
        s.on('data', c => {
          inBuf += c;
          const nl = inBuf.indexOf('\n');
          if (nl < 0) return;
          const line = inBuf.slice(0, nl);
          let cmd;
          try { cmd = JSON.parse(line); } catch { return; }
          if (cmd && cmd.type === 'activeTasksQuery') {
            // Stray broadcast first…
            s.write(JSON.stringify({type: 'event', name: 'noise'}) + '\n');
            // …then the real response.
            s.write(JSON.stringify({
              type: 'activeTasksResponse',
              count: 0,
              tabs: [],
            }) + '\n');
          }
        });
        s.on('error', () => {/* ignore */});
      });
      srv.once('error', reject);
      srv.listen(sockPath, () => resolve({
        close: () => new Promise(res => srv.close(() => {
          try { if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath); }
          catch {/* ignore */}
          res();
        })),
      }));
    });
    try {
      const res = await daemonHasActiveTasks(sockPath, 1500);
      assert.strictEqual(res.ok, true,
        `expected to skip stray broadcast and parse the real ` +
        `response; got: ${JSON.stringify(res)}`);
      assert.strictEqual(res.count, 0);
    } finally {
      await server.close();
    }
  });

  await test('daemonHasActiveTasks: treats an OLD-daemon "Unknown command: activeTasksQuery" error as count:0 (install.sh abort regression)', async () => {
    // Reproduces the install.sh failure in the user report:
    //
    //   kiss-web UDS probe at ~/.kiss/sorcar.sock returned unexpected
    //   message {'type': 'error', 'text': 'Unknown command:
    //   activeTasksQuery'}; refusing to kill.
    //
    // A daemon that predates the activeTasksQuery handler cannot have
    // in-flight-task accounting we need to defer to, so the only
    // remaining safety question is "can we kill it?".  The answer
    // must be "yes" or the user is stuck rerunning install.sh with
    // KISS_FORCE_RESTART=1 forever just to upgrade past the bug.
    const sockPath = path.join(tmpRoot, 'old-daemon.sock');
    const server = await listenUds(sockPath, {
      type: 'error',
      text: 'Unknown command: activeTasksQuery',
    });
    try {
      const res = await daemonHasActiveTasks(sockPath, 1500);
      assert.strictEqual(res.ok, true,
        `expected ok:true on old-daemon error; got: ${JSON.stringify(res)}`);
      assert.strictEqual(res.count, 0);
      assert.deepStrictEqual(res.tabs, []);
    } finally {
      await server.close();
    }
  });

  // -------------------------------------------------------------------
  // decideRestart — the actual guard decision used by
  // restartKissWebDaemon.  These cases LOCK IN the bug fix:
  //
  //   * Active-task: skip unconditionally (the OLD code didn't).
  //   * Unknown health + fingerprint matches: skip (the OLD code
  //     conflated "unknown" with "dead" and restarted).
  // -------------------------------------------------------------------

  await test('decideRestart: skips restart when the daemon reports active tasks (bug-fix scenario)', () => {
    const decision = decideRestart({
      fingerprintMatches: false, // even on a code change …
      health: 'alive',
      activeTasks: {ok: true, count: 1, tabs: ['x(task=74)']},
    });
    assert.strictEqual(decision.skip, true);
    assert.strictEqual(decision.reason, 'active-tasks');
  });

  await test('decideRestart: skips restart when health probe is UNKNOWN and fingerprint matches (the actual lsof-timeout regression)', () => {
    // This is the EXACT scenario from the bug report: lsof timed out
    // under load, the fingerprint was unchanged.  The OLD code
    // restarted; the new code MUST defer.
    const decision = decideRestart({
      fingerprintMatches: true,
      health: 'unknown',
      activeTasks: {ok: false, reason: 'not-probed'},
    });
    assert.strictEqual(decision.skip, true);
    assert.ok(decision.reason.startsWith('healthy-unchanged'));
  });

  await test('decideRestart: skips restart when health=alive and fingerprint matches (the happy path)', () => {
    const decision = decideRestart({
      fingerprintMatches: true,
      health: 'alive',
      activeTasks: {ok: true, count: 0, tabs: []},
    });
    assert.strictEqual(decision.skip, true);
  });

  await test('decideRestart: RESTARTS when fingerprint differs AND no active tasks AND health=alive', () => {
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'alive',
      activeTasks: {ok: true, count: 0, tabs: []},
    });
    assert.strictEqual(decision.skip, false);
  });

  await test('decideRestart: skips restart when daemon is ALIVE and activeTasks probe failed (bug: task 3192 SIGTERM regression)', () => {
    // Reproduces the bug behind the "Task interrupted by server
    // restart/shutdown" failure of task_history row 3192.  The kiss-
    // web daemon (pid 68477) was happily running a multi-minute
    // ``uv run check --full`` step on behalf of an agent task when
    // the extension's ``restartKissWebDaemon`` SIGTERMed it.  At the
    // moment of the SIGTERM the daemon's own log line reported
    // ``active_tasks=[a3b4ec24-...(task=3191)] rss=147.7MB`` —
    // proof that the daemon WAS running a task and that the
    // installer SHOULD have deferred.
    //
    // The pre-fix decision table restarted whenever
    // ``fingerprintMatches=false`` regardless of the
    // ``activeTasks.ok`` flag.  On an extension auto-update every
    // ``.py`` mtime in the bundled ``kiss_project`` changes so the
    // fingerprint never matches — and an in-flight task can make the
    // 1500 ms UDS round-trip miss its deadline (``ok:false,
    // reason:"timeout"``).  Those two failures combined SIGTERMed a
    // perfectly healthy, BUSY daemon.
    //
    // The fix: an ALIVE daemon whose active-tasks status cannot be
    // confirmed MUST NOT be SIGTERMed.  The fingerprint change is
    // applied lazily on the next activation that can prove the
    // daemon is idle (or has died on its own).
    const decision = decideRestart({
      fingerprintMatches: false, // extension auto-updated
      health: 'alive', // TCP probe accepted the connection
      activeTasks: {ok: false, reason: 'timeout'}, // UDS missed deadline
    });
    assert.strictEqual(decision.skip, true,
      `expected to skip restart of an alive daemon with unknown ` +
      `active-tasks status; got: ${JSON.stringify(decision)}`);
    assert.ok(/alive-uncertain/.test(decision.reason),
      `expected the skip reason to flag the uncertainty; got: ${decision.reason}`);
  });

  await test('decideRestart: RESTARTS when daemon is ALIVE but UDS socket file is missing (Update-button hang fix)', () => {
    // BUG FIX: After the Update button is clicked, ``install.sh``
    // kills the kiss-web daemon by port (``lsof -ti :8787 | xargs
    // kill``), waits, and removes ``~/.kiss/sorcar.sock``.  On macOS
    // launchd's ``KeepAlive`` (or systemd's ``Restart=always``) may
    // respawn the daemon BEFORE install.sh runs the ``rm -f``, so
    // the freshly-respawned daemon's UDS socket FILE is deleted out
    // from under it.  The daemon's open listening socket is in-
    // kernel and still healthy on port 8787, but new clients trying
    // to connect by path (``~/.kiss/sorcar.sock``) get ENOENT
    // forever.  The webview hangs on
    // "KISS Sorcar Server is starting ..." until the user restarts
    // VS Code — and even that did NOT help under the OLD decision
    // table, which lumped ``sock-missing`` together with the
    // ``timeout`` / generic UDS failures and SKIPPED the restart on
    // the assumption that the daemon was a "transient startup
    // window" away from binding.  In the install.sh race the
    // window is permanent — the daemon never re-binds on its own —
    // so the only way to recover is to force a restart, which
    // respawns a daemon whose ``_setup_server`` ``unlink`` + ``bind``
    // re-creates the UDS file.
    //
    // The fix carves ``sock-missing`` out from the
    // ``alive-uncertain`` skip and RESTARTS: this is safe because a
    // daemon with no UDS file cannot answer ``activeTasksQuery``
    // (the explicit ``active-tasks`` skip above always wins), and
    // the original "transient" window is microseconds wide, so a
    // false-positive restart there is a benign no-op.
    const decision = decideRestart({
      fingerprintMatches: true,  // the install.sh race leaves the bundle untouched
      health: 'alive',
      activeTasks: {ok: false, reason: 'sock-missing'},
    });
    assert.strictEqual(decision.skip, false,
      `expected restart when daemon is alive but UDS socket file is missing; ` +
      `got: ${JSON.stringify(decision)}`);
    assert.ok(/unreachable-uds/.test(decision.reason),
      `expected the restart reason to flag the unreachable UDS; got: ${decision.reason}`);
  });

  await test('decideRestart: still SKIPS restart when daemon is ALIVE and active-tasks probe TIMED OUT (task 3192 protection preserved)', () => {
    // Regression guard: the BUG FIX above ONLY carves out
    // ``sock-missing``.  The ``timeout`` failure mode is still
    // protective — a busy daemon under load can legitimately miss
    // the 1500 ms UDS deadline, and SIGTERMing it would abort the
    // in-flight agent task (the original task 3192 regression).
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'alive',
      activeTasks: {ok: false, reason: 'timeout'},
    });
    assert.strictEqual(decision.skip, true,
      `timeout must still defer to protect mid-task daemons; ` +
      `got: ${JSON.stringify(decision)}`);
    assert.ok(/alive-uncertain/.test(decision.reason));
  });

  await test('decideRestart: unreachable-uds restart wins over fingerprintMatches', () => {
    // A daemon that is alive on TCP but has no UDS socket file must
    // be restarted EVEN IF its source fingerprint matches the last
    // known good build.  Without this, ``restartKissWebDaemon``
    // would fall through to the ``healthy-unchanged`` skip and the
    // webview would stay on the loading overlay forever.
    const decision = decideRestart({
      fingerprintMatches: true,
      health: 'alive',
      activeTasks: {ok: false, reason: 'sock-missing'},
    });
    assert.strictEqual(decision.skip, false);
    assert.ok(/unreachable-uds/.test(decision.reason));
  });

  await test('decideRestart: explicit active-tasks reply still wins over sock-missing (defensive ordering)', () => {
    // Mostly theoretical: a daemon with no UDS file cannot return
    // ``ok: true``.  But pin the precedence anyway so a future
    // refactor that defaults ``activeTasks`` to ``ok:true`` cannot
    // accidentally restart a busy daemon.
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'alive',
      activeTasks: {ok: true, count: 1, tabs: ['x(task=1)']},
    });
    assert.strictEqual(decision.skip, true);
    assert.strictEqual(decision.reason, 'active-tasks');
  });

  await test('decideRestart: still RESTARTS when the daemon is fully unreachable (health=unknown + activeTasks failed + fingerprint mismatched)', () => {
    // Counter-test: the safety net must NOT block recovery when the
    // daemon really has gone away.  ``health='unknown'`` (e.g.
    // host overloaded, no ECONNREFUSED but no accept either) combined
    // with an active-tasks probe that could not connect and a code
    // change must still allow the installer to recycle the daemon.
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'unknown',
      activeTasks: {ok: false, reason: 'sock-missing'},
    });
    assert.strictEqual(decision.skip, false);
  });

  await test('decideRestart: RESTARTS when daemon is confirmed dead even if fingerprint matches', () => {
    const decision = decideRestart({
      fingerprintMatches: true,
      health: 'dead',
      activeTasks: {ok: false, reason: 'sock-missing'},
    });
    assert.strictEqual(decision.skip, false);
  });

  await test('decideRestart: active-tasks takes precedence over a dead probe result', () => {
    // Sanity: even if the probe somehow contradicts itself, an
    // explicit active-tasks signal must win — we never want to kill
    // a daemon that admits to running a task.
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'dead',
      activeTasks: {ok: true, count: 5, tabs: ['a', 'b', 'c', 'd', 'e']},
    });
    assert.strictEqual(decision.skip, true);
    assert.strictEqual(decision.reason, 'active-tasks');
  });

  // -------------------------------------------------------------------
  // End-to-end: probe a TCP server, query the UDS server, drive the
  // decision — i.e. the exact pipeline ``restartKissWebDaemon`` runs.
  // -------------------------------------------------------------------

  await test('end-to-end: alive TCP + UDS reporting 1 active task → skip("active-tasks")', async () => {
    const {port, close: closeTcp} = await listenTcp();
    const sockPath = path.join(tmpRoot, 'e2e.sock');
    const tabs = ['ad4ecb65-2878-4c2c-9736-3bb9be18814a(task=74)'];
    const uds = await listenUds(sockPath, {
      type: 'activeTasksResponse', count: 1, tabs,
    });
    try {
      const health = await probeDaemonHealth(port, 1500);
      const activeTasks = await daemonHasActiveTasks(sockPath, 1500);
      const decision = decideRestart({
        fingerprintMatches: false, // simulate a code change
        health,
        activeTasks,
      });
      assert.strictEqual(health, 'alive');
      assert.strictEqual(activeTasks.ok, true);
      assert.strictEqual(activeTasks.count, 1);
      assert.strictEqual(decision.skip, true);
      assert.strictEqual(decision.reason, 'active-tasks');
    } finally {
      await closeTcp();
      await uds.close();
    }
  });
})()
  .then(() => {
    try {
      fs.rmSync(tmpRoot, {recursive: true, force: true});
    } catch {/* ignore */}
    console.log(`\n${passed} passed, ${failures.length} failed`);
    if (failures.length > 0) {
      for (const f of failures) {
        console.error(`\n${f.name}:\n`, f.err);
      }
      process.exit(1);
    }
  })
  .catch(err => {
    console.error('unexpected runner error:', err);
    process.exit(1);
  });
