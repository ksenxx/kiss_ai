// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the "KISS Sorcar Server is starting ..."
// hang after clicking the Update button.
//
// Bug being locked in
// -------------------
// Clicking the chat panel's "Update" button spawns ``install.sh``
// detached from the kiss-web daemon (see ``_handle_run_update`` /
// ``_spawn_update_script`` in ``web_server.py``).  ``install.sh``:
//
//   1. ``lsof -ti :8787 | xargs kill`` — sends SIGTERM to the old
//      daemon and waits up to ~3 s for the port to free up.
//   2. ``rm -f "$HOME/.kiss/sorcar.sock"`` — pre-emptively removes
//      the stale UDS socket so the extension's reconnect does not
//      hit a dead socket of the just-killed daemon.
//   3. Writes ``~/.kiss/.extension-updated`` so the extension's
//      ``fs.watchFile`` poll fires ``workbench.action.reloadWindow``.
//
// On macOS the LaunchAgent's ``KeepAlive=true`` (and on Linux the
// systemd unit's ``Restart=always``) respawns kiss-web automatically.
// Crucially, the respawn timing is NOT synchronised with install.sh:
// in production we have repeatedly observed launchd respawning a
// fresh daemon DURING step 1's 3 s wait loop — that daemon's
// ``_setup_server`` ``unlink`` + ``asyncio.start_unix_server`` creates
// a new ``~/.kiss/sorcar.sock`` file before install.sh reaches step
// 2's ``rm -f``.  Step 2 then deletes the new daemon's UDS file
// instead of the old one's.
//
// The kernel-level listening socket survives the file removal (the
// daemon's open file descriptor is independent of the directory
// entry), so the daemon happily continues to accept on port 8787 —
// but every NEW ``connect(AF_UNIX, ~/.kiss/sorcar.sock)`` attempt
// from the extension's ``AgentClient`` fails with ``ENOENT`` until
// the daemon is killed again and respawned.
//
// In the VS Code window this manifests as the chat webview hanging
// forever on the "KISS Sorcar Server is starting ..." overlay —
// ``AgentClient``'s 500 ms reconnect loop keeps tripping ``ENOENT``
// and the ``daemonStatus: {connected: true}`` event that would hide
// the overlay is never broadcast.
//
// The "fix" the user falls back on (restart VS Code) used to be
// ineffective too, because ``ensureDependencies()``'s
// ``restartKissWebDaemon`` short-circuited via ``decideRestart``'s
// generic ``alive-uncertain`` skip — health=alive (port 8787 is up)
// combined with ``activeTasks={ok:false, reason:'sock-missing'}``
// was misclassified as "do not touch this daemon" instead of "this
// daemon is unreachable; force a fresh restart so it re-creates the
// UDS file".  This test exercises the corrected decision pipeline
// against a real TCP listener and a real UDS-socket-deleted-out-from-
// under-it scenario, and asserts both:
//
//   1. ``decideRestart`` now returns ``{skip:false,
//      reason:/unreachable-uds/}`` — i.e. the extension's
//      ``restartKissWebDaemon`` will recycle the broken daemon.
//   2. The original task-3192 protection still holds for the
//      ``timeout`` failure mode (a busy daemon must NEVER be
//      SIGTERMed).
//
// The test uses the REAL compiled ``daemonHealth.js`` — no mocks of
// project code — and only stubs the OS for the TCP-port allocation
// and UDS-socket-file lifecycle.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/updateButtonHangAfterRestart.test.js

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

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-update-hang-'));

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

/** Spin up a TCP server on a kernel-chosen port (simulates kiss-web's WSS). */
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
 * Spin up a UDS server (simulates kiss-web's ``~/.kiss/sorcar.sock``).
 * The returned ``deleteSocketFile`` helper removes the socket file from
 * disk WITHOUT closing the in-memory listening socket — i.e. the exact
 * state install.sh leaves the freshly-respawned daemon in.
 */
function listenUds(sockPath) {
  return new Promise((resolve, reject) => {
    try {
      if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
    } catch {/* ignore */}
    const server = net.createServer(sock => {
      sock.on('data', () => {/* accept and reply on demand */});
      sock.on('error', () => {/* ignore */});
    });
    server.once('error', reject);
    server.listen(sockPath, () => {
      resolve({
        deleteSocketFile: () => {
          // Mimic ``rm -f $HOME/.kiss/sorcar.sock``: detach the file
          // path from the listening socket without closing it.
          try {
            fs.unlinkSync(sockPath);
          } catch {/* ignore — may already be gone */}
        },
        close: () => new Promise(res => server.close(() => {
          try {
            if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
          } catch {/* ignore */}
          res();
        })),
      });
    });
  });
}

(async () => {

  // ---------------------------------------------------------------
  // Scenario 1 — REPRODUCE the bug exactly as the user observes it.
  //
  // 1. A kiss-web daemon is fully up: TCP listener on port 8787 +
  //    UDS socket file at ~/.kiss/sorcar.sock.
  // 2. install.sh's ``rm -f ~/.kiss/sorcar.sock`` runs AFTER launchd
  //    has respawned the daemon: the on-disk file is deleted but
  //    the daemon's open listening socket survives.
  // 3. ``restartKissWebDaemon`` re-runs (on the next VS Code
  //    activation triggered by the extension reload).  The TCP
  //    probe says ``alive`` and ``daemonHasActiveTasks`` reports
  //    ``{ok:false, reason:'sock-missing'}``.
  //
  // Before the fix: ``decideRestart`` returns ``{skip: true,
  //   reason: 'alive-uncertain'}`` — the daemon is NOT restarted,
  //   the webview hangs on the loading overlay, and the user has
  //   to restart VS Code (which under the OLD logic ALSO hit the
  //   same skip and did not help).
  //
  // After the fix: ``decideRestart`` returns ``{skip: false,
  //   reason: 'unreachable-uds (alive but socket file missing)'}``
  //   — ``restartKissWebDaemon`` recycles the daemon, the new
  //   daemon's ``_setup_server`` re-binds the UDS file, and
  //   ``AgentClient`` reconnects → ``daemonStatus:true`` →
  //   overlay hidden.
  // ---------------------------------------------------------------

  await test('Update-button hang — TCP alive + UDS file deleted ⇒ restart forced', async () => {
    const {port, close: closeTcp} = await listenTcp();
    const sockPath = path.join(tmpRoot, 'update-hang.sock');
    const uds = await listenUds(sockPath);
    try {
      // Sanity: with the socket file present a normal probe round-trip
      // would not even need an ``activeTasksResponse`` for the bug to
      // bite — we just need ``daemonHasActiveTasks`` to fail in the
      // specific ``sock-missing`` way once install.sh nukes the file.
      assert.ok(fs.existsSync(sockPath),
        'precondition: UDS socket file present before the simulated rm');

      // === Simulate install.sh's ``rm -f ~/.kiss/sorcar.sock`` ===
      uds.deleteSocketFile();
      assert.ok(!fs.existsSync(sockPath),
        'after the simulated rm the UDS file must be gone');

      // === Run the exact pipeline restartKissWebDaemon executes ===
      const health = await probeDaemonHealth(port, 1500);
      const activeTasks = await daemonHasActiveTasks(sockPath, 500);
      // ``health`` should remain ``alive`` — the kernel listening
      // socket survives ``unlink`` of its directory entry.
      assert.strictEqual(health, 'alive',
        `daemon's TCP listener should survive UDS file removal; got: ${health}`);
      assert.deepStrictEqual(activeTasks, {ok: false, reason: 'sock-missing'},
        `UDS probe should report sock-missing once install.sh has rm-ed ` +
        `the socket file; got: ${JSON.stringify(activeTasks)}`);

      // === The fix: decideRestart must FORCE a restart here.    ===
      // === (Pre-fix: decision.skip === true, which is the hang.)===
      const decision = decideRestart({
        // ``install.sh`` only re-installs the VSIX — the bundled
        // kiss source tree is byte-identical to the last fingerprint
        // recorded by the previous daemon, so the fingerprint MATCHES.
        // This is the worst case for the OLD code: even a code change
        // could not unstick it.
        fingerprintMatches: true,
        health,
        activeTasks,
      });
      assert.strictEqual(decision.skip, false,
        `restartKissWebDaemon must recycle a daemon whose UDS file is ` +
        `missing — otherwise the webview stays on "KISS Sorcar Server ` +
        `is starting ..." forever.  Got: ${JSON.stringify(decision)}`);
      assert.ok(/unreachable-uds/.test(decision.reason),
        `restart reason must flag the unreachable UDS so install.sh / ` +
        `restartKissWebDaemon logs are diagnosable; got: ${decision.reason}`);
    } finally {
      await uds.close();
      await closeTcp();
    }
  });

  // ---------------------------------------------------------------
  // Scenario 2 — REGRESSION GUARD for task 3192.
  //
  // A busy daemon under load can miss the 1500 ms UDS round-trip
  // deadline.  This MUST still defer the restart — SIGTERMing a
  // mid-task daemon was the original bug the ``alive-uncertain``
  // skip was introduced to fix.  The Update-button fix narrowly
  // carves out ``sock-missing`` only; ``timeout`` (and every other
  // UDS failure mode) keep the original protective behaviour.
  // ---------------------------------------------------------------

  await test('Task-3192 protection — TCP alive + UDS timeout ⇒ restart STILL deferred', async () => {
    const {port, close: closeTcp} = await listenTcp();
    const sockPath = path.join(tmpRoot, 'task3192.sock');

    // A UDS server that accepts the connection but NEVER replies.
    const server = await new Promise((resolve, reject) => {
      try { if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath); }
      catch {/* ignore */}
      const srv = net.createServer(s => {
        // Accept and idle — no reply to ``activeTasksQuery`` so the
        // probe runs out the clock and returns ``timeout``.
        s.on('data', () => {/* ignore */});
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
      const health = await probeDaemonHealth(port, 1500);
      const activeTasks = await daemonHasActiveTasks(sockPath, 150);
      assert.strictEqual(health, 'alive');
      assert.strictEqual(activeTasks.ok, false);
      assert.strictEqual(activeTasks.reason, 'timeout',
        `the timeout case must reach decideRestart as reason='timeout'; ` +
        `got: ${JSON.stringify(activeTasks)}`);

      const decision = decideRestart({
        fingerprintMatches: false, // extension auto-update changed every .py mtime
        health,
        activeTasks,
      });
      assert.strictEqual(decision.skip, true,
        `a daemon that is alive but cannot answer activeTasksQuery in ` +
        `time must NOT be SIGTERMed — that was the task-3192 regression. ` +
        `Got: ${JSON.stringify(decision)}`);
      assert.ok(/alive-uncertain/.test(decision.reason),
        `skip reason should still flag the uncertainty; got: ${decision.reason}`);
    } finally {
      await server.close();
      await closeTcp();
    }
  });

  // ---------------------------------------------------------------
  // Scenario 3 — Belt-and-suspenders: the new decision path must
  // also win when the source-tree fingerprint matches (which is the
  // realistic post-Update state — install.sh's only mutation is to
  // reinstall the SAME VSIX, so every bundled ``.py`` keeps its
  // pre-install content hash).  Without this, the
  // ``healthy-unchanged`` skip would still trap the user.
  // ---------------------------------------------------------------

  await test('Unreachable-UDS restart wins over healthy-unchanged skip', () => {
    const decision = decideRestart({
      fingerprintMatches: true,
      health: 'alive',
      activeTasks: {ok: false, reason: 'sock-missing'},
    });
    assert.strictEqual(decision.skip, false,
      `unreachable-uds must beat healthy-unchanged or the user remains ` +
      `stranded on the loading overlay forever; got: ${JSON.stringify(decision)}`);
  });

  // ---------------------------------------------------------------
  // Scenario 4 — Even an alive daemon that claims to have active
  // tasks must NOT be killed.  Defensive precedence test: a future
  // refactor that accidentally re-orders the decision table must
  // not let unreachable-uds override an explicit active-tasks reply.
  // ---------------------------------------------------------------

  await test('Active-tasks reply still wins over sock-missing (precedence pin)', () => {
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'alive',
      activeTasks: {ok: true, count: 2, tabs: ['a(task=1)', 'b(task=2)']},
    });
    assert.strictEqual(decision.skip, true);
    assert.strictEqual(decision.reason, 'active-tasks');
  });

  // ---------------------------------------------------------------
  // Scenario 5 — TOCTOU race: ``install.sh``'s ``rm -f`` lands AFTER
  // ``daemonHasActiveTasks`` has already checked ``fs.existsSync`` but
  // BEFORE its ``net.createConnection`` succeeds.  In that window the
  // kernel surfaces the deletion as an ``ENOENT`` on the connect
  // attempt.  Before the TOCTOU normalisation in ``daemonHealth.js``,
  // this reached ``decideRestart`` as ``reason:'error:ENOENT'`` —
  // NOT ``'sock-missing'`` — and hit the generic ``alive-uncertain``
  // skip, leaving the user stranded all over again.  The probe must
  // normalise ENOENT (and the closely-related ``ECONNREFUSED`` on a
  // stale leftover file) to ``'sock-missing'``.
  // ---------------------------------------------------------------

  await test('TOCTOU race — rm -f lands AFTER existsSync but BEFORE connect ⇒ reason normalised to sock-missing', async () => {
    const sockPath = path.join(tmpRoot, 'toctou.sock');
    // Pre-create a regular file at the socket path so ``existsSync``
    // returns true at probe entry, then unlink it before the connect
    // tries to resolve the path.  Using a regular file (rather than
    // a UDS server we delete the file out from under) is the simplest
    // way to deterministically force the connect to ENOENT regardless
    // of OS-level timing — the kernel's ``connect(AF_UNIX, ...)`` over
    // a path with no bound listener returns ENOENT once the directory
    // entry is removed.  The point of the test is the error path, not
    // the kernel mechanics: we need ``daemonHasActiveTasks`` to take
    // its ``sock.on('error', ENOENT)`` branch.
    fs.writeFileSync(sockPath, '');
    // Race: schedule the unlink to fire after existsSync but before
    // connect can resolve.  setImmediate runs after I/O — fast enough
    // to land inside daemonHasActiveTasks' tick.
    setImmediate(() => {
      try { fs.unlinkSync(sockPath); } catch {/* ignore */}
    });
    const res = await daemonHasActiveTasks(sockPath, 500);
    assert.strictEqual(res.ok, false,
      `TOCTOU probe must fail; got: ${JSON.stringify(res)}`);
    assert.strictEqual(res.reason, 'sock-missing',
      `the error path must normalise to 'sock-missing' so decideRestart's ` +
      `unreachable-uds branch fires; got reason='${res.reason}'`);

    // End-to-end: the normalised result must drive a restart.
    const decision = decideRestart({
      fingerprintMatches: true,
      health: 'alive',
      activeTasks: res,
    });
    assert.strictEqual(decision.skip, false,
      `TOCTOU race must STILL force a restart end-to-end; ` +
      `got: ${JSON.stringify(decision)}`);
    assert.ok(/unreachable-uds/.test(decision.reason),
      `decision reason must flag unreachable-uds; got: ${decision.reason}`);
  });

})()
  .then(() => {
    try {
      fs.rmSync(tmpRoot, {recursive: true, force: true});
    } catch {/* ignore */}
    if (failures.length > 0) {
      console.error(`\n${failures.length} FAIL(s):`);
      for (const f of failures) {
        console.error(`  - ${f.name}`);
        if (f.err && f.err.stack) console.error(`    ${f.err.stack}`);
      }
      process.exit(1);
    }
    console.log(`\nAll ${passed} tests passed`);
  })
  .catch(err => {
    console.error('runner error:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
