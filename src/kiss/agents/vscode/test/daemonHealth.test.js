// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

function listenTcp() {
  return new Promise((resolve, reject) => {
    const server = net.createServer(() => { });
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

function listenUds(sockPath, response) {
  return new Promise((resolve, reject) => {
    try {
      if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
    } catch {
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
          if (response === null) return;
          if (response === 'gibberish') {
            sock.write('not json\n');
            return;
          }
          sock.write(JSON.stringify(response) + '\n');
        }
      });
      sock.on('error', () => { });
    });
    server.once('error', reject);
    server.listen(sockPath, () => {
      resolve({
        close: () => new Promise(res => {
          server.close(() => {
            try {
              if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
            } catch { }
            res();
          });
        }),
      });
    });
  });
}

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-daemonhealth-'));

(async () => {

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
    const port = await pickClosedPort();
    const t0 = Date.now();
    await probeDaemonHealth(port, 1500);
    assert.ok(Date.now() - t0 < 750,
      `probe took ${Date.now() - t0}ms — too slow`);
  });

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
    const sockPath = path.join(tmpRoot, 'prefixed.sock');
    const server = await new Promise((resolve, reject) => {
      try {
        if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
      } catch { }
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
            s.write(JSON.stringify({type: 'event', name: 'noise'}) + '\n');
            s.write(JSON.stringify({
              type: 'activeTasksResponse',
              count: 0,
              tabs: [],
            }) + '\n');
          }
        });
        s.on('error', () => { });
      });
      srv.once('error', reject);
      srv.listen(sockPath, () => resolve({
        close: () => new Promise(res => srv.close(() => {
          try { if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath); }
          catch { }
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

  await test('daemonHasActiveTasks: an OLD-daemon "Unknown command: activeTasksQuery" error is INCONCLUSIVE (must not authorize a restart)', async () => {
    const sockPath = path.join(tmpRoot, 'old-daemon.sock');
    const server = await listenUds(sockPath, {
      type: 'error',
      text: 'Unknown command: activeTasksQuery',
    });
    try {
      const res = await daemonHasActiveTasks(sockPath, 1500);
      assert.strictEqual(res.ok, false,
        `expected ok:false on old-daemon error; got: ${JSON.stringify(res)}`);
      assert.strictEqual(res.reason, 'unsupported-query');
      // With an alive HTTP listener, this inconclusive probe must make
      // decideRestart defer instead of killing a possibly-busy daemon.
      const decision = decideRestart({
        fingerprintMatches: false,
        health: 'alive',
        activeTasks: res,
      });
      assert.strictEqual(decision.skip, true);
    } finally {
      await server.close();
    }
  });

  await test('decideRestart: skips restart when the daemon reports active tasks (bug-fix scenario)', () => {
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'alive',
      activeTasks: {ok: true, count: 1, tabs: ['x(task=74)']},
    });
    assert.strictEqual(decision.skip, true);
    assert.strictEqual(decision.reason, 'active-tasks');
  });

  await test('decideRestart: skips restart when health probe is UNKNOWN and fingerprint matches (the actual lsof-timeout regression)', () => {
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
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'alive',
      activeTasks: {ok: false, reason: 'timeout'},
    });
    assert.strictEqual(decision.skip, true,
      `expected to skip restart of an alive daemon with unknown ` +
      `active-tasks status; got: ${JSON.stringify(decision)}`);
    assert.ok(/alive-uncertain/.test(decision.reason),
      `expected the skip reason to flag the uncertainty; got: ${decision.reason}`);
  });

  await test('decideRestart: RESTARTS when daemon is ALIVE but UDS socket file is missing (Update-button hang fix)', () => {
    const decision = decideRestart({
      fingerprintMatches: true,
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
    const decision = decideRestart({
      fingerprintMatches: true,
      health: 'alive',
      activeTasks: {ok: false, reason: 'sock-missing'},
    });
    assert.strictEqual(decision.skip, false);
    assert.ok(/unreachable-uds/.test(decision.reason));
  });

  await test('decideRestart: explicit active-tasks reply still wins over sock-missing (defensive ordering)', () => {
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'alive',
      activeTasks: {ok: true, count: 1, tabs: ['x(task=1)']},
    });
    assert.strictEqual(decision.skip, true);
    assert.strictEqual(decision.reason, 'active-tasks');
  });

  await test('decideRestart: still RESTARTS when the daemon is fully unreachable (health=unknown + activeTasks failed + fingerprint mismatched)', () => {
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
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'dead',
      activeTasks: {ok: true, count: 5, tabs: ['a', 'b', 'c', 'd', 'e']},
    });
    assert.strictEqual(decision.skip, true);
    assert.strictEqual(decision.reason, 'active-tasks');
  });

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
        fingerprintMatches: false,
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
    } catch { }
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
