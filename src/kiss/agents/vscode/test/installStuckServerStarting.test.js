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

function startUdsHalf(sockPath, activeCount) {
  return new Promise((resolve, reject) => {
    try {
      fs.rmSync(sockPath, {force: true});
    } catch {
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
      conn.on('error', () => { });
    });
    uds.once('error', reject);
    uds.listen(sockPath, () => {
      resolve({
        close: async () => {
          await new Promise(res => uds.close(() => res()));
          try {
            fs.rmSync(sockPath, {force: true});
          } catch {
          }
        },
      });
    });
  });
}

function startTcpHalf(port) {
  return new Promise((resolve, reject) => {
    const tcp = net.createServer(() => { });
    tcp.once('error', reject);
    tcp.listen(port, '127.0.0.1', () => {
      resolve({
        close: () => new Promise(res => tcp.close(() => res())),
      });
    });
  });
}

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

function makeKissWebBin(name) {
  const bin = path.join(tmpRoot, name, '.venv', 'bin', 'kiss-web');
  fs.mkdirSync(path.dirname(bin), {recursive: true});
  fs.writeFileSync(bin, '#!/bin/sh\nexit 0\n', {mode: 0o755});
  return bin;
}

async function main() {
  await test(
    'reproduction: dead daemon + no recovery == stuck overlay (timeout)',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('repro-timeout');
      const sockPath = path.join(tmpRoot, 'repro-timeout.sock');
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

  await test(
    'async restart callback: rejection is logged, later attempt succeeds',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('async-restart');
      const sockPath = path.join(tmpRoot, 'async-restart.sock');
      let daemon = null;
      let calls = 0;
      let spawned = false;
      const logs = [];
      const res = await verifyDaemonStartup({
        binPath: bin,
        sockPath,
        port,
        restart: async () => {
          calls += 1;
          await new Promise(resolve => setTimeout(resolve, 50));
          if (calls === 1) {
            throw new Error('bootstrap still draining');
          }
          if (!spawned) {
            spawned = true;
            daemon = await startFakeDaemon(port, sockPath);
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
      assert.ok(calls >= 2, `calls=${calls}`);
      assert.ok(
        logs.some(
          m =>
            m.includes('re-restart attempt failed') &&
            m.includes('bootstrap still draining'),
        ),
        `the rejected async restart must be logged, got: ${JSON.stringify(logs)}`,
      );
      assert.ok(daemon, 'real daemon must have been started');
      await daemon.close();
    },
  );

  await test(
    'venv-wipe race: waits for binary to reappear, then restarts to alive',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('venv-wipe');
      const sockPath = path.join(tmpRoot, 'venv-wipe.sock');
      fs.rmSync(bin);
      let daemon = null;
      let restartsWhileBinMissing = 0;
      let spawned = false;
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

  await test(
    'alive TCP + missing UDS file reports sock-missing without restarts',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('sock-missing');
      const sockPath = path.join(tmpRoot, 'sock-missing.sock');
      const daemon = await startFakeDaemon(port, sockPath);
      fs.rmSync(sockPath);
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
        restartEveryMs: 60_000,
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

  await test('mid-boot daemon (UDS answering, TCP dead) is never bounced',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('mid-boot');
      const sockPath = path.join(tmpRoot, 'mid-boot.sock');
      const uds = await startUdsHalf(sockPath, 0);
      let tcp = null;
      const tcpTimer = setTimeout(() => {
        startTcpHalf(port).then(t => {
          tcp = t;
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
        restartEveryMs: 0,
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

  await test('UDS daemon with active tasks + dead TCP: no restart, timeout',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('active-veto');
      const sockPath = path.join(tmpRoot, 'active-veto.sock');
      const uds = await startUdsHalf(sockPath, 2);
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

  await test('stale UDS file does not fake success; restart still fires',
    async () => {
      const port = await freeTcpPort();
      const bin = makeKissWebBin('stale-sock');
      const sockPath = path.join(tmpRoot, 'stale-sock.sock');
      const dead = await startUdsHalf(sockPath, 0);
      await new Promise(res => setTimeout(res, 10));
      await dead.close();
      fs.writeFileSync(sockPath, '');
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
