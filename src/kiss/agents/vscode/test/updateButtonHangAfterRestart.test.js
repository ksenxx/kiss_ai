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

function listenUds(sockPath) {
  return new Promise((resolve, reject) => {
    try {
      if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
    } catch { }
    const server = net.createServer(sock => {
      sock.on('data', () => { });
      sock.on('error', () => { });
    });
    server.once('error', reject);
    server.listen(sockPath, () => {
      resolve({
        deleteSocketFile: () => {
          try {
            fs.unlinkSync(sockPath);
          } catch { }
        },
        close: () => new Promise(res => server.close(() => {
          try {
            if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath);
          } catch { }
          res();
        })),
      });
    });
  });
}

(async () => {

  await test('Update-button hang — TCP alive + UDS file deleted ⇒ restart forced', async () => {
    const {port, close: closeTcp} = await listenTcp();
    const sockPath = path.join(tmpRoot, 'update-hang.sock');
    const uds = await listenUds(sockPath);
    try {
      assert.ok(fs.existsSync(sockPath),
        'precondition: UDS socket file present before the simulated rm');

      uds.deleteSocketFile();
      assert.ok(!fs.existsSync(sockPath),
        'after the simulated rm the UDS file must be gone');

      const health = await probeDaemonHealth(port, 1500);
      const activeTasks = await daemonHasActiveTasks(sockPath, 500);
      assert.strictEqual(health, 'alive',
        `daemon's TCP listener should survive UDS file removal; got: ${health}`);
      assert.deepStrictEqual(activeTasks, {ok: false, reason: 'sock-missing'},
        `UDS probe should report sock-missing once install.sh has rm-ed ` +
        `the socket file; got: ${JSON.stringify(activeTasks)}`);

      const decision = decideRestart({
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

  await test('Task-3192 protection — TCP alive + UDS timeout ⇒ restart STILL deferred', async () => {
    const {port, close: closeTcp} = await listenTcp();
    const sockPath = path.join(tmpRoot, 'task3192.sock');

    const server = await new Promise((resolve, reject) => {
      try { if (fs.existsSync(sockPath)) fs.unlinkSync(sockPath); }
      catch { }
      const srv = net.createServer(s => {
        s.on('data', () => { });
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
      const health = await probeDaemonHealth(port, 1500);
      const activeTasks = await daemonHasActiveTasks(sockPath, 150);
      assert.strictEqual(health, 'alive');
      assert.strictEqual(activeTasks.ok, false);
      assert.strictEqual(activeTasks.reason, 'timeout',
        `the timeout case must reach decideRestart as reason='timeout'; ` +
        `got: ${JSON.stringify(activeTasks)}`);

      const decision = decideRestart({
        fingerprintMatches: false,
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

  await test('Active-tasks reply still wins over sock-missing (precedence pin)', () => {
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'alive',
      activeTasks: {ok: true, count: 2, tabs: ['a(task=1)', 'b(task=2)']},
    });
    assert.strictEqual(decision.skip, true);
    assert.strictEqual(decision.reason, 'active-tasks');
  });

  await test('TOCTOU race — rm -f lands AFTER existsSync but BEFORE connect ⇒ reason normalised to sock-missing', async () => {
    const sockPath = path.join(tmpRoot, 'toctou.sock');
    fs.writeFileSync(sockPath, '');
    setImmediate(() => {
      try { fs.unlinkSync(sockPath); } catch { }
    });
    const res = await daemonHasActiveTasks(sockPath, 500);
    assert.strictEqual(res.ok, false,
      `TOCTOU probe must fail; got: ${JSON.stringify(res)}`);
    assert.strictEqual(res.reason, 'sock-missing',
      `the error path must normalise to 'sock-missing' so decideRestart's ` +
      `unreachable-uds branch fires; got reason='${res.reason}'`);

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
    } catch { }
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
