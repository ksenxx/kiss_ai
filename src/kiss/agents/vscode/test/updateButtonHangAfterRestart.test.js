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
} = require('../out/daemonHealth');
const {createFakeDaemon, fakeEndpointPath} = require('./fakeDaemon');

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

function listenDaemon(endpointPath) {
  return new Promise((resolve, reject) => {
    try {
      if (fs.existsSync(endpointPath)) fs.unlinkSync(endpointPath);
    } catch { }
    // Authenticates, then never answers activeTasksQuery.
    const server = createFakeDaemon(sock => {
      sock.on('data', () => { });
      sock.on('error', () => { });
    });
    server.once('error', reject);
    server.listen(endpointPath, () => {
      resolve({
        deleteEndpointFile: () => {
          try {
            fs.unlinkSync(endpointPath);
          } catch { }
        },
        close: () => new Promise(res => server.close(() => {
          try {
            if (fs.existsSync(endpointPath)) fs.unlinkSync(endpointPath);
          } catch { }
          res();
        })),
      });
    });
  });
}

(async () => {

  await test('Update-button hang — TCP alive + endpoint file deleted ⇒ restart forced', async () => {
    const {port, close: closeTcp} = await listenTcp();
    const endpointPath = path.join(tmpRoot, 'update-hang.json');
    const daemon = await listenDaemon(endpointPath);
    try {
      assert.ok(fs.existsSync(endpointPath),
        'precondition: endpoint file present before the simulated rm');

      daemon.deleteEndpointFile();
      assert.ok(!fs.existsSync(endpointPath),
        'after the simulated rm the endpoint file must be gone');

      const health = await probeDaemonHealth(port, 1500);
      const activeTasks = await daemonHasActiveTasks(endpointPath, 500);
      assert.strictEqual(health, 'alive',
        `daemon's TCP listener should survive endpoint file removal; got: ${health}`);
      assert.deepStrictEqual(activeTasks, {ok: false, reason: 'endpoint-missing'},
        `endpoint probe should report endpoint-missing once install.sh has rm-ed ` +
        `the socket file; got: ${JSON.stringify(activeTasks)}`);

      const decision = decideRestart({
        fingerprintMatches: true,
        health,
        activeTasks,
      });
      assert.strictEqual(decision.skip, false,
        `restartKissWebDaemon must recycle a daemon whose endpoint file is ` +
        `missing — otherwise the webview stays on "KISS Sorcar Server ` +
        `is starting ..." forever.  Got: ${JSON.stringify(decision)}`);
      assert.ok(/unreachable-local/.test(decision.reason),
        `restart reason must flag the unreachable endpoint so install.sh / ` +
        `restartKissWebDaemon logs are diagnosable; got: ${decision.reason}`);
    } finally {
      await daemon.close();
      await closeTcp();
    }
  });

  await test('Task-3192 protection — TCP alive + endpoint timeout ⇒ restart STILL deferred', async () => {
    const {port, close: closeTcp} = await listenTcp();
    const endpointPath = fakeEndpointPath(tmpRoot, 'task3192.json');

    const server = await new Promise((resolve, reject) => {
      try { if (fs.existsSync(endpointPath)) fs.unlinkSync(endpointPath); }
      catch { }
      const srv = createFakeDaemon(s => {
        s.on('data', () => { });
        s.on('error', () => { });
      });
      srv.once('error', reject);
      srv.listen(endpointPath, () => resolve({
        close: () => new Promise(res => srv.close(() => {
          try { if (fs.existsSync(endpointPath)) fs.unlinkSync(endpointPath); }
          catch { }
          res();
        })),
      }));
    });

    try {
      const health = await probeDaemonHealth(port, 1500);
      const activeTasks = await daemonHasActiveTasks(endpointPath, 150);
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

  await test('Unreachable-endpoint restart wins over healthy-unchanged skip', () => {
    const decision = decideRestart({
      fingerprintMatches: true,
      health: 'alive',
      activeTasks: {ok: false, reason: 'endpoint-missing'},
    });
    assert.strictEqual(decision.skip, false,
      `unreachable-local must beat healthy-unchanged or the user remains ` +
      `stranded on the loading overlay forever; got: ${JSON.stringify(decision)}`);
  });

  await test('Active-tasks reply still wins over endpoint-missing (precedence pin)', () => {
    const decision = decideRestart({
      fingerprintMatches: false,
      health: 'alive',
      activeTasks: {ok: true, count: 2, tabs: ['a(task=1)', 'b(task=2)']},
    });
    assert.strictEqual(decision.skip, true);
    assert.strictEqual(decision.reason, 'active-tasks');
  });

  await test('TOCTOU race — rm -f lands AFTER existsSync but BEFORE connect ⇒ reason normalised to endpoint-missing', async () => {
    const endpointPath = path.join(tmpRoot, 'toctou.json');
    fs.writeFileSync(endpointPath, '');
    setImmediate(() => {
      try { fs.unlinkSync(endpointPath); } catch { }
    });
    const res = await daemonHasActiveTasks(endpointPath, 500);
    assert.strictEqual(res.ok, false,
      `TOCTOU probe must fail; got: ${JSON.stringify(res)}`);
    assert.strictEqual(res.reason, 'endpoint-missing',
      `the error path must normalise to 'endpoint-missing' so decideRestart's ` +
      `unreachable-local branch fires; got reason='${res.reason}'`);

    const decision = decideRestart({
      fingerprintMatches: true,
      health: 'alive',
      activeTasks: res,
    });
    assert.strictEqual(decision.skip, false,
      `TOCTOU race must STILL force a restart end-to-end; ` +
      `got: ${JSON.stringify(decision)}`);
    assert.ok(/unreachable-local/.test(decision.reason),
      `decision reason must flag unreachable-local; got: ${decision.reason}`);
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
