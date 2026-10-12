// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Audit 2026-10-12 (scope vscode-ts): adversarial end-to-end checks of
// AgentClient's connection single-flight against a REAL loopback daemon
// stand-in (test/fakeDaemon.js, no mocks):
//
// 1. N synchronous connect()/sendCommand() calls open ONE socket, and
//    every queued command reaches the daemon once, in order.
// 2. A daemon that drops every connection right after `auth_ok` never
//    sees two client sockets open at the same time (reconnects are
//    serialised, never stacked).
// 3. A daemon that never answers `auth` is abandoned by the auth timer
//    and retried — again one socket at a time — and dispose() in the
//    middle of that cycle stops everything: no socket opens afterwards,
//    no event fires afterwards, and the process has nothing left to
//    keep it alive (the test exits on its own: that IS the assertion
//    that no timer survived dispose()).
// 4. dispose() while the endpoint file is missing (the setImmediate
//    retry path) never connects once the file appears.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {createFakeDaemon} = require('./fakeDaemon');

const OUT_AGENT_CLIENT = path.join(__dirname, '..', 'out', 'AgentClient.js');
if (!fs.existsSync(OUT_AGENT_CLIENT)) {
  console.log('SKIP: out/AgentClient.js missing — run `npm run compile`');
  process.exit(0);
}
const {AgentClient} = require(OUT_AGENT_CLIENT);

const tmpDirs = [];
function tmpEndpoint() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit1012-ac-'));
  tmpDirs.push(dir);
  return path.join(dir, 'sorcar-local.json');
}
function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}
function listen(daemon, endpointPath) {
  return new Promise((res, rej) =>
    daemon.listen(endpointPath, err => (err ? rej(err) : res())),
  );
}
function closeDaemon(daemon) {
  daemon.destroyConnections();
  return new Promise(r => daemon.close(r));
}

/** Track how many raw TCP sockets the daemon has open at once. */
function trackConcurrency(daemon) {
  const stats = {open: 0, maxOpen: 0, total: 0};
  daemon._server.on('connection', sock => {
    stats.open++;
    stats.total++;
    stats.maxOpen = Math.max(stats.maxOpen, stats.open);
    sock.on('close', () => stats.open--);
  });
  return stats;
}

function makeClient(endpointPath, extra) {
  return new AgentClient(
    endpointPath,
    Object.assign(
      {
        reconnectBaseMs: 20,
        reconnectMaxMs: 40,
        endpointPollMs: 20,
        authTimeoutMs: 60,
      },
      extra || {},
    ),
  );
}

async function testBurstOpensOneSocket() {
  const endpointPath = tmpEndpoint();
  const received = [];
  const daemon = createFakeDaemon(conn => {
    conn.on('data', buf => {
      for (const line of buf.toString().split('\n')) {
        if (line.trim()) received.push(JSON.parse(line));
      }
    });
  });
  const stats = trackConcurrency(daemon);
  await listen(daemon, endpointPath);

  const client = makeClient(endpointPath);
  try {
    for (let i = 0; i < 5; i++) client.connect();
    for (let i = 0; i < 3; i++) {
      client.sendCommand({type: 'ping', seq: i});
    }
    for (let i = 0; i < 5; i++) client.connect();

    const deadline = Date.now() + 2000;
    while (received.length < 3 && Date.now() < deadline) await delay(10);
    assert.deepStrictEqual(
      received.map(m => m.seq),
      [0, 1, 2],
      'every queued command is delivered once, in order',
    );
    assert.strictEqual(stats.total, 1, 'a burst of connect() opens ONE socket');
  } finally {
    client.dispose();
    await closeDaemon(daemon);
  }
  console.log('ok - a burst of connect()/sendCommand() opens one socket');
}

async function testDroppingDaemonNeverStacksSockets() {
  const endpointPath = tmpEndpoint();
  // Drop each connection right after auth_ok: the client reconnects
  // over and over for the whole window.
  const daemon = createFakeDaemon(conn => conn.destroy());
  const stats = trackConcurrency(daemon);
  await listen(daemon, endpointPath);

  const client = makeClient(endpointPath);
  try {
    let disconnects = 0;
    client.on('disconnect', () => disconnects++);
    client.connect();
    await delay(600);
    assert.ok(
      stats.total >= 3,
      `expected several reconnects, saw ${stats.total}`,
    );
    assert.strictEqual(stats.maxOpen, 1, 'never two sockets open at once');
    assert.ok(disconnects >= 1, 'disconnect fired at least once');
  } finally {
    client.dispose();
    await closeDaemon(daemon);
  }
  console.log(
    'ok - a daemon that drops every connection never sees stacked sockets',
  );
}

async function testSilentAuthDisposeStopsEverything() {
  const endpointPath = tmpEndpoint();
  const daemon = createFakeDaemon(() => {}, {auth: 'silent'});
  const stats = trackConcurrency(daemon);
  await listen(daemon, endpointPath);

  const client = makeClient(endpointPath);
  try {
    const events = [];
    client.on('connect', () => events.push('connect'));
    client.on('disconnect', () => events.push('disconnect'));
    client.connect();
    // Let the auth timer expire a few times (60 ms each + 20-40 ms backoff).
    await delay(400);
    assert.ok(stats.total >= 2, `auth timeout must retry, saw ${stats.total}`);
    assert.strictEqual(stats.maxOpen, 1, 'auth retries are serialised');
    assert.ok(
      !events.includes('connect'),
      'silent auth never counts as connected',
    );

    const totalAtDispose = stats.total;
    const eventsAtDispose = events.length;
    client.dispose();
    // Dispose both inside and after a pending reconnect timer: nothing
    // may open or fire afterwards.
    client.connect();
    client.sendCommand({type: 'ping'});
    await delay(300);
    assert.strictEqual(
      stats.total,
      totalAtDispose,
      'no socket opens after dispose()',
    );
    assert.strictEqual(
      stats.open,
      0,
      'the in-flight socket was destroyed by dispose()',
    );
    assert.strictEqual(
      events.length,
      eventsAtDispose,
      'no event fires after dispose()',
    );
  } finally {
    client.dispose();
    await closeDaemon(daemon);
  }
  console.log(
    'ok - dispose() during the auth-timeout retry cycle stops every timer',
  );
}

async function testDisposeDuringEndpointPollNeverConnects() {
  const endpointPath = tmpEndpoint();
  const client = makeClient(endpointPath);
  const events = [];
  client.on('disconnect', () => events.push('disconnect'));
  client.connect(); // no endpoint file: schedules the setImmediate retry
  client.dispose(); // before the setImmediate runs

  const daemon = createFakeDaemon(() => {});
  const stats = trackConcurrency(daemon);
  try {
    await listen(daemon, endpointPath); // file appears now
    await delay(200);
    assert.strictEqual(stats.total, 0, 'disposed client never connects');
    assert.deepStrictEqual(events, [], 'no event fires after dispose()');
  } finally {
    await closeDaemon(daemon);
  }
  console.log(
    'ok - dispose() inside the endpoint-poll retry never connects later',
  );
}

(async () => {
  try {
    await testBurstOpensOneSocket();
    await testDroppingDaemonNeverStacksSockets();
    await testSilentAuthDisposeStopsEverything();
    await testDisposeDuringEndpointPollNeverConnects();
  } finally {
    for (const d of tmpDirs) fs.rmSync(d, {recursive: true, force: true});
  }
  console.log(
    'audit1012_vscode_ts_agent_client_single_flight: all tests passed',
  );
  // No process.exit(): a live timer left by the client would hang the
  // test, which is the leak detector for dispose().
})().catch(err => {
  console.error(err);
  process.exit(1);
});
