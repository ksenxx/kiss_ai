// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for AgentClient's behaviour across a daemon outage, driven
// against a REAL unix domain socket (no mocks):
//
// 1. Reconnects must back off. Every open VS Code window runs one of
//    these against ~/.kiss/sorcar.sock, so a fixed 500 ms retry meant N
//    windows knocked 2N times a second for the whole of every daemon
//    restart -- exactly while it was trying to bind.
// 2. A command queued against a daemon that then died must not be
//    replayed verbatim into a DIFFERENT daemon process minutes later. A
//    `run` delivered that way starts an agent nobody asked for, in a tab
//    that may no longer exist.
// 3. The queue must be bounded, so a long outage cannot grow it without
//    limit.

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

const OUT_AGENT_CLIENT = path.join(__dirname, '..', 'out', 'AgentClient.js');
if (!fs.existsSync(OUT_AGENT_CLIENT)) {
  console.log('SKIP: out/AgentClient.js missing — run `npm run compile`');
  process.exit(0);
}
const {AgentClient} = require(OUT_AGENT_CLIENT);

const tmpDirs = [];

function tmpSock(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-acr-'));
  tmpDirs.push(dir);
  return path.join(dir, name);
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function listen(server, sockPath) {
  return new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );
}

function close(server) {
  return new Promise(r => server.close(r));
}

// The daemon is down for the whole window: every attempt is refused, so
// the gaps between attempts are exactly the client's retry schedule.
async function testReconnectBacksOff() {
  const sockPath = tmpSock('storm.sock');
  const attempts = [];
  const server = net.createServer(conn => {
    attempts.push(Date.now());
    conn.destroy();
  });
  await listen(server, sockPath);

  const client = new AgentClient(sockPath);
  client.connect();
  await delay(3000);
  client.dispose();
  await close(server);

  assert.ok(attempts.length >= 3, 'the client must keep trying');
  assert.ok(
    attempts.length <= 8,
    `a 3s outage must not cost ~${attempts.length} connect attempts; a ` +
      'fixed retry is a connect storm once several windows are open',
  );
  const gaps = [];
  for (let i = 1; i < attempts.length; i += 1) {
    gaps.push(attempts[i] - attempts[i - 1]);
  }
  assert.ok(gaps.length >= 2, 'need at least two gaps to compare');
  assert.ok(
    gaps[gaps.length - 1] > gaps[0] * 1.4,
    `the retry delay must grow: gaps were ${JSON.stringify(gaps)}`,
  );
  console.log('ok - reconnects back off instead of hammering the socket');
}

async function testStaleQueuedCommandIsNotReplayed() {
  const sockPath = tmpSock('stale-run.sock');
  const client = new AgentClient(sockPath, {
    reconnectBaseMs: 40,
    reconnectMaxMs: 120,
    pendingTtlMs: 250,
  });

  // Nothing is listening: the frame is queued.
  client.sendCommand({type: 'run', task: 'do the thing', tabId: 't1'});
  await delay(600);

  // A NEW daemon comes up on the same socket.
  const received = [];
  const server = net.createServer(conn => {
    conn.on('data', d => received.push(d.toString()));
  });
  await listen(server, sockPath);

  await new Promise(resolve => {
    client.on('connect', resolve);
    client.connect();
  });
  client.sendCommand({type: 'ready', tabId: 't1'});
  await delay(200);
  client.dispose();
  await close(server);

  const wire = received.join('');
  assert.ok(
    wire.includes('"type":"ready"'),
    `a command sent after reconnecting must be delivered, got: ${wire}`,
  );
  assert.ok(
    !wire.includes('"type":"run"'),
    'a run queued against a daemon that died must NOT be replayed into ' +
      `the daemon that replaced it, got: ${wire}`,
  );
  console.log('ok - a stale queued command is not replayed into a new daemon');
}

async function testFreshQueuedCommandIsStillDelivered() {
  const sockPath = tmpSock('fresh-run.sock');
  const client = new AgentClient(sockPath, {
    reconnectBaseMs: 40,
    reconnectMaxMs: 120,
    pendingTtlMs: 5000,
  });

  client.sendCommand({type: 'run', task: 'do the thing', tabId: 't1'});
  const received = [];
  const server = net.createServer(conn => {
    conn.on('data', d => received.push(d.toString()));
  });
  await listen(server, sockPath);
  await new Promise(resolve => {
    client.on('connect', resolve);
    client.connect();
  });
  await delay(150);
  client.dispose();
  await close(server);

  assert.ok(
    received.join('').includes('"type":"run"'),
    'a command queued moments before the daemon answered must still be ' +
      'delivered — only a stale one is dropped',
  );
  console.log('ok - a fresh queued command is still delivered');
}

async function testQueueIsBounded() {
  const sockPath = tmpSock('bounded.sock');
  const client = new AgentClient(sockPath, {
    reconnectBaseMs: 40,
    reconnectMaxMs: 120,
    pendingTtlMs: 5000,
    maxPendingSends: 3,
  });
  for (let i = 0; i < 20; i += 1) {
    client.sendCommand({type: 'notice', n: i});
  }

  const received = [];
  const server = net.createServer(conn => {
    conn.on('data', d => received.push(d.toString()));
  });
  await listen(server, sockPath);
  await new Promise(resolve => {
    client.on('connect', resolve);
    client.connect();
  });
  await delay(150);
  client.dispose();
  await close(server);

  const lines = received
    .join('')
    .split('\n')
    .filter(l => l.trim());
  assert.strictEqual(
    lines.length,
    3,
    `the queue must be bounded, got ${lines.length} frames`,
  );
  assert.ok(
    lines[lines.length - 1].includes('"n":19'),
    'the newest commands are the ones kept',
  );
  console.log('ok - the pending-send queue is bounded');
}

(async () => {
  if (process.platform === 'win32') {
    console.log('SKIP: UDS tests require a POSIX platform');
    return;
  }
  try {
    await testReconnectBacksOff();
    await testStaleQueuedCommandIsNotReplayed();
    await testFreshQueuedCommandIsStillDelivered();
    await testQueueIsBounded();
    console.log('agentClientReconnect.test.js passed');
  } finally {
    for (const dir of tmpDirs.reverse()) {
      fs.rmSync(dir, {recursive: true, force: true});
    }
  }
})().catch(err => {
  console.error(err);
  process.exit(1);
});
