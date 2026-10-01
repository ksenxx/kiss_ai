// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for AgentClient.setPreamble(), driven against a REAL
// WebSocket daemon stand-in (test/fakeDaemon.js, no mocks).
//
// The daemon pins a connection's workspace folder from `setWorkDir` and
// stamps that pin on every later command sent without a workDir -- the
// extension host forwards its webview's submit / openFile / checkPaths
// exactly as sent, so they depend on it.  Commands queued while the
// daemon was down used to be flushed BEFORE the connect handler sent
// `setWorkDir`, so a prompt submitted or a file link clicked during an
// outage resolved against the daemon-global folder: another window's.
// The preamble is written first on every connection.

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
function tmpEndpoint(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-acp-'));
  tmpDirs.push(dir);
  return path.join(dir, name);
}
function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}
function listen(server, endpointPath) {
  return new Promise((res, rej) =>
    server.listen(endpointPath, err => (err ? rej(err) : res())),
  );
}
function close(server) {
  return new Promise(r => server.close(r));
}

// Collect the newline-framed commands of every authenticated connection
// a fake daemon sees, one array per connection, in arrival order.
function recordingServer() {
  const connections = [];
  const server = createFakeDaemon(conn => {
    const frames = [];
    connections.push({conn, frames});
    let buf = '';
    conn.on('data', d => {
      buf += d.toString();
      let idx;
      while ((idx = buf.indexOf('\n')) >= 0) {
        frames.push(JSON.parse(buf.slice(0, idx)));
        buf = buf.slice(idx + 1);
      }
    });
  });
  return {server, connections};
}

// 1. A command queued during an outage is delivered AFTER the preamble,
//    and a command sent from the connect handler after both.
async function testPreambleLeadsQueuedCommands() {
  const endpointPath = tmpEndpoint('preamble.json');
  const client = new AgentClient(endpointPath, {
    reconnectBaseMs: 40,
    reconnectMaxMs: 120,
    pendingTtlMs: 5000,
  });
  client.setPreamble({type: 'setWorkDir', workDir: '/ws/window-a'});
  client.on('connect', () => client.sendCommand({type: 'getModels'}));

  // Nothing is listening yet: the submit waits in the queue.
  client.sendCommand({type: 'submit', prompt: 'README.md', tabId: 't1'});
  await delay(100);

  const {server, connections} = recordingServer();
  await listen(server, endpointPath);
  await new Promise(resolve => {
    client.once('connect', resolve);
    client.connect();
  });
  await delay(150);

  assert.strictEqual(connections.length, 1, 'one connection expected');
  assert.deepStrictEqual(
    connections[0].frames.map(f => f.type),
    ['setWorkDir', 'submit', 'getModels'],
    'the pin must precede the queued submit, which precedes connect-time ' +
      `commands (got ${JSON.stringify(connections[0].frames)})`,
  );
  assert.strictEqual(connections[0].frames[0].workDir, '/ws/window-a');

  client.dispose();
  await close(server);
  console.log('  ok - the preamble leads the commands queued during an outage');
}

// 2. Every reconnect repeats the (current) preamble: a fresh daemon
//    connection starts with no pin.
async function testPreambleRepeatsOnReconnect() {
  const endpointPath = tmpEndpoint('preamble-again.json');
  const client = new AgentClient(endpointPath, {
    reconnectBaseMs: 40,
    reconnectMaxMs: 120,
    pendingTtlMs: 5000,
  });
  client.setPreamble({type: 'setWorkDir', workDir: '/ws/first'});

  const first = recordingServer();
  await listen(first.server, endpointPath);
  await new Promise(resolve => {
    client.once('connect', resolve);
    client.connect();
  });
  await delay(50);
  assert.deepStrictEqual(first.connections[0].frames, [
    {type: 'setWorkDir', workDir: '/ws/first'},
  ]);

  // The daemon dies; the window's folder changes meanwhile.
  const disconnected = new Promise(resolve =>
    client.once('disconnect', resolve),
  );
  first.connections[0].conn.destroy();
  await close(first.server);
  await disconnected;
  client.setPreamble({type: 'setWorkDir', workDir: '/ws/second'});
  client.sendCommand({type: 'checkPaths', paths: ['a.py'], tabId: 't1'});
  await delay(100);

  const second = recordingServer();
  await listen(second.server, endpointPath);
  await new Promise(resolve => client.once('connect', resolve));
  await delay(150);

  assert.deepStrictEqual(
    second.connections[0].frames.map(f => f.type),
    ['setWorkDir', 'checkPaths'],
    `the new connection must be pinned before the queued checkPaths (got ${
      JSON.stringify(second.connections[0].frames)})`,
  );
  assert.strictEqual(second.connections[0].frames[0].workDir, '/ws/second');

  client.dispose();
  await close(second.server);
  console.log('  ok - every reconnect is led by the current preamble');
}

// 3. No preamble (or a cleared one): nothing extra is written.
async function testClearedPreambleWritesNothing() {
  const endpointPath = tmpEndpoint('no-preamble.json');
  const client = new AgentClient(endpointPath, {pendingTtlMs: 5000});
  client.setPreamble({type: 'setWorkDir', workDir: '/ws/x'});
  client.setPreamble(null);
  client.sendCommand({type: 'getModels'});

  const {server, connections} = recordingServer();
  await listen(server, endpointPath);
  await new Promise(resolve => {
    client.once('connect', resolve);
    client.connect();
  });
  await delay(100);
  assert.deepStrictEqual(connections[0].frames, [{type: 'getModels'}]);

  client.dispose();
  await close(server);
  console.log('  ok - a cleared preamble writes nothing');
}

(async () => {
  try {
    await testPreambleLeadsQueuedCommands();
    await testPreambleRepeatsOnReconnect();
    await testClearedPreambleWritesNothing();
    console.log('agentClientPreamble.test.js: all tests passed');
  } catch (err) {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exitCode = 1;
  } finally {
    for (const dir of tmpDirs.slice().reverse()) {
      fs.rmSync(dir, {recursive: true, force: true});
    }
  }
})();
