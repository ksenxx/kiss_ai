// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E regression tests for the local-WSS transport review findings, all
// against a real TCP WebSocket peer (test/fakeDaemon.js):
// 1. AgentClient gives up on a daemon that upgrades but never answers
//    the auth frame, and retries instead of holding the dead socket.
// 2. daemonHasActiveTasks refuses `auth_ok local:false` (a remote
//    session is not proof that the local channel works).
// 3. WsClient closes with 1002 on a masked server frame and on a
//    fragmented control frame, and with 1007 on invalid UTF-8 text.
// 4. A Close received while a large message is still being written
//    under backpressure does not discard the message or the Close
//    reply: the peer receives both.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {createFakeDaemon} = require('./fakeDaemon');

const OUT = path.join(__dirname, '..', 'out');
for (const f of ['AgentClient.js', 'wsClient.js']) {
  if (!fs.existsSync(path.join(OUT, f))) {
    console.log(`SKIP: out/${f} missing — run \`npm run compile\``);
    process.exit(0);
  }
}
const {AgentClient} = require(path.join(OUT, 'AgentClient.js'));
const {WsClient} = require(path.join(OUT, 'wsClient.js'));
const {daemonHasActiveTasks} = require(path.join(OUT, 'daemonHealth.js'));

function tmpEndpoint(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wsfix-'));
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

async function waitFor(cond, what, ms = 2000) {
  const until = Date.now() + ms;
  while (!cond()) {
    if (Date.now() > until) throw new Error(`timed out waiting for ${what}`);
    await delay(10);
  }
}

function closeServer(server) {
  return new Promise(r => server.close(r));
}

/** Open a WsClient on the fake daemon's endpoint; resolve once open. */
function openClient(endpointPath) {
  const endpoint = JSON.parse(fs.readFileSync(endpointPath, 'utf8'));
  const ws = new WsClient({url: endpoint.url, connectTimeoutMs: 2000});
  const closed = new Promise(r => ws.on('close', r));
  ws.on('error', () => {});
  const opened = new Promise((res, rej) => {
    ws.on('open', res);
    ws.on('close', () => rej(new Error('closed before open')));
  });
  ws.connect();
  return {ws, opened, closed};
}

async function testAuthTimeoutRetries() {
  const endpointPath = tmpEndpoint('silent.json');
  let connections = 0;
  const server = createFakeDaemon(null, {auth: 'silent'});
  server.on('connection', () => connections++);
  await listen(server, endpointPath);

  const client = new AgentClient(endpointPath, {
    authTimeoutMs: 200,
    reconnectBaseMs: 50,
    reconnectMaxMs: 50,
  });
  let connects = 0;
  let disconnects = 0;
  client.on('connect', () => connects++);
  client.on('disconnect', () => disconnects++);
  client.connect();
  await delay(900);
  client.dispose();
  await closeServer(server);

  assert.strictEqual(connects, 0, 'a silent daemon must never count as connected');
  assert.ok(
    connections >= 2,
    `the client must drop the unanswered socket and retry (saw ${connections} connections)`,
  );
  assert.ok(disconnects >= 1, 'each abandoned attempt is reported as a disconnect');
  console.log('ok - auth timeout drops the silent socket and retries');
}

async function testHealthProbeRequiresLocal() {
  const endpointPath = tmpEndpoint('remote.json');
  let queries = 0;
  const server = createFakeDaemon(
    conn => {
      conn.on('data', () => {
        queries++;
        conn.write(JSON.stringify({type: 'activeTasksResponse', count: 0, tabs: []}));
      });
    },
    {auth: 'remote'},
  );
  await listen(server, endpointPath);
  const res = await daemonHasActiveTasks(endpointPath, 1500);
  await closeServer(server);

  assert.strictEqual(res.ok, false, `remote auth must not pass the probe: ${JSON.stringify(res)}`);
  assert.strictEqual(res.reason, 'auth-rejected');
  assert.strictEqual(queries, 0, 'no activeTasksQuery may be sent on a remote session');
  console.log('ok - health probe refuses auth_ok local:false');
}

async function testProtocolViolationsClose() {
  const cases = [
    {
      name: 'masked server frame',
      code: 1002,
      write: conn =>
        conn.writeRaw(
          Buffer.concat([Buffer.from([0x81, 0x82, 1, 2, 3, 4]), Buffer.from([0x7a, 0x7b])]),
        ),
    },
    {
      name: 'fragmented ping',
      code: 1002,
      write: conn => conn.writeRaw(Buffer.from([0x09, 0x01, 0x41])),
    },
    {
      name: 'oversized ping',
      code: 1002,
      write: conn =>
        conn.writeRaw(
          Buffer.concat([Buffer.from([0x89, 126, 0x00, 126]), Buffer.alloc(126, 0x41)]),
        ),
    },
    {
      name: 'invalid UTF-8 text',
      code: 1007,
      write: conn => conn.writeRaw(Buffer.from([0x81, 0x02, 0xff, 0xfe])),
    },
    // Headers whose first two bytes are already invalid must close at
    // once, not wait for the extended length that never arrives.
    {
      name: 'masked text header with 16-bit length pending',
      code: 1002,
      write: conn => conn.writeRaw(Buffer.from([0x81, 0xfe])),
    },
    {
      name: 'ping with a 16-bit length marker',
      code: 1002,
      write: conn => conn.writeRaw(Buffer.from([0x89, 0x7e])),
    },
    {
      name: 'fragmented ping with a 64-bit length marker',
      code: 1002,
      write: conn => conn.writeRaw(Buffer.from([0x09, 0x7f])),
    },
    {
      name: 'text frame after an invalid one in the same chunk',
      code: 1007,
      write: conn => conn.writeRaw(Buffer.from([0x81, 0x01, 0xff, 0x81, 0x01, 0x41])),
    },
  ];
  for (const c of cases) {
    const endpointPath = tmpEndpoint('proto.json');
    let conn;
    const server = createFakeDaemon(k => {
      conn = k;
    });
    await listen(server, endpointPath);
    const {ws, opened, closed} = openClient(endpointPath);
    const messages = [];
    ws.on('message', m => messages.push(m));
    await opened;
    ws.send(JSON.stringify({type: 'auth', token: server.token}));
    await waitFor(() => conn, 'authenticated connection');
    c.write(conn);
    const info = await Promise.race([closed, delay(1500).then(() => null)]);
    ws.destroy();
    await closeServer(server);
    assert.ok(info, `${c.name}: the client must close the connection`);
    assert.strictEqual(info.code, c.code, `${c.name}: close code`);
    const delivered = messages.filter(m => !m.includes('"auth_ok"'));
    assert.strictEqual(delivered.length, 0, `${c.name}: nothing may be delivered`);
    console.log(`ok - ${c.name} closes with ${c.code}`);
  }
}

async function testCloseUnderBackpressureFlushes() {
  const endpointPath = tmpEndpoint('flush.json');
  let conn;
  const server = createFakeDaemon(k => {
    conn = k;
  });
  await listen(server, endpointPath);
  const {ws, opened, closed} = openClient(endpointPath);
  await opened;
  ws.send(JSON.stringify({type: 'auth', token: server.token}));
  await waitFor(() => conn, 'authenticated connection');

  // Collect every byte the peer receives after it resumes, raw.
  const received = [];
  conn.socket.removeAllListeners('data');
  conn.socket.on('data', d => received.push(d));
  conn.pause();
  const big = 'x'.repeat(8 * 1024 * 1024);
  assert.ok(ws.send(big), 'send must be accepted');
  await delay(100);
  // The peer closes while the client's 8 MiB frame is still queued.
  conn.writeFrame(0x8, Buffer.from([0x03, 0xe8]));
  await delay(50);
  conn.resume();
  const info = await Promise.race([closed, delay(5000).then(() => null)]);
  await delay(100);
  await closeServer(server);

  assert.ok(info, 'the client must finish closing');
  assert.strictEqual(info.code, 1000);
  const bytes = Buffer.concat(received);
  // 10-byte header + 4-byte mask + payload, then the 2-byte-header +
  // 4-byte mask + 2-byte-code Close reply.
  const expected = 10 + 4 + big.length + 2 + 4 + 2;
  assert.strictEqual(
    bytes.length,
    expected,
    `the peer must receive the whole message and the Close reply (got ${bytes.length} of ${expected} bytes)`,
  );
  const closeHeader = bytes[bytes.length - 8];
  assert.strictEqual(closeHeader, 0x88, 'the last frame is a FIN Close');
  console.log('ok - Close under backpressure flushes the message and the reply');
}

(async () => {
  await testAuthTimeoutRetries();
  await testHealthProbeRequiresLocal();
  await testProtocolViolationsClose();
  await testCloseUnderBackpressureFlushes();
  console.log('ALL OK');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
