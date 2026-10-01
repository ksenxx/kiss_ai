// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for AgentClient transport correctness over a real WebSocket:
// 1. A UTF-8 code point split across WebSocket fragments (and TCP
//    chunks) must not be corrupted.
// 2. Frames received before `auth_ok` are the handshake, not events;
//    the first event after `auth_ok` reaches the listener intact even
//    when it arrives in the same TCP chunk.
// 3. dispose() while a connect is in flight must not emit 'connect' or
//    write queued commands to the socket.
// 4. The default endpoint path must honor $KISS_SORCAR_LOCAL and
//    $KISS_HOME.
// 5. A daemon that answers the token with anything but `auth_ok
//    local:true` gets dropped: no 'connect', no command delivered.
// 6. Large frames (>64 KiB, the 16-bit length form) round-trip both ways.

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

function tmpEndpoint(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ac-'));
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

async function testUtf8SplitAcrossFragments() {
  const endpointPath = tmpEndpoint('utf8.json');
  const text = 'emoji \u{1F600} end';
  const payload = Buffer.from(JSON.stringify({type: 'notice', text}));
  // Split inside the 4-byte emoji sequence: first fragment is a
  // non-final text frame, the rest a final continuation frame, and the
  // two go out in separate TCP writes.
  const emojiStart = payload.indexOf(Buffer.from('\u{1F600}'));
  const cut = emojiStart + 2;
  const server = createFakeDaemon(conn => {
    const first = payload.subarray(0, cut);
    const second = payload.subarray(cut);
    conn.writeRaw(Buffer.concat([Buffer.from([0x01, first.length]), first]));
    setTimeout(() => {
      conn.writeRaw(
        Buffer.concat([Buffer.from([0x80, second.length]), second]),
      );
    }, 30);
  });
  await listen(server, endpointPath);

  const client = new AgentClient(endpointPath);
  const messages = [];
  client.on('message', m => messages.push(m));
  client.connect();
  await delay(300);
  client.dispose();
  await new Promise(r => server.close(r));

  assert.strictEqual(messages.length, 1, 'expected exactly one message');
  assert.strictEqual(
    messages[0].text,
    text,
    `UTF-8 corrupted across fragments: ${JSON.stringify(messages[0].text)}`,
  );
  console.log('ok - UTF-8 code point split across fragments survives');
}

async function testAuthReplyNotDeliveredAsEvent() {
  const endpointPath = tmpEndpoint('auth.json');
  // `auth_ok` and the first event leave in ONE TCP write.
  const server = createFakeDaemon(conn => {
    conn.write('{"type":"notice","text":"first"}');
  });
  await listen(server, endpointPath);

  const client = new AgentClient(endpointPath);
  const messages = [];
  client.on('message', m => messages.push(m));
  const connected = new Promise(r => client.on('connect', r));
  client.connect();
  await connected;
  await delay(150);
  client.dispose();
  await new Promise(r => server.close(r));

  assert.deepStrictEqual(
    messages.map(m => m.type),
    ['notice'],
    `auth reply leaked as an event or event lost: ${JSON.stringify(messages)}`,
  );
  console.log('ok - auth_ok is consumed by the handshake, first event delivered');
}

async function testDisposeDuringConnect() {
  const endpointPath = tmpEndpoint('dispose.json');
  const received = [];
  const server = createFakeDaemon(conn => {
    conn.on('data', d => received.push(d.toString()));
  });
  await listen(server, endpointPath);

  const errors = [];
  const onUncaught = e => errors.push(e);
  process.on('uncaughtException', onUncaught);

  const client = new AgentClient(endpointPath);
  let connectEmitted = false;
  client.on('connect', () => {
    connectEmitted = true;
  });
  client.sendCommand({type: 'queued-while-connecting'});
  // Dispose synchronously before the async connect + handshake completes.
  client.dispose();
  await delay(200);
  process.removeListener('uncaughtException', onUncaught);
  await new Promise(r => server.close(r));

  assert.strictEqual(connectEmitted, false, "'connect' emitted after dispose()");
  assert.deepStrictEqual(
    errors,
    [],
    `uncaught exception after dispose: ${errors.map(e => e.message)}`,
  );
  assert.strictEqual(
    received.join(''),
    '',
    'queued command was written to the socket after dispose()',
  );
  console.log('ok - dispose() during connect neither emits nor writes');
}

function testDefaultEndpointPathHonorsEnv() {
  const oldLocal = process.env.KISS_SORCAR_LOCAL;
  const oldHome = process.env.KISS_HOME;
  try {
    process.env.KISS_SORCAR_LOCAL = '/tmp/custom-explicit.json';
    process.env.KISS_HOME = '/tmp/custom-kiss-home';
    let c = new AgentClient();
    assert.strictEqual(
      c.endpointPath,
      '/tmp/custom-explicit.json',
      'KISS_SORCAR_LOCAL override ignored',
    );
    c.dispose();

    delete process.env.KISS_SORCAR_LOCAL;
    c = new AgentClient();
    assert.strictEqual(
      c.endpointPath,
      path.join('/tmp/custom-kiss-home', 'sorcar-local.json'),
      'KISS_HOME override ignored',
    );
    c.dispose();

    delete process.env.KISS_HOME;
    c = new AgentClient();
    assert.strictEqual(
      c.endpointPath,
      path.join(os.homedir(), '.kiss', 'sorcar-local.json'),
      'default endpoint path wrong',
    );
    c.dispose();
  } finally {
    if (oldLocal !== undefined) process.env.KISS_SORCAR_LOCAL = oldLocal;
    else delete process.env.KISS_SORCAR_LOCAL;
    if (oldHome !== undefined) process.env.KISS_HOME = oldHome;
    else delete process.env.KISS_HOME;
  }
  console.log('ok - default endpoint path honors KISS_SORCAR_LOCAL / KISS_HOME');
}

async function testRejectedTokenNeverConnects() {
  const endpointPath = tmpEndpoint('reject.json');
  const received = [];
  let handshakes = 0;
  const server = createFakeDaemon(
    conn => {
      conn.on('data', d => received.push(d.toString()));
    },
    {auth: 'remote'},
  );
  server.on('handshake', () => {
    handshakes += 1;
  });
  await listen(server, endpointPath);

  const client = new AgentClient(endpointPath, {
    reconnectBaseMs: 50,
    reconnectMaxMs: 50,
  });
  let connectEmitted = false;
  client.on('connect', () => {
    connectEmitted = true;
  });
  client.sendCommand({type: 'getModels'});
  await delay(400);
  client.dispose();
  await new Promise(r => server.close(r));

  assert.strictEqual(connectEmitted, false, "'connect' despite auth_ok local:false");
  assert.strictEqual(received.length, 0, 'command delivered to a non-local session');
  assert.ok(handshakes >= 2, `expected reconnect attempts, saw ${handshakes}`);
  console.log('ok - a daemon that denies local status is dropped and retried');
}

async function testLargeFramesBothWays() {
  const endpointPath = tmpEndpoint('large.json');
  const big = 'y'.repeat(70_000);
  const received = [];
  const server = createFakeDaemon(conn => {
    conn.on('data', d => {
      received.push(JSON.parse(d.toString()));
      conn.write(JSON.stringify({type: 'echo', text: big}));
    });
  });
  await listen(server, endpointPath);

  const client = new AgentClient(endpointPath);
  const messages = [];
  client.on('message', m => messages.push(m));
  client.sendCommand({type: 'big', text: big});
  await delay(400);
  client.dispose();
  await new Promise(r => server.close(r));

  assert.strictEqual(received.length, 1, 'server did not get the big command');
  assert.strictEqual(received[0].text.length, big.length);
  assert.strictEqual(messages.length, 1, 'client did not get the big reply');
  assert.strictEqual(messages[0].text.length, big.length);
  console.log('ok - 70 KB frames round-trip in both directions');
}

(async () => {
  await testUtf8SplitAcrossFragments();
  await testAuthReplyNotDeliveredAsEvent();
  await testDisposeDuringConnect();
  testDefaultEndpointPathHonorsEnv();
  await testRejectedTokenNeverConnects();
  await testLargeFramesBothWays();
  console.log('agentClientTransport.test.js passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
