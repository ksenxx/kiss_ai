// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for AgentClient transport correctness over a real UDS:
// 1. UTF-8 code points split across socket chunks must not be corrupted.
// 2. A partial line left over from a dead connection must not contaminate
//    the next connection's first message.
// 3. dispose() while a connect is in flight must not emit 'connect' or
//    write queued commands to the ended socket.
// 4. The default socket path must honor $KISS_SORCAR_SOCK and $KISS_HOME.

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

function tmpSock(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ac-'));
  return path.join(dir, name);
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function testUtf8SplitAcrossChunks() {
  const sockPath = tmpSock('utf8.sock');
  const text = 'emoji \u{1F600} end';
  const line = Buffer.from(JSON.stringify({type: 'notice', text}) + '\n');
  // Split inside the 4-byte emoji sequence.
  const emojiStart = line.indexOf(Buffer.from('\u{1F600}'));
  const cut = emojiStart + 2;

  const server = net.createServer(conn => {
    conn.write(line.subarray(0, cut));
    setTimeout(() => conn.write(line.subarray(cut)), 30);
  });
  await new Promise(r => server.listen(sockPath, r));

  const client = new AgentClient(sockPath);
  const msg = await new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('no message')), 3000);
    client.on('message', m => {
      clearTimeout(t);
      resolve(m);
    });
    client.connect();
  });
  client.dispose();
  await new Promise(r => server.close(r));
  assert.strictEqual(
    msg.text,
    text,
    `UTF-8 split across chunks corrupted text: ${JSON.stringify(msg.text)}`,
  );
  console.log('ok - UTF-8 code point split across chunks survives intact');
}

async function testStaleBufferClearedOnReconnect() {
  const sockPath = tmpSock('stale.sock');
  let connCount = 0;
  const server = net.createServer(conn => {
    connCount++;
    if (connCount === 1) {
      // Send half a JSON line, then die without the newline.
      conn.write('{"type":"notice","text":"HALF');
      setTimeout(() => conn.destroy(), 30);
    } else {
      conn.write('{"type":"notice","text":"clean"}\n');
    }
  });
  await new Promise(r => server.listen(sockPath, r));

  const client = new AgentClient(sockPath);
  const messages = [];
  client.on('message', m => messages.push(m));
  client.connect();
  // Wait for the reconnect (500ms delay) and second message.
  const deadline = Date.now() + 5000;
  while (messages.length === 0 && Date.now() < deadline) await delay(50);
  client.dispose();
  await new Promise(r => server.close(r));
  assert.ok(messages.length >= 1, 'expected a message on second connection');
  assert.strictEqual(
    messages[0].text,
    'clean',
    'first message on new connection was contaminated by stale buffer: ' +
      JSON.stringify(messages[0]),
  );
  console.log('ok - partial line from dead connection does not leak');
}

async function testDisposeDuringConnect() {
  const sockPath = tmpSock('dispose.sock');
  const received = [];
  const server = net.createServer(conn => {
    conn.on('data', d => received.push(d.toString()));
  });
  await new Promise(r => server.listen(sockPath, r));

  const errors = [];
  const onUncaught = e => errors.push(e);
  process.on('uncaughtException', onUncaught);

  const client = new AgentClient(sockPath);
  let connectEmitted = false;
  client.on('connect', () => {
    connectEmitted = true;
  });
  client.sendCommand({type: 'queued-while-connecting'});
  // Dispose synchronously before the async UDS connect completes.
  client.dispose();
  await delay(200);
  process.removeListener('uncaughtException', onUncaught);
  await new Promise(r => server.close(r));

  assert.strictEqual(
    connectEmitted,
    false,
    "'connect' emitted after dispose()",
  );
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

function testDefaultSockPathHonorsEnv() {
  const oldSock = process.env.KISS_SORCAR_SOCK;
  const oldHome = process.env.KISS_HOME;
  try {
    process.env.KISS_SORCAR_SOCK = '/tmp/custom-explicit.sock';
    process.env.KISS_HOME = '/tmp/custom-kiss-home';
    let c = new AgentClient();
    assert.strictEqual(
      c._sockPath,
      '/tmp/custom-explicit.sock',
      'KISS_SORCAR_SOCK override ignored',
    );
    c.dispose();

    delete process.env.KISS_SORCAR_SOCK;
    c = new AgentClient();
    assert.strictEqual(
      c._sockPath,
      path.join('/tmp/custom-kiss-home', 'sorcar.sock'),
      'KISS_HOME override ignored',
    );
    c.dispose();

    delete process.env.KISS_HOME;
    c = new AgentClient();
    assert.strictEqual(
      c._sockPath,
      path.join(os.homedir(), '.kiss', 'sorcar.sock'),
      'default sock path wrong',
    );
    c.dispose();
  } finally {
    if (oldSock !== undefined) process.env.KISS_SORCAR_SOCK = oldSock;
    else delete process.env.KISS_SORCAR_SOCK;
    if (oldHome !== undefined) process.env.KISS_HOME = oldHome;
    else delete process.env.KISS_HOME;
  }
  console.log('ok - default socket path honors KISS_SORCAR_SOCK / KISS_HOME');
}

(async () => {
  if (process.platform === 'win32') {
    console.log('SKIP: UDS tests require a POSIX platform');
    return;
  }
  await testUtf8SplitAcrossChunks();
  await testStaleBufferClearedOnReconnect();
  await testDisposeDuringConnect();
  testDefaultSockPathHonorsEnv();
  console.log('agentClientTransport.test.js passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
