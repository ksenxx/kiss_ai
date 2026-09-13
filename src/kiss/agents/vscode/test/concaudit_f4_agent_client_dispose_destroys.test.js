// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: AgentClient.dispose() must tear the
// connection down even when the daemon has stopped reading.
//
// The bug: dispose() called socket.end(), which is a GRACEFUL close --
// Node keeps the socket (and the process) alive until every buffered
// byte has been written.  A daemon that is wedged and no longer reads
// its socket, plus one large buffered command (a prompt with base64
// attachments), meant the disposed client's socket lingered for ever:
// the extension host kept a dead connection open per disposed view.
//
// This test runs the REAL compiled AgentClient in a child process
// against a Unix socket server in this process that never reads.  The
// child sends a 64 MiB command, disposes the client and then has
// nothing else keeping its event loop alive -- so it exits promptly
// only if dispose() really destroyed the socket.  A regressed build
// never exits (the write buffer can never drain) and the test's
// deadline fails it.

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

const OUT_AGENT_CLIENT = path.join(__dirname, '..', 'out', 'AgentClient.js');
if (!fs.existsSync(OUT_AGENT_CLIENT)) {
  console.log('SKIP: out/AgentClient.js missing — run `npm run compile`');
  process.exit(0);
}
if (process.platform === 'win32') {
  console.log('SKIP: Unix domain sockets only');
  process.exit(0);
}

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ac-dispose-'));
const sockPath = path.join(tmpDir, 'daemon.sock');

const CHILD = `
'use strict';
const {AgentClient} = require(process.env.KISS_TEST_MODULE);
const client = new AgentClient(process.env.KISS_TEST_SOCK);
client.on('connect', () => {
  // One command far larger than any socket buffer, against a peer that
  // never reads: most of it stays queued inside the client's socket.
  client.sendCommand({type: 'run', prompt: 'x'.repeat(64 * 1024 * 1024)});
  setImmediate(() => {
    client.dispose();
    process.stdout.write('DISPOSED\\n');
    // Nothing else holds the event loop: the process exits as soon as
    // the (destroyed) socket handle is gone.
  });
});
client.connect();
`;

async function main() {
  const connections = [];
  const server = net.createServer(conn => {
    // A wedged daemon: accept, then never read a byte.
    conn.pause();
    conn.on('error', () => {});
    connections.push(conn);
  });
  await new Promise(r => server.listen(sockPath, r));

  const started = Date.now();
  const result = await new Promise(resolve => {
    const child = spawn(process.execPath, ['-e', CHILD], {
      stdio: ['ignore', 'pipe', 'inherit'],
      env: {
        ...process.env,
        KISS_TEST_MODULE: OUT_AGENT_CLIENT,
        KISS_TEST_SOCK: sockPath,
      },
    });
    let out = '';
    child.stdout.on('data', d => {
      out += d.toString();
    });
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      resolve({out, code: null, timedOut: true});
    }, 15000);
    child.on('close', code => {
      clearTimeout(timer);
      resolve({out, code, timedOut: false});
    });
  });

  assert.ok(
    result.out.includes('DISPOSED'),
    `child never reached dispose(): ${result.out}`,
  );
  assert.ok(
    !result.timedOut,
    'the disposed client kept the process alive for 15s: dispose() is ' +
      'waiting for a non-reading daemon to drain the write buffer again',
  );
  assert.strictEqual(result.code, 0, `child exited ${result.code}`);
  console.log(
    `  ok - disposed client with a wedged peer exited in ${Date.now() - started}ms`,
  );

  for (const c of connections) c.destroy();
  await new Promise(r => server.close(r));
  fs.rmSync(tmpDir, {recursive: true, force: true});
}

main()
  .then(() => {
    console.log(
      'concaudit_f4_agent_client_dispose_destroys: all assertions passed',
    );
  })
  .catch(err => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
