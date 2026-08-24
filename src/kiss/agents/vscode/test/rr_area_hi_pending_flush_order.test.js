// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// H-RC2: AgentClient must flush commands queued while the daemon was
// unreachable BEFORE it emits 'connect'. Connect handlers immediately
// write fresh commands (the sidebar re-sends getModels on every
// connect); if those overtook older queued frames, the daemon's reply
// to the fresh getModels could repaint the model picker with the model
// a queued selectModel was about to replace.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

if (process.platform === 'win32') {
  console.log('skipped on win32 (UDS test)');
  process.exit(0);
}

const {AgentClient} = require(
  path.join(__dirname, '..', 'out', 'AgentClient.js'),
);

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rrhi-flush-'));
const sockPath = path.join(tmpDir, 'sorcar.sock');

function waitFor(predicate, message, timeout = 5000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (predicate()) return resolve();
      if (Date.now() - start > timeout) {
        return reject(new Error(message));
      }
      setTimeout(tick, 10);
    };
    tick();
  });
}

async function main() {
  const received = [];
  const server = net.createServer(sock => {
    let buf = '';
    sock.on('data', chunk => {
      buf += chunk.toString();
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.trim()) received.push(JSON.parse(line));
      }
    });
    sock.on('error', () => {});
  });
  await new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(sockPath, err => (err ? reject(err) : resolve()));
  });

  const client = new AgentClient(sockPath);

  // The sidebar's real connect handler sends init commands; mirror it.
  client.on('connect', () => {
    client.sendCommand({type: 'getModels'});
  });

  // Queued while disconnected: the user picked a model during an outage.
  client.sendCommand({type: 'selectModel', model: 'queued-model'});

  await waitFor(
    () => received.length >= 2,
    'daemon never received both commands',
  );

  assert.deepStrictEqual(
    received.map(m => m.type),
    ['selectModel', 'getModels'],
    'queued frames must be delivered before connect-handler commands',
  );

  client.dispose();
  await new Promise(resolve => server.close(resolve));
  fs.rmSync(tmpDir, {recursive: true, force: true});
  console.log('rr_area_hi_pending_flush_order: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
