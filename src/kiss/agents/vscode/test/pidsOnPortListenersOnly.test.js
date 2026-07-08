// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for ``pidsOnPort`` — the helper behind
// ``killProcessOnPort`` in ``DependencyInstaller``.
//
// Contract locked in here: ``pidsOnPort(port)`` returns only processes
// LISTENING on the port, never TCP *clients* connected to it.  The old
// ``lsof -ti :port`` (no ``-sTCP:LISTEN`` filter) also returned client
// PIDs — cloudflared proxying the remote tunnel, or a browser holding
// a WebSocket to localhost:8787 — and ``restartKissWebDaemon`` would
// SIGTERM/SIGKILL them along with the daemon.
//
// The test spawns a REAL listener (a child node process) and connects
// a REAL client from this test process, then asserts the child PID is
// reported and this process's PID is not.
//
// Runs against the compiled extension under ``out/``:
//
//     tsc -p . && node test/pidsOnPortListenersOnly.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const path = require('path');
const {spawn, execFileSync} = require('child_process');
const Module = require('module');

// DependencyInstaller requires 'vscode' at load time; redirect to the
// shared on-disk stub (none of the code under test touches vscode).
// ``_vscode-stub.js`` is a git-tracked fixture shared by tests running
// in parallel; it already re-exports ``global.__kissVscodeStub || {}`` —
// never rewrite or delete it here (writeFileSync truncates first, racing
// a concurrent ``require('vscode')`` in sibling test processes).
global.__kissVscodeStub = {
  workspace: {isTrusted: false, getConfiguration: () => ({get: () => undefined})},
  Uri: {joinPath: () => ({fsPath: ''})},
  ProgressLocation: {Notification: 15},
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

const sourcePath = path.join(__dirname, '..', 'out', 'DependencyInstaller.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
);
const {pidsOnPort} = require(sourcePath);
assert.strictEqual(
  typeof pidsOnPort,
  'function',
  'pidsOnPort must be exported from the compiled DependencyInstaller',
);

// lsof must exist for this test to be meaningful (it is the tool the
// production code shells out to).
try {
  execFileSync('lsof', ['-v'], {stdio: 'ignore'});
} catch {
  console.log('  ok - SKIPPED (lsof unavailable on this host)');
  console.log('\n1 passed, 0 failed');
  process.exit(0);
}

async function main() {
  // Child process: real TCP listener on an ephemeral port.
  const listenerSrc =
    "const net=require('net');" +
    "const s=net.createServer(c=>{});" +
    "s.listen(0,'127.0.0.1',()=>console.log(s.address().port));" +
    'setTimeout(()=>process.exit(0),30000);';
  const child = spawn(process.execPath, ['-e', listenerSrc], {
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  const port = await new Promise((resolve, reject) => {
    let buf = '';
    child.stdout.on('data', d => {
      buf += d.toString();
      const m = /^(\d+)\n/.exec(buf);
      if (m) resolve(parseInt(m[1], 10));
    });
    child.on('exit', () => reject(new Error('listener died prematurely')));
    setTimeout(() => reject(new Error('listener start timeout')), 10000);
  });

  // Real client connection from THIS process (plays the role of
  // cloudflared / the browser in production).
  const client = net.connect({host: '127.0.0.1', port});
  await new Promise((resolve, reject) => {
    client.once('connect', resolve);
    client.once('error', reject);
  });

  let passed = 0;
  const failures = [];
  function test(name, fn) {
    try {
      fn();
      passed += 1;
      console.log(`  ok - ${name}`);
    } catch (err) {
      failures.push({name, err});
      console.log(`  FAIL - ${name}: ${err && err.message}`);
    }
  }

  const pids = pidsOnPort(port);

  test('reports the listening process', () => {
    assert.ok(
      pids.includes(String(child.pid)),
      `listener pid ${child.pid} missing from ${JSON.stringify(pids)}`,
    );
  });

  test('does NOT report a mere TCP client of the port', () => {
    assert.ok(
      !pids.includes(String(process.pid)),
      `client pid ${process.pid} (this test process) wrongly reported ` +
        `in ${JSON.stringify(pids)} — killProcessOnPort would kill ` +
        'cloudflared / the browser',
    );
  });

  // Find a genuinely free port (bind ephemeral, then release it).
  const freePort = await new Promise(resolve => {
    const free = net.createServer();
    free.listen(0, '127.0.0.1', () => {
      const p = free.address().port;
      free.close(() => resolve(p));
    });
  });
  test('returns empty for a port nobody listens on', () => {
    assert.deepStrictEqual(pidsOnPort(freePort), []);
  });

  client.destroy();
  child.kill('SIGKILL');

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length > 0) {
    for (const f of failures) console.error(`\n${f.name}:\n`, f.err);
    process.exit(1);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
