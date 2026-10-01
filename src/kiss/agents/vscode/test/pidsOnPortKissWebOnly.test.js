// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: pidsOnPort() (the source of the PIDs that
// killProcessOnPort() SIGTERMs/SIGKILLs before a daemon restart) must
// name only kiss-web processes.
//
// The bug: `lsof -ti tcp:8787 -sTCP:LISTEN` lists EVERY listener on the
// port.  On macOS a VS Code Remote-SSH window forwarding a remote
// kiss-web binds 127.0.0.1:8787 inside VS Code's own extension host, so
// a daemon restart would have killed that extension host (and with it
// the remote window).  The fix filters the lsof PIDs through
// `ps -o pid=,command=` and keeps those whose command line names
// kiss-web.
//
// This test runs the REAL compiled pidsOnPort against a fake `lsof`
// (real lsof needs a real listener on a fixed port and its output format
// differs across platforms) and the REAL `ps`, with two live stand-in
// processes: one whose argv contains "kiss-web" and one that does not.
// Both are plain `node` children of this test, so nothing real is
// touched; pidsOnPort() only reads, it never signals.

const assert = require('assert');
const {spawn, spawnSync} = require('child_process');
const fs = require('fs');
const Module = require('module');
const os = require('os');
const path = require('path');

const OUT = path.join(__dirname, '..', 'out', 'DependencyInstaller.js');
if (!fs.existsSync(OUT)) {
  console.log(
    'SKIP: out/DependencyInstaller.js missing — run `npm run compile`',
  );
  process.exit(0);
}
if (process.platform === 'win32') {
  console.log('SKIP: lsof/ps are not used on Windows');
  process.exit(0);
}

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
const {pidsOnPort} = require(OUT);
assert.strictEqual(typeof pidsOnPort, 'function');

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-pidsonport-'));
const fakeBin = path.join(tmpRoot, 'bin');
fs.mkdirSync(fakeBin);
const pidsFile = path.join(tmpRoot, 'pids.txt');

// Two idle stand-ins: the "daemon" carries kiss-web on its command line
// (like `.venv/bin/kiss-web` does), the "forwarder" does not.
const idle = 'setInterval(() => {}, 1000); process.stdout.write("READY\\n");';
function startStandIn(extraArgv) {
  const child = spawn(process.execPath, ['-e', idle, '--', ...extraArgv], {
    stdio: ['ignore', 'pipe', 'ignore'],
  });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('stand-in never reported readiness')),
      10000,
    );
    child.stdout.on('data', d => {
      if (d.toString().includes('READY')) {
        clearTimeout(timer);
        resolve(child);
      }
    });
    child.on('exit', code => {
      clearTimeout(timer);
      reject(new Error(`stand-in exited early (code ${code})`));
    });
  });
}

function writeFakeLsof(body) {
  const fakeLsof = path.join(fakeBin, 'lsof');
  fs.writeFileSync(fakeLsof, `#!/bin/sh\n${body}\n`);
  fs.chmodSync(fakeLsof, 0o755);
}

async function main() {
  const daemon = await startStandIn(['/tmp/fake/.venv/bin/kiss-web']);
  const forwarder = await startStandIn(['vscode-port-forward']);
  // Mentions kiss-web only inside an argument: not a daemon.
  const lookalike = await startStandIn(['--directory', '/x/kiss-web-client']);
  process.env.PATH = `${fakeBin}:${process.env.PATH}`;
  try {
    // Real `ps` must see the stand-ins the way pidsOnPort() will.
    const ps = spawnSync('ps', ['-o', 'pid=,command=', '-p',
      `${daemon.pid},${forwarder.pid},${lookalike.pid}`], {encoding: 'utf8'});
    assert.strictEqual(ps.status, 0, `ps failed: ${ps.stderr}`);
    assert.ok(ps.stdout.includes('kiss-web-client'), `ps output: ${ps.stdout}`);

    // 1. All listed by lsof -> only the kiss-web entry point survives.
    fs.writeFileSync(pidsFile, `${daemon.pid}\n${forwarder.pid}\n${lookalike.pid}\n`);
    writeFakeLsof(`cat "${pidsFile}"; exit 0`);
    assert.deepStrictEqual(await pidsOnPort(8787), [String(daemon.pid)]);

    // 2. Only the forwarder listed -> nothing to kill.
    fs.writeFileSync(pidsFile, `${forwarder.pid}\n`);
    assert.deepStrictEqual(await pidsOnPort(8787), []);

    // 3. lsof reports no listener (exit 1, as real lsof does) -> [].
    writeFakeLsof('exit 1');
    assert.deepStrictEqual(await pidsOnPort(8787), []);

    // 4. lsof succeeds with empty output -> [] without running ps.
    writeFakeLsof('exit 0');
    assert.deepStrictEqual(await pidsOnPort(8787), []);

    // 5. lsof names a PID that no longer exists -> ps fails -> [].
    forwarder.kill('SIGKILL');
    await new Promise(resolve => forwarder.once('exit', resolve));
    fs.writeFileSync(pidsFile, `${forwarder.pid}\n`);
    writeFakeLsof(`cat "${pidsFile}"; exit 0`);
    assert.deepStrictEqual(await pidsOnPort(8787), []);

    // 6. lsof itself is missing -> [] (the catch branch).
    fs.unlinkSync(path.join(fakeBin, 'lsof'));
    process.env.PATH = fakeBin;
    assert.deepStrictEqual(await pidsOnPort(8787), []);
  } finally {
    daemon.kill('SIGKILL');
    lookalike.kill('SIGKILL');
    if (forwarder.exitCode === null) forwarder.kill('SIGKILL');
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  }
  console.log('  ok - pidsOnPort names only kiss-web listeners');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
