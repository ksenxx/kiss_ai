// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for ``downloadFile`` in ``DependencyInstaller``.
//
// Bug locked in: a mid-body connection failure (server truncates the
// response before the advertised Content-Length) left the returned
// promise PENDING FOREVER — ``pipe`` does not propagate source errors
// to the write stream, and once the socket is gone the 60 s inactivity
// timeout can never fire.  In production this wedged installUv /
// installNode and the "KISS Sorcar: Setting up" progress notification
// indefinitely.  The promise must settle (reject) promptly and clean up
// the partial file.
//
// The test runs a REAL local TLS server (self-signed cert generated
// with openssl; skipped when openssl is unavailable) and exercises the
// real compiled helper — no mocks.
//
// Run directly with ``node`` (after ``npm run compile``):
//
//     node src/kiss/agents/vscode/test/downloadFileTruncatedBody.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const https = require('https');
const os = require('os');
const path = require('path');
const {execFileSync} = require('child_process');
const Module = require('module');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-dl-'));

// --- Self-signed cert (skip when openssl is unavailable) -----------------
const keyPath = path.join(tmpDir, 'key.pem');
const certPath = path.join(tmpDir, 'cert.pem');
try {
  execFileSync(
    'openssl',
    [
      'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
      '-keyout', keyPath, '-out', certPath,
      '-days', '1', '-subj', '/CN=127.0.0.1',
    ],
    {stdio: 'ignore', timeout: 30000},
  );
} catch {
  console.log('  ok - SKIPPED (openssl unavailable on this host)');
  fs.rmSync(tmpDir, {recursive: true, force: true});
  process.exit(0);
}

// downloadFile has no CA injection point; trust the self-signed cert for
// this test process only.
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

// --- vscode stub so the compiled module loads outside the host -----------
const stubPath = path.join(__dirname, '_vscode-stub.js');
fs.writeFileSync(
  stubPath,
  `'use strict';\nmodule.exports = global.__kissVscodeStub || {};\n`,
);
global.__kissVscodeStub = {
  workspace: {isTrusted: false, getConfiguration: () => ({get: () => undefined})},
  ProgressLocation: {Notification: 15},
  Uri: {joinPath: () => ({fsPath: ''})},
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

const sourcePath = path.join(__dirname, '..', 'out', 'DependencyInstaller.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`npm run compile\` first`,
);
const {downloadFile} = require(sourcePath);
assert.strictEqual(
  typeof downloadFile,
  'function',
  'downloadFile must be exported from the compiled DependencyInstaller',
);

const FULL_BODY = Buffer.from('kiss-sorcar-download-test-payload\n'.repeat(64));

const server = https.createServer(
  {key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath)},
  (req, res) => {
    if (req.url === '/truncated') {
      // Advertise far more than we send, then kill the socket mid-body.
      res.writeHead(200, {'Content-Length': String(FULL_BODY.length * 100)});
      res.write(FULL_BODY);
      setTimeout(() => res.destroy(), 30);
      return;
    }
    res.writeHead(200, {'Content-Length': String(FULL_BODY.length)});
    res.end(FULL_BODY);
  },
);

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function runTests() {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  const base = `https://127.0.0.1:${port}`;

  // --- Bug: truncated body must settle the promise promptly ------------
  const dest1 = path.join(tmpDir, 'truncated.bin');
  const outcome = await Promise.race([
    downloadFile(`${base}/truncated`, dest1).then(
      () => 'resolved',
      () => 'rejected',
    ),
    sleep(5000).then(() => 'hung'),
  ]);
  assert.strictEqual(
    outcome,
    'rejected',
    `BUG: downloadFile must reject on a mid-body connection failure — ` +
      `got '${outcome}' (a 'hung' outcome is the production wedge: the ` +
      `install and its progress notification stall forever)`,
  );
  assert.ok(
    !fs.existsSync(dest1),
    'the partial download must be cleaned up on mid-body failure',
  );

  // --- Success path still works ----------------------------------------
  const dest2 = path.join(tmpDir, 'ok.bin');
  await downloadFile(`${base}/ok`, dest2);
  assert.ok(fs.existsSync(dest2), 'successful download must exist');
  assert.deepStrictEqual(
    fs.readFileSync(dest2),
    FULL_BODY,
    'successful download must match the served body',
  );

  console.log('\nAll downloadFile truncated-body tests passed');
}

runTests().then(
  () => {
    server.close();
    fs.rmSync(tmpDir, {recursive: true, force: true});
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err);
    server.close();
    fs.rmSync(tmpDir, {recursive: true, force: true});
    process.exit(1);
  },
);
