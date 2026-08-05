// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end test for the SHA-256 manifest fetchers in
// DependencyInstaller: both fetchUvStyleSha256 and fetchNodeSha256 must
// share one HTTPS text transport (httpsGetText) and parse their
// respective manifest formats served by a real local TLS server.

'use strict';

const assert = require('assert');
const fs = require('fs');
const https = require('https');
const os = require('os');
const path = require('path');
const {execFileSync} = require('child_process');
const Module = require('module');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sha-'));
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

process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

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
const {httpsGetText, fetchUvStyleSha256, fetchNodeSha256} =
  require(sourcePath);
assert.strictEqual(typeof httpsGetText, 'function');
assert.strictEqual(typeof fetchUvStyleSha256, 'function');
assert.strictEqual(typeof fetchNodeSha256, 'function');

const DIGEST_A = 'a'.repeat(64);
const DIGEST_B = 'b'.repeat(64);

const server = https.createServer(
  {key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath)},
  (req, res) => {
    if (req.url === '/tool.tar.gz.sha256') {
      res.writeHead(200);
      res.end(`${DIGEST_A}  tool.tar.gz\n`);
    } else if (req.url === '/SHASUMS256.txt') {
      res.writeHead(200);
      res.end(
        `${DIGEST_A}  node-v0.0.0-linux-x64.tar.gz\n` +
          `${DIGEST_B}  node-v0.0.0-darwin-arm64.tar.gz\n`,
      );
    } else {
      res.writeHead(404);
      res.end('not found');
    }
  },
);

async function main() {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `https://127.0.0.1:${server.address().port}`;

  const text = await httpsGetText(`${base}/SHASUMS256.txt`);
  assert.ok(text && text.includes(DIGEST_B), 'httpsGetText returns body');
  assert.strictEqual(
    await httpsGetText(`${base}/missing`),
    null,
    'httpsGetText returns null on non-200',
  );
  assert.strictEqual(
    await httpsGetText('not a url at all'),
    null,
    'httpsGetText resolves null (never rejects) on a malformed URL',
  );
  assert.strictEqual(
    await httpsGetText('http://127.0.0.1:1/insecure'),
    null,
    'httpsGetText resolves null on a non-HTTPS URL',
  );

  assert.strictEqual(
    await fetchUvStyleSha256(`${base}/tool.tar.gz`),
    DIGEST_A,
    'uv-style .sha256 manifest parsed',
  );
  assert.strictEqual(
    await fetchUvStyleSha256(`${base}/nothere.tar.gz`),
    null,
    'uv-style fetch returns null on 404',
  );

  assert.strictEqual(
    await fetchNodeSha256(
      'node-v0.0.0-darwin-arm64.tar.gz',
      `${base}/SHASUMS256.txt`,
    ),
    DIGEST_B,
    'node SHASUMS256 manifest parsed per asset',
  );
  assert.strictEqual(
    await fetchNodeSha256('node-unknown.tar.gz', `${base}/SHASUMS256.txt`),
    null,
    'unlisted asset yields null',
  );

  server.close();
  fs.rmSync(tmpDir, {recursive: true, force: true});
  console.log('  ok - sha fetchers share one HTTPS transport and parse');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
