// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Review finding 6: checkForExtensionUpdate() stored its cooldown cache
// under os.homedir()/.kiss regardless of $KISS_HOME, while every other
// extension path resolves the state directory through kissHomeDir().
// The default cache path must honor $KISS_HOME (and fall back to
// ~/.kiss when it is unset).

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');

const {checkForExtensionUpdate} = require('../src/UpdateChecker.js');

function startPypiStub(payload) {
  const state = {payload, hits: 0, server: null, url: ''};
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      state.hits += 1;
      const body = Buffer.from(JSON.stringify(state.payload), 'utf-8');
      res.statusCode = 200;
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Content-Length', String(body.length));
      res.end(body);
    });
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const {port} = server.address();
      state.server = server;
      state.url = `http://127.0.0.1:${port}/pypi/kiss-agent-framework/json`;
      resolve(state);
    });
  });
}

function stopStub(state) {
  return new Promise(resolve => state.server.close(() => resolve()));
}

async function runCheck(stubUrl) {
  return checkForExtensionUpdate({
    pypiUrl: stubUrl,
    cooldownMs: 60_000,
    currentVersion: '2026.6.30',
    notify: () => {},
    now: () => 1_000_000,
  });
}

async function main() {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rrupd-'));
  const fakeHome = path.join(tmpRoot, 'home');
  const customKissHome = path.join(tmpRoot, 'custom-kiss-home');
  fs.mkdirSync(fakeHome, {recursive: true});
  fs.mkdirSync(customKissHome, {recursive: true});
  const savedHome = process.env.HOME;
  const savedUserProfile = process.env.USERPROFILE;
  const savedKissHome = process.env.KISS_HOME;
  // os.homedir() reads $HOME on POSIX, so the fallback path is
  // sandboxed too — the test never touches the real ~/.kiss.
  process.env.HOME = fakeHome;
  process.env.USERPROFILE = fakeHome;
  const stub = await startPypiStub({info: {version: '2099.1.1'}});
  try {
    // With KISS_HOME set, the cooldown cache must live under it.
    process.env.KISS_HOME = customKissHome;
    let result = await runCheck(stub.url);
    assert.strictEqual(result.checked, true, `check failed: ${result.reason}`);
    const customCache = path.join(customKissHome, '.update-check.json');
    assert.ok(
      fs.existsSync(customCache),
      `cooldown cache missing from $KISS_HOME: ${customCache}`,
    );
    assert.ok(
      !fs.existsSync(path.join(fakeHome, '.kiss', '.update-check.json')),
      'cooldown cache leaked into ~/.kiss despite $KISS_HOME',
    );
    console.log('  ok - cooldown cache lives under $KISS_HOME when set');

    // A rerun inside the cooldown window must replay the cached
    // decision from the SAME $KISS_HOME path (no extra PyPI hit).
    const hitsBefore = stub.hits;
    result = await runCheck(stub.url);
    assert.ok(
      result.reason === 'cooldown' || result.reason === 'cooldown-replay',
      `expected a cooldown skip, got: ${result.reason}`,
    );
    assert.strictEqual(stub.hits, hitsBefore, 'cooldown must skip PyPI');
    console.log('  ok - cooldown replay reads the $KISS_HOME cache');

    // Without KISS_HOME the default falls back to ~/.kiss.
    delete process.env.KISS_HOME;
    result = await runCheck(stub.url);
    assert.strictEqual(result.checked, true, `check failed: ${result.reason}`);
    assert.ok(
      fs.existsSync(path.join(fakeHome, '.kiss', '.update-check.json')),
      'cooldown cache missing from the ~/.kiss fallback',
    );
    console.log('  ok - cooldown cache falls back to ~/.kiss when unset');
  } finally {
    await stopStub(stub);
    if (savedHome === undefined) delete process.env.HOME;
    else process.env.HOME = savedHome;
    if (savedUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = savedUserProfile;
    if (savedKissHome === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = savedKissHome;
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  }
  console.log('rr_review_update_cooldown_kiss_home: all assertions passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
