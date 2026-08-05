// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {extensionFileSize, pathExists, isReloadReady} = require('../src/reloadGuard');

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

const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-reloadguard-'));
const extJsPath = path.join(workDir, 'out', 'extension.js');
const sockPath = path.join(workDir, 'sorcar.sock');
fs.mkdirSync(path.dirname(extJsPath), {recursive: true});

function setExtJs(contents) {
  fs.writeFileSync(extJsPath, contents);
}
function removeExtJs() {
  if (fs.existsSync(extJsPath)) fs.rmSync(extJsPath);
}
function setSocket() {
  fs.writeFileSync(sockPath, '');
}
function removeSocket() {
  if (fs.existsSync(sockPath)) fs.rmSync(sockPath);
}

function shouldReload(state) {
  const {codeReady, socketUp, codeStableForMs, waitedMs, graceMs, timeoutMs} =
    state;
  return (
    (codeReady && (socketUp || codeStableForMs >= graceMs)) ||
    waitedMs >= timeoutMs
  );
}

function runSettleLoop({intervalMs, graceMs, timeoutMs, onPoll}) {
  let prevSize = -1;
  let waited = 0;
  let codeReadySince = -1;
  for (let guard = 0; guard < 1000; guard++) {
    waited += intervalMs;
    if (onPoll) onPoll(waited);
    const {codeReady, socketUp, size} = isReloadReady(
      extJsPath,
      sockPath,
      prevSize,
    );
    prevSize = size;
    if (codeReady && codeReadySince < 0) codeReadySince = waited;
    const codeStableForMs = codeReadySince < 0 ? 0 : waited - codeReadySince;
    if (
      shouldReload({
        codeReady,
        socketUp,
        codeStableForMs,
        waitedMs: waited,
        graceMs,
        timeoutMs,
      })
    ) {
      return waited;
    }
  }
  throw new Error('reload never fired within the guard bound');
}

const INTERVAL = 500;
const GRACE = 3_000;
const TIMEOUT = 15_000;

test('extensionFileSize returns -1 for a missing file', () => {
  removeExtJs();
  assert.strictEqual(extensionFileSize(extJsPath), -1);
});

test('extensionFileSize returns -1 for a directory', () => {
  assert.strictEqual(extensionFileSize(path.dirname(extJsPath)), -1);
});

test('extensionFileSize returns the byte size of a regular file', () => {
  setExtJs('console.log(1);');
  assert.strictEqual(extensionFileSize(extJsPath), 'console.log(1);'.length);
});

test('pathExists reflects socket presence', () => {
  removeSocket();
  assert.strictEqual(pathExists(sockPath), false);
  setSocket();
  assert.strictEqual(pathExists(sockPath), true);
});

test('isReloadReady: missing entry file is never code-ready', () => {
  removeExtJs();
  setSocket();
  const r = isReloadReady(extJsPath, sockPath, -1);
  assert.strictEqual(r.codeReady, false);
  assert.strictEqual(r.ready, false);
  assert.strictEqual(r.size, -1);
});

test('isReloadReady: empty entry file is never code-ready', () => {
  setExtJs('');
  setSocket();
  const r = isReloadReady(extJsPath, sockPath, 0);
  assert.strictEqual(r.codeReady, false);
  assert.strictEqual(r.ready, false);
});

test('isReloadReady: a still-growing file is not code-ready', () => {
  setExtJs('abc');
  const r = isReloadReady(extJsPath, sockPath, 1);
  assert.strictEqual(r.codeReady, false);
});

test('isReloadReady: stable file WITHOUT socket is code-ready but not ready', () => {
  setExtJs('stable-bytes');
  removeSocket();
  const size = extensionFileSize(extJsPath);
  const r = isReloadReady(extJsPath, sockPath, size);
  assert.strictEqual(r.codeReady, true, 'code should be ready');
  assert.strictEqual(r.socketUp, false, 'socket should be down');
  assert.strictEqual(r.ready, false, 'strict ready requires the socket');
});

test('isReloadReady: stable file WITH socket is fully ready', () => {
  setExtJs('stable-bytes');
  setSocket();
  const size = extensionFileSize(extJsPath);
  const r = isReloadReady(extJsPath, sockPath, size);
  assert.strictEqual(r.codeReady, true);
  assert.strictEqual(r.socketUp, true);
  assert.strictEqual(r.ready, true);
});

test('reload fires immediately once code is stable AND socket is up', () => {
  setExtJs('console.log("ext");');
  setSocket();
  const firedAt = runSettleLoop({
    intervalMs: INTERVAL,
    graceMs: GRACE,
    timeoutMs: TIMEOUT,
    onPoll: () => {},
  });
  assert.strictEqual(firedAt, INTERVAL * 2);
});

test('reload fires after the socket grace when the daemon never returns (deadlock broken)', () => {
  setExtJs('console.log("new ext");');
  removeSocket();
  const firedAt = runSettleLoop({
    intervalMs: INTERVAL,
    graceMs: GRACE,
    timeoutMs: TIMEOUT,
    onPoll: () => {
      removeSocket();
    },
  });
  assert.strictEqual(firedAt, INTERVAL * 2 + GRACE);
  assert.ok(firedAt < TIMEOUT, 'must fire well before the absolute timeout');
});

test('a late-returning socket triggers reload before the grace elapses', () => {
  setExtJs('console.log("new ext");');
  removeSocket();
  const firedAt = runSettleLoop({
    intervalMs: INTERVAL,
    graceMs: GRACE,
    timeoutMs: TIMEOUT,
    onPoll: waited => {
      if (waited >= 1_000) setSocket();
      else removeSocket();
    },
  });
  assert.ok(
    firedAt <= 1_500,
    `expected an early reload on socket return, got ${firedAt}`,
  );
  assert.ok(firedAt < INTERVAL * 2 + GRACE);
});

test('absolute timeout reloads even if the code never settles', () => {
  removeExtJs();
  removeSocket();
  let n = 0;
  const firedAt = runSettleLoop({
    intervalMs: INTERVAL,
    graceMs: GRACE,
    timeoutMs: TIMEOUT,
    onPoll: () => {
      n += 1;
      setExtJs('x'.repeat(n));
    },
  });
  assert.strictEqual(firedAt, TIMEOUT);
});

fs.rmSync(workDir, {recursive: true, force: true});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`\n${f.name}:\n`, f.err);
  }
  process.exit(1);
}
