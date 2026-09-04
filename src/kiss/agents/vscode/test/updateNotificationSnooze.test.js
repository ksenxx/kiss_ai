// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the "Remind me later" snooze: clicking it must
// suppress the update popup for 24 hours (across window reloads AND
// across the 6h PyPI re-fetch), after which the popup returns.  A
// release NEWER than the snoozed one must break through the snooze.

'use strict';

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');

const {
  checkForExtensionUpdate,
  snoozeUpdateNotification,
} = require('../src/UpdateChecker.js');

const HOUR_MS = 60 * 60 * 1000;

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

const tmpDirs = [];
function makeCachePath(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kiss-snooze-${tag}-`));
  tmpDirs.push(dir);
  return path.join(dir, '.update-check.json');
}

let passed = 0;
const failures = [];
async function test(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  FAIL - ${name}: ${err && err.message}`);
  }
}

// One "window activation" of the production check with a fake clock.
function activate(stub, cachePath, nowMs, notified) {
  return checkForExtensionUpdate({
    pypiUrl: stub.url,
    cacheFilePath: cachePath,
    cooldownMs: 6 * HOUR_MS,
    currentVersion: '2026.9.1',
    notify: params => notified.push(params),
    now: () => nowMs,
  });
}

async function testSnoozeSuppressesReloadReplay() {
  const stub = await startPypiStub({info: {version: '2026.9.2'}});
  const cachePath = makeCachePath('replay');
  try {
    const notified = [];
    const t0 = 1_000_000;
    const first = await activate(stub, cachePath, t0, notified);
    assert.strictEqual(first.reason, 'update-available');
    assert.strictEqual(notified.length, 1);

    // User clicks "Remind me later".
    const snooze = snoozeUpdateNotification({
      latest: '2026.9.2',
      cacheFilePath: cachePath,
      now: () => t0,
    });
    assert.strictEqual(snooze.snoozeUntilMs, t0 + 24 * HOUR_MS);
    assert.strictEqual(snooze.snoozedLatest, '2026.9.2');

    // Window reload 1 hour later: inside the fetch cooldown, previously
    // a 'cooldown-replay' popup — now silenced.
    const second = await activate(stub, cachePath, t0 + HOUR_MS, notified);
    assert.strictEqual(second.reason, 'snoozed');
    assert.strictEqual(second.notified, false);
    assert.strictEqual(second.latest, '2026.9.2');
    assert.strictEqual(notified.length, 1, 'snoozed reload must not notify');
    assert.strictEqual(stub.hits, 1, 'snoozed reload must not re-hit PyPI');

    // Cooldown fields must survive the snooze write.
    const cache = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    assert.strictEqual(cache.lastCheckMs, t0);
    assert.strictEqual(cache.lastLatest, '2026.9.2');
  } finally {
    await stopStub(stub);
  }
}

async function testSnoozeSurvivesCooldownRefetch() {
  const stub = await startPypiStub({info: {version: '2026.9.2'}});
  const cachePath = makeCachePath('refetch');
  try {
    const notified = [];
    const t0 = 1_000_000;
    await activate(stub, cachePath, t0, notified);
    snoozeUpdateNotification({
      latest: '2026.9.2',
      cacheFilePath: cachePath,
      now: () => t0,
    });

    // 7 hours later the 6h cooldown has lapsed: PyPI is re-fetched and
    // the cache rewritten, but the 24h snooze must still silence the
    // popup AND survive in the rewritten cache file.
    const later = await activate(stub, cachePath, t0 + 7 * HOUR_MS, notified);
    assert.strictEqual(later.checked, true);
    assert.strictEqual(later.reason, 'snoozed');
    assert.strictEqual(stub.hits, 2, 'cooldown expiry must re-hit PyPI');
    assert.strictEqual(notified.length, 1);

    const cache = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    assert.strictEqual(cache.snoozeUntilMs, t0 + 24 * HOUR_MS);
    assert.strictEqual(cache.snoozedLatest, '2026.9.2');
    assert.strictEqual(cache.lastCheckMs, t0 + 7 * HOUR_MS);

    // A reload right after (inside the new cooldown) is still silent.
    const reload = await activate(stub, cachePath, t0 + 8 * HOUR_MS, notified);
    assert.strictEqual(reload.reason, 'snoozed');
    assert.strictEqual(notified.length, 1);
  } finally {
    await stopStub(stub);
  }
}

async function testSnoozeExpiresAfter24Hours() {
  const stub = await startPypiStub({info: {version: '2026.9.2'}});
  const cachePath = makeCachePath('expiry');
  try {
    const notified = [];
    const t0 = 1_000_000;
    await activate(stub, cachePath, t0, notified);
    snoozeUpdateNotification({
      latest: '2026.9.2',
      cacheFilePath: cachePath,
      now: () => t0,
    });

    // 25 hours later the snooze has expired: the popup returns via a
    // fresh fetch, and the expired snooze is dropped from the cache.
    const after = await activate(stub, cachePath, t0 + 25 * HOUR_MS, notified);
    assert.strictEqual(after.reason, 'update-available');
    assert.strictEqual(after.notified, true);
    assert.strictEqual(notified.length, 2);

    const cache = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    assert.strictEqual(cache.snoozeUntilMs, undefined);
    assert.strictEqual(cache.snoozedLatest, undefined);

    // ... and replays again on the next reload, as before the snooze.
    const reload = await activate(
      stub, cachePath, t0 + 25 * HOUR_MS + 1, notified,
    );
    assert.strictEqual(reload.reason, 'cooldown-replay');
    assert.strictEqual(notified.length, 3);
  } finally {
    await stopStub(stub);
  }
}

async function testSnoozeExpiryInsideCooldownReplays() {
  const stub = await startPypiStub({info: {version: '2026.9.2'}});
  const cachePath = makeCachePath('shortsnooze');
  try {
    const notified = [];
    const t0 = 1_000_000;
    await activate(stub, cachePath, t0, notified);
    // A short custom snooze that ends while the fetch cooldown is
    // still running: the replay path must notify again.
    snoozeUpdateNotification({
      latest: '2026.9.2',
      cacheFilePath: cachePath,
      snoozeMs: HOUR_MS,
      now: () => t0,
    });
    const during = await activate(
      stub, cachePath, t0 + HOUR_MS - 1, notified,
    );
    assert.strictEqual(during.reason, 'snoozed');
    const after = await activate(stub, cachePath, t0 + HOUR_MS, notified);
    assert.strictEqual(after.reason, 'cooldown-replay');
    assert.strictEqual(after.notified, true);
    assert.strictEqual(stub.hits, 1);
    assert.strictEqual(notified.length, 2);
  } finally {
    await stopStub(stub);
  }
}

async function testNewerReleaseBreaksThroughSnooze() {
  const stub = await startPypiStub({info: {version: '2026.9.2'}});
  const cachePath = makeCachePath('breakthrough');
  try {
    const notified = [];
    const t0 = 1_000_000;
    await activate(stub, cachePath, t0, notified);
    snoozeUpdateNotification({
      latest: '2026.9.2',
      cacheFilePath: cachePath,
      now: () => t0,
    });

    // A NEWER release ships inside the snooze window; after the fetch
    // cooldown the user must be told about it despite the snooze.
    stub.payload = {info: {version: '2026.9.3'}};
    const later = await activate(stub, cachePath, t0 + 7 * HOUR_MS, notified);
    assert.strictEqual(later.reason, 'update-available');
    assert.strictEqual(later.latest, '2026.9.3');
    assert.strictEqual(notified.length, 2);
    assert.deepStrictEqual(notified[1], {
      latest: '2026.9.3',
      current: '2026.9.1',
    });
  } finally {
    await stopStub(stub);
  }
}

async function testSnoozeWithoutPriorCacheOrVersion() {
  // Defensive path: "Remind me later" clicked when no cache file exists
  // and no version is passed — everything stays suppressed for 24h
  // (unparsable snoozedLatest snoozes any version until expiry).
  const stub = await startPypiStub({info: {version: '2026.9.2'}});
  const cachePath = makeCachePath('nocache');
  try {
    const t0 = 1_000_000;
    const snooze = snoozeUpdateNotification({
      cacheFilePath: cachePath,
      now: () => t0,
    });
    assert.strictEqual(snooze.snoozedLatest, '');
    const cache = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    assert.strictEqual(cache.lastCheckMs, 0);
    assert.strictEqual(cache.lastLatest, '');

    // lastCheckMs is 0, so the next activation takes the fetch path;
    // the version-less snooze must still silence the fetched update.
    const notified = [];
    const result = await activate(stub, cachePath, t0 + 7 * HOUR_MS, notified);
    assert.strictEqual(stub.hits, 1);
    assert.strictEqual(result.reason, 'snoozed');
    assert.strictEqual(result.latest, '2026.9.2');
    assert.strictEqual(notified.length, 0);
  } finally {
    await stopStub(stub);
  }
}

async function testSnoozeRecordedDuringFetchIsNotErased() {
  // Regression (review finding): checkForExtensionUpdate used to
  // rewrite the cache from its PRE-fetch snapshot, so a sibling
  // window's "Remind me later" recorded while this window's PyPI
  // fetch was in flight got erased — and the popup fired anyway.
  const cachePath = makeCachePath('race');
  const t0 = 1_000_000;
  const notified = [];
  const result = await checkForExtensionUpdate({
    cacheFilePath: cachePath,
    cooldownMs: 6 * HOUR_MS,
    currentVersion: '2026.9.1',
    notify: params => notified.push(params),
    now: () => t0,
    fetchLatest: async () => {
      snoozeUpdateNotification({
        latest: '2026.9.2',
        cacheFilePath: cachePath,
        now: () => t0,
      });
      return '2026.9.2';
    },
  });
  assert.strictEqual(result.reason, 'snoozed');
  assert.strictEqual(result.notified, false);
  assert.strictEqual(notified.length, 0);

  const cache = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
  assert.strictEqual(cache.snoozeUntilMs, t0 + 24 * HOUR_MS);
  assert.strictEqual(cache.snoozedLatest, '2026.9.2');
  assert.strictEqual(cache.lastCheckMs, t0);
  assert.strictEqual(cache.lastLatest, '2026.9.2');
}

async function testDefaultsUseKissHomeAndRealClock() {
  // Production defaults: no cacheFilePath / now / snoozeMs — the snooze
  // lands in $KISS_HOME/.update-check.json, 24h from the real clock.
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-snooze-home-'));
  tmpDirs.push(home);
  const prevKissHome = process.env.KISS_HOME;
  process.env.KISS_HOME = home;
  try {
    const before = Date.now();
    const snooze = snoozeUpdateNotification({latest: '2026.9.2'});
    const after = Date.now();
    assert.ok(snooze.snoozeUntilMs >= before + 24 * HOUR_MS);
    assert.ok(snooze.snoozeUntilMs <= after + 24 * HOUR_MS);
    const cache = JSON.parse(
      fs.readFileSync(path.join(home, '.update-check.json'), 'utf-8'),
    );
    assert.strictEqual(cache.snoozedLatest, '2026.9.2');
    assert.strictEqual(cache.snoozeUntilMs, snooze.snoozeUntilMs);
  } finally {
    if (prevKissHome === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prevKissHome;
  }
}

async function runTests() {
  await test(
    'reproduce: Remind me later silences the reload replay popup',
    testSnoozeSuppressesReloadReplay,
  );
  await test(
    'snooze survives the 6h cooldown re-fetch and cache rewrite',
    testSnoozeSurvivesCooldownRefetch,
  );
  await test(
    'popup returns after the 24h snooze expires (fetch path)',
    testSnoozeExpiresAfter24Hours,
  );
  await test(
    'popup returns after snooze expiry inside the cooldown (replay path)',
    testSnoozeExpiryInsideCooldownReplays,
  );
  await test(
    'a release newer than the snoozed one breaks through the snooze',
    testNewerReleaseBreaksThroughSnooze,
  );
  await test(
    'snooze without prior cache/version suppresses until expiry',
    testSnoozeWithoutPriorCacheOrVersion,
  );
  await test(
    'a snooze recorded during the in-flight fetch is not erased',
    testSnoozeRecordedDuringFetchIsNotErased,
  );
  await test(
    'defaults: KISS_HOME cache path, real clock, 24h duration',
    testDefaultsUseKissHomeAndRealClock,
  );
}

runTests()
  .then(() => {
    for (const dir of tmpDirs) {
      try {
        fs.rmSync(dir, {recursive: true, force: true});
      } catch {
      }
    }
    console.log(`\n${passed} passed, ${failures.length} failed`);
    if (failures.length > 0) {
      for (const f of failures) {
        console.error(`\n${f.name}:\n`, f.err);
      }
      process.exit(1);
    }
    process.exit(0);
  })
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });
