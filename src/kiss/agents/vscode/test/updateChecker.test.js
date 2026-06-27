// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the KISS Sorcar on-activation PyPI
// update check.
//
// Bug reproduced
// --------------
// Before this fix, when VS Code was launched and KISS Sorcar was NOT
// installing (the "all deps present" fast path in
// ``DependencyInstaller.ensureDependenciesImpl``) the extension did
// nothing to tell the user a newer release was available.  The
// kiss-web daemon polls PyPI hourly and broadcasts an
// ``update_available`` event to connected webview clients — but on
// the fast path the daemon is not restarted, so its cached "no
// update" answer can be up to an hour stale.  Users who launched
// VS Code only briefly never saw the next poll fire at all.
//
// This test pins the new ``checkForExtensionUpdate`` helper from
// ``src/UpdateChecker.js`` (which the extension's ``activate``
// invokes on every launch) by driving it directly against a real,
// loopback HTTP server impersonating the PyPI JSON endpoint.  No
// vscode is required — the helper accepts a ``notify`` callback so
// the test can observe its behavior end-to-end without involving the
// VS Code API or the user's ``~/.kiss/``.
//
// Coverage
// --------
//   1. Stale local version + newer PyPI version → ``notify`` is called
//      with the right ``{latest, current}`` payload AND the JSON cache
//      file is written (regression locks in the fix).
//   2. Within cooldown the cached answer is replayed and PyPI is NOT
//      hit a second time (rate-limit guard).
//   3. After cooldown elapses with PyPI now reporting the current
//      version, ``notify`` is NOT called (no false positive after the
//      user updates).
//   4. Network error: ``notify`` is NOT called, no exception escapes,
//      the cache file is NOT poisoned with an empty answer.
//   5. ``_version.py``-based current-version resolution finds the
//      installed kiss_project version (covers the production code path
//      that ``extension.ts`` calls into).
//   6. ``compareVersions`` unit-style coverage so a malformed PyPI
//      payload never produces a false positive.
//
// Runs under bare ``node`` (no TypeScript compile):
//
//     node test/updateChecker.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');

const {
  checkForExtensionUpdate,
  compareVersions,
  resolveCurrentVersion,
} = require('../src/UpdateChecker.js');

// ---------------------------------------------------------------------------
// PyPI stub: a real loopback HTTP server whose payload and hit counter
// can be inspected per-test.  Mirrors the Python ``_PypiStub`` used by
// ``tests/agents/vscode/test_update_available_check.py`` so the
// extension- and daemon-side regression tests sit on the same shape of
// fixture.
// ---------------------------------------------------------------------------
function startPypiStub(payload, status = 200) {
  const state = {payload, status, hits: 0, server: null, url: ''};
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      state.hits += 1;
      if (state.payload === null) {
        res.statusCode = state.status;
        res.end();
        return;
      }
      const body = Buffer.from(JSON.stringify(state.payload), 'utf-8');
      res.statusCode = state.status;
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

// ---------------------------------------------------------------------------
// Per-test scratch directory so the helper's cache file does not leak
// across cases or contaminate the developer's real ``~/.kiss/``.
// ---------------------------------------------------------------------------
const tmpDirs = [];
function makeCachePath(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kiss-updcheck-${tag}-`));
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

// ---------------------------------------------------------------------------
// 1.  Stale current vs newer PyPI version: ``notify`` MUST fire.
//     This is the bug reproduction — before the fix nothing called
//     ``notify`` because the helper did not exist.
// ---------------------------------------------------------------------------
async function testNotifiesWhenUpdateAvailable() {
  const stub = await startPypiStub({info: {version: '2099.1.1'}});
  const cachePath = makeCachePath('newer');
  try {
    const notified = [];
    const result = await checkForExtensionUpdate({
      pypiUrl: stub.url,
      cacheFilePath: cachePath,
      cooldownMs: 60_000,
      currentVersion: '2026.6.30',
      notify: params => notified.push(params),
      now: () => 1_000_000,
    });

    assert.strictEqual(stub.hits, 1, 'PyPI must have been hit exactly once');
    assert.strictEqual(result.checked, true);
    assert.strictEqual(result.notified, true);
    assert.strictEqual(result.latest, '2099.1.1');
    assert.strictEqual(result.current, '2026.6.30');
    assert.strictEqual(result.reason, 'update-available');
    assert.strictEqual(notified.length, 1, 'notify should fire exactly once');
    assert.deepStrictEqual(notified[0], {
      latest: '2099.1.1',
      current: '2026.6.30',
    });

    // Cache file is now written so the next activation rate-limits.
    const cache = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    assert.strictEqual(cache.lastCheckMs, 1_000_000);
    assert.strictEqual(cache.lastLatest, '2099.1.1');
  } finally {
    await stopStub(stub);
  }
}

// ---------------------------------------------------------------------------
// 2.  Within cooldown: cached answer is replayed, PyPI is NOT hit again.
// ---------------------------------------------------------------------------
async function testCooldownReplaysCachedDecision() {
  const stub = await startPypiStub({info: {version: '2099.1.1'}});
  const cachePath = makeCachePath('cooldown');
  try {
    // First call populates the cache.
    await checkForExtensionUpdate({
      pypiUrl: stub.url,
      cacheFilePath: cachePath,
      cooldownMs: 60_000,
      currentVersion: '2026.6.30',
      notify: () => {},
      now: () => 1_000_000,
    });
    assert.strictEqual(stub.hits, 1);

    // Second call within cooldown: must use cache, replay notify, and
    // NOT make a second PyPI request.
    const notified = [];
    const result = await checkForExtensionUpdate({
      pypiUrl: stub.url,
      cacheFilePath: cachePath,
      cooldownMs: 60_000,
      currentVersion: '2026.6.30',
      notify: params => notified.push(params),
      now: () => 1_010_000,
    });
    assert.strictEqual(stub.hits, 1, 'PyPI must NOT be hit again in cooldown');
    assert.strictEqual(result.checked, false);
    assert.strictEqual(result.notified, true);
    assert.strictEqual(result.reason, 'cooldown-replay');
    assert.deepStrictEqual(notified[0], {
      latest: '2099.1.1',
      current: '2026.6.30',
    });
  } finally {
    await stopStub(stub);
  }
}

// ---------------------------------------------------------------------------
// 3.  After the user updates, PyPI reports the same version we have →
//     ``notify`` must NOT fire (no stale "update available" toast).
// ---------------------------------------------------------------------------
async function testNoNotifyWhenUpToDate() {
  const stub = await startPypiStub({info: {version: '2026.6.30'}});
  const cachePath = makeCachePath('current');
  try {
    const notified = [];
    const result = await checkForExtensionUpdate({
      pypiUrl: stub.url,
      cacheFilePath: cachePath,
      cooldownMs: 60_000,
      currentVersion: '2026.6.30',
      notify: params => notified.push(params),
      now: () => 1_000_000,
    });
    assert.strictEqual(stub.hits, 1);
    assert.strictEqual(result.checked, true);
    assert.strictEqual(result.notified, false);
    assert.strictEqual(result.reason, 'up-to-date');
    assert.strictEqual(notified.length, 0);
  } finally {
    await stopStub(stub);
  }
}

// ---------------------------------------------------------------------------
// 4.  Network failure (PyPI returns 500): ``notify`` must NOT fire, the
//     promise must NOT reject, and the cache file must NOT be poisoned.
// ---------------------------------------------------------------------------
async function testFetchFailureDoesNotCrashOrNotify() {
  const stub = await startPypiStub(null, 500);
  const cachePath = makeCachePath('fail');
  try {
    const notified = [];
    const result = await checkForExtensionUpdate({
      pypiUrl: stub.url,
      cacheFilePath: cachePath,
      cooldownMs: 60_000,
      currentVersion: '2026.6.30',
      notify: params => notified.push(params),
      now: () => 1_000_000,
    });
    assert.strictEqual(result.checked, true);
    assert.strictEqual(result.notified, false);
    assert.strictEqual(result.reason, 'fetch-failed');
    assert.strictEqual(notified.length, 0);
    assert.strictEqual(
      fs.existsSync(cachePath),
      false,
      'fetch failure must not write a poisoned cache file',
    );
  } finally {
    await stopStub(stub);
  }
}

// ---------------------------------------------------------------------------
// 5.  Production current-version resolution via ``_version.py``: this
//     covers the code path that ``extension.ts`` exercises when it
//     hands ``kissProjectPath`` to the helper.
// ---------------------------------------------------------------------------
async function testResolvesCurrentVersionFromVersionPy() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-updcheck-vpy-'));
  tmpDirs.push(root);
  const versionPyDir = path.join(root, 'src', 'kiss');
  fs.mkdirSync(versionPyDir, {recursive: true});
  fs.writeFileSync(
    path.join(versionPyDir, '_version.py'),
    "__version__ = '2099.9.9'\n",
  );
  assert.strictEqual(resolveCurrentVersion(root), '2099.9.9');
  assert.strictEqual(resolveCurrentVersion(undefined), null);

  // And end-to-end: the helper uses it when no ``currentVersion`` is
  // passed explicitly.
  const stub = await startPypiStub({info: {version: '2099.9.10'}});
  const cachePath = makeCachePath('vpy');
  try {
    const notified = [];
    const result = await checkForExtensionUpdate({
      pypiUrl: stub.url,
      cacheFilePath: cachePath,
      cooldownMs: 60_000,
      kissProjectPath: root,
      notify: params => notified.push(params),
      now: () => 1_000_000,
    });
    assert.strictEqual(result.current, '2099.9.9');
    assert.strictEqual(result.latest, '2099.9.10');
    assert.strictEqual(notified.length, 1);
    assert.deepStrictEqual(notified[0], {
      latest: '2099.9.10',
      current: '2099.9.9',
    });
  } finally {
    await stopStub(stub);
  }
}

// ---------------------------------------------------------------------------
// 6.  compareVersions edge cases — keep parity with the Python helper.
// ---------------------------------------------------------------------------
async function testCompareVersions() {
  assert.strictEqual(compareVersions('2026.6.30', '2026.6.29'), 1);
  assert.strictEqual(compareVersions('2026.6.30', '2026.6.31'), -1);
  assert.strictEqual(compareVersions('2026.6.30', '2026.6.30'), 0);
  // Shorter tuples are right-padded with zeros.
  assert.strictEqual(compareVersions('2026.6', '2026.6.0'), 0);
  assert.strictEqual(compareVersions('2026.7', '2026.6.9'), 1);
  // Garbage falls back to equality so we never raise a false positive.
  assert.strictEqual(compareVersions('bad', '2026.6.30'), 0);
  assert.strictEqual(compareVersions('2026.6.30', ''), 0);
}

// ---------------------------------------------------------------------------
// 7.  No current version available → graceful skip, no notify, no fetch.
// ---------------------------------------------------------------------------
async function testSkipsWhenCurrentVersionUnknown() {
  let fetched = 0;
  const notified = [];
  const cachePath = makeCachePath('nover');
  const result = await checkForExtensionUpdate({
    cacheFilePath: cachePath,
    currentVersion: '', // explicit empty + no kissProjectPath = unknown
    fetchLatest: () => {
      fetched += 1;
      return Promise.resolve('9999.9.9');
    },
    notify: params => notified.push(params),
    now: () => 1_000_000,
  });
  assert.strictEqual(result.checked, false);
  assert.strictEqual(result.notified, false);
  assert.strictEqual(result.reason, 'unknown-current-version');
  assert.strictEqual(fetched, 0, 'must not hit PyPI when local version unknown');
  assert.strictEqual(notified.length, 0);
}

async function runTests() {
  await test(
    'reproduce: stale local version triggers update notification',
    testNotifiesWhenUpdateAvailable,
  );
  await test(
    'within cooldown, cached decision is replayed and PyPI is not re-hit',
    testCooldownReplaysCachedDecision,
  );
  await test(
    'no notification fires when PyPI reports the current version',
    testNoNotifyWhenUpToDate,
  );
  await test(
    'fetch failure is swallowed and does not poison the cache',
    testFetchFailureDoesNotCrashOrNotify,
  );
  await test(
    'production code path: current version is read from _version.py',
    testResolvesCurrentVersionFromVersionPy,
  );
  await test('compareVersions matches the Python helper', testCompareVersions);
  await test(
    'helper skips check when the local version is unknown',
    testSkipsWhenCurrentVersionUnknown,
  );
}

runTests()
  .then(() => {
    for (const dir of tmpDirs) {
      try {
        fs.rmSync(dir, {recursive: true, force: true});
      } catch {
        /* ignore */
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
