// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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
  scanInstalledExtensionVersions,
} = require('../src/UpdateChecker.js');

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

    const cache = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    assert.strictEqual(cache.lastCheckMs, 1_000_000);
    assert.strictEqual(cache.lastLatest, '2099.1.1');
  } finally {
    await stopStub(stub);
  }
}

async function testCooldownReplaysCachedDecision() {
  const stub = await startPypiStub({info: {version: '2099.1.1'}});
  const cachePath = makeCachePath('cooldown');
  try {
    await checkForExtensionUpdate({
      pypiUrl: stub.url,
      cacheFilePath: cachePath,
      cooldownMs: 60_000,
      currentVersion: '2026.6.30',
      notify: () => {},
      now: () => 1_000_000,
    });
    assert.strictEqual(stub.hits, 1);

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

async function testResolvesCurrentVersionFromVersionPy() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-updcheck-vpy-'));
  tmpDirs.push(root);
  const versionPyDir = path.join(root, 'src', 'kiss');
  fs.mkdirSync(versionPyDir, {recursive: true});
  fs.writeFileSync(
    path.join(versionPyDir, '_version.py'),
    "__version__ = '2099.9.9'\n",
  );
  const emptyExtRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), 'kiss-updcheck-noext-'),
  );
  tmpDirs.push(emptyExtRoot);
  assert.strictEqual(resolveCurrentVersion(root, emptyExtRoot), '2099.9.9');
  assert.strictEqual(resolveCurrentVersion(undefined, emptyExtRoot), null);

  const stub = await startPypiStub({info: {version: '2099.9.10'}});
  const cachePath = makeCachePath('vpy');
  try {
    const notified = [];
    const result = await checkForExtensionUpdate({
      pypiUrl: stub.url,
      cacheFilePath: cachePath,
      cooldownMs: 60_000,
      kissProjectPath: root,
      extensionsRoot: emptyExtRoot,
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

async function testScansMaxInstalledExtensionVersion() {
  const extRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), 'kiss-updcheck-extroot-'),
  );
  tmpDirs.push(extRoot);
  for (const ver of ['2026.6.30', '2026.7.5']) {
    const d = path.join(
      extRoot,
      `ksenxx.kiss-sorcar-${ver}`,
      'kiss_project',
      'src',
      'kiss',
    );
    fs.mkdirSync(d, {recursive: true});
    fs.writeFileSync(
      path.join(d, '_version.py'),
      `__version__ = '${ver}'\n`,
    );
  }
  fs.mkdirSync(path.join(extRoot, 'someone.other-extension-1.0.0'));
  fs.mkdirSync(
    path.join(extRoot, 'ksenxx.kiss-sorcar-broken', 'kiss_project'),
    {recursive: true},
  );
  const bad = path.join(
    extRoot,
    'ksenxx.kiss-sorcar-2026.6.99',
    'kiss_project',
    'src',
    'kiss',
  );
  fs.mkdirSync(bad, {recursive: true});
  fs.writeFileSync(path.join(bad, '_version.py'), '# no version here\n');
  fs.writeFileSync(path.join(extRoot, 'not-a-directory.txt'), 'junk');

  const seen = scanInstalledExtensionVersions(extRoot);
  seen.sort();
  assert.deepStrictEqual(
    seen,
    ['2026.6.30', '2026.7.5'],
    'scanner must return only the parseable KISS Sorcar versions',
  );

  const oldKissProject = path.join(
    extRoot,
    'ksenxx.kiss-sorcar-2026.6.30',
    'kiss_project',
  );
  assert.strictEqual(
    resolveCurrentVersion(oldKissProject, extRoot),
    '2026.7.5',
  );

  assert.deepStrictEqual(
    scanInstalledExtensionVersions(
      path.join(extRoot, 'does-not-exist'),
    ),
    [],
  );
  assert.strictEqual(
    resolveCurrentVersion(
      oldKissProject,
      path.join(extRoot, 'does-not-exist'),
    ),
    '2026.6.30',
  );
}

async function testCompareVersions() {
  assert.strictEqual(compareVersions('2026.6.30', '2026.6.29'), 1);
  assert.strictEqual(compareVersions('2026.6.30', '2026.6.31'), -1);
  assert.strictEqual(compareVersions('2026.6.30', '2026.6.30'), 0);
  assert.strictEqual(compareVersions('2026.6', '2026.6.0'), 0);
  assert.strictEqual(compareVersions('2026.7', '2026.6.9'), 1);
  assert.strictEqual(compareVersions('bad', '2026.6.30'), 0);
  assert.strictEqual(compareVersions('2026.6.30', ''), 0);
}

async function testSkipsWhenCurrentVersionUnknown() {
  let fetched = 0;
  const notified = [];
  const cachePath = makeCachePath('nover');
  const emptyExtRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), 'kiss-updcheck-nover-ext-'),
  );
  tmpDirs.push(emptyExtRoot);
  const result = await checkForExtensionUpdate({
    cacheFilePath: cachePath,
    currentVersion: '',
    extensionsRoot: emptyExtRoot,
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
  await test(
    'reproduces stale-daemon bug: max installed extension version wins',
    testScansMaxInstalledExtensionVersion,
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
