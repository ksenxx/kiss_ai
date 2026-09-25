// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-02 (vscode-ext partition): UpdateChecker's cooldown cache
// is written by EVERY VS Code window at activation, all to the same
// $KISS_HOME/.update-check.json.  writeCache() wrote a temp file and
// renamed it over the cache, but the temp file had one FIXED name
// (`<cache>.tmp`): two windows writing at once truncated and renamed each
// other's temp file, and the cache spent a good share of its time as an
// empty (or half-written) file that readCache() could not parse -- which
// silently voided the six-hour cooldown and re-hit PyPI.  Sibling
// writeKissConfig() in DependencyInstaller.ts already uses a per-process
// temp name; this makes the update cache do the same.
//
// Two real processes hammer checkForExtensionUpdate() (cooldown disabled
// so every call writes) while this process reads the cache file as fast
// as it can: every read must parse and hold one of the two writers'
// payloads whole.

'use strict';

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const UPDATE_CHECKER = path.join(__dirname, '..', 'src', 'UpdateChecker.js');
const uc = require(UPDATE_CHECKER);

const WORKER_SRC = `
'use strict';
// node -e: argv[1] is the first extra argument (there is no script path).
const uc = require(process.argv[1]);
const cachePath = process.argv[2];
const tag = process.argv[3];
const stopPath = process.argv[4];
// One byte on stdout per write, so the parent can watch progress live and
// stop the race only once every side has done enough work.
(async () => {
  while (!require('fs').existsSync(stopPath)) {
    const r = await uc.checkForExtensionUpdate({
      cacheFilePath: cachePath,
      currentVersion: '1.0.0',
      cooldownMs: 0,
      fetchLatest: async () => tag,
      notify: () => {},
    });
    if (r.checked) process.stdout.write('w');
  }
})();
`;

// A worker's write count is read live from ``progress.writes`` and settles
// when the returned promise resolves.
function runWorker(cachePath, tag, stopPath, progress) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      ['-e', WORKER_SRC, UPDATE_CHECKER, cachePath, tag, stopPath],
      {stdio: ['ignore', 'pipe', 'inherit']},
    );
    child.stdout.on('data', d => {
      progress.writes += d.length;
    });
    child.on('error', reject);
    child.on('exit', code => {
      if (code !== 0) reject(new Error(`worker ${tag} exited ${code}`));
      else resolve(progress.writes);
    });
  });
}

// The race runs for at least MIN_RACE_MS, and longer (up to MAX_RACE_MS) on
// a machine whose file system is slow or busy, until the reader and both
// writers have each done MIN_OPS operations: a fixed 2.5 s window produced
// only 43 reads on a loaded Windows VM, which says nothing about atomicity.
const MIN_RACE_MS = 2500;
const MAX_RACE_MS = 30000;
const MIN_OPS = 50;

async function main() {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit-uc-'));
  const cachePath = path.join(home, '.update-check.json');
  // Long payloads widen the window between truncate and write.
  const tagA = '9.' + '1'.repeat(400);
  const tagB = '9.' + '2'.repeat(400);
  try {
    // A valid cache exists before the race starts, so a missing file is
    // never a legitimate observation.
    const seed = await uc.checkForExtensionUpdate({
      cacheFilePath: cachePath,
      currentVersion: '1.0.0',
      cooldownMs: 0,
      fetchLatest: async () => tagA,
      notify: () => {},
    });
    assert.strictEqual(seed.reason, 'update-available');
    assert.ok(fs.existsSync(cachePath), 'seed write left no cache file');

    const stopPath = path.join(home, 'stop');
    const startMs = Date.now();
    const progressA = {writes: 0};
    const progressB = {writes: 0};
    const workers = Promise.all([
      runWorker(cachePath, tagA, stopPath, progressA),
      runWorker(cachePath, tagB, stopPath, progressB),
    ]);

    let reads = 0;
    let corrupt = 0;
    const samples = [];
    const raceDone = () => {
      const elapsed = Date.now() - startMs;
      if (elapsed >= MAX_RACE_MS) return true;
      return (
        elapsed >= MIN_RACE_MS &&
        reads >= MIN_OPS &&
        progressA.writes >= MIN_OPS &&
        progressB.writes >= MIN_OPS
      );
    };
    while (!raceDone()) {
      let text;
      try {
        text = fs.readFileSync(cachePath, 'utf-8');
      } catch (err) {
        corrupt++;
        if (samples.length < 5) samples.push(`read failed: ${err.code}`);
        continue;
      }
      reads++;
      let parsed = null;
      try {
        parsed = JSON.parse(text);
      } catch {
        corrupt++;
        if (samples.length < 5) {
          samples.push(
            text.length === 0
              ? 'empty file'
              : `unparsable (${text.length} bytes)`,
          );
        }
        continue;
      }
      if (
        !parsed ||
        typeof parsed.lastCheckMs !== 'number' ||
        (parsed.lastLatest !== tagA && parsed.lastLatest !== tagB)
      ) {
        corrupt++;
        if (samples.length < 5) samples.push('torn payload');
      }
      // Yield so the workers are not starved of the CPU.
      await new Promise(resolve => setImmediate(resolve));
    }
    fs.writeFileSync(stopPath, '');
    const [writesA, writesB] = await workers;
    fs.rmSync(stopPath);
    console.log(
      `  reads=${reads} corrupt=${corrupt} writesA=${writesA} writesB=${writesB}`,
    );
    assert.ok(writesA >= MIN_OPS && writesB >= MIN_OPS, 'workers barely ran');
    assert.ok(reads >= MIN_OPS, 'reader barely ran');
    assert.strictEqual(
      corrupt,
      0,
      `concurrent writeCache() left the update cache unreadable ` +
        `${corrupt} times out of ${reads} reads: ${samples.join('; ')}`,
    );

    // After the dust settles the cache is whole and the cooldown works:
    // no stray temp files, and a replay under cooldown does not fetch.
    const leftovers = fs
      .readdirSync(home)
      .filter(n => n !== '.update-check.json');
    assert.deepStrictEqual(
      leftovers,
      [],
      `temp files left behind: ${leftovers}`,
    );
    let fetched = 0;
    const replay = await uc.checkForExtensionUpdate({
      cacheFilePath: cachePath,
      currentVersion: '1.0.0',
      fetchLatest: async () => {
        fetched++;
        return '1.0.0';
      },
      notify: () => {},
    });
    assert.strictEqual(fetched, 0, 'cooldown cache was not honoured');
    assert.strictEqual(replay.reason, 'cooldown-replay');
    console.log('  ✓ concurrent update-cache writes never leave a torn file');

    // A cache path that cannot be renamed over (here: an existing,
    // non-empty directory) must not strand the per-process temp file.
    const blocked = path.join(home, 'blocked-cache');
    fs.mkdirSync(path.join(blocked, 'occupied'), {recursive: true});
    const res = await uc.checkForExtensionUpdate({
      cacheFilePath: blocked,
      currentVersion: '1.0.0',
      cooldownMs: 0,
      fetchLatest: async () => '2.0.0',
      notify: () => {},
    });
    assert.strictEqual(res.reason, 'update-available');
    const stray = fs.readdirSync(home).filter(n => n.endsWith('.tmp'));
    assert.deepStrictEqual(stray, [], `temp file stranded: ${stray}`);
    // And one whose parent is a plain file cannot even be created.
    const parentFile = path.join(home, 'not-a-dir');
    fs.writeFileSync(parentFile, 'x');
    const res2 = await uc.checkForExtensionUpdate({
      cacheFilePath: path.join(parentFile, 'cache.json'),
      currentVersion: '1.0.0',
      cooldownMs: 0,
      fetchLatest: async () => '2.0.0',
      notify: () => {},
    });
    assert.strictEqual(res2.reason, 'update-available');
    console.log('  ✓ a failed cache write leaves no temp file behind');
  } finally {
    fs.rmSync(home, {recursive: true, force: true});
  }
  console.log('audit0902_vscode_ext_update_cache_atomic: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
