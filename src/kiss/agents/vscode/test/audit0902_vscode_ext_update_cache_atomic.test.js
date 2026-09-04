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
const untilMs = Number(process.argv[4]);
(async () => {
  let writes = 0;
  while (Date.now() < untilMs) {
    const r = await uc.checkForExtensionUpdate({
      cacheFilePath: cachePath,
      currentVersion: '1.0.0',
      cooldownMs: 0,
      fetchLatest: async () => tag,
      notify: () => {},
    });
    if (r.checked) writes++;
  }
  process.stdout.write(String(writes));
})();
`;

function runWorker(cachePath, tag, untilMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      ['-e', WORKER_SRC, UPDATE_CHECKER, cachePath, tag, String(untilMs)],
      {stdio: ['ignore', 'pipe', 'inherit']},
    );
    let out = '';
    child.stdout.on('data', d => {
      out += d;
    });
    child.on('error', reject);
    child.on('exit', code => {
      if (code !== 0) reject(new Error(`worker ${tag} exited ${code}`));
      else resolve(Number(out));
    });
  });
}

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

    const untilMs = Date.now() + 2500;
    const workers = Promise.all([
      runWorker(cachePath, tagA, untilMs),
      runWorker(cachePath, tagB, untilMs),
    ]);

    let reads = 0;
    let corrupt = 0;
    const samples = [];
    while (Date.now() < untilMs) {
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
    const [writesA, writesB] = await workers;
    console.log(
      `  reads=${reads} corrupt=${corrupt} writesA=${writesA} writesB=${writesB}`,
    );
    assert.ok(writesA > 50 && writesB > 50, 'workers barely ran');
    assert.ok(reads > 50, 'reader barely ran');
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
