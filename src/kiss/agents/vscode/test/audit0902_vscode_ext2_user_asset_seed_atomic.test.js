// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-02 (vscode-ext2): ensureUserAssetFromDefault() in
// src/userAssets.ts seeds ~/.kiss/MY_INJECTION.md and MY_TASK_TEMPLATES.md
// with a truncating fs.writeFileSync after an existsSync check.  The
// daemon's autocomplete worker reads MY_INJECTION.md on every keystroke
// (server/user_assets.py::read_tricks), so a reader that lands between
// open(O_TRUNC) and the write sees an EMPTY file and drops every user
// trick for that completion; and two seeders (extension host + daemon on
// a fresh install) that both pass the existsSync check truncate and
// overwrite each other's payload -- when the payloads differ in length
// the file ends up as one content with the tail of the other.  The Python
// twin stages the default in a sibling temp file and os.link()s it into
// place (the loser gets EEXIST and keeps the winner's content); this
// suite pins the same contract on the compiled out/userAssets.js.
//
// Scenario 1: a reader child process spins on each asset while this
// process seeds it; every observation must be either ENOENT or the whole
// default -- never '' or a prefix.
// Scenario 2: two seeder child processes race on the same names (paced to
// the same millisecond tick) with defaults of different lengths; both must
// return the path and every file must hold exactly one of the two
// defaults.
// Scenario 3: a dangling symlink at the asset path (exists() is false but
// the directory entry is taken) must be treated as "someone else won" and
// return the path; an unwritable ~/.kiss/ must still yield null.

/* global require, process, console, __dirname, setTimeout */

'use strict';

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const USER_ASSETS = path.join(__dirname, '..', 'out', 'userAssets.js');
const ua = require(USER_ASSETS);

const N_READ = 300;
const N_SEED = 400;
// Wide payloads widen the window between truncate and write.
const CONTENT_A = '## Trick\n\n' + 'A'.repeat(600_000) + '\n';
const CONTENT_B = '## Trick\n\n' + 'B'.repeat(200_000) + '\n';

const READER_SRC = `
'use strict';
const fs = require('fs');
const path = require('path');
const dir = process.argv[1];
const n = Number(process.argv[2]);
const expected = fs.readFileSync(process.argv[3], 'utf-8');
const untilMs = Date.now() + 20000;
let torn = 0;
let whole = 0;
const samples = [];
for (let i = 0; i < n && Date.now() < untilMs; i++) {
  const p = path.join(dir, 'asset-' + i + '.md');
  for (;;) {
    let text;
    try {
      text = fs.readFileSync(p, 'utf-8');
    } catch (err) {
      if (err.code === 'ENOENT') continue;
      throw err;
    }
    if (text === expected) {
      whole++;
      break;
    }
    torn++;
    if (samples.length < 5) {
      samples.push(text.length === 0 ? 'asset-' + i + ': empty' : 'asset-' + i + ': ' + text.length + ' bytes');
    }
    if (text.length === expected.length) break; // torn, but no more writes will come
  }
}
process.stdout.write(JSON.stringify({torn, whole, samples}));
`;

const SEEDER_SRC = `
'use strict';
const fs = require('fs');
const ua = require(process.argv[1]);
const n = Number(process.argv[2]);
const content = fs.readFileSync(process.argv[3], 'utf-8');
const startMs = Number(process.argv[4]);
const stepMs = Number(process.argv[5]);
let nulls = 0;
for (let i = 0; i < n; i++) {
  // Both seeders release on the same millisecond tick so their
  // existsSync checks overlap as often as possible.
  const gate = startMs + i * stepMs;
  while (Date.now() < gate) {
    // spin
  }
  const r = ua.ensureUserAssetFromDefault('asset-' + i + '.md', content);
  if (r === null) nulls++;
}
process.stdout.write(JSON.stringify({nulls}));
`;

function runChild(src, args, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['-e', src, ...args], {
      stdio: ['ignore', 'pipe', 'inherit'],
      env: {...process.env, ...env},
    });
    let out = '';
    child.stdout.on('data', d => {
      out += d;
    });
    child.on('error', reject);
    child.on('exit', code => {
      if (code !== 0) reject(new Error(`child exited ${code}`));
      else resolve(JSON.parse(out));
    });
  });
}

async function readerVsSeeder(root) {
  const home = path.join(root, 'home-reader');
  fs.mkdirSync(home, {recursive: true});
  const contentFile = path.join(root, 'content-a.txt');
  fs.writeFileSync(contentFile, CONTENT_A);
  const reader = runChild(READER_SRC, [home, String(N_READ), contentFile], {});
  // Give the reader a head start so it is already spinning on asset-0.
  await new Promise(r => setTimeout(r, 300));
  const prev = process.env.KISS_HOME;
  process.env.KISS_HOME = home;
  try {
    for (let i = 0; i < N_READ; i++) {
      const r = ua.ensureUserAssetFromDefault(`asset-${i}.md`, CONTENT_A);
      assert.strictEqual(r, path.join(home, `asset-${i}.md`));
      // Let the reader catch up so it observes every file's birth.
      const born = Date.now();
      while (Date.now() - born < 2) {
        // spin
      }
    }
  } finally {
    if (prev === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prev;
  }
  const res = await reader;
  console.log(`  reader: whole=${res.whole} torn=${res.torn}`);
  assert.ok(res.whole > N_READ / 2, 'reader barely ran');
  assert.strictEqual(
    res.torn,
    0,
    `a concurrent reader saw an empty/partial seed ${res.torn} times: ` +
      res.samples.join('; '),
  );
  const leftovers = fs
    .readdirSync(home)
    .filter(n => !/^asset-\d+\.md$/.test(n));
  assert.deepStrictEqual(leftovers, [], `temp files left behind: ${leftovers}`);
  console.log('  ✓ a concurrent reader never sees an empty or partial seed');
}

async function twoSeeders(root) {
  const home = path.join(root, 'home-seeders');
  fs.mkdirSync(home, {recursive: true});
  const fileA = path.join(root, 'seed-a.txt');
  const fileB = path.join(root, 'seed-b.txt');
  fs.writeFileSync(fileA, CONTENT_A);
  fs.writeFileSync(fileB, CONTENT_B);
  const startMs = Date.now() + 1500;
  const stepMs = 4;
  const [ra, rb] = await Promise.all([
    runChild(
      SEEDER_SRC,
      [USER_ASSETS, String(N_SEED), fileA, String(startMs), String(stepMs)],
      {KISS_HOME: home},
    ),
    runChild(
      SEEDER_SRC,
      [USER_ASSETS, String(N_SEED), fileB, String(startMs), String(stepMs)],
      {KISS_HOME: home},
    ),
  ]);
  assert.strictEqual(ra.nulls, 0, 'seeder A returned null');
  assert.strictEqual(rb.nulls, 0, 'seeder B returned null');
  let a = 0;
  let b = 0;
  const bad = [];
  for (let i = 0; i < N_SEED; i++) {
    const text = fs.readFileSync(path.join(home, `asset-${i}.md`), 'utf-8');
    if (text === CONTENT_A) a++;
    else if (text === CONTENT_B) b++;
    else bad.push(`asset-${i}: ${text.length} bytes`);
  }
  console.log(`  seeders: A won ${a}, B won ${b}, corrupt ${bad.length}`);
  assert.strictEqual(
    bad.length,
    0,
    `concurrent seeders left mixed content: ${bad.slice(0, 5).join('; ')}`,
  );
  const leftovers = fs
    .readdirSync(home)
    .filter(n => !/^asset-\d+\.md$/.test(n));
  assert.deepStrictEqual(leftovers, [], `temp files left behind: ${leftovers}`);
  console.log('  ✓ concurrent seeders leave exactly one default intact');
}

function edgeCases(root) {
  const home = path.join(root, 'home-edge');
  fs.mkdirSync(home, {recursive: true});
  const prev = process.env.KISS_HOME;
  process.env.KISS_HOME = home;
  try {
    // Existing file: returned untouched.
    fs.writeFileSync(path.join(home, 'keep.md'), 'mine');
    assert.strictEqual(
      ua.ensureUserAssetFromDefault('keep.md', 'default'),
      path.join(home, 'keep.md'),
    );
    assert.strictEqual(
      fs.readFileSync(path.join(home, 'keep.md'), 'utf-8'),
      'mine',
    );

    // Dangling symlink: exists() says no, but the directory entry is
    // taken, so the link fails with EEXIST -- "someone else won".
    fs.symlinkSync(path.join(home, 'nowhere'), path.join(home, 'dangling.md'));
    assert.strictEqual(
      ua.ensureUserAssetFromDefault('dangling.md', 'default'),
      path.join(home, 'dangling.md'),
    );
    assert.ok(
      !fs.existsSync(path.join(home, 'nowhere')),
      'must not write through the foreign entry',
    );
    const leftovers = fs
      .readdirSync(home)
      .filter(n => n !== 'keep.md' && n !== 'dangling.md');
    assert.deepStrictEqual(
      leftovers,
      [],
      `temp files left behind: ${leftovers}`,
    );

    // ~/.kiss/ that cannot be created (its parent is a plain file).
    const blocked = path.join(root, 'not-a-dir');
    fs.writeFileSync(blocked, 'x');
    process.env.KISS_HOME = path.join(blocked, 'kiss');
    assert.strictEqual(ua.ensureUserAssetFromDefault('x.md', 'default'), null);
    console.log(
      '  ✓ existing file wins, foreign entry wins, unwritable home yields null',
    );
  } finally {
    if (prev === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prev;
  }
}

async function main() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit-ua-'));
  try {
    await readerVsSeeder(root);
    await twoSeeders(root);
    edgeCases(root);
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
  console.log('audit0902_vscode_ext2_user_asset_seed_atomic: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
