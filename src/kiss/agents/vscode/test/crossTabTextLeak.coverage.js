// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the cross-tab text leak
// guard code in BOTH media/main.js and media/voice.js (the regions fenced by
// `// tableak-coverage:start` / `// tableak-coverage:end`) when running the
// functional jsdom suite in test/crossTabTextLeak.test.js.
//
// The guard code is the whole point of the fix, so an unexecuted guard line
// is an untested claim. Both files are scanned because the leak is fixed in
// both: main.js decides what may touch a shared surface, voice.js decides
// which conversation a transcript belongs to.

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const TEST_FILE = path.join(__dirname, 'crossTabTextLeak.test.js');
const START_MARK = '// tableak-coverage:start';
const END_MARK = '// tableak-coverage:end';

// Every fenced file, with the sourceURL pragma the test appends when it
// evals it, and the exact number of fenced regions it must contain. The
// counts are asserted rather than lower-bounded so that deleting a guard
// cannot silently shrink what the gate measures.
const TARGETS = [
  {
    file: path.join(__dirname, '..', 'media', 'main.js'),
    sourceUrl: 'tableak-main.js',
    regions: 24,
  },
  {
    file: path.join(__dirname, '..', 'media', 'voice.js'),
    sourceUrl: 'tableak-voice.js',
    regions: 4,
  },
];

/**
 * The interior of every fenced region, as zero-based half-open [start, end)
 * line intervals. `start` is the line after the opening fence and `end` is
 * the index of the closing fence, so the whole interior is included.
 */
function findRegions(lines, name, expected) {
  const regions = [];
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === START_MARK) {
      assert.strictEqual(start, -1, `nested tableak-coverage:start in ${name}`);
      start = i + 1;
    } else if (t === END_MARK) {
      assert.ok(start >= 0, `tableak-coverage:end without start in ${name}`);
      regions.push([start, i]);
      start = -1;
    }
  }
  assert.strictEqual(start, -1, `unclosed tableak-coverage region in ${name}`);
  assert.strictEqual(
    regions.length,
    expected,
    `${name} must contain exactly ${expected} fenced guard regions`,
  );
  return regions;
}

function paintInstance(functions, length) {
  const painted = new Uint8Array(length);
  const ranges = [];
  for (const fn of functions) for (const r of fn.ranges) ranges.push(r);
  ranges.sort(
    (a, b) => a.startOffset - b.startOffset || b.endOffset - a.endOffset,
  );
  for (const r of ranges) {
    const v = r.count > 0 ? 1 : 0;
    const end = Math.min(r.endOffset, length);
    for (let i = Math.max(0, r.startOffset); i < end; i++) painted[i] = v;
  }
  return painted;
}

/** Union the V8 byte-level coverage of every eval of one source file. */
function collectCoverage(covDir, sourceUrl, length) {
  const covered = new Uint8Array(length);
  let instances = 0;
  for (const f of fs.readdirSync(covDir)) {
    const report = JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8'));
    for (const script of report.result || []) {
      if (!script.url || !script.url.endsWith(sourceUrl)) continue;
      instances++;
      const painted = paintInstance(script.functions, length);
      for (let i = 0; i < length; i++) if (painted[i]) covered[i] = 1;
    }
  }
  return {covered, instances};
}

/** Byte offset of the first character of every line. */
function lineOffsets(lines) {
  const starts = new Array(lines.length);
  let offset = 0;
  for (let n = 0; n < lines.length; n++) {
    starts[n] = offset;
    offset += lines[n].length + 1;
  }
  return starts;
}

function measure(target, covDir) {
  const name = path.basename(target.file);
  const src = fs.readFileSync(target.file, 'utf-8');
  const lines = src.split('\n');
  const regions = findRegions(lines, name, target.regions);
  const starts = lineOffsets(lines);
  const {covered, instances} = collectCoverage(
    covDir,
    target.sourceUrl,
    src.length,
  );
  assert.ok(
    instances > 0,
    `no ${target.sourceUrl} coverage entries found -- did the test stop ` +
      'eval-ing it with the sourceURL pragma?',
  );

  let total = 0;
  let hit = 0;
  const missed = [];
  for (const [a, b] of regions) {
    for (let n = a; n < b; n++) {
      const line = lines[n];
      if (line.trim() === '') continue;
      total++;
      let lineCovered = false;
      for (let i = 0; i < line.length; i++) {
        if (/\s/.test(line[i])) continue;
        if (covered[starts[n] + i]) {
          lineCovered = true;
          break;
        }
      }
      if (lineCovered) hit++;
      else missed.push({n: n + 1, line});
    }
  }
  assert.ok(total > 0, `${name} fenced regions must not be empty`);
  return {name, total, hit, missed, instances, regions: regions.length};
}

function main() {
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tableak-cov-'));
  const res = spawnSync(process.execPath, [TEST_FILE], {
    env: Object.assign({}, process.env, {NODE_V8_COVERAGE: covDir}),
    encoding: 'utf-8',
  });
  process.stdout.write(res.stdout || '');
  process.stderr.write(res.stderr || '');
  if (res.status !== 0) {
    console.error('coverage gate: the functional test itself FAILED');
    process.exit(res.status || 1);
  }

  let results;
  try {
    results = TARGETS.map(t => measure(t, covDir));
  } finally {
    fs.rmSync(covDir, {recursive: true, force: true});
  }

  let failed = false;
  console.log('');
  for (const r of results) {
    const pct = ((100 * r.hit) / r.total).toFixed(1);
    console.log(
      `cross-tab leak guard coverage ${r.name}: ${r.hit}/${r.total} ` +
        `(${pct}%) over ${r.regions} regions, ${r.instances} eval instances`,
    );
    if (!r.missed.length) continue;
    failed = true;
    console.error(`\nUNCOVERED CROSS-TAB GUARD LINES in ${r.name}:`);
    for (const m of r.missed) {
      console.error(`  ${String(m.n).padStart(5)}: ${m.line}`);
    }
  }
  if (failed) {
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }
  console.log(
    'coverage gate passed: 100% line coverage of the cross-tab ' +
      'text leak guard code in every fenced file.',
  );
}

main();
