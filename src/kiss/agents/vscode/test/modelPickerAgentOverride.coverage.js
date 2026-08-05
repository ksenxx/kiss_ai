// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the model-picker override code
// in media/main.js (the regions fenced by `// modelpick-coverage:start` /
// `// modelpick-coverage:end`) when running the functional jsdom suite in
// test/modelPickerAgentOverride.test.js.
//
// That code is the whole point of the feature -- it decides whether the picker
// shows the user's own model or the one a running agent switched itself to --
// so an unexecuted line there is an untested claim about what the user sees.

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const TEST_FILE = path.join(__dirname, 'modelPickerAgentOverride.test.js');
const START_MARK = '// modelpick-coverage:start';
const END_MARK = '// modelpick-coverage:end';

// The fenced file, the sourceURL pragma the test appends when it evals it,
// and the exact number of fenced regions it must contain. The count is
// asserted rather than lower-bounded so deleting a region cannot silently
// shrink what the gate measures.
const TARGET = {
  file: path.join(__dirname, '..', 'media', 'main.js'),
  sourceUrl: 'modelpick-main.js',
  regions: 4,
};

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
      assert.strictEqual(start, -1, `nested modelpick-coverage:start in ${name}`);
      start = i + 1;
    } else if (t === END_MARK) {
      assert.ok(start >= 0, `modelpick-coverage:end without start in ${name}`);
      regions.push([start, i]);
      start = -1;
    }
  }
  assert.strictEqual(start, -1, `unclosed modelpick-coverage region in ${name}`);
  assert.strictEqual(
    regions.length,
    expected,
    `${name} must contain exactly ${expected} fenced regions`,
  );
  return regions;
}

/** Paint one script instance's V8 ranges onto a byte-level covered map. */
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

/** Measure fenced-region line coverage of TARGET from the V8 reports. */
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
      if (line.trim() === '' || line.trim().startsWith('//')) continue;
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
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'modelpick-cov-'));
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

  let r;
  try {
    r = measure(TARGET, covDir);
  } finally {
    fs.rmSync(covDir, {recursive: true, force: true});
  }

  const pct = ((100 * r.hit) / r.total).toFixed(1);
  console.log('');
  console.log(
    `model picker override coverage ${r.name}: ${r.hit}/${r.total} ` +
      `(${pct}%) over ${r.regions} regions, ${r.instances} eval instances`,
  );
  if (r.missed.length) {
    console.error(`\nUNCOVERED MODEL PICKER LINES in ${r.name}:`);
    for (const m of r.missed) {
      console.error(`  ${String(m.n).padStart(5)}: ${m.line}`);
    }
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }
  console.log(
    'coverage gate passed: 100% line coverage of the model picker ' +
      'override code.',
  );
}

main();
