// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Coverage gate for the per-panel event-timestamp feature code:
// re-runs ``panelEventTimestamp.test.js`` under V8's built-in
// coverage (``NODE_V8_COVERAGE``) and FAILS unless every non-blank
// line between the ``// panelts-coverage:start`` /
// ``// panelts-coverage:end`` markers of ``media/panelCopy.js``
// (formatEventTs + addPanelTimestamp) and ``media/main.js``
// (normalizeEventTs) was executed.  The test evals both files with
// ``//# sourceURL=panelts-*.js`` pragmas, which is how their coverage
// JSON entries are identified here.
//
// Run with:
//
//     node src/kiss/agents/vscode/test/panelEventTimestamp.coverage.js

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const MEDIA = path.join(__dirname, '..', 'media');
const TEST_FILE = path.join(__dirname, 'panelEventTimestamp.test.js');
const START_MARK = '// panelts-coverage:start';
const END_MARK = '// panelts-coverage:end';

// The gated sources and the sourceURL pragma each is eval'd under.
const TARGETS = [
  {file: path.join(MEDIA, 'panelCopy.js'), url: 'panelts-panelCopy.js'},
  {file: path.join(MEDIA, 'main.js'), url: 'panelts-main.js'},
];

/**
 * Return the [startLine, endLine] (1-based, exclusive of the marker
 * lines themselves) of every panelts-coverage region in *lines*.
 */
function findRegions(lines, file) {
  const regions = [];
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === START_MARK) {
      assert.strictEqual(start, -1, `nested panelts-coverage:start (${file})`);
      start = i + 1;
    } else if (t === END_MARK) {
      assert.ok(start >= 0, `panelts-coverage:end without start (${file})`);
      regions.push([start + 1, i]); // exclusive of both marker lines
      start = -1;
    }
  }
  assert.strictEqual(start, -1, `unclosed panelts-coverage region (${file})`);
  assert.ok(regions.length >= 1, `expected a panelts region in ${file}`);
  return regions;
}

// Paint one eval-instance's char coverage from its V8 ranges.  Ranges
// nest; applying them sorted by (startOffset asc, endOffset desc)
// paints outer ranges first so inner ranges override, matching V8
// block-coverage semantics.
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

/**
 * Enforce 100% line coverage of every panelts region of one target.
 * Returns {hit, total, missed} for the summary line.
 */
function gateTarget(target, reports) {
  const src = fs.readFileSync(target.file, 'utf-8');
  const lines = src.split('\n');
  const regions = findRegions(lines, target.file);

  // A char is covered if ANY eval instance executed it.  The eval'd
  // source carries the appended sourceURL pragma line; coverage
  // offsets index that eval text, whose first src.length chars match
  // the file exactly.
  const covered = new Uint8Array(src.length);
  let instances = 0;
  for (const report of reports) {
    for (const script of report.result || []) {
      if (!script.url || !script.url.endsWith(target.url)) continue;
      instances++;
      const painted = paintInstance(script.functions, src.length);
      for (let i = 0; i < src.length; i++) if (painted[i]) covered[i] = 1;
    }
  }
  assert.ok(
    instances > 0,
    `no ${target.url} coverage entries found — did the test stop ` +
      'eval-ing with the sourceURL pragma?',
  );

  // Line gate: every non-blank line inside every panelts-coverage
  // region must have at least one covered non-whitespace character.
  let offset = 0;
  const lineStart = new Array(lines.length);
  for (let n = 0; n < lines.length; n++) {
    lineStart[n] = offset;
    offset += lines[n].length + 1;
  }
  let total = 0;
  let hit = 0;
  const missed = [];
  for (const [a, b] of regions) {
    for (let n = a - 1; n < b - 1; n++) {
      const line = lines[n];
      if (line.trim() === '') continue;
      total++;
      let lineCovered = false;
      for (let i = 0; i < line.length; i++) {
        if (/\s/.test(line[i])) continue;
        if (covered[lineStart[n] + i]) {
          lineCovered = true;
          break;
        }
      }
      if (lineCovered) hit++;
      else missed.push({n: n + 1, line});
    }
  }
  return {hit, total, missed};
}

function main() {
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'panelts-cov-'));

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

  const reports = fs
    .readdirSync(covDir)
    .map(f => JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8')));
  fs.rmSync(covDir, {recursive: true, force: true});

  let failed = false;
  for (const target of TARGETS) {
    const {hit, total, missed} = gateTarget(target, reports);
    const pct = ((100 * hit) / total).toFixed(1);
    console.log(
      `\n${path.basename(target.file)} panelts line coverage: ` +
        `${hit}/${total} (${pct}%)`,
    );
    if (missed.length) {
      console.error('UNCOVERED PANELTS LINES:');
      for (const m of missed) {
        console.error(`  ${String(m.n).padStart(5)}: ${m.line}`);
      }
      failed = true;
    }
  }
  if (failed) {
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }
  console.log(
    '\ncoverage gate passed: 100% line coverage of the panel-timestamp ' +
      'feature code.',
  );
}

main();
