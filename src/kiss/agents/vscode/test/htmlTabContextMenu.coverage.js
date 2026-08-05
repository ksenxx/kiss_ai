// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the html-tab context menu
// feature — the regions of media/main.js fenced by
// `// ctxmenu-coverage:start` / `// ctxmenu-coverage:end` plus the whole of
// media/contentContextMenu.js — when running the functional jsdom suite in
// test/htmlTabContextMenu.test.js.

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const MEDIA = path.join(__dirname, '..', 'media');
const MAIN_JS = path.join(MEDIA, 'main.js');
const MODULE_JS = path.join(MEDIA, 'contentContextMenu.js');
const TEST_FILE = path.join(__dirname, 'htmlTabContextMenu.test.js');
const MAIN_URL = 'ctxmenu-main.js';
const MODULE_URL = 'contentContextMenu.js';
const START_MARK = '// ctxmenu-coverage:start';
const END_MARK = '// ctxmenu-coverage:end';
const EXPECTED_REGIONS = 3;

function findRegions(lines) {
  const regions = [];
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === START_MARK) {
      assert.strictEqual(start, -1, 'nested ctxmenu-coverage:start');
      start = i + 1;
    } else if (t === END_MARK) {
      assert.ok(start >= 0, 'ctxmenu-coverage:end without start');
      regions.push([start + 1, i]);
      start = -1;
    }
  }
  assert.strictEqual(start, -1, 'unclosed ctxmenu-coverage region');
  assert.strictEqual(
    regions.length,
    EXPECTED_REGIONS,
    `expected ${EXPECTED_REGIONS} ctxmenu-coverage regions in main.js`,
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

// The sandboxed-iframe copy of the menu is a Function.prototype.toString
// re-serialisation evaluated inside a `srcdoc` document, so V8 attributes it
// to the iframe URL, not to contentContextMenu.js.  Coverage is therefore
// collected from the module instances the parent documents load, which the
// suite exercises through the very same public API.
function collectCovered(covDir, urlSuffix, length) {
  const covered = new Uint8Array(length);
  let instances = 0;
  for (const f of fs.readdirSync(covDir)) {
    const report = JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8'));
    for (const script of report.result || []) {
      if (!script.url || !script.url.endsWith(urlSuffix)) continue;
      instances++;
      const painted = paintInstance(script.functions, length);
      for (let i = 0; i < length; i++) if (painted[i]) covered[i] = 1;
    }
  }
  return {covered, instances};
}

function lineStarts(lines) {
  const starts = new Array(lines.length);
  let offset = 0;
  for (let n = 0; n < lines.length; n++) {
    starts[n] = offset;
    offset += lines[n].length + 1;
  }
  return starts;
}

function isCoveredLine(line, start, covered) {
  for (let i = 0; i < line.length; i++) {
    if (/\s/.test(line[i])) continue;
    if (covered[start + i]) return true;
  }
  return false;
}

// A line that only closes a block or holds a comment carries no executable
// code, so V8 never attributes a range to it.
function isExecutable(line) {
  const t = line.trim();
  if (t === '') return false;
  if (t.startsWith('//') || t.startsWith('*') || t.startsWith('/*')) {
    return false;
  }
  return !/^[)\]}',;]+$/.test(t);
}

function measure(label, file, regions, covered) {
  const lines = fs.readFileSync(file, 'utf-8').split('\n');
  const starts = lineStarts(lines);
  let total = 0;
  let hit = 0;
  const missed = [];
  for (const [a, b] of regions) {
    for (let n = a - 1; n < b - 1; n++) {
      if (!isExecutable(lines[n])) continue;
      total++;
      if (isCoveredLine(lines[n], starts[n], covered)) hit++;
      else missed.push({n: n + 1, line: lines[n]});
    }
  }
  const pct = ((100 * hit) / total).toFixed(1);
  console.log(`${label} line coverage: ${hit}/${total} (${pct}%)`);
  if (missed.length) {
    console.error(`\nUNCOVERED ${label.toUpperCase()} LINES:`);
    for (const m of missed) {
      console.error(`  ${String(m.n).padStart(5)}: ${m.line}`);
    }
  }
  return missed.length === 0;
}

function main() {
  const mainSrc = fs.readFileSync(MAIN_JS, 'utf-8');
  const moduleSrc = fs.readFileSync(MODULE_JS, 'utf-8');
  const regions = findRegions(mainSrc.split('\n'));
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ctxmenu-cov-'));

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

  const main0 = collectCovered(covDir, MAIN_URL, mainSrc.length);
  const mod0 = collectCovered(covDir, MODULE_URL, moduleSrc.length);
  fs.rmSync(covDir, {recursive: true, force: true});
  assert.ok(
    main0.instances > 0,
    'no ctxmenu-main.js coverage entries found — did the test stop ' +
      'eval-ing main.js with the sourceURL pragma?',
  );
  assert.ok(
    mod0.instances > 0,
    'no contentContextMenu.js coverage entries found',
  );

  const moduleLines = moduleSrc.split('\n').length;
  const okMain = measure('main.js ctxmenu region', MAIN_JS, regions, main0.covered);
  const okModule = measure(
    'contentContextMenu.js',
    MODULE_JS,
    [[1, moduleLines + 1]],
    mod0.covered,
  );
  if (!okMain || !okModule) {
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }
  console.log(
    'coverage gate passed: 100% line coverage of the html-tab ' +
      'context menu feature code.',
  );
}

main();
