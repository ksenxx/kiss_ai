// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const MAIN_JS = path.join(__dirname, '..', 'media', 'main.js');
const TEST_FILE = path.join(__dirname, 'taskPanelSettingsInfo.test.js');
const SOURCE_URL = 'taskinfo-main.js';
const START_MARK = '// taskinfo-coverage:start';
const END_MARK = '// taskinfo-coverage:end';

function findRegions(lines) {
  const regions = [];
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === START_MARK) {
      assert.strictEqual(start, -1, 'nested taskinfo-coverage:start');
      start = i + 1;
    } else if (t === END_MARK) {
      assert.ok(start >= 0, 'taskinfo-coverage:end without start');
      regions.push([start + 1, i]);
      start = -1;
    }
  }
  assert.strictEqual(start, -1, 'unclosed taskinfo-coverage region');
  assert.ok(regions.length >= 3, 'expected the taskinfo regions');
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

function main() {
  const src = fs.readFileSync(MAIN_JS, 'utf-8');
  const lines = src.split('\n');
  const regions = findRegions(lines);
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'drawer-cov-'));

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

  const covered = new Uint8Array(src.length);
  let instances = 0;
  for (const f of fs.readdirSync(covDir)) {
    const report = JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8'));
    for (const script of report.result || []) {
      if (!script.url || !script.url.endsWith(SOURCE_URL)) continue;
      instances++;
      const painted = paintInstance(script.functions, src.length);
      for (let i = 0; i < src.length; i++) if (painted[i]) covered[i] = 1;
    }
  }
  fs.rmSync(covDir, {recursive: true, force: true});
  assert.ok(
    instances > 0,
    'no taskinfo-main.js coverage entries found — did the test stop ' +
      'eval-ing main.js with the sourceURL pragma?',
  );

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

  const pct = ((100 * hit) / total).toFixed(1);
  console.log(
    `\ntask-settings info feature line coverage: ${hit}/${total} (${pct}%) ` +
      `across ${instances} eval instances`,
  );
  if (missed.length) {
    console.error('\nUNCOVERED TASKINFO LINES:');
    for (const m of missed) {
      console.error(`  ${String(m.n).padStart(5)}: ${m.line}`);
    }
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }
  console.log(
    'coverage gate passed: 100% line coverage of the task-settings info feature code.',
  );
}

main();
