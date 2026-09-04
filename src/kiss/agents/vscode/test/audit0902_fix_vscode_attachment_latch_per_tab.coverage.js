// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate for the per-tab attachment latch in media/main.js (the
// region fenced by `// attachlatch-coverage:start` / `:end` in
// sendMessage()) when running
// test/audit0902_fix_vscode_attachment_latch_per_tab.test.js.
//
// Stricter than a line gate: every non-whitespace byte of every code line
// in the region must have executed.  V8 block coverage assigns a zero
// count to an untaken `return;`, `finally` body or `||` right operand
// even when it shares its line with executed code, so an unexercised
// branch fails the gate.

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const MAIN_JS = path.join(__dirname, '..', 'media', 'main.js');
const TEST_FILE = path.join(
  __dirname,
  'audit0902_fix_vscode_attachment_latch_per_tab.test.js',
);
const SOURCE_URL = 'audit0902-attachlatch-main.js';
const START_MARK = '// attachlatch-coverage:start';
const END_MARK = '// attachlatch-coverage:end';

function findRegions(lines) {
  const regions = [];
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === START_MARK) {
      assert.strictEqual(start, -1, 'nested attachlatch-coverage:start');
      start = i + 1;
    } else if (t === END_MARK) {
      assert.ok(start >= 0, 'attachlatch-coverage:end without start');
      regions.push([start + 1, i]);
      start = -1;
    }
  }
  assert.strictEqual(start, -1, 'unclosed attachlatch-coverage region');
  assert.ok(regions.length >= 1, 'no attachlatch-coverage region found');
  return regions;
}

// Later (inner, shorter) ranges override earlier ones, as in V8's model.
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
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'attachlatch-cov-'));

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
  assert.ok(instances > 0, `no ${SOURCE_URL} coverage entries found`);

  const lineStart = new Array(lines.length);
  let offset = 0;
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
      const t = line.trim();
      if (t === '' || t.startsWith('//')) continue;
      total++;
      const uncovered = [];
      for (let i = 0; i < line.length; i++) {
        if (/\s/.test(line[i])) continue;
        if (!covered[lineStart[n] + i]) uncovered.push(i);
      }
      if (uncovered.length === 0) hit++;
      else missed.push({n: n + 1, line, from: uncovered[0]});
    }
  }

  const pct = ((100 * hit) / total).toFixed(1);
  console.log(
    `\nattachment-latch byte-level line coverage: ${hit}/${total} (${pct}%)`,
  );
  if (missed.length) {
    console.error(
      '\nUNCOVERED ATTACHLATCH CODE (line: text, first uncovered column):',
    );
    for (const m of missed) {
      console.error(
        `  ${String(m.n).padStart(5)}: ${m.line}   [col ${m.from}]`,
      );
    }
    console.error(
      '\ncoverage gate FAILED: every branch of the region must run',
    );
    process.exit(1);
  }
  console.log('coverage gate passed: 100% of the attachment-latch code ran.');
}

main();
