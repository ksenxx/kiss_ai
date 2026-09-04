// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate for the audit0903 fixes: requires 100% line coverage
// of every fenced region introduced by the flash-timer race fixes and
// the composer-reset extraction, when running the functional jsdom
// suites:
//
//   media/panelCopy.js  copyflash0903      audit0903_panelcopy_flash_timer
//   media/tips.js       tipsflash0903      audit0903_tips_copy_flash_timer
//   media/main.js       shareflash0903     audit0903_main_flash_timers
//   media/main.js       urlflash0903       audit0903_main_flash_timers
//   media/main.js       sidebarflash0903   audit0903_main_flash_timers
//   media/main.js       composerreset0903  audit0903_composer_reset
//
// Each suite evals its media file with a distinct //# sourceURL
// pragma so the V8 coverage entries can be mapped back to the file.

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const MEDIA = path.join(__dirname, '..', 'media');

const TESTS = [
  'audit0903_panelcopy_flash_timer.test.js',
  'audit0903_tips_copy_flash_timer.test.js',
  'audit0903_main_flash_timers.test.js',
  'audit0903_composer_reset.test.js',
];

const TARGETS = [
  {
    file: path.join(MEDIA, 'panelCopy.js'),
    sourceUrl: 'audit0903-panelcopy.js',
    markers: ['copyflash0903'],
  },
  {
    file: path.join(MEDIA, 'tips.js'),
    sourceUrl: 'audit0903-tips.js',
    markers: ['tipsflash0903'],
  },
  {
    file: path.join(MEDIA, 'main.js'),
    sourceUrl: 'audit0903-main.js',
    markers: [
      'shareflash0903',
      'urlflash0903',
      'sidebarflash0903',
      'composerreset0903',
    ],
  },
];

function findRegions(lines, marker) {
  const startMark = '// ' + marker + '-coverage:start';
  const endMark = '// ' + marker + '-coverage:end';
  const regions = [];
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === startMark) {
      assert.strictEqual(start, -1, 'nested ' + startMark);
      start = i + 1;
    } else if (t === endMark) {
      assert.ok(start >= 0, endMark + ' without start');
      regions.push([start + 1, i]);
      start = -1;
    }
  }
  assert.strictEqual(start, -1, 'unclosed ' + marker + '-coverage region');
  assert.ok(regions.length >= 1, 'expected a ' + marker + ' region');
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

function checkTarget(target, covDir) {
  const src = fs.readFileSync(target.file, 'utf-8');
  const lines = src.split('\n');
  const covered = new Uint8Array(src.length);
  let instances = 0;
  for (const f of fs.readdirSync(covDir)) {
    const report = JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8'));
    for (const script of report.result || []) {
      if (!script.url || !script.url.endsWith(target.sourceUrl)) continue;
      instances++;
      const painted = paintInstance(script.functions, src.length);
      for (let i = 0; i < src.length; i++) if (painted[i]) covered[i] = 1;
    }
  }
  assert.ok(
    instances > 0,
    'no ' +
      target.sourceUrl +
      ' coverage entries found — did a test stop eval-ing ' +
      path.basename(target.file) +
      ' with the sourceURL pragma?',
  );

  let offset = 0;
  const lineStart = new Array(lines.length);
  for (let n = 0; n < lines.length; n++) {
    lineStart[n] = offset;
    offset += lines[n].length + 1;
  }

  let failed = false;
  for (const marker of target.markers) {
    const regions = findRegions(lines, marker);
    let total = 0;
    let hit = 0;
    const missed = [];
    for (const [a, b] of regions) {
      for (let n = a - 1; n < b - 1; n++) {
        const line = lines[n];
        const t = line.trim();
        if (
          t === '' ||
          t.startsWith('//') ||
          t.startsWith('*') ||
          t === '/**'
        ) {
          continue;
        }
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
    const pct = total ? ((100 * hit) / total).toFixed(1) : '100.0';
    console.log(
      `${path.basename(target.file)} ${marker}: ${hit}/${total} (${pct}%)`,
    );
    if (missed.length) {
      failed = true;
      console.error(`\nUNCOVERED ${marker} LINES:`);
      for (const m of missed) {
        console.error(`  ${String(m.n).padStart(5)}: ${m.line}`);
      }
    }
  }
  return failed;
}

function main() {
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'audit0903-cov-'));
  for (const test of TESTS) {
    const res = spawnSync(process.execPath, [path.join(__dirname, test)], {
      env: Object.assign({}, process.env, {NODE_V8_COVERAGE: covDir}),
      encoding: 'utf-8',
    });
    process.stdout.write(res.stdout || '');
    process.stderr.write(res.stderr || '');
    if (res.status !== 0) {
      console.error(`coverage gate: ${test} itself FAILED`);
      process.exit(res.status || 1);
    }
  }

  let failed = false;
  for (const target of TARGETS) {
    if (checkTarget(target, covDir)) failed = true;
  }
  fs.rmSync(covDir, {recursive: true, force: true});
  if (failed) {
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }
  console.log(
    'coverage gate passed: 100% line coverage of every audit0903 region.',
  );
}

main();
