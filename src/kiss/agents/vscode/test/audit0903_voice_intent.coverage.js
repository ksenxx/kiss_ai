// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the code changed by the
// 2026-09-03 vscode-main audit — the regions fenced by
// `// audit0903-coverage:start` / `// audit0903-coverage:end` in
// out/SorcarSidebarView.js (the _voiceEnabled intent flag: voiceToggle,
// the voiceSensitivity restart gate, the hide/show suspension, and the
// webview/view dispose paths) — when running the functional suite
// audit0903_voice_intent.test.js.
//
// Same V8-coverage pattern as audit0902_*.coverage.js; that helper's
// marks are hardcoded to audit0902, hence this gate carries its own
// audit0903 marks and exports its own runGate for the other audit0903
// gates (same sharing convention as the 0902 files).

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {pathToFileURL} = require('url');

const START_MARK = '// audit0903-coverage:start';
const END_MARK = '// audit0903-coverage:end';

function findRegions(lines) {
  const regions = [];
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === START_MARK) {
      assert.strictEqual(start, -1, 'nested audit0903-coverage:start');
      start = i + 1;
    } else if (t === END_MARK) {
      assert.ok(start >= 0, 'audit0903-coverage:end without start');
      regions.push([start + 1, i]);
      start = -1;
    }
  }
  assert.strictEqual(start, -1, 'unclosed audit0903-coverage region');
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

function isCodeLine(line) {
  const t = line.trim();
  if (!t) return false;
  if (t.startsWith('//')) return false;
  if (t === '}' || t === '});' || t === '} else {') return false;
  return true;
}

/**
 * Run *testFiles* under NODE_V8_COVERAGE and require every code line of
 * the audit0903 fenced regions in *targets* (absolute paths of the files
 * the tests exercise) to have executed in at least one of them.  Exits
 * non-zero on a miss.
 *
 * @param {string} label Gate name for the report lines.
 * @param {string[]} testFiles Functional suites to run, in order.
 * @param {string[]} targets Files whose fenced regions must be covered.
 */
function runGate(label, testFiles, targets) {
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'audit0903-cov-'));
  for (const testFile of testFiles) {
    const res = spawnSync(process.execPath, [testFile], {
      env: Object.assign({}, process.env, {NODE_V8_COVERAGE: covDir}),
      encoding: 'utf-8',
    });
    process.stdout.write(res.stdout || '');
    process.stderr.write(res.stderr || '');
    if (res.status !== 0) {
      console.error(`coverage gate: ${path.basename(testFile)} itself FAILED`);
      process.exit(res.status || 1);
    }
  }

  const reports = [];
  for (const f of fs.readdirSync(covDir)) {
    reports.push(JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8')));
  }
  fs.rmSync(covDir, {recursive: true, force: true});

  let failed = false;
  for (const file of targets) {
    const name = path.basename(file);
    const src = fs.readFileSync(file, 'utf-8');
    const url = pathToFileURL(file).href;
    const covered = new Uint8Array(src.length);
    let instances = 0;
    for (const report of reports) {
      for (const script of report.result || []) {
        if (script.url !== url) continue;
        instances++;
        const painted = paintInstance(script.functions, src.length);
        for (let i = 0; i < src.length; i++) if (painted[i]) covered[i] = 1;
      }
    }
    assert.ok(instances > 0, `no coverage entries for ${url}`);

    const lines = src.split('\n');
    const regions = findRegions(lines);
    assert.ok(regions.length >= 1, `no audit0903 regions in ${name}`);
    const lineStart = [];
    let offset = 0;
    for (const line of lines) {
      lineStart.push(offset);
      offset += line.length + 1;
    }
    let total = 0;
    let hit = 0;
    const missed = [];
    for (const [from, to] of regions) {
      for (let n = from; n <= to; n++) {
        const line = lines[n - 1];
        if (!isCodeLine(line)) continue;
        total++;
        const s = lineStart[n - 1];
        const e = s + line.length;
        let any = false;
        for (let i = s; i < e; i++) {
          if (covered[i]) {
            any = true;
            break;
          }
        }
        if (any) hit++;
        else missed.push(`${name}:${n}: ${line.trim()}`);
      }
    }
    const pct = total ? (100 * hit) / total : 100;
    console.log(
      `audit0903 coverage ${name}: ${hit}/${total} lines (${pct.toFixed(1)}%)`,
    );
    if (missed.length) {
      failed = true;
      for (const m of missed) console.log('  MISSED ' + m);
    }
  }
  if (failed) {
    console.error(`${label}: FAILED`);
    process.exit(1);
  }
  console.log(`${label}: 100% ok`);
}

module.exports = {runGate};

if (require.main === module) {
  const TARGET = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  if (process.platform === 'win32') {
    console.log('SKIP: the functional suite needs POSIX process groups');
    process.exit(0);
  }
  if (!fs.existsSync(TARGET)) {
    console.log(`SKIP: ${TARGET} missing — run \`npm run compile\``);
    process.exit(0);
  }
  runGate(
    'audit0903_voice_intent.coverage',
    [path.join(__dirname, 'audit0903_voice_intent.test.js')],
    [TARGET],
  );
}
