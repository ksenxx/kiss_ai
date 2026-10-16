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

const MEDIA = path.resolve(__dirname, '..', 'media');
const TEST_FILE = path.join(__dirname, 'remotePasswordBypass.test.js');
const CHILD_TIMEOUT_MS = 5 * 60 * 1000;
// The two page scripts of media/ that the functional test eval's, each
// under its own sourceURL pragma.
const SCRIPTS = [
  {name: 'transport.js', url: 'ws-shim.js'},
  {name: 'assetLoadGuard.js', url: 'asset-guard.js'},
];

function readMediaJs(name) {
  return fs.readFileSync(path.join(MEDIA, name), 'utf-8');
}

function paintInstance(functions, length) {
  const painted = new Uint8Array(length);
  const ranges = [];
  for (const fn of functions) for (const r of fn.ranges) ranges.push(r);
  ranges.sort((a, b) => a.startOffset - b.startOffset ||
                        b.endOffset - a.endOffset);
  for (const r of ranges) {
    const v = r.count > 0 ? 1 : 0;
    const end = Math.min(r.endOffset, length);
    for (let i = Math.max(0, r.startOffset); i < end; i++) painted[i] = v;
  }
  return painted;
}

function main() {
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ws-shim-cov-'));

  // The child inherits this process's V8 flags (run-all.js passes
  // --no-maglev --no-concurrent-sparkplug to dodge a V8 exit hang) and is
  // killed after CHILD_TIMEOUT_MS so a hung functional test is reported
  // here by name instead of stalling the whole run.
  const res = spawnSync(process.execPath, [...process.execArgv, TEST_FILE], {
    env: Object.assign({}, process.env, {NODE_V8_COVERAGE: covDir}),
    encoding: 'utf-8',
    timeout: CHILD_TIMEOUT_MS,
    killSignal: 'SIGKILL',
  });
  process.stdout.write(res.stdout || '');
  process.stderr.write(res.stderr || '');
  if (res.status !== 0) {
    fs.rmSync(covDir, {recursive: true, force: true});
    console.error(
      res.error && res.error.code === 'ETIMEDOUT'
        ? `coverage gate: the functional test did not exit within ${CHILD_TIMEOUT_MS / 1000}s, killed`
        : 'coverage gate: the functional test itself FAILED',
    );
    process.exit(res.status || 1);
  }

  const reports = fs.readdirSync(covDir).map((f) =>
    JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8')));
  fs.rmSync(covDir, {recursive: true, force: true});
  for (const {name, url} of SCRIPTS) gate(readMediaJs(name), url, reports);
}

function gate(shim, url, reports) {
  const covered = new Uint8Array(shim.length);
  let instances = 0;
  for (const report of reports) {
    for (const script of report.result || []) {
      if (script.url !== url) continue;
      instances++;
      const painted = paintInstance(script.functions, shim.length);
      for (let i = 0; i < shim.length; i++) if (painted[i]) covered[i] = 1;
    }
  }
  assert.ok(instances > 0,
    `no ${url} coverage entries found — did the test stop eval-ing ` +
    'the script with the sourceURL pragma?');

  const lines = shim.split('\n');
  let offset = 0;
  let total = 0;
  let hit = 0;
  const missed = [];
  for (let n = 0; n < lines.length; n++) {
    const line = lines[n];
    const start = offset;
    offset += line.length + 1;
    if (line.trim() === '') continue;
    total++;
    let lineCovered = false;
    for (let i = 0; i < line.length; i++) {
      if (/\s/.test(line[i])) continue;
      if (covered[start + i]) { lineCovered = true; break; }
    }
    if (lineCovered) hit++;
    else missed.push({n: n + 1, line});
  }

  const pct = ((100 * hit) / total).toFixed(1);
  console.log(
    `\n${url} line coverage: ${hit}/${total} (${pct}%) ` +
    `across ${instances} eval instances`);
  if (missed.length) {
    console.error(`\nUNCOVERED ${url} LINES:`);
    for (const m of missed) {
      console.error(`  ${String(m.n).padStart(4)}: ${m.line}`);
    }
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }

  const badSpans = [];
  let i = 0;
  while (i < shim.length) {
    if (covered[i] || /\s/.test(shim[i])) { i++; continue; }
    let start = i;
    while (i < shim.length && !covered[i]) i++;
    const text = shim.slice(start, i).trim();
    const context = shim.slice(Math.max(0, start - 80), start);
    const isReloadCatch =
      /window\.location\.reload\(\)/.test(context) &&
      /^catch\s*\(_e\)\s*\{\}$/.test(text);
    if (!isReloadCatch) badSpans.push({start, text});
  }
  if (badSpans.length) {
    console.error(`\nUNCOVERED ${url} BRANCHES:`);
    for (const s of badSpans) {
      console.error(`  at char ${s.start}: ${JSON.stringify(s.text)}`);
    }
    console.error('\ncoverage gate FAILED: only the location.reload() ' +
      'catch (unreachable by spec) may be uncovered');
    process.exit(1);
  }
  console.log(
    `coverage gate passed: 100% line coverage of ${url}; all branches ` +
    'covered except the spec-unreachable location.reload() catch.');
}

main();
