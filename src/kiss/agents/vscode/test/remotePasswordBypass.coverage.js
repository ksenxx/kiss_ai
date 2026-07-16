// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Coverage gate for the remote-password shim (``_WS_SHIM_JS`` in
// ``web_server.py``): re-runs ``remotePasswordBypass.test.js`` under
// V8's built-in coverage (``NODE_V8_COVERAGE``) and FAILS unless every
// line of the shim was executed.  The test evals the shim with a
// ``//# sourceURL=ws-shim.js`` pragma, which is how the coverage JSON
// entries for the shim are identified here.
//
// Run with:
//
//     node src/kiss/agents/vscode/test/remotePasswordBypass.coverage.js

'use strict';

const assert = require('assert');
const {spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const WEB_SERVER_PY = path.resolve(
  __dirname, '..', '..', '..', 'server', 'web_server.py',
);
const TEST_FILE = path.join(__dirname, 'remotePasswordBypass.test.js');
const SHIM_URL = 'ws-shim.js';

function readShimJs() {
  const src = fs.readFileSync(WEB_SERVER_PY, 'utf-8');
  const m = src.match(/_WS_SHIM_JS\s*=\s*r"""([\s\S]*?)"""/);
  assert.ok(m, 'could not locate _WS_SHIM_JS literal in web_server.py');
  return m[1];
}

// Paint one eval-instance's char coverage from its V8 ranges.  Ranges
// nest; applying them sorted by (startOffset asc, endOffset desc)
// paints outer ranges first so inner ranges override, matching V8
// block-coverage semantics.
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
  const shim = readShimJs();
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ws-shim-cov-'));

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

  // A shim char is covered if ANY eval instance executed it.
  const covered = new Uint8Array(shim.length);
  let instances = 0;
  for (const f of fs.readdirSync(covDir)) {
    const report = JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf-8'));
    for (const script of report.result || []) {
      if (script.url !== SHIM_URL) continue;
      instances++;
      const painted = paintInstance(script.functions, shim.length);
      for (let i = 0; i < shim.length; i++) if (painted[i]) covered[i] = 1;
    }
  }
  fs.rmSync(covDir, {recursive: true, force: true});
  assert.ok(instances > 0,
    'no ws-shim.js coverage entries found — did the test stop eval-ing ' +
    'the shim with the sourceURL pragma?');

  // Line coverage: every non-blank line must have at least one covered
  // non-whitespace character.
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
    `\nws-shim.js line coverage: ${hit}/${total} (${pct}%) ` +
    `across ${instances} eval instances`);
  if (missed.length) {
    console.error('\nUNCOVERED SHIM LINES:');
    for (const m of missed) {
      console.error(`  ${String(m.n).padStart(4)}: ${m.line}`);
    }
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }

  // Branch-level gate: every uncovered char span must be the single
  // spec-unreachable branch — the defensive catch around
  // ``window.location.reload()``.  ``Location.reload`` is an
  // unforgeable platform API (its property cannot be replaced, and it
  // does not throw when invoked on the page's own location), so this
  // catch exists only for exotic embedded browsers and cannot be
  // executed from a test.  Everything else must be covered.
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
      /^catch\s*\(e\)\s*\{\}$/.test(text);
    if (!isReloadCatch) badSpans.push({start, text});
  }
  if (badSpans.length) {
    console.error('\nUNCOVERED SHIM BRANCHES:');
    for (const s of badSpans) {
      console.error(`  at char ${s.start}: ${JSON.stringify(s.text)}`);
    }
    console.error('\ncoverage gate FAILED: only the location.reload() ' +
      'catch (unreachable by spec) may be uncovered');
    process.exit(1);
  }
  console.log(
    'coverage gate passed: 100% line coverage of the shim; all branches ' +
    'covered except the spec-unreachable location.reload() catch.');
}

main();
