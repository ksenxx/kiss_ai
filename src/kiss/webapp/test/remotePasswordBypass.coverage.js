// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the WHOLE of
// media/transport.js and media/assetLoadGuard.js (not a fenced region) when
// running test/remotePasswordBypass.test.js together with
// test/transportReloadOnNewDaemon.test.js (the shim's GET /api/daemon
// reload check, whose pages have a fetch), plus every branch except the
// spec-unreachable `location.reload()` catch.

'use strict';

const {runCovered, measure} = require('./coverage_gate');

// The two page scripts of media/ that the functional tests eval, each
// under its own sourceURL pragma.
const SCRIPTS = [
  {file: 'transport.js', sourceUrl: 'ws-shim.js', wholeFile: true},
  {file: 'assetLoadGuard.js', sourceUrl: 'asset-guard.js', wholeFile: true},
];

const SUITES = [
  'remotePasswordBypass.test.js',
  'transportReloadOnNewDaemon.test.js',
];

function main() {
  let reports = [];
  for (const suite of SUITES) {
    reports = reports.concat(runCovered(suite, 'ws-shim'));
  }
  for (const target of SCRIPTS) gate(target, reports);
}

/**
 * Require every code line of `target` hit, then every non-whitespace
 * byte span except the `location.reload()` catch body.
 */
function gate(target, reports) {
  const url = target.sourceUrl;
  const r = measure(target, 'ws-shim', reports, false);
  const pct = ((100 * r.hit) / r.total).toFixed(1);
  console.log(
    `\n${url} line coverage: ${r.hit}/${r.total} (${pct}%) ` +
      `across ${r.instances} eval instances`,
  );
  if (r.missed.length) {
    console.error(`\nUNCOVERED ${url} LINES:`);
    for (const m of r.missed) {
      console.error(`  ${String(m.n).padStart(4)}: ${m.line}`);
    }
    console.error('\ncoverage gate FAILED: 100% line coverage required');
    process.exit(1);
  }

  const {src, covered} = r;
  const badSpans = [];
  let i = 0;
  while (i < src.length) {
    if (covered[i] || /\s/.test(src[i])) {
      i++;
      continue;
    }
    const start = i;
    while (i < src.length && !covered[i]) i++;
    const text = src.slice(start, i).trim();
    const context = src.slice(Math.max(0, start - 80), start);
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
    console.error(
      '\ncoverage gate FAILED: only the location.reload() ' +
        'catch (unreachable by spec) may be uncovered',
    );
    process.exit(1);
  }
  console.log(
    `coverage gate passed: 100% line coverage of ${url}; all branches ` +
      'covered except the spec-unreachable location.reload() catch.',
  );
}

main();
