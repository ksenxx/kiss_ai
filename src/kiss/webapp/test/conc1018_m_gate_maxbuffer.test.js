// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for coverage_gate.js's `runCovered`: a gated suite
// that passes but prints more than spawnSync's default 1 MiB maxBuffer
// must still be collected (before the fix the child was SIGKILLed with
// ENOBUFS and the gate blamed "the functional test itself FAILED"), and
// a suite that overruns even the raised 64 MiB bound must be reported
// with the real spawn error instead of that misleading line.
//
// `runCovered` exits the process on failure, so each case runs a tiny
// gate script in a child node process.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');

const GATE_HELPER = path.join(__dirname, 'coverage_gate.js');

function makeScratch() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'conc1018-gate-'));
  fs.writeFileSync(
    path.join(dir, 'gate.js'),
    [
      "'use strict';",
      `const {runCovered} = require(${JSON.stringify(GATE_HELPER)});`,
      "const reports = runCovered(process.argv[2], 'conc1018m');",
      "console.log('reports=' + reports.length);",
      '',
    ].join('\n'),
  );
  return dir;
}

/** A passing suite that prints `mib` MiB on stdout, then a marker. */
function writeChattySuite(dir, name, mib) {
  const file = path.join(dir, name);
  fs.writeFileSync(
    file,
    [
      "'use strict';",
      `process.stdout.write('x'.repeat(${mib} << 20));`,
      `console.log('\\n${name} done');`,
      '',
    ].join('\n'),
  );
  // runCovered resolves its argument against the test directory.
  return path.relative(__dirname, file);
}

function runGate(dir, suite) {
  const res = spawnSync(process.execPath, [path.join(dir, 'gate.js'), suite], {
    encoding: 'utf8',
    timeout: 60000,
    maxBuffer: 256 << 20,
  });
  assert.ok(!res.error, `the gate process ran: ${res.error}`);
  return res;
}

function main() {
  const dir = makeScratch();
  try {
    const big = runGate(dir, writeChattySuite(dir, 'big.test.js', 2));
    assert.strictEqual(
      big.status,
      0,
      `a passing suite with 2 MiB of output passes the gate\n${big.stderr}`,
    );
    assert.ok(big.stdout.includes('big.test.js done'), 'output forwarded');
    assert.match(big.stdout, /reports=[1-9]/, 'coverage reports collected');
    assert.ok(!big.stderr.includes('FAILED'), big.stderr);
    console.log('ok - a 2 MiB suite output no longer fails the gate');

    const huge = runGate(dir, writeChattySuite(dir, 'huge.test.js', 66));
    assert.strictEqual(huge.status, 1, 'an overrun is still a gate failure');
    assert.ok(
      huge.stderr.includes('ENOBUFS'),
      `the real spawn error is reported\n${huge.stderr.slice(-500)}`,
    );
    assert.ok(
      !huge.stderr.includes('the functional test itself FAILED'),
      `the suite is not blamed for the overrun\n${huge.stderr.slice(-500)}`,
    );
    console.log('ok - an overrun past 64 MiB names ENOBUFS, not the suite');
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

main();
