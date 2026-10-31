// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for test/run-all.js's choice of suites: a suite that a
// *.coverage.js gate runs is run through that gate only, whether it is
// the gate's own X.test.js or another suite the gate names, as
// remotePasswordBypass.coverage.js names transportReloadOnNewDaemon,
// transportAuthBatch_sweep1030 and transportFramePassword.  The runner
// used to skip only X.test.js, so every other suite of a multi-suite gate
// ran twice per `npm test`.
//
// The runner is copied into a scratch directory holding a gate,
// a.coverage.js, that runs a.test.js and b.test.js the way
// coverage_gate.runCovered does (one child process each), and a lone
// c.test.js.  Every suite appends its name to a log, so the log lists
// the suite runs.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');

const RUNNER = path.join(__dirname, 'run-all.js');

/** Write the suite `name` into `dir`: it appends its name to `log`. */
function writeSuite(dir, name, log) {
  fs.writeFileSync(
    path.join(dir, name),
    `require('fs').appendFileSync(${JSON.stringify(log)}, '${name}\\n');\n`,
  );
}

/** Write the gate a.coverage.js into `dir`: it runs `suites`, in order. */
function writeGate(dir, suites) {
  fs.writeFileSync(
    path.join(dir, 'a.coverage.js'),
    [
      "'use strict';",
      "const path = require('path');",
      "const {execFileSync} = require('child_process');",
      `const SUITES = [${suites.map(s => `'${s}'`).join(', ')}];`,
      'for (const suite of SUITES) {',
      '  execFileSync(process.execPath, [path.join(__dirname, suite)]);',
      '}',
      '',
    ].join('\n'),
  );
}

function testGatedSuitesRunOnce() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'runall-gated-'));
  try {
    const log = path.join(dir, 'runs.log');
    fs.copyFileSync(RUNNER, path.join(dir, 'run-all.js'));
    for (const name of ['a.test.js', 'b.test.js', 'c.test.js']) {
      writeSuite(dir, name, log);
    }
    writeGate(dir, ['a.test.js', 'b.test.js']);
    const res = spawnSync(process.execPath, [path.join(dir, 'run-all.js')], {
      encoding: 'utf8',
      timeout: 60000,
      killSignal: 'SIGKILL',
    });
    const out = (res.stdout || '') + (res.stderr || '');
    assert.strictEqual(res.status, 0, out);
    assert.match(out, /\n2\/2 suites passed/, 'the gate and c.test.js only');
    const runs = fs.readFileSync(log, 'utf8').split('\n').filter(Boolean);
    assert.deepStrictEqual(
      runs.sort(),
      ['a.test.js', 'b.test.js', 'c.test.js'],
      'every suite ran exactly once',
    );
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
  console.log('ok - a suite a gate names runs through the gate only');
}

testGatedSuitesRunOnce();
console.log('runAllGatedSuites_wave2: all tests passed');
