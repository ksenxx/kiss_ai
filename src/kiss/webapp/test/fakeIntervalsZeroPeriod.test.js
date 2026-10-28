// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// simplify2_harness.fakeIntervals: a page interval armed with a 0 ms (or
// missing, or non-numeric) period must fire once per millisecond of
// `tick`, as a browser clamps it, not stay due forever.  Before the clamp
// `tick` never returned on such an interval, so the probe runs in a child
// process under a deadline: a hang is reported as a timeout instead of
// stalling the suite.  Runs with: node test/fakeIntervalsZeroPeriod.test.js

'use strict';

const assert = require('assert');
const path = require('path');
const {spawnSync} = require('child_process');

const PROBE = `
  const {newDom, fakeIntervals} = require(${JSON.stringify(
    path.join(__dirname, 'simplify2_harness.js'),
  )});
  const win = newDom();
  const {tick, intervals} = fakeIntervals(win);
  const fired = {zero: 0, missing: 0, nan: 0, two: 0, cleared: 0};
  win.setInterval(() => fired.zero++, 0);
  win.setInterval(() => fired.missing++);
  win.setInterval(() => fired.nan++, 'soon');
  win.setInterval(n => (fired.two += n), 2, 1);
  const id = win.setInterval(() => fired.cleared++, 1);
  win.clearInterval(id);
  win.clearInterval(id);
  tick(5);
  console.log(JSON.stringify({fired, periods: intervals.map(iv => iv.ms)}));
  win.close();
`;

function runProbe() {
  return spawnSync(process.execPath, ['-e', PROBE], {
    encoding: 'utf8',
    timeout: 20000,
  });
}

(function main() {
  const res = runProbe();
  assert.strictEqual(
    res.signal,
    null,
    'tick(5) over a 0 ms interval must return: ' +
      (res.signal === 'SIGTERM' ? 'the probe hung until its deadline' : ''),
  );
  assert.strictEqual(res.status, 0, 'probe failed:\n' + res.stderr);
  const out = JSON.parse(res.stdout.trim());
  assert.deepStrictEqual(
    out.fired,
    {zero: 5, missing: 5, nan: 5, two: 2, cleared: 0},
    'each interval fires once per period inside the ticked span',
  );
  assert.deepStrictEqual(
    out.periods,
    [1, 1, 1, 2],
    'a 0 ms, missing or non-numeric period is clamped to 1 ms; the ' +
      'cleared interval is gone',
  );
  console.log('fakeIntervalsZeroPeriod: all passed');
})();
