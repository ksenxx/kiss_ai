// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The suites below assert day separators, "N minutes ago" labels, running
// durations and the page's wall-clock windows (the 15 s launch window, the
// 5 s close shield, the 30 s history flight, the 10 s rename overlay, the
// transport shim's overlay, lockout and reload budgets).  They used to date
// their fixtures from Node's clock and let the page read its own, so they
// failed whenever the two readings fell on different sides of a boundary: a
// run across local midnight filed "today" under Yesterday, a render 16 s
// after its fixture made a 45 s old task "1 minute ago", a 16 s stall
// expired the launch window.  Each now freezes its pages' clock at the
// instant its fixtures are dated from; here each runs in a child process
// whose pages would otherwise read a hostile clock
// (hostile_clock_preload.js).

'use strict';

const assert = require('assert');
const path = require('path');
const {spawn} = require('child_process');

const PRELOAD = path.join(__dirname, 'hostile_clock_preload.js');
const SUITES = [
  'historyGroups.test.js',
  'historyTaskLaunchedAgo.test.js',
  'historyTagsChatLaunched.test.js',
  'launchTabSwitch.test.js',
  'mainJsDaemonReconnectRaces.test.js',
  'historyRefreshSingleFlight.test.js',
  'historyRenameSurvivesRebuild.test.js',
  'remotePasswordBypass.test.js',
];
// Under run-all.js's own two minutes for this suite.
const CHILD_TIMEOUT_MS = 110000;

/** Run `suite` under the hostile clock; resolves with `{suite, code, out}`. */
function run(suite) {
  return new Promise(resolve => {
    const child = spawn(
      process.execPath,
      [...process.execArgv, '-r', PRELOAD, path.join(__dirname, suite)],
      {
        stdio: ['ignore', 'pipe', 'pipe'],
        timeout: CHILD_TIMEOUT_MS,
        killSignal: 'SIGKILL',
      },
    );
    let out = '';
    child.stdout.on('data', chunk => (out += chunk));
    child.stderr.on('data', chunk => (out += chunk));
    child.on('close', code => resolve({suite, code, out}));
  });
}

async function main() {
  for (const {suite, code, out} of await Promise.all(SUITES.map(run))) {
    assert.strictEqual(code, 0, `${suite} depends on the host clock:\n${out}`);
    console.log(`ok - ${suite} passes under a hostile clock`);
  }
  console.log('suitesUnderHostileClock: all passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
