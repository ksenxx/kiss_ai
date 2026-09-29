// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: retrying a torn config.json read must not
// freeze the extension host.
//
// The bug: readKissConfig() retried an empty / unparsable config.json
// five times with a SYNCHRONOUS 100ms sleep (Atomics.wait) between
// attempts, and ensureRemotePassword() reads the config three times on
// its way to the prompt -- so a config.json caught mid-write (or left
// empty) stalled the whole event loop for ~400ms per read.
//
// The test drives the REAL compiled ensureRemotePassword() in a child
// process against a fake HOME whose config.json is empty, with the
// input box stubbed to "Esc" (so nothing is saved), while a 10ms
// heartbeat records the longest event-loop stall.  A regressed build
// stalls for hundreds of milliseconds; the fixed build only ever pauses
// for the few milliseconds a synchronous file read takes.
//
// The heartbeat measures wall-clock time, so a child starved of CPU by
// other test suites running alongside (a 262ms gap was seen with eleven
// suites sharing ten cores) looks like a stall too.  A regressed build
// stalls for >= 400ms on EVERY run, while starvation is transient, so
// the measurement is repeated up to MAX_RUNS times and the test fails
// only when no run stays under the threshold.

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OUT = path.join(__dirname, '..', 'out', 'DependencyInstaller.js');
if (!fs.existsSync(OUT)) {
  console.log('SKIP: out/DependencyInstaller.js missing — run `npm run compile`');
  process.exit(0);
}

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-cfgread-'));

// Each run gets its own HOME: ensureRemotePassword() records a
// "prompt declined" marker under ~/.kiss, so a second child in the same
// HOME would skip the prompt instead of re-reading the config.
function makeHome(run) {
  const tmpHome = path.join(tmpRoot, `home${run}`);
  const kissDir = path.join(tmpHome, '.kiss');
  fs.mkdirSync(kissDir, {recursive: true});
  // An empty config.json: the "torn write" case the retry loop exists for.
  fs.writeFileSync(path.join(kissDir, 'config.json'), '');
  return {tmpHome, kissDir};
}

const CHILD = `
'use strict';
const path = require('path');
const Module = require('module');
const stubPath = path.join(process.env.KISS_TEST_DIR, '_vscode-stub.js');
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return stubPath;
  return origResolve.call(this, request, parent, ...rest);
};
let prompted = 0;
global.__kissVscodeStub = {
  window: {
    showInputBox: () => {
      prompted += 1;
      return Promise.resolve(undefined);
    },
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
  },
  workspace: {workspaceFolders: undefined},
  ProgressLocation: {Notification: 15},
};
let last = Date.now();
let maxStallMs = 0;
const beat = setInterval(() => {
  const now = Date.now();
  if (now - last > maxStallMs) maxStallMs = now - last;
  last = now;
}, 10);
const installer = require(process.env.KISS_TEST_MODULE);
installer
  .ensureRemotePassword(null, process.env.KISS_TEST_DIR, undefined, 5000, 50)
  .then(() => {
    clearInterval(beat);
    process.stdout.write(
      '\\nPROMPTED=' + prompted + '\\nMAXSTALL=' + maxStallMs + '\\nDONE');
  })
  .catch(err => {
    clearInterval(beat);
    process.stdout.write('FAIL:' + (err && err.message));
    process.exitCode = 1;
  });
`;

function cleanup() {
  try {
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  } catch {}
}

const STALL_LIMIT_MS = 250;
const MAX_RUNS = 3;

// Runs the child once and returns the longest event-loop stall it saw.
async function measureStall(run) {
  const {tmpHome, kissDir} = makeHome(run);
  const out = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['-e', CHILD], {
      stdio: ['ignore', 'pipe', 'inherit'],
      env: {
        ...process.env,
        HOME: tmpHome,
        USERPROFILE: tmpHome,
        KISS_HOME: kissDir,
        KISS_TEST_DIR: __dirname,
        KISS_TEST_MODULE: OUT,
      },
    });
    let buf = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error('ensureRemotePassword did not finish within 60s'));
    }, 60000);
    child.stdout.on('data', d => {
      buf += d.toString();
    });
    child.on('close', () => {
      clearTimeout(timer);
      resolve(buf.trim());
    });
  });

  assert.ok(out.endsWith('DONE'), `ensureRemotePassword failed: ${out}`);

  // The empty config really was treated as "no password": the retry
  // loop ran out and the user was prompted (and pressed Esc).
  const p = /PROMPTED=(\d+)/.exec(out);
  assert.ok(p && parseInt(p[1], 10) === 1, `expected one prompt, got: ${out}`);
  console.log('  ok - an empty config.json falls through to the prompt');

  const m = /MAXSTALL=(\d+)/.exec(out);
  assert.ok(m, `child did not report its heartbeat: ${out}`);
  return parseInt(m[1], 10);
}

async function main() {
  const stalls = [];
  for (let run = 0; run < MAX_RUNS; run++) {
    const maxStallMs = await measureStall(run);
    stalls.push(maxStallMs);
    if (maxStallMs < STALL_LIMIT_MS) {
      const starved = stalls.slice(0, -1).join(', ');
      console.log(
        `  ok - longest event-loop stall while reading config: ${maxStallMs}ms` +
          (run > 0 ? ` (after ${run} CPU-starved run(s): ${starved}ms)` : ''),
      );
      return;
    }
    console.log(
      `  warn - run ${run + 1} stalled for ${maxStallMs}ms` +
        (run + 1 < MAX_RUNS ? ', retrying' : ''),
    );
  }
  assert.fail(
    `event loop stalled for ${stalls.join(', ')}ms in ${MAX_RUNS} runs while ` +
      're-reading config.json — the retry backoff is blocking the extension host again',
  );
}

main()
  .then(() => {
    cleanup();
    console.log('concaudit_w7_config_read_no_block: all assertions passed');
  })
  .catch(err => {
    cleanup();
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
