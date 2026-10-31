// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for how coverage_gate.js's `runCovered` starts the
// functional suite:
//
// * a gate run as `node -e` runs the suite once.  The child used to get
//   this process's execArgv, `-e <gate code>` included, so it ran the
//   gate code again instead of the suite, and so on without end;
// * the suite's output reaches the gate's stdout while the suite runs.
//   It used to be buffered until the suite exited, so a hung suite killed
//   with its process group (run-all.js's suite timeout) left no output.
//
// Each case runs node in its own process group with TMPDIR in a scratch
// directory and kills the group when it is done, so a regression (an
// endless chain of gates, a hung suite) is bounded and cleaned up.  A
// last case checks that this harness keeps output written after the
// process exits.
//
// The gate's own CHILD_TIMEOUT_MS kill (90 s) is exercised, on a copy
// with the timeout lowered, by runAllRetiresSuiteGroup_wave3.test.js.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawn} = require('child_process');

const GATE = path.join(__dirname, 'coverage_gate.js');
const WAIT_MS = 20000;

/**
 * Start node with `args` in a new process group and collect its stdout
 * and stderr until it has exited and its pipes have closed (`close`: a
 * process it left behind may still write), `until` matches the output,
 * or WAIT_MS pass; then kill the whole group.  Resolves with `{out,
 * code, timedOut}`, `code` null unless it ended on `close`.
 */
function runGroup(args, env, until) {
  return new Promise(resolve => {
    const child = spawn(process.execPath, args, {
      detached: true,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    let done = false;
    const timer = setTimeout(() => finish(null, true), WAIT_MS);
    function finish(code, timedOut) {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try {
        process.kill(-child.pid, 'SIGKILL');
      } catch {
        // The group has exited already.
      }
      resolve({out, code, timedOut});
    }
    function collect(chunk) {
      out += chunk;
      if (until && until.test(out)) finish(null, false);
    }
    child.stdout.on('data', collect);
    child.stderr.on('data', collect);
    child.on('close', code => finish(code, false));
  });
}

/** A scratch dir with `files` written into it, and an env using it as TMPDIR. */
function scratch(files) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gate-child-'));
  for (const [name, text] of Object.entries(files)) {
    fs.writeFileSync(path.join(dir, name), text);
  }
  return {dir, env: Object.assign({}, process.env, {TMPDIR: dir})};
}

/** `file` in `dir` as runCovered takes it: relative to this directory. */
function suiteArg(dir, file) {
  return path.relative(__dirname, path.join(dir, file));
}

async function testEvalGateRunsTheSuiteOnce() {
  const {dir, env} = scratch({'once.test.js': "console.log('suite ran');\n"});
  try {
    const code =
      `require(${JSON.stringify(GATE)}).runCovered(` +
      `${JSON.stringify(suiteArg(dir, 'once.test.js'))}, 'evalgate');` +
      "console.log('gate done');";
    const res = await runGroup(['-e', code], env);
    assert.ok(!res.timedOut, `the -e gate did not finish:\n${res.out}`);
    assert.strictEqual(res.code, 0, res.out);
    assert.strictEqual(res.out, 'suite ran\ngate done\n');
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
  console.log('ok - a gate run with node -e runs the suite once');
}

async function testHungSuiteOutputStreams() {
  const {dir, env} = scratch({
    // Hangs well past WAIT_MS; exits on its own if nothing kills it.
    'hung.test.js':
      "console.log('ok - first');\n" +
      'setTimeout(() => process.exit(3), 60000);\n',
  });
  fs.writeFileSync(
    path.join(dir, 'gate.js'),
    `require(${JSON.stringify(GATE)}).runCovered(` +
      `${JSON.stringify(suiteArg(dir, 'hung.test.js'))}, 'hunggate');\n`,
  );
  try {
    const res = await runGroup([path.join(dir, 'gate.js')], env, /ok - first/);
    assert.ok(
      !res.timedOut,
      "the hung suite's output did not reach the gate's stdout while it ran",
    );
    assert.strictEqual(res.code, null, 'the suite was still running');
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
  console.log("ok - a hung suite's output streams through the gate");
}

async function testRunGroupKeepsOutputWrittenAfterTheExit() {
  // The process exits at once; the child it leaves behind holding its
  // pipes writes 300 ms later.  Resolving on `exit` returned ''.
  if (process.platform === 'win32') {
    // On Windows the pipes end with the process itself (`close` follows
    // `exit` within a millisecond, the leftover's handle notwithstanding),
    // so no harness can collect what it writes afterwards.
    console.log('ok - runGroup keeps the output written after the exit (skipped on Windows)');
    return;
  }
  const late = "setTimeout(() => console.log('late line'), 300)";
  const code =
    "require('child_process').spawn(process.execPath, " +
    `['-e', ${JSON.stringify(late)}], ` +
    "{stdio: ['ignore', 'inherit', 'inherit']}).unref();";
  const res = await runGroup(['-e', code], process.env);
  assert.ok(!res.timedOut, `the run did not finish:\n${res.out}`);
  assert.strictEqual(res.code, 0, res.out);
  assert.strictEqual(res.out, 'late line\n');
  console.log('ok - runGroup keeps the output written after the exit');
}

async function main() {
  await testEvalGateRunsTheSuiteOnce();
  await testHungSuiteOutputStreams();
  await testRunGroupKeepsOutputWrittenAfterTheExit();
  console.log('coverageGateChild_sweep1030: all tests passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
