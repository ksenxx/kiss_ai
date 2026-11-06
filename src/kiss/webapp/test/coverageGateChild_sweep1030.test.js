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
// endless chain of gates, a hung suite) is bounded and cleaned up; the
// hung suite's case also kills the group of the gate's guard, which the
// gate's does not include.  Process groups are POSIX: the cases that
// leave a process to kill are skipped on Windows.  A last case checks
// that this harness keeps output written after the process exits.
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
const {killGroup} = require(GATE);
const WAIT_MS = 20000;

/** Resolves after `ms`, without holding the process open. */
function delay(ms) {
  return new Promise(settle => setTimeout(settle, ms).unref());
}

/**
 * Start node with `args` in a new process group and collect its stdout
 * and stderr until it has exited and its pipes have closed (`close`: a
 * process it left behind may still write), `until` matches the output,
 * or WAIT_MS pass; then kill the whole group.  Resolves with `{out,
 * code, timedOut, closed}`, `code` null unless it ended on `close`;
 * `closed` resolves when the pipes have closed, which a process outside
 * the group that inherited them (the gate's guard leads its own group)
 * may delay past the kill.
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
    const closed = new Promise(settle => child.on('close', () => settle()));
    const timer = setTimeout(() => finish(null, true), WAIT_MS);
    function finish(code, timedOut) {
      if (done) return;
      done = true;
      clearTimeout(timer);
      killGroup(child.pid);
      resolve({out, code, timedOut, closed});
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
  if (process.platform === 'win32') {
    // runCovered runs the suite without a guard or process group there,
    // and `process.kill` of a group throws, so the hung suite would run
    // its full 60 s holding the gate's pipes.
    console.log(
      "ok - a hung suite's output streams through the gate (skipped on Windows)",
    );
    return;
  }
  const {dir, env} = scratch({
    // Names its guard (its parent, the leader of its process group) and
    // hangs well past WAIT_MS; exits on its own if nothing kills it.  It
    // stops the guard first: a guard that cannot react to the gate's
    // death stands for one too slow to do so before this test cleans up,
    // so the cleanup below has to retire it itself.  The guard's pid goes
    // to a file before the stop, so the cleanup finds it even when the
    // output never reached this process (the failure the test detects):
    // a stopped guard that only the captured output named leaked for ever.
    'hung.test.js':
      "require('fs').writeFileSync(" +
      "require('path').join(__dirname, 'guard.pid'), String(process.ppid));\n" +
      "process.kill(process.ppid, 'SIGSTOP');\n" +
      "console.log('ok - first ' + process.ppid);\n" +
      'setTimeout(() => process.exit(3), 60000);\n',
  });
  fs.writeFileSync(
    path.join(dir, 'gate.js'),
    `require(${JSON.stringify(GATE)}).runCovered(` +
      `${JSON.stringify(suiteArg(dir, 'hung.test.js'))}, 'hunggate');\n`,
  );
  const res = await runGroup(
    [path.join(dir, 'gate.js')],
    env,
    /ok - first \d+\n/,
  );
  const guard = /ok - first (\d+)\n/.exec(res.out);
  const pidFile = path.join(dir, 'guard.pid');
  let closed = false;
  try {
    assert.ok(
      !res.timedOut,
      "the hung suite's output did not reach the gate's stdout while it ran",
    );
    assert.strictEqual(res.code, null, 'the suite was still running');
    assert.strictEqual(
      fs.readFileSync(pidFile, 'utf8'),
      guard[1],
      'the suite named its guard in the pid file before stopping it',
    );
  } finally {
    // The kill reached the gate only.  Its guard leads its own group,
    // holds the gate's pipes, and once it notices the gate is gone
    // writes its V8 coverage into the scratch TMPDIR before killing its
    // group (`process.kill` of one's own group runs the exit hooks
    // first): removing the directory before that write recreated it,
    // with the coverage file in it, one directory per run.  So kill the
    // guard's group here and wait for the pipes to close, which only the
    // exit of their last holder does, before removing the directory.
    // The pid file, not the captured output, names the guard: a SIGKILL
    // reaches a stopped process, so no SIGCONT is needed.
    if (fs.existsSync(pidFile)) killGroup(Number(fs.readFileSync(pidFile, 'utf8')));
    closed = await Promise.race([
      res.closed.then(() => true),
      delay(WAIT_MS).then(() => false),
    ]);
    fs.rmSync(dir, {recursive: true, force: true});
  }
  assert.ok(closed, "the guard and the suite released the gate's pipes");
  assert.ok(!fs.existsSync(dir), 'nothing recreated the scratch directory');
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
