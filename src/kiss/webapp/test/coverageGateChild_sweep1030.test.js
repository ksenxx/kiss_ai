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
//   with its process group (run-all.js's suite timeout) left no output;
// * a target media file rewritten while its suite ran is refused by
//   name.  The gate used to map the coverage of the text the suite had
//   eval'd onto the new text, and reported lines the suite ran as
//   uncovered.
//
// Each case runs node in its own process group with TMPDIR in a scratch
// directory and kills the group when it is done, so a regression (an
// endless chain of gates, a hung suite) is bounded and cleaned up; the
// hung suite's case also kills the group of the gate's guard, which the
// gate's does not include.  The suite names that guard in a pid file,
// published by a rename so the file is complete or absent, never the
// empty file a `writeFileSync` paused between truncating and writing
// left: `Number('')` is 0, and `process.kill(-0)` signals the group of
// this very process.  The cleanup refuses such a file, and `killGroup`
// itself refuses anything but a positive integer; a case run in a process
// group of its own checks each.  Process groups are POSIX: the cases
// that leave a process to kill are skipped on Windows.  A last case
// checks that this harness keeps output written after the process exits.
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
// argv[2] of a run of this file that runs emptyPidFileScenario only.
const EMPTY_PID_FILE = '--empty-pid-file';

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

/** Write `gate.js` into `dir`: a gate running `dir`'s hung.test.js. */
function writeHungGate(dir) {
  fs.writeFileSync(
    path.join(dir, 'gate.js'),
    `require(${JSON.stringify(GATE)}).runCovered(` +
      `${JSON.stringify(suiteArg(dir, 'hung.test.js'))}, 'hunggate');\n`,
  );
  return path.join(dir, 'gate.js');
}

/**
 * The pid published in `pidFile`, or null when the file is absent or
 * holds anything but a positive integer: `Number('')` is 0, no pid, and
 * `process.kill(-0)` signals the group of the process calling it, this
 * one (`killGroup` refuses such a value too).
 */
function publishedPid(pidFile) {
  if (!fs.existsSync(pidFile)) return null;
  const pid = Number(fs.readFileSync(pidFile, 'utf8'));
  return Number.isSafeInteger(pid) && pid > 0 ? pid : null;
}

/**
 * Retire what a run of a hung gate left behind.  The kill reached the
 * gate only.  Its guard leads its own group, holds the gate's pipes, and
 * once it notices the gate is gone writes its V8 coverage into the
 * scratch TMPDIR before killing its group (`process.kill` of one's own
 * group runs the exit hooks first): removing the directory before that
 * write recreated it, with the coverage file in it, one directory per
 * run.  So kill the guard's group when `pidFile` names it, wait for the
 * pipes to close, which only the exit of their last holder does, and
 * then remove `dir`.  The pid file, not the captured output, names the
 * guard: a SIGKILL reaches a stopped process, so no SIGCONT is needed.
 * Resolves with whether the pipes closed within WAIT_MS.
 */
async function retireGuard(res, pidFile, dir) {
  const pid = publishedPid(pidFile);
  if (pid !== null) killGroup(pid);
  const closed = await Promise.race([
    res.closed.then(() => true),
    delay(WAIT_MS).then(() => false),
  ]);
  fs.rmSync(dir, {recursive: true, force: true});
  return closed;
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

async function testAReportCutShortLeavesNoCoverageDirectory() {
  // The suite passes but leaves a report cut short beside its own, as a
  // process killed while writing its coverage does.  The gate fails on
  // it, and used to leave its coverage directory behind.
  const {dir, env} = scratch({
    'cut.test.js':
      "require('fs').writeFileSync(require('path').join(" +
      "process.env.NODE_V8_COVERAGE, 'cut.json'), '{');\n",
  });
  try {
    const code =
      `require(${JSON.stringify(GATE)}).runCovered(` +
      `${JSON.stringify(suiteArg(dir, 'cut.test.js'))}, 'cutgate');`;
    const res = await runGroup(['-e', code], env);
    assert.ok(!res.timedOut, `the gate did not finish:\n${res.out}`);
    assert.strictEqual(res.code, 1, res.out);
    assert.match(res.out, /SyntaxError/);
    assert.deepStrictEqual(
      fs.readdirSync(dir).filter(f => f.includes('-cov-')),
      [],
      'the gate removed its coverage directory',
    );
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
  console.log('ok - a gate failing on a report cut short removes its directory');
}

/**
 * Run a gate in a scratch tree of its own: a copy of coverage_gate.js in
 * `test/`, a fenced `media/target.js`, and a suite that evals the target
 * under its sourceURL and then runs `afterEval` (source text; `target`
 * holds the target's path).  Resolves with runGroup's result.
 */
async function runTargetGate(afterEval) {
  const {dir, env} = scratch({});
  try {
    const target = path.join(dir, 'media', 'target.js');
    fs.mkdirSync(path.join(dir, 'media'));
    fs.mkdirSync(path.join(dir, 'test'));
    fs.copyFileSync(GATE, path.join(dir, 'test', 'coverage_gate.js'));
    fs.writeFileSync(
      target,
      '// edit-coverage:start\nglobalThis.targetRan = true;\n' +
        '// edit-coverage:end\n',
    );
    fs.writeFileSync(
      path.join(dir, 'test', 'suite.test.js'),
      "const fs = require('fs');\n" +
        `const target = ${JSON.stringify(target)};\n` +
        "(0, eval)(fs.readFileSync(target, 'utf8') + " +
        "'\\n//# sourceURL=target.js');\n" +
        afterEval,
    );
    fs.writeFileSync(
      path.join(dir, 'test', 'gate.js'),
      "require('./coverage_gate').gateRegions({testFile: 'suite.test.js', " +
        "mark: 'edit', label: 'edit', " +
        "targets: [{file: 'target.js', sourceUrl: 'target.js', regions: 1}]});\n",
    );
    return await runGroup([path.join(dir, 'test', 'gate.js')], env);
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

async function testATargetEditedWhileItsSuiteRanIsRefused() {
  // Untouched, the target passes the gate.
  const kept = await runTargetGate('');
  assert.strictEqual(kept.code, 0, kept.out);
  assert.match(kept.out, /coverage gate passed/);
  // The suite rewrites the target after eval-ing it, as an editor saving
  // the file while the suite runs does.
  const edited = await runTargetGate(
    "fs.writeFileSync(target, '// a line added mid-run\\n'.repeat(40) + " +
      "fs.readFileSync(target, 'utf8'));\n",
  );
  assert.strictEqual(edited.code, 1, edited.out);
  assert.match(edited.out, /media\/target\.js changed/);
  assert.ok(!edited.out.includes('UNCOVERED'), edited.out);
  console.log('ok - a target edited while its suite ran is refused by name');
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
    // A rename publishes the file whole: a suite paused between
    // truncating and writing it (then timed out) left it empty, and the
    // cleanup's `killGroup(Number(''))` signalled this process's group.
    'hung.test.js':
      "const fs = require('fs');\n" +
      "const pidFile = require('path').join(__dirname, 'guard.pid');\n" +
      "fs.writeFileSync(pidFile + '.tmp', String(process.ppid));\n" +
      "fs.renameSync(pidFile + '.tmp', pidFile);\n" +
      "process.kill(process.ppid, 'SIGSTOP');\n" +
      "console.log('ok - first ' + process.ppid);\n" +
      'setTimeout(() => process.exit(3), 60000);\n',
  });
  const res = await runGroup([writeHungGate(dir)], env, /ok - first \d+\n/);
  const guard = /ok - first (\d+)\n/.exec(res.out);
  const pidFile = path.join(dir, 'guard.pid');
  let closed;
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
    assert.ok(
      !fs.existsSync(pidFile + '.tmp'),
      'the rename took the temp file',
    );
  } finally {
    closed = await retireGuard(res, pidFile, dir);
  }
  assert.ok(closed, "the guard and the suite released the gate's pipes");
  assert.ok(!fs.existsSync(dir), 'nothing recreated the scratch directory');
  console.log("ok - a hung suite's output streams through the gate");
}

/**
 * The hung suite's cleanup with an empty guard.pid.  The window that
 * left one, inside the suite's `writeFileSync`, cannot be opened from
 * here (a rename now closes it), so the suite writes the empty file
 * itself, prints, and hangs without stopping its guard.  Run by
 * testEmptyPidFileRetiresNothing in a process group of its own, which a
 * cleanup signalling group 0 then kills instead of the suite's.
 */
async function emptyPidFileScenario() {
  const {dir, env} = scratch({
    'hung.test.js':
      "require('fs').writeFileSync(" +
      "require('path').join(__dirname, 'guard.pid'), '');\n" +
      "console.log('ok - empty');\n" +
      'setTimeout(() => process.exit(3), 60000);\n',
  });
  const res = await runGroup([writeHungGate(dir)], env, /ok - empty\n/);
  const pidFile = path.join(dir, 'guard.pid');
  let closed;
  try {
    assert.ok(!res.timedOut, `the suite did not run:\n${res.out}`);
    assert.strictEqual(fs.readFileSync(pidFile, 'utf8'), '');
  } finally {
    closed = await retireGuard(res, pidFile, dir);
  }
  // Nothing stopped the guard: it saw the gate die and killed its group.
  assert.ok(closed, "the guard released the gate's pipes on its own");
  assert.ok(!fs.existsSync(dir), 'nothing recreated the scratch directory');
  console.log('ok - the cleanup outlived an empty pid file');
}

async function testEmptyPidFileRetiresNothing() {
  if (process.platform === 'win32') {
    console.log('ok - an empty pid file retires nothing (skipped on Windows)');
    return;
  }
  const res = await runGroup(
    [...process.execArgv, __filename, EMPTY_PID_FILE],
    process.env,
  );
  assert.ok(!res.timedOut, `the scenario did not finish:\n${res.out}`);
  assert.strictEqual(
    res.code,
    0,
    `the cleanup of an empty pid file killed its own group (exit ${res.code}):\n${res.out}`,
  );
  assert.match(res.out, /ok - the cleanup outlived an empty pid file\n/);
  console.log('ok - an empty pid file retires nothing');
}

/**
 * `killGroup` refuses what is no pid.  Without its guard `process.kill(-x)`
 * of 0, -0, '', null, false or [] signals group 0, the caller's own, a
 * negative number signals the process it negates to and a numeric string
 * the group it negates to; NaN, undefined and 1.5 only throw, which it
 * swallows.  The caller runs in a process group of its own, so a signal
 * that gets through kills it, not this process.
 */
async function testKillGroupRefusesWhatIsNoPid() {
  const code =
    `const {killGroup} = require(${JSON.stringify(GATE)});` +
    "for (const v of [0, -0, '', null, false, [], NaN, undefined, 1.5, " +
    '-process.pid, String(process.pid)]) killGroup(v);' +
    "console.log('alive');";
  const res = await runGroup(['-e', code], process.env);
  assert.ok(!res.timedOut, `the caller did not finish:\n${res.out}`);
  assert.strictEqual(
    res.code,
    0,
    `killGroup signalled its caller (exit ${res.code}):\n${res.out}`,
  );
  assert.strictEqual(res.out, 'alive\n');
  console.log('ok - killGroup refuses what is no pid');
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
  if (process.argv[2] === EMPTY_PID_FILE) {
    await emptyPidFileScenario();
    return;
  }
  await testEvalGateRunsTheSuiteOnce();
  await testAReportCutShortLeavesNoCoverageDirectory();
  await testATargetEditedWhileItsSuiteRanIsRefused();
  await testHungSuiteOutputStreams();
  await testEmptyPidFileRetiresNothing();
  await testKillGroupRefusesWhatIsNoPid();
  await testRunGroupKeepsOutputWrittenAfterTheExit();
  console.log('coverageGateChild_sweep1030: all tests passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
