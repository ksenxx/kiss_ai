// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for test/run-all.js and processes a suite leaves
// behind holding its stdout/stderr pipes (spawned with inherited stdio):
//
//  1. a suite that hangs with such a grandchild is reported as a
//     timed-out failure, the grandchild is killed with it, and the
//     suite process is reaped by the runner;
//  2. a suite that EXITS at once but left such a grandchild in its own
//     process group (`detached`) is still reported, with its output, at
//     the timeout; before the fix the runner only reported from `exit`
//     or `close`, and neither fired again after the early exit;
//  3. Ctrl-C on the runner kills the suites' process groups and the
//     runner reaps its suite processes before exiting 130.
//
// Whether a process was killed, and who reaped it, is checked without
// relying on the host's init: each runner is started under a Python
// helper that makes itself a child subreaper (prctl), so every orphan
// the suites leave is adopted by the helper, which waits for each one
// and reports its termination status.  `process.kill(pid, 0)` would
// not do: it succeeds for a dead, unreaped zombie.  A suite pid the
// helper cannot wait for was reaped by the runner, its parent.
//
// The runner is copied into a scratch directory holding one suite, with
// its 2-minute SUITE_TIMEOUT_MS lowered to SHORT_TIMEOUT_MS, 1 s (the
// constant is not configurable, by design: the copy is the only knob).

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');

const RUNNER = path.join(__dirname, 'run-all.js');
// Long enough for the scratch suite to start node, spawn its `sleep`
// grandchild and write the pid file on a loaded host; the runner's real
// cap is its own business (makeScratchRunner patches only this value).
const SHORT_TIMEOUT_MS = 1000;

// prctl(PR_SET_CHILD_SUBREAPER) is a Linux system call: on macOS and
// Windows the helper below cannot adopt the suites' orphans, so the
// fates it reports would be meaningless.
if (process.platform !== 'linux') {
  console.log(
    'skip - conc1018_m_runall_grandchild needs prctl(PR_SET_CHILD_SUBREAPER) (Linux only)',
  );
  process.exit(0);
}

// Runs `node <runner>` as a child subreaper.  `mode` "sigint" sends the
// runner SIGINT once the suite's pid file exists; if it has not
// appeared after 10 s the runner is still interrupted (so it kills its
// suites' groups), killed if it lingers, and the status is
// "no-pid-file".  After the runner has
// exited (or 30 s passed, in which case it is SIGKILLed and status is
// "hung"), every pid in the pid file is waited for: "not-ours" means
// its parent (the runner) reaped it; "signal N"/"exit N" is how the
// helper saw it end; "alive" means it was still running 3 s later and
// the helper killed it.  The result is written to `result_file` as JSON.
const SUBREAPER_HELPER = `
import ctypes, json, os, signal, subprocess, sys, time
node, runner, mode, pid_file, result_file = sys.argv[1:6]
PR_SET_CHILD_SUBREAPER = 36
ctypes.CDLL(None).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
proc = subprocess.Popen([node, runner])
def pid_file_written():
    return os.path.exists(pid_file) and os.path.getsize(pid_file) > 0

if mode == "sigint":
    for _ in range(200):
        if pid_file_written():
            break
        time.sleep(0.05)
    if not pid_file_written():
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        with open(result_file, "w") as f:
            json.dump({"status": "no-pid-file", "fates": {}}, f)
        sys.exit(0)
    proc.send_signal(signal.SIGINT)
try:
    status = proc.wait(timeout=30)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
    status = "hung"

def fate(pid):
    for _ in range(60):
        try:
            wpid, st = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return "not-ours"
        if wpid == pid:
            if os.WIFSIGNALED(st):
                return "signal %d" % os.WTERMSIG(st)
            return "exit %d" % os.WEXITSTATUS(st)
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    return "alive"

pids = open(pid_file).read().split() if os.path.exists(pid_file) else []
fates = {name: fate(int(pid)) for name, pid in zip(["suite", "grandchild"], pids)}
with open(result_file, "w") as f:
    json.dump({"status": status, "fates": fates}, f)
`;

function makeScratchRunner() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'conc1018-runall-'));
  const src = fs.readFileSync(RUNNER, 'utf8');
  // The cap's value is the runner's business; only the declaration matters.
  const needle = /const SUITE_TIMEOUT_MS = [^;]+;/;
  assert.ok(needle.test(src), 'run-all.js still declares the timeout');
  fs.writeFileSync(
    path.join(dir, 'run-all.js'),
    src.replace(needle, `const SUITE_TIMEOUT_MS = ${SHORT_TIMEOUT_MS};`),
  );
  return dir;
}

/**
 * Write the one suite of the scratch runner: it hands its stdio to a
 * long-lived `sleep` grandchild and records "<suite pid> <grandchild
 * pid>" in `pidFile`.  `detached` puts the grandchild in its own
 * process group; `exits` makes the suite exit right away instead of
 * hanging.
 */
function writeSuite(dir, pidFile, {detached, exits}) {
  fs.writeFileSync(
    path.join(dir, 'zz_stall.test.js'),
    [
      "'use strict';",
      "const {spawn} = require('child_process');",
      "const fs = require('fs');",
      `const child = spawn('sleep', ['300'], {stdio: 'inherit', detached: ${detached}});`,
      `fs.writeFileSync(${JSON.stringify(pidFile)}, process.pid + ' ' + child.pid);`,
      "console.log('stalling suite started');",
      exits ? 'process.exit(0);' : 'setInterval(function () {}, 1000);',
      '',
    ].join('\n'),
  );
}

/** Run the scratch runner in `dir` under the subreaper helper; returns
 * `{out, status, fates, elapsed}`. */
function runUnderSubreaper(dir, mode) {
  const pidFile = path.join(dir, 'pids');
  const resultFile = path.join(dir, 'result.json');
  const started = Date.now();
  const res = spawnSync(
    'python3',
    [
      '-c',
      SUBREAPER_HELPER,
      process.execPath,
      path.join(dir, 'run-all.js'),
      mode,
      pidFile,
      resultFile,
    ],
    {encoding: 'utf8', timeout: 60000, killSignal: 'SIGKILL'},
  );
  const out = (res.stdout || '') + (res.stderr || '');
  assert.ok(!res.error, `the helper ran: ${res.error}\n${out}`);
  assert.strictEqual(res.status, 0, `the helper finished\n${out}`);
  const result = JSON.parse(fs.readFileSync(resultFile, 'utf8'));
  return {out, elapsed: Date.now() - started, ...result};
}

function withScratchRunner(suiteOptions, mode, check) {
  const dir = makeScratchRunner();
  try {
    writeSuite(dir, path.join(dir, 'pids'), suiteOptions);
    check(runUnderSubreaper(dir, mode));
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

function assertTimedOutRun(r, message) {
  assert.notStrictEqual(
    r.status,
    'hung',
    `the runner exited on its own\n${r.out}`,
  );
  assert.strictEqual(r.status, 1, `a timed-out suite fails the run\n${r.out}`);
  assert.ok(r.out.includes(message), `the timed-out suite is named\n${r.out}`);
  assert.ok(
    r.out.includes('stalling suite started'),
    `output flushed\n${r.out}`,
  );
  assert.ok(r.out.includes('0/1 suites passed'), r.out);
  // `status === 1` and the reaped fate already prove the runner acted on
  // the timeout; the elapsed bound only guards against a runner that
  // waits for the stalled suite anyway, so it allows for three process
  // start-ups under a loaded 32-way parallel run.
  assert.ok(
    r.elapsed < SHORT_TIMEOUT_MS + 30000,
    `the runner reported soon after the timeout (${r.elapsed} ms)`,
  );
  assert.strictEqual(r.fates.suite, 'not-ours', 'the runner reaped the suite');
}

function checkHangingSuiteTimesOut() {
  withScratchRunner({detached: false, exits: false}, 'none', r => {
    assertTimedOutRun(
      r,
      `zz_stall.test.js: no exit after ${SHORT_TIMEOUT_MS / 1000}s, killed`,
    );
    assert.strictEqual(
      r.fates.grandchild,
      'signal 9',
      'the grandchild holding the pipe was killed with the suite',
    );
  });
  console.log(
    'ok - run-all.js reports and kills a hung suite whose grandchild holds its pipe',
  );
}

function checkExitedSuiteWithPipeHolderTimesOut() {
  withScratchRunner({detached: true, exits: true}, 'none', r => {
    assertTimedOutRun(
      r,
      'zz_stall.test.js: exited, but a process it started still held its ' +
        `output ${SHORT_TIMEOUT_MS / 1000}s later, killed`,
    );
  });
  console.log(
    'ok - run-all.js reports an exited suite whose detached grandchild holds its pipe',
  );
}

function checkSigintKillsAndReaps() {
  withScratchRunner({detached: false, exits: false}, 'sigint', r => {
    assert.strictEqual(
      r.status,
      130,
      `the runner exits with 128 + SIGINT\n${r.out}`,
    );
    assert.strictEqual(
      r.fates.suite,
      'not-ours',
      'the runner reaped the suite before exiting',
    );
    assert.strictEqual(
      r.fates.grandchild,
      'signal 9',
      'the grandchild was killed with its group',
    );
  });
  console.log(
    'ok - SIGINT on run-all.js kills the suites it started and reaps them',
  );
}

checkHangingSuiteTimesOut();
checkExitedSuiteWithPipeHolderTimesOut();
checkSigintKillsAndReaps();
