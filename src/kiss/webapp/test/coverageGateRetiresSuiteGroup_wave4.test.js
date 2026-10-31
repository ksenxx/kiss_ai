// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: a coverage gate run on its own (`node <gate>`, not
// through test/run-all.js) leaves nothing its suite started running once
// it returns, including a worker that does not hold the gate's output
// (stdio 'ignore').
//
// Before the fix `runCovered` killed only the suite process at its
// CHILD_TIMEOUT_MS and left the rest to run-all.js's process-group kill,
// so a standalone gate returned with the worker still alive, after a
// timeout and after a passing suite alike.  A third case checks that the
// suite and its worker still die with the gate when the gate itself is
// SIGKILLed with its process group mid-run (run-all.js stopping early, a
// terminal's Ctrl-C), which a suite moved out of the gate's group without
// a guard would survive.
//
// Who killed each process is checked as in runAllRetiresSuiteGroup_wave3:
// the gate runs under a Python helper that makes itself a child subreaper
// (prctl), so each orphaned process is adopted by the helper, which waits
// for it and reports how it ended.  The helper never kills a process
// group before it has the fates, so it cannot hide a leak.  The gate
// helper is copied into a scratch directory with its 90 s timeout
// lowered (the copy is the only knob).
//
// Not reproducible end to end: `runCovered` skips the group kill when
// spawnSync could not start the guard at all (pid 0, which as a process
// group would name the gate's own group); starting `process.execPath`
// does not fail on a working host.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');

if (process.platform !== 'linux') {
  console.log(
    'skip - coverageGateRetiresSuiteGroup_wave4 needs prctl(PR_SET_CHILD_SUBREAPER) (Linux only)',
  );
  process.exit(0);
}

// Long enough for the suite to start node, spawn its worker and write
// its pid files on a loaded host.
const SHORT_TIMEOUT_MS = 2000;
// Far longer than the helper needs to kill the gate mid-run.
const LONG_TIMEOUT_MS = 60000;

// Runs `node <gate>` in a new session as a child subreaper, writing its
// output to `log`.  With mode "kill" it SIGKILLs the gate's process group
// once the worker's pid file exists, as run-all.js does to a gate it
// stops.  Then it waits for each process in `names` (comma-separated)
// whose pid the file of that name in `pid_dir` holds: "signal N"/"exit N"
// is how it ended, "alive" that it still ran 3 s after the gate ended
// (the helper then kills it), "not-ours" that it never became the
// helper's child (its parent reaped it, or still runs).  Prints
// {"status": <gate exit status>, "fates": {name: fate}}.
const SUBREAPER_HELPER = `
import ctypes, json, os, signal, subprocess, sys, time
node, gate, pid_dir, log, mode, names = sys.argv[1:7]
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
with open(log, "w") as out:
    proc = subprocess.Popen(
        [node, gate], stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True)
if mode == "kill":
    deadline = time.monotonic() + 30
    while not os.path.exists(os.path.join(pid_dir, "worker")):
        if time.monotonic() > deadline:
            break
        time.sleep(0.05)
    os.killpg(proc.pid, signal.SIGKILL)
status = proc.wait(timeout=60)

def fate(pid):
    ours = False
    for _ in range(60):
        try:
            wpid, st = os.waitpid(pid, os.WNOHANG)
            ours = True
        except ChildProcessError:
            wpid = 0  # not adopted yet: its parent is still exiting
        if wpid == pid:
            if os.WIFSIGNALED(st):
                return "signal %d" % os.WTERMSIG(st)
            return "exit %d" % os.WEXITSTATUS(st)
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if not ours:
        return "not-ours"
    os.waitpid(pid, 0)
    return "alive"

fates = {}
for name in names.split(","):
    try:
        pid = int(open(os.path.join(pid_dir, name)).read())
    except OSError:
        fates[name] = "no pid file"
        continue
    fates[name] = fate(pid)
print(json.dumps({"status": status, "fates": fates}))
`;

/** A suite that records its own pid, starts a `sleep` worker with stdio
 * 'ignore' (left in the suite's process group), records the worker's
 * pid, then hangs or exits 0. */
function suiteSource(pidDir, hangs) {
  return [
    "'use strict';",
    "const {spawn} = require('child_process');",
    "const fs = require('fs');",
    "const path = require('path');",
    'function record(name, pid) {',
    `  const file = path.join(${JSON.stringify(pidDir)}, name);`,
    "  fs.writeFileSync(file + '.tmp', String(pid));",
    "  fs.renameSync(file + '.tmp', file);",
    '}',
    "record('suite', process.pid);",
    "const worker = spawn('sleep', ['300'], {stdio: 'ignore'});",
    "record('worker', worker.pid);",
    hangs ? 'setInterval(function () {}, 1000);' : 'process.exit(0);',
    '',
  ].join('\n');
}

/**
 * Copy coverage_gate.js into a scratch directory with CHILD_TIMEOUT_MS
 * set to `timeoutMs`, add a suite (hanging or not) and a gate script that
 * runs it, run the gate under the subreaper helper in `mode`, and return
 * `{status, fates, out}` with the fates of the processes in `names`.
 */
function runGate(timeoutMs, hangs, mode, names) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gate-wave4-'));
  const pidDir = path.join(dir, 'pids');
  fs.mkdirSync(pidDir);
  try {
    const gate = fs.readFileSync(
      path.join(__dirname, 'coverage_gate.js'),
      'utf8',
    );
    const needle = /const CHILD_TIMEOUT_MS = [^;]+;/;
    assert.ok(needle.test(gate), 'coverage_gate.js still declares the timeout');
    fs.writeFileSync(
      path.join(dir, 'coverage_gate.js'),
      gate.replace(needle, `const CHILD_TIMEOUT_MS = ${timeoutMs};`),
    );
    fs.writeFileSync(
      path.join(dir, 'suite.test.js'),
      suiteSource(pidDir, hangs),
    );
    fs.writeFileSync(
      path.join(dir, 'gate.js'),
      "require('./coverage_gate').runCovered('suite.test.js', 'wave4');\n" +
        "console.log('gate returned');\n",
    );
    const log = path.join(dir, 'gate.log');
    const res = spawnSync(
      'python3',
      [
        '-c',
        SUBREAPER_HELPER,
        process.execPath,
        path.join(dir, 'gate.js'),
        pidDir,
        log,
        mode,
        names.join(','),
      ],
      {
        encoding: 'utf8',
        env: Object.assign({}, process.env, {TMPDIR: dir}),
        timeout: 90000,
        killSignal: 'SIGKILL',
      },
    );
    const out = fs.existsSync(log) ? fs.readFileSync(log, 'utf8') : '';
    const diag = `${res.stdout || ''}${res.stderr || ''}\ngate output:\n${out}`;
    assert.ok(!res.error, `the helper ran: ${res.error}\n${diag}`);
    assert.strictEqual(res.status, 0, `the helper finished\n${diag}`);
    const result = JSON.parse(res.stdout.trim().split('\n').pop());
    return {status: result.status, fates: result.fates, out: diag};
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

function testATimedOutSuiteTakesItsWorkerWithIt() {
  const {status, fates, out} = runGate(SHORT_TIMEOUT_MS, true, 'run', [
    'suite',
    'worker',
  ]);
  assert.ok(
    out.includes(`did not exit within ${SHORT_TIMEOUT_MS / 1000}s, killed`),
    `the gate's own timeout killed its hung suite\n${out}`,
  );
  assert.strictEqual(status, 1, `the timed-out gate fails\n${out}`);
  assert.deepStrictEqual(
    fates,
    {suite: 'signal 9', worker: 'signal 9'},
    `the gate killed the suite's whole group at its timeout\n${out}`,
  );
  console.log('ok - a standalone gate kills a timed-out suite with its worker');
}

function testAPassingSuiteLeavesNoWorkerBehind() {
  // The suite exits 0 and is reaped by its parent: only the worker is
  // left for the helper.
  const {status, fates, out} = runGate(LONG_TIMEOUT_MS, false, 'run', [
    'worker',
  ]);
  assert.ok(out.includes('gate returned'), `runCovered returned\n${out}`);
  assert.strictEqual(status, 0, `the gate passed\n${out}`);
  assert.deepStrictEqual(
    fates,
    {worker: 'signal 9'},
    `the gate killed the worker its passing suite left running\n${out}`,
  );
  console.log(
    'ok - a standalone gate retires what a passing suite left running',
  );
}

function testKillingTheGateKillsTheSuiteGroup() {
  const {fates, out} = runGate(LONG_TIMEOUT_MS, true, 'kill', [
    'suite',
    'worker',
  ]);
  assert.deepStrictEqual(
    fates,
    {suite: 'signal 9', worker: 'signal 9'},
    `the suite and its worker died with the SIGKILLed gate\n${out}`,
  );
  console.log('ok - SIGKILLing a gate with its group still ends its suite');
}

testATimedOutSuiteTakesItsWorkerWithIt();
testAPassingSuiteLeavesNoWorkerBehind();
testKillingTheGateKillsTheSuiteGroup();
console.log('coverageGateRetiresSuiteGroup_wave4: all tests passed');
