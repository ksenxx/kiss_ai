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
// a guard would survive, and that the gate's coverage directory goes with
// them: the killed gate cannot remove it, and the guard that killed the
// group died with it, so the directory (and the reports written into it)
// stayed in the temporary directory.  The process that now removes it
// must end once it is done, whatever a NODE_OPTIONS preload left running
// in it (PRELOAD).  A fourth runs it against a group that does not go away.
// A fifth stops this test itself as run-all.js stops a suite (SIGKILL to
// its process group, which runs no `finally`): the gate it was running, in
// a session of its own, and the group of the fourth case, in a group of
// its own, were out of that kill's reach and lived on (the gate until its
// timeout, the group for five minutes); both must end with the test.
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
const {spawn, spawnSync} = require('child_process');

if (process.platform !== 'linux') {
  console.log(
    'skip - coverageGateRetiresSuiteGroup_wave4 needs prctl(PR_SET_CHILD_SUBREAPER) (Linux only)',
  );
  process.exit(0);
}

// The gate arms its timeout when it starts the guard, before the suite
// has started node, spawned its worker and written its pid files (the
// copy's constant is the only knob, so it cannot be armed once the suite
// is ready): a run the suite did not start within it is repeated with
// twice the timeout, up to MAX_TIMEOUT_MS, as `runTimedOutGate` does.
const SHORT_TIMEOUT_MS = 2000;
const MAX_TIMEOUT_MS = 16000;
// Far longer than the helper needs to kill the gate mid-run.
const LONG_TIMEOUT_MS = 60000;

// Runs `node <gate>` in a new session as a child subreaper, writing its
// output to `log`.  The gate dies with the helper (PR_SET_PDEATHSIG): a
// kill of this test's process group, the helper in it, does not reach a
// session of its own.  With mode "kill" it SIGKILLs the gate's process group
// once the worker's pid file exists, as run-all.js does to a gate it
// stops.  Then it waits for each process in `names` (comma-separated)
// whose pid the file of that name in `pid_dir` holds: "signal N"/"exit N"
// is how it ended, "alive" that it still ran 3 s after the gate ended
// (the helper then kills it), "not-ours" that it never became the
// helper's child (its parent reaped it, or still runs).  Prints
// {"status": <gate exit status>, "fates": {name: fate}} once every other
// process it adopted has ended too: the cleanup a killed gate's guard
// starts is one, and what it removes is looked at when it is done.
const SUBREAPER_HELPER = `
import ctypes, json, os, signal, subprocess, sys, time
node, gate, pid_dir, log, mode, names = sys.argv[1:7]
libc = ctypes.CDLL(None)
libc.prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
helper = os.getpid()

def die_with_helper():
    libc.prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
    if os.getppid() != helper:  # the helper died before the prctl
        os._exit(1)

with open(log, "w") as out:
    proc = subprocess.Popen(
        [node, gate], stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True, preexec_fn=die_with_helper)
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
try:
    while True:
        os.waitpid(-1, 0)
except ChildProcessError:
    pass
print(json.dumps({"status": status, "fates": fates}))
`;

/** A suite that, `startDelayMs` after node started it, records its own
 * pid, starts a `sleep` worker with stdio 'ignore' (left in the suite's
 * process group), records the worker's pid, then hangs or exits 0. */
function suiteSource(pidDir, hangs, startDelayMs) {
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
    'Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ' +
      `${startDelayMs});`,
    "record('suite', process.pid);",
    "const worker = spawn('sleep', ['300'], {stdio: 'ignore'});",
    "record('worker', worker.pid);",
    hangs ? 'setInterval(function () {}, 1000);' : 'process.exit(0);',
    '',
  ].join('\n');
}

// A NODE_OPTIONS preload reaches every node process a gate starts.  In
// the cleanup of a killed gate, which runs where nothing would kill it,
// this one leaves the mark `cleanup-preloaded` beside itself and a
// pending timer that adds `cleanup-lingered` if the process is still
// there to run it (a repeating timer would keep such a process for good).
const PRELOAD = `
if (process.argv[2] === '--cleanup') {
  const fs = require('fs');
  fs.writeFileSync(__dirname + '/cleanup-preloaded', '');
  setTimeout(fs.writeFileSync, 1000, __dirname + '/cleanup-lingered', '');
}
`;

// `spawn` arguments of a process group of its own that stays until it is
// killed or the process that started it is gone, however that ended (a
// SIGKILL runs no `finally`): `cat` reads a pipe whose other end only
// that process holds.
const STAYING_GROUP = [
  'cat',
  [],
  {detached: true, stdio: ['pipe', 'ignore', 'ignore']},
];

/**
 * A stand-in for this test, for the helper to stop as run-all.js stops a
 * suite (SIGKILL to its process group): a node process that starts the
 * staying group, records its pid as `group`, and runs the gate in `dir`
 * under a helper of its own.
 */
function stoppedTestSource(dir, pidDir) {
  const helper = [
    '-c',
    SUBREAPER_HELPER,
    process.execPath,
    path.join(dir, 'gate.js'),
    pidDir,
    path.join(dir, 'inner.log'),
    'run',
    '',
  ];
  return [
    "'use strict';",
    "const {spawn, spawnSync} = require('child_process');",
    `const group = spawn(...${JSON.stringify(STAYING_GROUP)});`,
    `const file = ${JSON.stringify(path.join(pidDir, 'group'))};`,
    "require('fs').writeFileSync(file, String(group.pid));",
    `spawnSync('python3', ${JSON.stringify(helper)});`,
    '',
  ].join('\n');
}

/**
 * Copy coverage_gate.js into a scratch directory with CHILD_TIMEOUT_MS
 * set to `timeoutMs`, add a suite (hanging or not, started `startDelayMs`
 * late) and a gate script that records its pid as `gate` and runs it, run
 * the gate under the subreaper helper in `mode` with PRELOAD in its
 * NODE_OPTIONS, and return `{status, fates, out, leftovers, marks}` with
 * the fates of the processes in `names`, the coverage directories
 * (`<mark>-cov-*`) the gate left in its temporary directory and the
 * preload's marks.  Mode "stop" runs the helper in mode "kill" on a
 * stand-in for this test running the gate (`stoppedTestSource`).
 */
function runGate(timeoutMs, hangs, mode, names, startDelayMs = 0) {
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
      suiteSource(pidDir, hangs, startDelayMs),
    );
    fs.writeFileSync(
      path.join(dir, 'gate.js'),
      `const file = ${JSON.stringify(path.join(pidDir, 'gate'))};\n` +
        "require('fs').writeFileSync(file, String(process.pid));\n" +
        "require('./coverage_gate').runCovered('suite.test.js', 'wave4');\n" +
        "console.log('gate returned');\n",
    );
    const stop = mode === 'stop';
    if (stop) {
      fs.writeFileSync(
        path.join(dir, 'test.js'),
        stoppedTestSource(dir, pidDir),
      );
    }
    const preload = path.join(dir, 'preload.js');
    fs.writeFileSync(preload, PRELOAD);
    const log = path.join(dir, 'gate.log');
    const res = spawnSync(
      'python3',
      [
        '-c',
        SUBREAPER_HELPER,
        process.execPath,
        path.join(dir, stop ? 'test.js' : 'gate.js'),
        pidDir,
        log,
        stop ? 'kill' : mode,
        names.join(','),
      ],
      {
        encoding: 'utf8',
        env: Object.assign({}, process.env, {
          TMPDIR: dir,
          NODE_OPTIONS: `--require ${JSON.stringify(preload)}`,
        }),
        timeout: 90000,
        killSignal: 'SIGKILL',
      },
    );
    const out = fs.existsSync(log) ? fs.readFileSync(log, 'utf8') : '';
    const diag = `${res.stdout || ''}${res.stderr || ''}\ngate output:\n${out}`;
    assert.ok(!res.error, `the helper ran: ${res.error}\n${diag}`);
    assert.strictEqual(res.status, 0, `the helper finished\n${diag}`);
    const result = JSON.parse(res.stdout.trim().split('\n').pop());
    const leftovers = fs.readdirSync(dir).filter(f => f.includes('-cov-'));
    const marks = fs.readdirSync(dir).filter(f => f.startsWith('cleanup-'));
    return {...result, out: diag, leftovers, marks};
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

/**
 * `runGate` on a hanging suite that starts `startDelayMs` late, with
 * `timeoutMs` doubled (up to MAX_TIMEOUT_MS) and the run repeated while
 * the helper found no pid file: the suite had not started within the
 * timeout, which says nothing about what the gate killed.  Adds the
 * `timeoutMs` of the run returned.
 */
function runTimedOutGate(timeoutMs, startDelayMs) {
  for (;;) {
    const res = runGate(
      timeoutMs,
      true,
      'run',
      ['suite', 'worker'],
      startDelayMs,
    );
    const started = !Object.values(res.fates).includes('no pid file');
    if (started || timeoutMs >= MAX_TIMEOUT_MS) return {...res, timeoutMs};
    timeoutMs *= 2;
  }
}

function testATimedOutSuiteTakesItsWorkerWithIt() {
  const {status, fates, out, timeoutMs} = runTimedOutGate(SHORT_TIMEOUT_MS, 0);
  assert.ok(
    out.includes(`did not exit within ${timeoutMs / 1000}s, killed`),
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

function testASuiteSlowToStartIsJudgedOnceStarted() {
  // A host too loaded for the suite to start within the timeout: the
  // gate times out before the suite recorded a pid, and the run is
  // repeated with a timeout the suite does start within.
  const {fates, out, timeoutMs} = runTimedOutGate(500, 700);
  assert.ok(timeoutMs > 500, `the run without pid files was repeated\n${out}`);
  assert.ok(
    out.includes(`did not exit within ${timeoutMs / 1000}s, killed`),
    `the gate's own timeout killed its hung suite\n${out}`,
  );
  assert.deepStrictEqual(
    fates,
    {suite: 'signal 9', worker: 'signal 9'},
    `the gate killed the suite's whole group at its timeout\n${out}`,
  );
  console.log('ok - a suite slow to start is judged once it started');
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
  const {fates, out, leftovers, marks} = runGate(
    LONG_TIMEOUT_MS,
    true,
    'kill',
    ['suite', 'worker'],
  );
  assert.deepStrictEqual(
    fates,
    {suite: 'signal 9', worker: 'signal 9'},
    `the suite and its worker died with the SIGKILLed gate\n${out}`,
  );
  assert.deepStrictEqual(
    leftovers,
    [],
    `the SIGKILLed gate's coverage directory was removed\n${out}`,
  );
  assert.deepStrictEqual(
    marks,
    ['cleanup-preloaded'],
    `the cleanup ran under the preload and did not wait for its timer\n${out}`,
  );
  console.log('ok - SIGKILLing a gate with its group still ends its suite');
}

function testTheCleanupDoesNotWaitForEverForAGroupToGo() {
  // The cleanup removes the directory once the group it is given is gone.
  // A group that stays (a live `cat` here; a zombie nobody reaps counts
  // as well) must not keep it waiting: the directory goes all the same,
  // and the group is left alone.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gate-wave4-'));
  const sleeper = spawn(...STAYING_GROUP);
  try {
    const res = spawnSync(
      process.execPath,
      [
        path.join(__dirname, 'coverage_gate.js'),
        '--cleanup',
        String(sleeper.pid),
        dir,
      ],
      {encoding: 'utf8', timeout: 60000, killSignal: 'SIGKILL'},
    );
    assert.strictEqual(res.status, 0, `the cleanup ended\n${res.stderr}`);
    assert.ok(!fs.existsSync(dir), 'the cleanup removed the directory');
    process.kill(sleeper.pid, 0); // throws if the cleanup killed the group
  } finally {
    process.kill(-sleeper.pid, 'SIGKILL');
    fs.rmSync(dir, {recursive: true, force: true});
  }
  console.log('ok - the cleanup of a killed gate is bounded');
}

function testStoppingTheTestEndsItsGateAndItsStayingGroup() {
  const {fates, out} = runGate(LONG_TIMEOUT_MS, true, 'stop', [
    'gate',
    'suite',
    'worker',
    'group',
  ]);
  assert.deepStrictEqual(
    fates,
    {gate: 'signal 9', suite: 'signal 9', worker: 'signal 9', group: 'exit 0'},
    `what the stopped test had started outside its group ended with it\n${out}`,
  );
  console.log('ok - stopping the test ends its gate and its staying group');
}

testATimedOutSuiteTakesItsWorkerWithIt();
testASuiteSlowToStartIsJudgedOnceStarted();
testAPassingSuiteLeavesNoWorkerBehind();
testKillingTheGateKillsTheSuiteGroup();
testTheCleanupDoesNotWaitForEverForAGroupToGo();
testStoppingTheTestEndsItsGateAndItsStayingGroup();
console.log('coverageGateRetiresSuiteGroup_wave4: all tests passed');
