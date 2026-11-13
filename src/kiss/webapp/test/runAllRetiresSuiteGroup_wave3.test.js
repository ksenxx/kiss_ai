// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: test/run-all.js ends every process a suite leaves in
// its process group once it reports the suite, including processes that
// do not hold the suite's stdout/stderr (stdio 'ignore').
//
// Such a process never holds `close` back, so before the fix the runner
// reported the suite at once and never killed its group: the process
// lived on.  Two suites leave one behind:
//
//  1. hung.coverage.js, a coverage gate that starts the worker and then
//     runs its suite, which hangs: the gate's CHILD_TIMEOUT_MS (90 s,
//     below the runner's 120 s cap) ends the suite, then the gate exits
//     and is reported, and only the runner's group kill ends the worker.
//     The gate starts the worker itself, before its timeout is armed,
//     so that the worker's pid is on record whatever the suite does: a
//     suite slow to start on a loaded host could otherwise be killed
//     before writing it, and the runner be blamed for a worker that was
//     never started.  (What a gate's suite leaves behind is cleaned up
//     by the gate itself: coverageGateRetiresSuiteGroup_wave4.)
//  2. leaky.test.js, a plain suite that exits leaving a worker running.
//
// Who killed each worker is checked as in conc1018_m_runall_grandchild:
// the runner runs under a Python helper that makes itself a child
// subreaper (prctl), so each orphaned worker is adopted by the helper,
// which waits for it and reports how it ended.  The runner and the gate
// helper are copied into a scratch directory, the gate's 90 s timeout
// lowered to GATE_TIMEOUT_MS (the copy is the only knob).

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawn, spawnSync} = require('child_process');

// The gate's cap on its hung suite; nothing has to happen within it (the
// gate records its worker's pid before arming it), so it is just short.
const GATE_TIMEOUT_MS = 2000;

if (process.platform !== 'linux') {
  console.log(
    'skip - runAllRetiresSuiteGroup_wave3 needs prctl(PR_SET_CHILD_SUBREAPER) (Linux only)',
  );
  process.exit(0);
}

// Runs `node <runner>` as a child subreaper, then waits for each worker
// whose pid a file in `pid_dir` names: "signal N"/"exit N" is how it
// ended, "alive" means it still ran 3 s after the runner exited (the
// helper then kills it), "not-ours" that the helper could not wait for
// it.  Prints {"status": <runner exit status>, "fates": {file: fate}}.
// A runner that outlives the deadline (an optional 4th argument, 60 s by
// default) is SIGTERMed, which lets run-all.js's killAllAndExit SIGKILL
// its suites' groups, then SIGKILLed if it stays, and reported with
// status "timeout".  The deadline starts once the pid files an optional
// 5th argument names (comma-separated) are all there, so that a host slow
// to start node cannot use it up before the processes the test asks about
// exist.  A runner that ends first, or does not get that far within the
// 6th argument (20 s by default), goes the way of one past its deadline
// and is reported with status "not ready: <the names missing>".  The
// suites a killed runner left (each in a process group of its own, which
// no kill of the runner reaches) are adopted by the helper, which kills
// their groups; the workers on record are then waited for and killed, and
// whatever else was adopted is killed and reaped, so a regression of the
// cleanup under test never leaks a suite or a worker past the suite.  The
// runner gets a session of its own, so no group the helper kills is the
// helper's.
const SUBREAPER_HELPER = `
import ctypes, json, os, signal, subprocess, sys, time
node, runner, pid_dir = sys.argv[1:4]
deadline = float(sys.argv[4]) if len(sys.argv) > 4 else 60
ready = sys.argv[5].split(",") if len(sys.argv) > 5 else []
startup = float(sys.argv[6]) if len(sys.argv) > 6 else 20
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER

def adopted():
    """Pids of this process's children: what the runner left behind."""
    me = str(os.getpid())
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % name) as f:
                stat = f.read()
        except OSError:
            continue
        if stat[stat.rindex(")") + 2:].split()[1] == me:
            pids.append(int(name))
    return pids

def kill_group(pid):
    """SIGKILL pid's process group, or pid alone when that is our group."""
    try:
        pgid = os.getpgid(pid)
        if pgid == os.getpgid(0):
            os.kill(pid, signal.SIGKILL)
        else:
            os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass

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

def missing():
    """The names in ready whose pid file is not there (yet)."""
    return [n for n in ready if not os.path.exists(os.path.join(pid_dir, n))]

status = "timeout"
proc = subprocess.Popen(
    [node, runner], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    start_new_session=True,
)
ready_by = time.monotonic() + startup
while missing() and proc.poll() is None and time.monotonic() < ready_by:
    time.sleep(0.02)
not_ready = missing()
try:
    out, err = proc.communicate(timeout=0 if not_ready else deadline)
    status = proc.returncode
except subprocess.TimeoutExpired:
    proc.terminate()
    try:
        out, err = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        for pid in adopted():
            kill_group(pid)
        out, err = proc.communicate()
sys.stderr.write(out + err)
if not_ready:
    status = "not ready: " + ",".join(not_ready)
fates = {
    name: fate(int(open(os.path.join(pid_dir, name)).read()))
    for name in sorted(os.listdir(pid_dir)) if not name.endswith(".tmp")
}
for pid in adopted():
    kill_group(pid)
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
print(json.dumps({"status": status, "fates": fates}))
`;

/** SIGKILL every pid a file in `pidDir` names that is still alive. */
function killRecordedWorkers(pidDir) {
  for (const name of fs.readdirSync(pidDir)) {
    const pid = parseInt(fs.readFileSync(path.join(pidDir, name), 'utf8'), 10);
    try {
      if (pid > 0) process.kill(pid, 'SIGKILL');
    } catch {
      // ESRCH: already gone.
    }
  }
}

/** A suite that, `startDelayMs` after node started it, starts a `sleep`
 * worker with stdio 'ignore' (left in the suite's process group), records
 * the worker's pid in `pidFile`, then runs the statement `then`.  A pid
 * file appears whole, by a rename (`record`, which `then` may use too):
 * the helper takes its being there for the process being on record. */
function suiteSource(pidFile, then, startDelayMs = 0) {
  return [
    "'use strict';",
    "const {spawn} = require('child_process');",
    "const fs = require('fs');",
    'function record(file, pid) {',
    "  fs.writeFileSync(file + '.tmp', String(pid));",
    "  fs.renameSync(file + '.tmp', file);",
    '}',
    'Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ' +
      `${startDelayMs});`,
    "const worker = spawn('sleep', ['300'], {stdio: 'ignore'});",
    `record(${JSON.stringify(pidFile)}, worker.pid);`,
    then,
    '',
  ].join('\n');
}

function testReportedSuitesTakeTheirWorkersWithThem() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'runall-wave3-'));
  const pidDir = path.join(dir, 'pids');
  fs.mkdirSync(pidDir);
  try {
    fs.copyFileSync(
      path.join(__dirname, 'run-all.js'),
      path.join(dir, 'run-all.js'),
    );
    const gate = fs.readFileSync(
      path.join(__dirname, 'coverage_gate.js'),
      'utf8',
    );
    const needle = /const CHILD_TIMEOUT_MS = [^;]+;/;
    assert.ok(needle.test(gate), 'coverage_gate.js still declares the timeout');
    fs.writeFileSync(
      path.join(dir, 'coverage_gate.js'),
      gate.replace(needle, `const CHILD_TIMEOUT_MS = ${GATE_TIMEOUT_MS};`),
    );
    fs.writeFileSync(
      path.join(dir, 'hung.coverage.js'),
      suiteSource(
        path.join(pidDir, 'hung'),
        "require('./coverage_gate').runCovered('hung.test.js', 'hung');",
      ),
    );
    fs.writeFileSync(
      path.join(dir, 'hung.test.js'),
      "'use strict';\nsetInterval(function () {}, 1000);\n",
    );
    fs.writeFileSync(
      path.join(dir, 'leaky.test.js'),
      suiteSource(path.join(pidDir, 'leaky'), 'process.exit(0);'),
    );
    const res = spawnSync(
      'python3',
      [
        '-c',
        SUBREAPER_HELPER,
        process.execPath,
        path.join(dir, 'run-all.js'),
        pidDir,
      ],
      {encoding: 'utf8', timeout: 90000, killSignal: 'SIGKILL'},
    );
    const out = `${res.stdout || ''}${res.stderr || ''}`;
    assert.ok(!res.error, `the helper ran: ${res.error}\n${out}`);
    assert.strictEqual(res.status, 0, `the helper finished\n${out}`);
    const result = JSON.parse(res.stdout.trim().split('\n').pop());
    assert.ok(
      out.includes(`did not exit within ${GATE_TIMEOUT_MS / 1000}s, killed`),
      `the gate's own timeout killed its hung suite\n${out}`,
    );
    assert.strictEqual(result.status, 1, `the hung gate fails the run\n${out}`);
    assert.deepStrictEqual(
      result.fates,
      {hung: 'signal 9', leaky: 'signal 9'},
      `each worker was killed with its reported suite's group\n${out}`,
    );
  } finally {
    killRecordedWorkers(pidDir);
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

// The helper's own cleanup: a runner that hangs past the deadline is
// killed, and the worker it left on record is killed and reported too,
// instead of surviving the suite (as it did when the deadline fired
// before any cleanup ran).
function testAHungRunnerStillGetsItsWorkersCleanedUp() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'runall-wave3-hang-'));
  const pidDir = path.join(dir, 'pids');
  fs.mkdirSync(pidDir);
  try {
    // A "runner" that starts a worker, records it, and never exits.  It
    // starts the worker 1.5 s late, as a loaded host would: the 1 s
    // deadline, armed when the runner was spawned, then ended a runner
    // with no worker to report, which failed the test for no fault of the
    // cleanup.  The deadline now starts once the worker is on record.
    fs.writeFileSync(
      path.join(dir, 'run-all.js'),
      suiteSource(
        path.join(pidDir, 'stuck'),
        'setInterval(() => {}, 1000);',
        1500,
      ),
    );
    const res = spawnSync(
      'python3',
      [
        '-c',
        SUBREAPER_HELPER,
        process.execPath,
        path.join(dir, 'run-all.js'),
        pidDir,
        '1',
        'stuck',
      ],
      {encoding: 'utf8', timeout: 30000, killSignal: 'SIGKILL'},
    );
    const out = `${res.stdout || ''}${res.stderr || ''}`;
    assert.strictEqual(res.status, 0, `the helper finished\n${out}`);
    const result = JSON.parse(res.stdout.trim().split('\n').pop());
    assert.strictEqual(result.status, 'timeout', `the runner timed out\n${out}`);
    assert.deepStrictEqual(result.fates, {stuck: 'alive'}, out);
    const pid = parseInt(fs.readFileSync(path.join(pidDir, 'stuck'), 'utf8'), 10);
    assert.throws(
      () => process.kill(pid, 0),
      {code: 'ESRCH'},
      'the worker the hung runner left behind is gone',
    );
  } finally {
    killRecordedWorkers(pidDir);
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

// A runner that never records what the helper waits for does not hold it
// for ever, whether it hangs (the wait is bounded, 1 s here) or has
// ended: it goes the way of one past its deadline, and the failure is
// reported as the setup's.
function testARunnerThatNeverGetsReadyIsReportedAsSuch(runnerSource) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'runall-wave3-never-'));
  const pidDir = path.join(dir, 'pids');
  fs.mkdirSync(pidDir);
  try {
    fs.writeFileSync(path.join(dir, 'run-all.js'), runnerSource);
    const res = spawnSync(
      'python3',
      [
        '-c',
        SUBREAPER_HELPER,
        process.execPath,
        path.join(dir, 'run-all.js'),
        pidDir,
        '1',
        'stuck',
        '1',
      ],
      {encoding: 'utf8', timeout: 30000, killSignal: 'SIGKILL'},
    );
    const out = `${res.stdout || ''}${res.stderr || ''}`;
    assert.strictEqual(res.status, 0, `the helper finished\n${out}`);
    assert.deepStrictEqual(
      JSON.parse(res.stdout.trim().split('\n').pop()),
      {status: 'not ready: stuck', fates: {}},
      out,
    );
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

// The helper's timeout path against the tree the real runner leaves: a
// suite in a process group of its own (spawned `detached`), with a
// worker of its own.  Killing the runner at the deadline reached neither;
// the suite was adopted by the helper, which waited for the recorded
// worker only ("not-ours": a grandchild) and exited, and the outer
// cleanup kills recorded pids, never the suite: both lived on.  The helper
// now SIGTERMs the runner first, so run-all.js's killAllAndExit retires
// its suites' groups, and after the forced kill of a runner that cannot
// (here one ignoring SIGTERM) it kills and reaps the groups of the
// children it adopted.  `fates.suite` tells who retired the suite:
// "not-ours" when the runner reaped it, "signal 9" when the helper did.
function testAHungRunnersSuiteTreeIsRetired(runnerSource, fates, why) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'runall-wave3-tree-'));
  const pidDir = path.join(dir, 'pids');
  fs.mkdirSync(pidDir);
  try {
    fs.writeFileSync(path.join(dir, 'run-all.js'), runnerSource);
    fs.writeFileSync(
      path.join(dir, 'hang.test.js'),
      suiteSource(
        path.join(pidDir, 'worker'),
        `record(${JSON.stringify(path.join(pidDir, 'suite'))}, ` +
          'process.pid);\nsetInterval(() => {}, 1000);',
      ),
    );
    // The deadline starts once the suite has recorded both pids, however
    // long a loaded host takes to start the runner and the suite.
    const res = spawnSync(
      'python3',
      [
        '-c',
        SUBREAPER_HELPER,
        process.execPath,
        path.join(dir, 'run-all.js'),
        pidDir,
        '1',
        'suite,worker',
      ],
      {encoding: 'utf8', timeout: 30000, killSignal: 'SIGKILL'},
    );
    const out = `${res.stdout || ''}${res.stderr || ''}`;
    assert.strictEqual(res.status, 0, `${why}: the helper finished\n${out}`);
    const result = JSON.parse(res.stdout.trim().split('\n').pop());
    assert.strictEqual(
      result.status,
      'timeout',
      `${why}: the runner timed out\n${out}`,
    );
    assert.deepStrictEqual(result.fates, fates, `${why}\n${out}`);
    for (const name of ['suite', 'worker']) {
      const pid = parseInt(
        fs.readFileSync(path.join(pidDir, name), 'utf8'),
        10,
      );
      assert.throws(
        () => process.kill(pid, 0),
        {code: 'ESRCH'},
        `${why}: the ${name} the hung runner left behind is gone`,
      );
    }
  } finally {
    killRecordedWorkers(pidDir);
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

// A runner that starts the suite as run-all.js does, in a process group
// of its own, and cannot clean up: it ignores SIGTERM.
const STUBBORN_RUNNER = [
  "'use strict';",
  "const {spawn} = require('child_process');",
  "const path = require('path');",
  "process.on('SIGTERM', () => {});",
  "spawn(process.execPath, [path.join(__dirname, 'hang.test.js')], " +
    "{stdio: 'ignore', detached: true});",
  'setInterval(() => {}, 1000);',
  '',
].join('\n');

// How a failing run ends: with status 1 and its whole report.  CAP_SUITE,
// run first (--serial), sets its parent's RLIMIT_NOFILE to 0 (once sure the
// parent is that runner: an orphan's is someone else), so the next spawns
// fail with EMFILE and leave a child no stdout or stderr: the runner used
// to die there of a TypeError, status 143.  It also used to end with
// process.exit(1), which drops what a pipe has not accepted yet: stdout is
// left unread here until the runner's last write, the FAILED SUITES list,
// is on stderr.  Not reachable without a test double: spawn() throwing
// while such a child awaits its 'error' (why runSuite checks child.pid).
const CAP_SUITE = `
console.log('x'.repeat(2e6));
const runner = process.ppid;
const cmdline = require('fs').readFileSync('/proc/' + runner + '/cmdline');
if (!cmdline.includes(require('path').basename(__dirname))) process.exit(1);
require('child_process').execFileSync('python3', [
  '-c',
  'import resource as r; r.prlimit(' + runner + ', r.RLIMIT_NOFILE, (0, 0))',
]);
`;

function testAFailingRunReportsInFull(done) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'runall-wave3-end-'));
  process.once('exit', () => fs.rmSync(dir, {recursive: true, force: true}));
  const runner = path.join(dir, 'run-all.js');
  fs.copyFileSync(path.join(__dirname, 'run-all.js'), runner);
  const opts = {timeout: 60000, killSignal: 'SIGKILL'};
  const {status, stdout, stderr} = spawnSync(process.execPath, [runner], opts);
  assert.match(`${status}:${stdout}${stderr}`, /^1:no test files found in /);
  fs.writeFileSync(path.join(dir, 'a.test.js'), CAP_SUITE);
  for (const s of ['b', 'c']) fs.writeFileSync(`${dir}/${s}.test.js`, '');
  const child = spawn(process.execPath, [runner, '--serial'], opts);
  let out = '';
  let err = '';
  child.stdout.pause().on('data', chunk => (out += chunk));
  child.stderr.on('data', chunk => {
    err += chunk;
    if (err.includes('\n  c.test.js\n')) child.stdout.resume();
  });
  child.on('close', code => {
    assert.strictEqual(code, 1, `two suites fail, not the runner\n${err}`);
    assert.match(err, /b\.test\.js: Error: spawn .+ EMFILE\nc\.test\.js: /);
    assert.ok(out.endsWith('\n1/3 suites passed\n'), 'the summary arrived');
    done();
  });
}

testReportedSuitesTakeTheirWorkersWithThem();
testAHungRunnerStillGetsItsWorkersCleanedUp();
testARunnerThatNeverGetsReadyIsReportedAsSuch(
  'setInterval(() => {}, 1000);\n',
);
testARunnerThatNeverGetsReadyIsReportedAsSuch('process.exit(3);\n');
testAHungRunnersSuiteTreeIsRetired(
  fs.readFileSync(path.join(__dirname, 'run-all.js'), 'utf8'),
  {suite: 'not-ours', worker: 'signal 9'},
  'the real runner, SIGTERMed, retires its suite',
);
testAHungRunnersSuiteTreeIsRetired(
  STUBBORN_RUNNER,
  {suite: 'signal 9', worker: 'signal 9'},
  'the helper retires the suite of a runner that ignores SIGTERM',
);
testAFailingRunReportsInFull(() =>
  console.log('runAllRetiresSuiteGroup_wave3: all tests passed'),
);
