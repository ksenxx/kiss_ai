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
const {spawnSync} = require('child_process');

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
// status "timeout".  The suites a killed runner left (each in a process
// group of its own, which no kill of the runner reaches) are adopted by
// the helper, which kills their groups; the workers on record are then
// waited for and killed, and whatever else was adopted is killed and
// reaped, so a regression of the cleanup under test never leaks a suite
// or a worker past the suite.  The runner gets a session of its own, so
// no group the helper kills is the helper's.
const SUBREAPER_HELPER = `
import ctypes, json, os, signal, subprocess, sys, time
node, runner, pid_dir = sys.argv[1:4]
deadline = float(sys.argv[4]) if len(sys.argv) > 4 else 60
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

status = "timeout"
proc = subprocess.Popen(
    [node, runner], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    start_new_session=True,
)
try:
    out, err = proc.communicate(timeout=deadline)
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
fates = {
    name: fate(int(open(os.path.join(pid_dir, name)).read()))
    for name in sorted(os.listdir(pid_dir))
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

/** A suite that starts a `sleep` worker with stdio 'ignore' (left in the
 * suite's process group), records the worker's pid in `pidFile`, then
 * runs the statement `then`. */
function suiteSource(pidFile, then) {
  return [
    "'use strict';",
    "const {spawn} = require('child_process');",
    "const fs = require('fs');",
    "const worker = spawn('sleep', ['300'], {stdio: 'ignore'});",
    `fs.writeFileSync(${JSON.stringify(pidFile)}, String(worker.pid));`,
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
    // A "runner" that starts a worker, records it, and never exits.
    fs.writeFileSync(
      path.join(dir, 'run-all.js'),
      suiteSource(path.join(pidDir, 'stuck'), 'setInterval(() => {}, 1000);'),
    );
    const res = spawnSync(
      'python3',
      ['-c', SUBREAPER_HELPER, process.execPath, path.join(dir, 'run-all.js'), pidDir, '1'],
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
        `fs.writeFileSync(${JSON.stringify(path.join(pidDir, 'suite'))}, ` +
          'String(process.pid));\nsetInterval(() => {}, 1000);',
      ),
    );
    // The deadline leaves a loaded host time to start the runner and the
    // suite, which records both pids as soon as it runs.
    const res = spawnSync(
      'python3',
      [
        '-c',
        SUBREAPER_HELPER,
        process.execPath,
        path.join(dir, 'run-all.js'),
        pidDir,
        '5',
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

testReportedSuitesTakeTheirWorkersWithThem();
testAHungRunnerStillGetsItsWorkersCleanedUp();
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
console.log('runAllRetiresSuiteGroup_wave3: all tests passed');
