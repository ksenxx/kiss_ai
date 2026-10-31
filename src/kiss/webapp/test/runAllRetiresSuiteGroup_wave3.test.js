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
//  1. hung.coverage.js, a coverage gate whose suite hangs.  The gate's
//     CHILD_TIMEOUT_MS (90 s, below the runner's 120 s cap) kills only
//     the suite process, then the gate exits and is reported; with the
//     old 300 s gate timeout the runner's group kill came first and
//     ended the suite's worker with the gate.
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

// Long enough for the gate's suite to start node, spawn its worker and
// write its pid file on a loaded host.
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
const SUBREAPER_HELPER = `
import ctypes, json, os, signal, subprocess, sys, time
node, runner, pid_dir = sys.argv[1:4]
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
proc = subprocess.run([node, runner], capture_output=True, text=True, timeout=60)
sys.stderr.write(proc.stdout + proc.stderr)

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

fates = {
    name: fate(int(open(os.path.join(pid_dir, name)).read()))
    for name in sorted(os.listdir(pid_dir))
}
print(json.dumps({"status": proc.returncode, "fates": fates}))
`;

/** A suite that starts a `sleep` worker with stdio 'ignore' (left in the
 * suite's process group), records the worker's pid in `pidFile`, then
 * hangs or exits. */
function suiteSource(pidFile, hangs) {
  return [
    "'use strict';",
    "const {spawn} = require('child_process');",
    "const fs = require('fs');",
    "const worker = spawn('sleep', ['300'], {stdio: 'ignore'});",
    `fs.writeFileSync(${JSON.stringify(pidFile)}, String(worker.pid));`,
    hangs ? 'setInterval(function () {}, 1000);' : 'process.exit(0);',
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
      "require('./coverage_gate').runCovered('hung.test.js', 'hung');\n",
    );
    fs.writeFileSync(
      path.join(dir, 'hung.test.js'),
      suiteSource(path.join(pidDir, 'hung'), true),
    );
    fs.writeFileSync(
      path.join(dir, 'leaky.test.js'),
      suiteSource(path.join(pidDir, 'leaky'), false),
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
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

testReportedSuitesTakeTheirWorkersWithThem();
console.log('runAllRetiresSuiteGroup_wave3: all tests passed');
