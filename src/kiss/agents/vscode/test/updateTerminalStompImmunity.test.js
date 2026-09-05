// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Regression (2026-09-05): clicking Update on a machine whose workspace has
// a selected Python venv printed the update command at the zsh prompt, then
// `^C`, then ` source .../.venv/bin/activate` — and the update never ran.
// The Python(-Environments) extension "activates" the venv in every new
// terminal; that activation makes VS Code send Ctrl+C first (core clears
// what it believes is leftover prompt input, microsoft/vscode#287139) and
// the ^C cancelled the sendText'd update command before it executed.
//
// The fix: runUpdate() no longer types the command into an interactive
// shell.  The terminal is created with shellPath='/bin/bash' and
// shellArgs=['-c', "trap '' INT TERM HUP; <installer>; <hold-open tail>"],
// so the installer IS the terminal process (no prompt to stomp on) and a
// stray \x03 written into the PTY cannot SIGINT it.
//
// These tests reproduce the stomp against a REAL pseudo-terminal (python3's
// pty module — the same line discipline that turns \x03 into SIGINT for the
// foreground process group), running the REAL compiled runUpdate() to obtain
// the exact terminal process argv:
//   1. Control: an unguarded `bash -c 'sleep; touch marker'` under the same
//      PTY dies on the injected \x03 and never writes its marker — proving
//      the harness really delivers SIGINT (the original bug mechanism).
//   2. The captured update terminal process survives the same \x03 plus an
//      injected ` source .../activate` line and completes the install.
//   3. A failing installer keeps the pane open (the hold-open tail), prints
//      the exit status, survives the stomp, and forwards the exit code
//      after Enter.
//
// No mocks beyond the vscode module stub required to load the compiled
// sidebar; every process, PTY, signal and file is real.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const {spawnSync} = require('child_process');

if (process.platform === 'win32') {
  console.log('SKIP: POSIX PTY test, not applicable on Windows');
  process.exit(0);
}

const OUT_SIDEBAR = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
if (!fs.existsSync(OUT_SIDEBAR)) {
  console.log(`SKIP: ${OUT_SIDEBAR} missing — run \`npm run compile\``);
  process.exit(0);
}

class StubEventEmitter {
  constructor() {
    this._listeners = [];
    this.event = cb => {
      this._listeners.push(cb);
      return {
        dispose: () => {
          const idx = this._listeners.indexOf(cb);
          if (idx >= 0) this._listeners.splice(idx, 1);
        },
      };
    };
  }
  fire(arg) {
    for (const cb of this._listeners.slice()) cb(arg);
  }
  dispose() {
    this._listeners = [];
  }
}

const terminals = [];

global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => ({fsPath: p, scheme: 'file'}),
    joinPath: (uri, ...parts) => ({
      fsPath: path.join(uri.fsPath, ...parts),
      scheme: uri.scheme || 'file',
    }),
  },
  ProgressLocation: {Notification: 15},
  window: {
    createTerminal(opts) {
      const t = {
        name: opts && opts.name,
        cwd: opts && opts.cwd,
        shellPath: opts && opts.shellPath,
        shellArgs: opts && opts.shellArgs,
        sent: [],
        show() {},
        sendText(text) {
          this.sent.push(text);
        },
        dispose() {},
      };
      terminals.push(t);
      return t;
    },
    onDidCloseTerminal: () => ({dispose: () => {}}),
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
  },
  commands: {executeCommand: () => Promise.resolve()},
};
const realResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return realResolve.call(this, request, ...rest);
};

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-upd-stomp-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_SORCAR_SOCK = path.join(tmpHome, 'no-daemon.sock');
const installRoot = path.join(tmpHome, '.kiss', 'kiss_ai');
const scriptsDir = path.join(installRoot, 'scripts');
fs.mkdirSync(scriptsDir, {recursive: true});
// findInstallScript() requires the root install.sh; runUpdate() then prefers
// the locked scripts/install.sh, which is the fake installer under test.
fs.writeFileSync(path.join(installRoot, 'install.sh'), '#!/bin/sh\nexit 0\n');
const bootstrapPath = path.join(scriptsDir, 'install.sh');

const {SorcarSidebarView} = require(OUT_SIDEBAR);

// PTY harness: fork the given argv onto a real pseudo-terminal, write the
// scheduled injections (\x03, activation text, Enter) into the master side,
// stream the child's output, and exit with the child's exit code (or
// 128+signal).  python3's pty module is the lightest real-PTY driver that is
// present on every macOS and Linux machine this suite runs on.  Injection
// offsets are measured from the moment the READY TOKEN appears in the
// child's output, not from fork(): on a loaded runner exec can lag, and an
// injection landing before the command under test has started would test
// nothing (or kill the wrong process).
const PTY_HARNESS = `
import json, os, pty, select, sys, time
spec = json.loads(sys.argv[1])
cwd = sys.argv[2]
token = sys.argv[3].encode()
cmd = sys.argv[4:]
pid, fd = pty.fork()
if pid == 0:
    os.chdir(cwd)
    os.execvp(cmd[0], cmd)
born = time.time()
start = None
out = b""
i = 0
inj = sorted(spec, key=lambda x: x["at"])
while True:
    now = time.time()
    if start is not None:
        while i < len(inj) and inj[i]["at"] <= now - start:
            os.write(fd, inj[i]["data"].encode())
            i += 1
    if now - born > 25:
        os.kill(pid, 9)
        break
    r, _, _ = select.select([fd], [], [], 0.05)
    if r:
        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break
        out += data
    if start is None and token in out:
        start = time.time()
_, status = os.waitpid(pid, 0)
sys.stdout.buffer.write(out)
code = os.waitstatus_to_exitcode(status)
sys.exit(128 - code if code < 0 else code)
`;

function runInPty(argv, cwd, readyToken, injections) {
  const startedAt = Date.now();
  const res = spawnSync(
    'python3',
    ['-c', PTY_HARNESS, JSON.stringify(injections), cwd, readyToken, ...argv],
    {encoding: 'utf8', timeout: 60000},
  );
  assert.strictEqual(res.error, undefined, `pty harness failed: ${res.error}`);
  return {
    code: res.status,
    output: res.stdout + res.stderr,
    elapsedMs: Date.now() - startedAt,
  };
}

// The stomp the Python extension performs on every new terminal: Ctrl+C to
// clear "pending input", then the venv activation line (leading space keeps
// it out of shell history — exactly as in the bug report's transcript).
const STOMP = [
  {at: 0.6, data: '\u0003'},
  {at: 0.7, data: ' source /nonexistent/.venv/bin/activate\n'},
];

function captureUpdateTerminal() {
  const view = new SorcarSidebarView({
    fsPath: path.resolve(__dirname, '..'),
    scheme: 'file',
  });
  const before = terminals.length;
  view.runUpdate();
  view.dispose();
  assert.strictEqual(terminals.length, before + 1, 'no update terminal');
  return terminals[terminals.length - 1];
}

function testControlProvesPtyDeliversSigint() {
  // The reproduction: an UNGUARDED command at the mercy of the PTY — the
  // moral equivalent of the sendText'd command — is killed by the injected
  // \x03 before it finishes.  This validates that the harness actually
  // exercises the bug mechanism the fix defends against.
  const marker = path.join(tmpHome, 'control.marker');
  const {code} = runInPty(
    ['/bin/bash', '-c', `echo CONTROL_READY; sleep 2; : > '${marker}'`],
    tmpHome,
    'CONTROL_READY',
    STOMP,
  );
  assert.ok(
    !fs.existsSync(marker),
    'control run survived \\x03 — the PTY harness is not delivering SIGINT',
  );
  assert.notStrictEqual(code, 0, 'control run must die on the injected ^C');
  console.log('  ✓ control: an unguarded command dies on the injected ^C');
}

function testUpdateTerminalSurvivesStomp() {
  const marker = path.join(tmpHome, 'install-done.marker');
  fs.writeFileSync(
    bootstrapPath,
    '#!/bin/bash\necho INSTALL_STARTED\nsleep 2\n' +
      `echo "installed home=$KISS_HOME nonint=$KISS_NONINTERACTIVE"\n: > '${marker}'\n`,
  );
  const term = captureUpdateTerminal();
  assert.strictEqual(term.sent.length, 0, 'update must not use sendText');
  assert.strictEqual(term.shellPath, '/bin/bash');
  const {code, output} = runInPty(
    [term.shellPath, ...term.shellArgs],
    term.cwd,
    'INSTALL_STARTED',
    STOMP,
  );
  assert.ok(
    fs.existsSync(marker),
    `installer was killed by the stomp; output:\n${output}`,
  );
  assert.strictEqual(code, 0, `update must exit 0, got ${code}:\n${output}`);
  assert.ok(
    output.includes(`installed home=${path.join(tmpHome, '.kiss')} nonint=1`),
    `installer must run with the pinned environment:\n${output}`,
  );
  console.log('  ✓ the update terminal survives ^C + venv-activation stomp');
}

function testFailingInstallHoldsPaneOpenAndForwardsExitCode() {
  fs.writeFileSync(
    bootstrapPath,
    '#!/bin/bash\necho "boom before failing"\nexit 7\n',
  );
  const term = captureUpdateTerminal();
  const {code, output, elapsedMs} = runInPty(
    [term.shellPath, ...term.shellArgs],
    term.cwd,
    'boom before failing',
    // The stomp arrives right after the failure, ANOTHER delayed
    // activation line arrives at 2.2s (a slow Python extension), and the
    // user presses Enter at 3.5s.  Only the empty Enter may close the
    // pane — the non-empty activation lines must be ignored, whenever
    // they arrive (gpt-5.6-sol review finding #1).
    STOMP.concat([
      {at: 2.2, data: ' source /nonexistent/.venv/bin/activate\n'},
      {at: 3.5, data: '\r'},
    ]),
  );
  assert.strictEqual(code, 7, `exit status must be forwarded:\n${output}`);
  assert.ok(
    output.includes('update exited with status 7'),
    `failure must stay visible until Enter:\n${output}`,
  );
  assert.ok(
    output.includes('boom before failing'),
    `installer output must be shown:\n${output}`,
  );
  // If any injected non-empty line had closed the pane, the process would
  // have exited by ~2.3s; surviving until the 3.5s Enter proves the pane
  // stayed open for the user.
  assert.ok(
    elapsedMs > 3000,
    `pane closed before the user's Enter (${elapsedMs}ms):\n${output}`,
  );
  console.log('  ✓ a failing install holds the pane open and forwards status');
}

function main() {
  try {
    testControlProvesPtyDeliversSigint();
    testUpdateTerminalSurvivesStomp();
    testFailingInstallHoldsPaneOpenAndForwardsExitCode();
  } finally {
    fs.rmSync(tmpHome, {recursive: true, force: true});
  }
  console.log('updateTerminalStompImmunity: all tests passed');
}

main();
