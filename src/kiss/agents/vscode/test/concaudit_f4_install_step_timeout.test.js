// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end regression test: a stalled install step must not leave
// setup pending for ever.
//
// The bug: every runAsync() call (`uv sync`, Playwright installs) passed
// `timeoutMs: 0`, i.e. no deadline at all.  A `uv sync` that hung --
// stuck on its environment lock, a dead mirror, a wedged build backend
// -- left ensureDependencies() and its "Setting up" progress toast
// pending for the lifetime of the extension host, with no error and
// nothing to retry.
//
// The fix gives each step a 30-minute ceiling, kills the step's whole
// process group when it expires and surfaces a retryable error.  The
// test drives the REAL compiled ensureDependencies() in a child process
// against a fake HOME / KISS project whose `uv sync` hangs (and forks a
// grandchild, to prove the GROUP is killed, not just `uv`).  A real
// 30-minute wait is not acceptable in the suite, so the compiled module
// is copied to a temp dir with that ONE constant lowered (the copy
// asserts the substitution matched, so a rename fails loudly).

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OUT_DIR = path.join(__dirname, '..', 'out');
if (!fs.existsSync(path.join(OUT_DIR, 'DependencyInstaller.js'))) {
  console.log(
    'SKIP: out/DependencyInstaller.js missing — run `npm run compile`',
  );
  process.exit(0);
}
if (process.platform === 'win32') {
  console.log('SKIP: POSIX process groups only');
  process.exit(0);
}

const SHORT_CEILING_MS = 1500;

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-installto-'));
const tmpHome = path.join(tmpRoot, 'home');
const kissHome = path.join(tmpHome, '.kiss');
const fakeBin = path.join(tmpRoot, 'bin');
const fakeProj = path.join(tmpRoot, 'proj');
const stateDir = path.join(tmpRoot, 'state');
const outCopy = path.join(tmpRoot, 'out');
for (const d of [
  path.join(tmpHome, '.local', 'bin'),
  kissHome,
  fakeBin,
  fakeProj,
  stateDir,
  outCopy,
]) {
  fs.mkdirSync(d, {recursive: true});
}

// The compiled extension with the install-step ceiling lowered.
for (const f of fs.readdirSync(OUT_DIR)) {
  if (!f.endsWith('.js')) continue;
  let src = fs.readFileSync(path.join(OUT_DIR, f), 'utf-8');
  if (f === 'DependencyInstaller.js') {
    const needle = 'const INSTALL_STEP_TIMEOUT_MS = 30 * 60_000;';
    assert.ok(
      src.includes(needle),
      'DependencyInstaller.js no longer defines INSTALL_STEP_TIMEOUT_MS as expected',
    );
    src = src.replace(
      needle,
      `const INSTALL_STEP_TIMEOUT_MS = ${SHORT_CEILING_MS};`,
    );
  }
  fs.writeFileSync(path.join(outCopy, f), src);
}

// A KISS project with no .venv, so setup takes the `uv sync` path.
fs.writeFileSync(
  path.join(fakeProj, 'pyproject.toml'),
  '[project]\nname = "kiss"\nversion = "0"\n',
);

// Fake uv: `sync` records its pid, forks a grandchild and hangs; every
// other invocation succeeds instantly.
const fakeUv = path.join(tmpHome, '.local', 'bin', 'uv');
fs.writeFileSync(
  fakeUv,
  `#!/bin/sh
case "$1" in
  sync)
    echo $$ > "${stateDir}/uv.pid"
    sleep 600 &
    echo $! > "${stateDir}/child.pid"
    wait
    ;;
esac
exit 0
`,
);
fs.chmodSync(fakeUv, 0o755);
// A `code` on PATH keeps setup away from the VS Code CLI installers.
const fakeCode = path.join(fakeBin, 'code');
fs.writeFileSync(fakeCode, '#!/bin/sh\nexit 0\n');
fs.chmodSync(fakeCode, 0o755);

const CHILD = `
'use strict';
const path = require('path');
const Module = require('module');
const stubPath = path.join(process.env.KISS_TEST_DIR, '_vscode-stub.js');
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return stubPath;
  return origResolve.call(this, request, parent, ...rest);
};
const noop = () => Promise.resolve(undefined);
global.__kissVscodeStub = {
  window: {
    withProgress: (opts, task) =>
      task({report: () => {}}, {isCancellationRequested: false}),
    showErrorMessage: noop,
    showWarningMessage: noop,
    showInformationMessage: noop,
    showInputBox: noop,
  },
  workspace: {
    isTrusted: true,
    workspaceFolders: undefined,
    getConfiguration: () => ({get: () => undefined}),
  },
  ProgressLocation: {Notification: 15},
};
const installer = require(process.env.KISS_TEST_MODULE);
installer
  .ensureDependencies()
  .then(() => {
    process.stdout.write('RESOLVED');
  })
  .catch(err => {
    process.stdout.write('REJECTED:' + (err && err.message));
  })
  .finally(() => process.exit(0));
`;

function readPid(name) {
  try {
    return parseInt(fs.readFileSync(path.join(stateDir, name), 'utf-8'), 10);
  } catch {
    return 0;
  }
}

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function waitUntilDead(pid, ms) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (!alive(pid)) return true;
    await new Promise(r => setTimeout(r, 50));
  }
  return !alive(pid);
}

function cleanup() {
  for (const name of ['uv.pid', 'child.pid']) {
    const pid = readPid(name);
    if (pid > 0) {
      try {
        process.kill(pid, 'SIGKILL');
      } catch {}
    }
  }
  try {
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  } catch {}
}

async function main() {
  const started = Date.now();
  const out = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['-e', CHILD], {
      stdio: ['ignore', 'pipe', 'inherit'],
      env: {
        ...process.env,
        HOME: tmpHome,
        USERPROFILE: tmpHome,
        KISS_HOME: kissHome,
        KISS_PROJECT_PATH: fakeProj,
        PATH: `${fakeBin}:${process.env.PATH}`,
        KISS_TEST_DIR: __dirname,
        KISS_TEST_MODULE: path.join(outCopy, 'DependencyInstaller.js'),
      },
    });
    let buf = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(
        new Error(
          'ensureDependencies() did not settle within 60s: the stalled ' +
            '`uv sync` is holding setup pending again',
        ),
      );
    }, 60000);
    child.stdout.on('data', d => {
      buf += d.toString();
    });
    child.on('close', () => {
      clearTimeout(timer);
      resolve(buf.trim());
    });
  });
  const elapsed = Date.now() - started;

  // The installer's log() also writes to stdout; the verdict is its own line.
  const verdict = out.split('\n').find(l => /^(RESOLVED|REJECTED:)/.test(l));
  assert.ok(
    verdict && verdict.startsWith('REJECTED:'),
    `setup must fail with a retryable error, got: ${out}`,
  );
  assert.match(
    verdict,
    /uv sync did not finish within [\d.]+ minutes and was killed; reload the window to retry/,
    `unexpected error text: ${verdict}`,
  );
  console.log(
    `  ok - stalled \`uv sync\` surfaced a retryable error after ${elapsed}ms`,
  );

  const uvPid = readPid('uv.pid');
  const childPid = readPid('child.pid');
  assert.ok(uvPid > 0 && childPid > 0, 'the fake uv never ran its sync branch');
  assert.ok(
    await waitUntilDead(uvPid, 3000),
    'the stalled `uv` is still alive',
  );
  assert.ok(
    await waitUntilDead(childPid, 3000),
    "the stalled uv's grandchild survived: only the direct child was killed, " +
      'not the process group',
  );
  console.log('  ok - the whole process group of the stalled step was killed');
}

main()
  .then(() => {
    cleanup();
    console.log('concaudit_f4_install_step_timeout: all assertions passed');
  })
  .catch(err => {
    cleanup();
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
