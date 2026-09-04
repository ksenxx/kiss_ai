// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-03 (vscode-installer): installCliScript() and
// writeShellRc() in src/DependencyInstaller.ts replaced two files other
// processes read at arbitrary moments — the executable
// `~/.local/bin/sorcar` (exec'ed by user shells) and the user's shell rc
// (sourced by every new terminal) — with a bare fs.writeFileSync, which
// truncates before writing.  A reader in that window got an EMPTY or
// truncated file: a shell exec'ing `sorcar` mid-write silently ran a
// no-op (or partial!) script, and a terminal opened mid-write sourced an
// rc without the user's PATH and API keys.  Every VS Code window runs
// runFinalization at activation, so two windows also hammered the same
// paths concurrently.  The fix routes both through writeFileAtomicSync
// (unique temp name + rename, symlink- and mode-preserving).
//
// Scenario 1: two REAL child processes hammer installCliScript() with
// alternating contents while this process reads the wrapper as fast as
// it can: every read must be one writer's payload, whole, and the file
// must stay 0755.
// Scenario 2: same two-process hammer on writeShellRc(): every read of
// the rc is a complete payload.
// Scenario 3: a symlinked rc (dotfile repo) keeps being a symlink and
// the file it points at gets the content; permission bits (0600) are
// preserved across the replace.
// Scenario 4: a fresh rc path (no existing file) is created; an
// explicit-mode wrapper write lands 0755 regardless of prior mode.
// Scenario 5: installCliScript resolves a relative uv through `which`,
// falls back to ~/.local/bin/uv when `which` fails, returns without
// writing when HOME is empty (child process), and logs instead of
// throwing when the wrapper path is unwritable (a directory squatting
// on ~/.local/bin/sorcar).
// Scenario 6: a DANGLING symlinked rc (dotfile repo with the target not
// checked out yet) stays a symlink; the missing referent is created
// with the content — including multi-hop chains and a bounded fallback
// for link loops.
// Scenario 7: a write that creates the temp file and then fails
// (RLIMIT_FSIZE → EFBIG, the ENOSPC class) leaves no temp litter and
// keeps the old target intact.
// Scenario 8: writeFileAtomicSync leaves no temp litter when the rename
// fails (target is a directory) and rethrows the failure.
//
// The win32 branch of installCliScript (sorcar.cmd) is unreachable on
// the Linux/macOS hosts this suite runs on and is not exercised.

/* global require, process, console, __dirname, global, setTimeout */

'use strict';

const assert = require('assert');
const {spawn, spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT = path.join(__dirname, '..', 'out');
const DI = path.join(OUT, 'DependencyInstaller.js');

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit-atomic-'));
const home = path.join(root, 'home');
const kissHome = path.join(root, 'kiss-home');
fs.mkdirSync(home);
fs.mkdirSync(kissHome);
process.env.HOME = home;
process.env.KISS_HOME = kissHome;

global.__kissVscodeStub = {};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') {
    return origResolve.call(
      this,
      path.join(__dirname, '_vscode-stub.js'),
      ...rest,
    );
  }
  return origResolve.call(this, request, ...rest);
};

const di = require(DI);

const SORCAR = path.join(home, '.local', 'bin', 'sorcar');
// Long payload segments widen the truncate-to-write window the old code
// exposed (same trick as the 0902 update-cache test).
const PROJ_A = path.join(root, 'proj-' + 'a'.repeat(2000));
const PROJ_B = path.join(root, 'proj-' + 'b'.repeat(2000));
const UV_ABS = path.join(root, 'fake-uv');

// Worker hammering one exported function in its own process.  argv:
// [DI path, stub path, fn, target/home, contentTag, untilMs]
const WORKER_SRC = `
'use strict';
const Module = require('module');
const path = require('path');
global.__kissVscodeStub = {};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') {
    return origResolve.call(this, process.argv[2], ...rest);
  }
  return origResolve.call(this, request, ...rest);
};
const di = require(process.argv[1]);
const fn = process.argv[3];
const arg1 = process.argv[4];
const arg2 = process.argv[5];
const untilMs = Number(process.argv[6]);
let writes = 0;
while (Date.now() < untilMs) {
  if (fn === 'installCliScript') di.installCliScript(arg1, arg2);
  else di.writeShellRc(arg1, arg2);
  writes++;
}
process.stdout.write(String(writes));
`;

function runWorker(fn, arg1, arg2, untilMs, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      [
        '-e',
        WORKER_SRC,
        DI,
        path.join(__dirname, '_vscode-stub.js'),
        fn,
        arg1,
        arg2,
        String(untilMs),
      ],
      {stdio: ['ignore', 'pipe', 'inherit'], env: {...process.env, ...env}},
    );
    let out = '';
    child.stdout.on('data', d => {
      out += d;
    });
    child.on('error', reject);
    child.on('exit', code => {
      if (code !== 0) reject(new Error(`worker ${fn} exited ${code}`));
      else resolve(Number(out));
    });
  });
}

function assertWholeScript(text, reads) {
  assert.ok(text.length > 0, `read ${reads}: wrapper is EMPTY (torn write)`);
  assert.ok(
    text.startsWith('#!/bin/bash\n'),
    `read ${reads}: wrapper missing shebang: ${JSON.stringify(text.slice(0, 40))}`,
  );
  assert.ok(
    text.endsWith(' sorcar "$@"\n'),
    `read ${reads}: wrapper truncated (${text.length} bytes)`,
  );
  const hasA = text.includes(PROJ_A);
  const hasB = text.includes(PROJ_B);
  assert.ok(
    hasA !== hasB,
    `read ${reads}: wrapper holds ${hasA && hasB ? 'BOTH' : 'NEITHER'} payload`,
  );
}

async function scenarioCliHammer() {
  const untilMs = Date.now() + 2000;
  const writersDone = Promise.all([
    runWorker('installCliScript', PROJ_A, UV_ABS, untilMs),
    runWorker('installCliScript', PROJ_B, UV_ABS, untilMs),
  ]);
  // Wait for the first write, then read as fast as possible.
  while (!fs.existsSync(SORCAR) && Date.now() < untilMs) {
    await new Promise(r => setTimeout(r, 5));
  }
  let reads = 0;
  while (Date.now() < untilMs) {
    let text;
    try {
      text = fs.readFileSync(SORCAR, 'utf-8');
    } catch (err) {
      // rename() never unlinks the target; a missing file means a
      // non-atomic replace.
      assert.fail(`read failed mid-hammer: ${err.message}`);
    }
    assertWholeScript(text, reads);
    reads++;
    if (reads % 500 === 0) await new Promise(r => setTimeout(r, 0));
  }
  const [writesA, writesB] = await writersDone;
  assert.ok(writesA > 10 && writesB > 10, `writers too slow: ${writesA}/${writesB}`);
  assert.ok(reads > 100, `reader too slow: ${reads}`);
  const mode = fs.statSync(SORCAR).mode & 0o777;
  assert.strictEqual(mode, 0o755, `wrapper mode ${mode.toString(8)}`);
  console.log(`  cli hammer ok: ${writesA + writesB} writes, ${reads} clean reads`);
}

async function scenarioRcHammer() {
  const rc = path.join(root, 'rc-hammer', '.bashrc');
  fs.mkdirSync(path.dirname(rc));
  const payloadA = '# rc A\n' + 'export KISS_A=1 # ' + 'a'.repeat(4000) + '\n';
  const payloadB = '# rc B\n' + 'export KISS_B=1 # ' + 'b'.repeat(4000) + '\n';
  const untilMs = Date.now() + 2000;
  const writersDone = Promise.all([
    runWorker('writeShellRc', rc, payloadA, untilMs),
    runWorker('writeShellRc', rc, payloadB, untilMs),
  ]);
  while (!fs.existsSync(rc) && Date.now() < untilMs) {
    await new Promise(r => setTimeout(r, 5));
  }
  let reads = 0;
  while (Date.now() < untilMs) {
    const text = fs.readFileSync(rc, 'utf-8');
    assert.ok(
      text === payloadA || text === payloadB,
      `read ${reads}: rc is neither writer's payload whole ` +
        `(${text.length} bytes)`,
    );
    reads++;
    if (reads % 500 === 0) await new Promise(r => setTimeout(r, 0));
  }
  await writersDone;
  assert.ok(reads > 100, `reader too slow: ${reads}`);
  console.log(`  rc hammer ok: ${reads} clean reads`);
}

function scenarioSymlinkAndMode() {
  const dir = path.join(root, 'rc-symlink');
  fs.mkdirSync(dir);
  const realRc = path.join(dir, 'real-bashrc');
  const linkRc = path.join(dir, '.bashrc');
  fs.writeFileSync(realRc, '# original\n', {mode: 0o600});
  fs.chmodSync(realRc, 0o600);
  fs.symlinkSync(realRc, linkRc);

  di.writeShellRc(linkRc, '# replaced through the symlink');

  assert.ok(
    fs.lstatSync(linkRc).isSymbolicLink(),
    'the dotfile symlink was replaced by a plain file',
  );
  assert.strictEqual(
    fs.readFileSync(realRc, 'utf-8'),
    '# replaced through the symlink\n',
  );
  assert.strictEqual(
    fs.statSync(realRc).mode & 0o777,
    0o600,
    'permission bits were not preserved across the atomic replace',
  );
  console.log('  symlink + mode preservation ok');
}

function scenarioFreshFileAndExplicitMode() {
  const rc = path.join(root, 'rc-fresh', '.zshrc');
  fs.mkdirSync(path.dirname(rc));
  di.writeShellRc(rc, 'export FRESH=1');
  assert.strictEqual(fs.readFileSync(rc, 'utf-8'), 'export FRESH=1\n');

  // Explicit mode overrides whatever the target had.
  const bin = path.join(root, 'rc-fresh', 'tool');
  fs.writeFileSync(bin, 'old', {mode: 0o644});
  di.writeFileAtomicSync(bin, '#!/bin/bash\n', 0o755);
  assert.strictEqual(fs.statSync(bin).mode & 0o777, 0o755);
  console.log('  fresh file + explicit mode ok');
}

function scenarioCliBranches() {
  // Relative uv resolved through `which`: a shim dir first on PATH.
  const shimDir = path.join(root, 'shim');
  fs.mkdirSync(shimDir);
  const shimUv = path.join(shimDir, 'kiss-shim-uv');
  fs.writeFileSync(shimUv, '#!/bin/bash\nexit 0\n', {mode: 0o755});
  const oldPath = process.env.PATH;
  process.env.PATH = `${shimDir}:${oldPath}`;
  try {
    di.installCliScript('/proj', 'kiss-shim-uv');
    assert.ok(
      fs.readFileSync(SORCAR, 'utf-8').includes(`"${shimUv}" run`),
      'relative uv was not resolved through `which`',
    );

    // `which` failure falls back to ~/.local/bin/uv.
    di.installCliScript('/proj', 'kiss-no-such-uv-anywhere');
    assert.ok(
      fs
        .readFileSync(SORCAR, 'utf-8')
        .includes(`"${path.join(home, '.local', 'bin', 'uv')}" run`),
      'failed `which` did not fall back to ~/.local/bin/uv',
    );
  } finally {
    process.env.PATH = oldPath;
  }

  // Empty HOME: returns without writing (fresh process, HOME/USERPROFILE
  // removed before the module loads).
  const env = {...process.env, KISS_HOME: kissHome};
  delete env.HOME;
  delete env.USERPROFILE;
  const probe = spawnSync(
    process.execPath,
    [
      '-e',
      `
'use strict';
const Module = require('module');
global.__kissVscodeStub = {};
const orig = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return orig.call(this, process.argv[2], ...rest);
  return orig.call(this, request, ...rest);
};
const di = require(process.argv[1]);
di.installCliScript('/proj', '/abs/uv');
`,
      DI,
      path.join(__dirname, '_vscode-stub.js'),
    ],
    {env, encoding: 'utf-8'},
  );
  assert.strictEqual(probe.status, 0, probe.stderr);

  // Unwritable wrapper path (a directory squats on it): logged, not thrown.
  fs.rmSync(SORCAR, {force: true});
  fs.mkdirSync(SORCAR);
  try {
    di.installCliScript('/proj', '/abs/uv');
  } finally {
    fs.rmdirSync(SORCAR);
  }
  console.log('  cli branch coverage ok');
}

// A DANGLING dotfile symlink (`.bashrc -> dotfiles/bashrc` whose target
// does not exist yet) made fs.realpathSync throw, so the atomic writer
// fell back to renaming over the LINK itself: the symlink silently
// became a regular file and the dotfile repo never saw the content.
// The old fs.writeFileSync followed the link and created its target, so
// this was a regression.  The writer must resolve the link chain itself
// and create the missing referent.
function scenarioDanglingSymlink() {
  const dir = path.join(root, 'rc-dangling');
  fs.mkdirSync(path.join(dir, 'dotfiles'), {recursive: true});
  const linkRc = path.join(dir, '.bashrc');
  fs.symlinkSync(path.join('dotfiles', 'bashrc'), linkRc); // relative + dangling

  di.writeShellRc(linkRc, 'export DANGLE=1');

  assert.ok(
    fs.lstatSync(linkRc).isSymbolicLink(),
    'a dangling dotfile symlink was replaced by a regular file',
  );
  assert.strictEqual(
    fs.readFileSync(path.join(dir, 'dotfiles', 'bashrc'), 'utf-8'),
    'export DANGLE=1\n',
    'the dangling link\u2019s target was not created with the content',
  );
  assert.strictEqual(fs.readFileSync(linkRc, 'utf-8'), 'export DANGLE=1\n');

  // A multi-hop chain ending in a missing file: every link survives and
  // the final referent is created.
  const a = path.join(dir, 'a');
  const b = path.join(dir, 'b');
  const c = path.join(dir, 'c');
  fs.symlinkSync('b', a);
  fs.symlinkSync('c', b);
  di.writeFileAtomicSync(a, 'chain');
  assert.ok(fs.lstatSync(a).isSymbolicLink(), 'link a was destroyed');
  assert.ok(fs.lstatSync(b).isSymbolicLink(), 'link b was destroyed');
  assert.strictEqual(fs.readFileSync(c, 'utf-8'), 'chain');

  // A pathological self-loop cannot be followed; the bounded resolver
  // gives up on a node in the cycle and the write replaces it (the old
  // writeFileSync would have thrown ELOOP; either way no litter).
  const loop = path.join(dir, 'loop');
  fs.symlinkSync('loop', loop);
  di.writeFileAtomicSync(loop, 'loop-content');
  assert.strictEqual(fs.readFileSync(loop, 'utf-8'), 'loop-content');
  const litter = fs.readdirSync(dir).filter(f => f.includes('.tmp'));
  assert.deepStrictEqual(litter, [], `temp litter left behind: ${litter}`);
  console.log('  dangling symlink preservation ok');
}

// Scenario 7: a write that CREATES the temp file and then fails (the
// EFBIG/ENOSPC class) must not leave the temp behind.  The old code ran
// fs.writeFileSync(tmp, ...) outside the try whose catch unlinked tmp,
// so a partial write littered the directory.  RLIMIT_FSIZE in a child
// process (with SIGXFSZ handled so write() returns EFBIG) produces the
// real partial-write failure.
function scenarioWriteFailureLeavesNoLitter() {
  const dir = path.join(root, 'write-fail');
  fs.mkdirSync(dir);
  const target = path.join(dir, 'rc');
  fs.writeFileSync(target, 'old content\n');

  const childSrc = `
'use strict';
process.on('SIGXFSZ', () => {});
const Module = require('module');
const fs = require('fs');
global.__kissVscodeStub = {};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') {
    return origResolve.call(this, process.argv[2], ...rest);
  }
  return origResolve.call(this, request, ...rest);
};
const di = require(process.argv[1]);
let code = 'none';
try {
  di.writeFileAtomicSync(process.argv[3], 'x'.repeat(64 * 1024));
} catch (err) {
  code = err.code;
}
process.stdout.write(JSON.stringify({code}));
`;
  const probe = spawnSync(
    'bash',
    [
      '-c',
      `ulimit -f 4 && exec "${process.execPath}" -e "$KISS_CHILD_SRC" ` +
        `"${DI}" "${path.join(__dirname, '_vscode-stub.js')}" "${target}"`,
    ],
    {
      // No NODE_V8_COVERAGE here: writing the (large) coverage JSON at
      // exit would itself trip RLIMIT_FSIZE and SIGXFSZ-kill the child.
      // The cleanup lines this exercises are also covered in-process by
      // the rename-failure and missing-parent-dir cases below.
      env: {...process.env, KISS_CHILD_SRC: childSrc, NODE_V8_COVERAGE: ''},
      encoding: 'utf-8',
    },
  );
  assert.strictEqual(probe.status, 0, probe.stderr);
  const {code} = JSON.parse(probe.stdout);
  assert.strictEqual(code, 'EFBIG', `expected EFBIG, got ${code}`);
  assert.strictEqual(
    fs.readFileSync(target, 'utf-8'),
    'old content\n',
    'a failed write clobbered the target',
  );
  const litter = fs.readdirSync(dir).filter(f => f.endsWith('.tmp'));
  assert.deepStrictEqual(litter, [], `temp litter left behind: ${litter}`);

  // A write that fails BEFORE the temp exists (missing parent
  // directory): the failure propagates and the cleanup is a no-op.
  assert.throws(() => di.writeFileAtomicSync(path.join(dir, 'nosuch', 'rc'), 'x'));
  console.log('  write-failure cleanup ok');
}

function scenarioRenameFailureLeavesNoLitter() {
  const dir = path.join(root, 'rename-fail');
  fs.mkdirSync(dir);
  const target = path.join(dir, 'occupied');
  fs.mkdirSync(target); // realpath ok, rename file->dir must fail
  assert.throws(() => di.writeShellRc(target, 'boom'));
  const litter = fs.readdirSync(dir).filter(f => f.endsWith('.tmp'));
  assert.deepStrictEqual(litter, [], `temp litter left behind: ${litter}`);
  console.log('  rename-failure cleanup ok');
}

async function main() {
  await scenarioCliHammer();
  await scenarioRcHammer();
  scenarioSymlinkAndMode();
  scenarioFreshFileAndExplicitMode();
  scenarioCliBranches();
  scenarioDanglingSymlink();
  scenarioWriteFailureLeavesNoLitter();
  scenarioRenameFailureLeavesNoLitter();
  fs.rmSync(root, {recursive: true, force: true});
  console.log('audit0903_cli_shellrc_atomic_write: all scenarios passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
