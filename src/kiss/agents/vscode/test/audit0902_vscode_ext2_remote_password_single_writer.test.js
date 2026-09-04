// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-02 (vscode-ext2): ensureRemotePassword() in
// src/DependencyInstaller.ts saved the remote-access password with an
// unlocked read-modify-replace of $KISS_HOME/config.json
// (`cfg = readKissConfig(); cfg.remote_password = pw; writeKissConfig(cfg)`)
// while the daemon's single writer, kiss/core/vscode_config.py::save_config,
// merges under an fcntl.flock on .config.lock.  The two interleave freely:
// the daemon's save between the extension's read and write is rolled back
// to stale values, and the extension's password vanishes when the daemon
// read the file just before the extension replaced it.  Node has no flock,
// so the fix makes the Python side the only writer: saveKissConfig() hands
// the update to save_config() through `uv run python` (payload on stdin).
//
// Review follow-up (review-vscode.md #4/#5): the first fix still fell back
// to the unlocked write whenever uv was missing or Python failed (the same
// race, plus a password file created with umask permissions), and it ran
// `uv` through execFileSync, freezing the extension host for up to 60 s.
// Now the save is asynchronous and there is no fallback: a failed save
// leaves config.json untouched and tells the user to set the password in
// the settings panel (whose Remote password field goes through the daemon).
//
// Scenario 1: saveKissConfig() with the repo's real uv merges the password
// into a pre-populated config.json and preserves every other key.
// Scenario 2 (the race): a Python child runs save_config() in a tight loop,
// checking its own write landed, while this process saves passwords
// through saveKissConfig(); no update on either side may be lost.
// Scenario 3: no fallback -- uv missing, uv binary unusable (no stderr) and
// Python failing (stderr) all reject, log the reason and leave config.json
// byte-for-byte untouched (no password file appears when there was none).
// Scenario 4: ensureRemotePassword() end to end through the vscode stub:
// already set / found on retry / prompted and saved (trimmed) / prompt
// dismissed / save failed -> actionable error notification.
// Scenario 5: the event loop keeps turning while a slow uv runs (a 10 ms
// timer fires long before the 1 s fake uv exits).
//
// The extension resolves $KISS_HOME when out/DependencyInstaller.js loads,
// so the module is required after pointing KISS_HOME at a temp dir.

/* global require, process, console, __dirname, global, setTimeout */

'use strict';

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT = path.join(__dirname, '..', 'out');
// src/kiss/agents/vscode/test -> repo root, which IS the kiss project.
const KISS_PROJECT = path.resolve(__dirname, '..', '..', '..', '..', '..');
assert.ok(
  fs.existsSync(path.join(KISS_PROJECT, 'pyproject.toml')),
  `kiss project not found at ${KISS_PROJECT}`,
);

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit-pw-'));
const kissHome = path.join(root, 'kiss-home');
fs.mkdirSync(kissHome);
process.env.KISS_HOME = kissHome;
const CONFIG = path.join(kissHome, 'config.json');
const BASE_CONFIG = {
  email: 'user@example.com',
  tunnel_token: 'tok-123',
  max_budget: 42,
  last_model: 'm-init',
  seq: -1,
};

const prompts = [];
const infos = [];
const errors = [];
let nextPrompt = undefined;
global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => undefined}),
  },
  ProgressLocation: {Notification: 15},
  window: {
    showInputBox: async opts => {
      prompts.push(opts.title);
      return nextPrompt;
    },
    showInformationMessage: msg => {
      infos.push(msg);
      return Promise.resolve(undefined);
    },
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: msg => {
      errors.push(msg);
      return Promise.resolve(undefined);
    },
  },
  commands: {executeCommand: () => Promise.resolve()},
};
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

const di = require(path.join(OUT, 'DependencyInstaller.js'));
const {findUvPath} = require(path.join(OUT, 'kissPaths.js'));
const UV = findUvPath();
assert.ok(UV, 'this suite needs a real uv binary');

function resetConfig() {
  fs.writeFileSync(CONFIG, JSON.stringify(BASE_CONFIG, null, 2) + '\n');
}

function readConfig() {
  return JSON.parse(fs.readFileSync(CONFIG, 'utf-8'));
}

function assertBaseKeysPreserved(cfg, where) {
  assert.strictEqual(cfg.email, BASE_CONFIG.email, `${where}: email lost`);
  assert.strictEqual(
    cfg.tunnel_token,
    BASE_CONFIG.tunnel_token,
    `${where}: tunnel_token lost`,
  );
  assert.strictEqual(cfg.max_budget, 42, `${where}: max_budget lost`);
}

function assertNoStrayFiles(where) {
  const stray = fs
    .readdirSync(kissHome)
    .filter(
      n => n !== 'config.json' && n !== '.config.lock' && n !== 'install.log',
    );
  assert.deepStrictEqual(stray, [], `${where}: stray files ${stray}`);
}

// --- Scenario 1 -----------------------------------------------------------
async function singleSave() {
  resetConfig();
  await di.saveKissConfig({remote_password: 'pw-one'}, UV, KISS_PROJECT);
  const cfg = readConfig();
  assert.strictEqual(cfg.remote_password, 'pw-one');
  assertBaseKeysPreserved(cfg, 'single save');
  assert.strictEqual(cfg.seq, -1, 'unknown keys must pass through save_config');
  assert.ok(
    fs.existsSync(path.join(kissHome, '.config.lock')),
    'save_config takes the daemon lock file',
  );
  assertNoStrayFiles('single save');
  console.log(
    '  ✓ saveKissConfig() saves through the daemon writer, preserving keys',
  );
}

// --- Scenario 2 -----------------------------------------------------------
const DAEMON_LOOP_PY = `
import json, os, sys
from pathlib import Path
from kiss.core.vscode_config import save_config
stop = Path(sys.argv[1])
cfg_path = Path(os.environ["KISS_HOME"]) / "config.json"
i = 0
lost = 0
samples = []
while not stop.exists():
    save_config({"seq": i, "last_model": f"m{i}"})
    got = json.loads(cfg_path.read_text())
    bad = []
    if got.get("seq") != i:
        bad.append(f"seq={got.get('seq')!r} wanted {i}")
    for k in ("email", "tunnel_token", "max_budget"):
        if k not in got:
            bad.append(f"{k} missing")
    if bad:
        lost += 1
        if len(samples) < 5:
            samples.append("; ".join(bad))
    i += 1
print(json.dumps({"writes": i, "lost": lost, "samples": samples}))
`;

function startDaemonLoop(stopFile) {
  const script = path.join(root, 'daemon_loop.py');
  fs.writeFileSync(script, DAEMON_LOOP_PY);
  return new Promise((resolve, reject) => {
    const child = spawn(UV, ['run', 'python', script, stopFile], {
      cwd: KISS_PROJECT,
      env: {...process.env, KISS_HOME: kissHome},
      stdio: ['ignore', 'pipe', 'inherit'],
    });
    let out = '';
    child.stdout.on('data', d => {
      out += d;
    });
    child.on('error', reject);
    child.on('exit', code => {
      if (code !== 0) reject(new Error(`daemon loop exited ${code}`));
      else resolve(JSON.parse(out));
    });
  });
}

async function raceWithDaemon() {
  resetConfig();
  const stopFile = path.join(root, 'stop');
  const loop = startDaemonLoop(stopFile);
  // Wait until the Python loop is actually writing.
  const started = Date.now();
  while (readConfig().seq < 0) {
    assert.ok(Date.now() - started < 60_000, 'daemon loop never started');
    await new Promise(r => setTimeout(r, 20));
  }
  const EXT_WRITES = 12;
  let lastSeq = -1;
  const problems = [];
  try {
    for (let j = 0; j < EXT_WRITES; j++) {
      const pw = `pw-${j}`;
      await di.saveKissConfig({remote_password: pw}, UV, KISS_PROJECT);
      const cfg = readConfig();
      if (cfg.remote_password !== pw) {
        problems.push(`write ${j}: password is ${cfg.remote_password}`);
      }
      if (cfg.seq < lastSeq) {
        problems.push(
          `write ${j}: daemon seq went back ${lastSeq} -> ${cfg.seq}`,
        );
      }
      lastSeq = cfg.seq;
      // Let the daemon loop advance between extension saves.
      await new Promise(r => setTimeout(r, 30));
      const later = readConfig();
      if (later.remote_password !== pw) {
        problems.push(
          `after write ${j}: daemon rolled password back to ${later.remote_password}`,
        );
      }
    }
  } finally {
    fs.writeFileSync(stopFile, '');
  }
  const res = await loop;
  console.log(
    `  race: daemon writes=${res.writes} daemon lost=${res.lost} ` +
      `extension problems=${problems.length}`,
  );
  assert.ok(res.writes > 50, 'daemon loop barely ran');
  assert.strictEqual(
    res.lost,
    0,
    `daemon updates lost: ${res.samples.join(' | ')}`,
  );
  assert.deepStrictEqual(
    problems,
    [],
    `extension updates lost:\n${problems.join('\n')}`,
  );
  const final = readConfig();
  assert.strictEqual(final.remote_password, `pw-${EXT_WRITES - 1}`);
  assert.strictEqual(final.seq, res.writes - 1);
  assertBaseKeysPreserved(final, 'race');
  assertNoStrayFiles('race');
  console.log(
    '  ✓ concurrent daemon saves and extension saves never lose an update',
  );
}

// --- Scenario 3 -----------------------------------------------------------
async function rejects(promise, pattern, what) {
  let err;
  try {
    await promise;
  } catch (e) {
    err = e;
  }
  assert.ok(err instanceof Error, `${what}: saveKissConfig did not reject`);
  assert.match(err.message, pattern, `${what}: unexpected reason`);
}

async function noFallback() {
  const logFile = path.join(kissHome, 'install.log');

  // No config.json at all: a failed save must not create one (the
  // fallback used to create a password-bearing file with umask perms).
  fs.rmSync(CONFIG, {force: true});
  await rejects(
    di.saveKissConfig({remote_password: 'no-uv'}, null, KISS_PROJECT),
    /uv not found/,
    'no uv',
  );
  assert.ok(
    !fs.existsSync(CONFIG),
    'a failed save must not create config.json',
  );
  assert.ok(
    fs.readFileSync(logFile, 'utf-8').includes('uv not found'),
    'missing uv must be logged',
  );

  // A uv path that does not exist: spawn fails, no stderr.
  resetConfig();
  const before = fs.readFileSync(CONFIG, 'utf-8');
  await rejects(
    di.saveKissConfig(
      {remote_password: 'bad-uv'},
      path.join(root, 'no-such-uv'),
      KISS_PROJECT,
    ),
    /ENOENT/,
    'bad uv',
  );
  assert.strictEqual(
    fs.readFileSync(CONFIG, 'utf-8'),
    before,
    'bad uv: config.json must be untouched',
  );
  assert.ok(
    fs.readFileSync(logFile, 'utf-8').includes('ENOENT'),
    'spawn failure must be logged',
  );

  // Python itself failing: node stands in for uv and rejects the `run`
  // argument on stderr, which must end up in the log and the reason.
  await rejects(
    di.saveKissConfig(
      {remote_password: 'py-fail'},
      process.execPath,
      KISS_PROJECT,
    ),
    /exited with code \d+[\s\S]*run/,
    'python failure',
  );
  assert.strictEqual(
    fs.readFileSync(CONFIG, 'utf-8'),
    before,
    'python failure: config.json must be untouched',
  );
  assert.ok(
    /save_config failed[\s\S]*run/.test(fs.readFileSync(logFile, 'utf-8')),
    'python failure and its stderr must be logged',
  );
  assertNoStrayFiles('no fallback');
  console.log(
    '  ✓ no fallback: missing uv, unusable uv and failing Python leave config.json alone',
  );
}

// A stand-in uv: sleeps, prints on stderr, exits non-zero.
function writeSlowUv(seconds) {
  const script = path.join(root, 'slow-uv.js');
  fs.writeFileSync(
    script,
    "process.stdin.resume(); setTimeout(() => { process.stderr.write('slow uv gave up'); " +
      `process.exit(3); }, ${seconds * 1000});\n`,
  );
  const sh = path.join(root, 'slow-uv');
  fs.writeFileSync(
    sh,
    `#!/bin/sh\nexec '${process.execPath}' '${script}' "$@"\n`,
    {mode: 0o755},
  );
  return sh;
}

// --- Scenario 5 -----------------------------------------------------------
async function eventLoopNotBlocked() {
  resetConfig();
  const before = fs.readFileSync(CONFIG, 'utf-8');
  const slowUv = writeSlowUv(1);
  let timerFiredAt = 0;
  const t0 = Date.now();
  setTimeout(() => {
    timerFiredAt = Date.now();
  }, 10);
  const pending = di.saveKissConfig(
    {remote_password: 'slow'},
    slowUv,
    KISS_PROJECT,
  );
  await rejects(pending, /code 3[\s\S]*slow uv gave up/, 'slow uv');
  const elapsed = Date.now() - t0;
  assert.ok(elapsed >= 900, `fake uv exited too early (${elapsed} ms)`);
  assert.ok(timerFiredAt > 0, 'the 10 ms timer never fired');
  assert.ok(
    timerFiredAt - t0 < 500,
    `event loop was blocked: timer fired after ${timerFiredAt - t0} ms`,
  );
  assert.strictEqual(fs.readFileSync(CONFIG, 'utf-8'), before);
  console.log('  ✓ a slow uv does not block the extension host event loop');
}

// --- Scenario 4 -----------------------------------------------------------
async function promptFlow() {
  const logFile = path.join(kissHome, 'install.log');

  // Already set: no prompt, no wait.
  resetConfig();
  await di.saveKissConfig({remote_password: 'preset'}, UV, KISS_PROJECT);
  const t0 = Date.now();
  await di.ensureRemotePassword(UV, KISS_PROJECT);
  assert.ok(Date.now() - t0 < 1500, 'must not wait when the password is set');
  assert.deepStrictEqual(prompts, []);
  assert.strictEqual(readConfig().remote_password, 'preset');

  // Found on retry: the daemon writes the password during the 2 s grace.
  resetConfig();
  setTimeout(() => {
    void di.saveKissConfig({remote_password: 'late'}, UV, KISS_PROJECT);
  }, 500);
  await di.ensureRemotePassword(UV, KISS_PROJECT);
  assert.deepStrictEqual(prompts, [], 'must not prompt when found on retry');
  assert.ok(fs.readFileSync(logFile, 'utf-8').includes('found on retry'));

  // Prompted and saved through the daemon writer, trimmed.
  resetConfig();
  nextPrompt = '  typed-pw  ';
  await di.ensureRemotePassword(UV, KISS_PROJECT);
  assert.strictEqual(prompts.length, 1);
  assert.match(prompts[0], /Remote Access Password/);
  let cfg = readConfig();
  assert.strictEqual(cfg.remote_password, 'typed-pw');
  assertBaseKeysPreserved(cfg, 'prompt');
  assert.ok(
    fs.readFileSync(logFile, 'utf-8').includes('Remote access password saved'),
    'the prompted password must be reported as saved',
  );
  assert.ok(
    fs.existsSync(path.join(kissHome, '.config.lock')),
    'the prompted password must go through the daemon writer (lock file)',
  );

  // Prompt dismissed (Esc) and blank answer: nothing written, a hint shown.
  resetConfig();
  nextPrompt = undefined;
  await di.ensureRemotePassword(UV, KISS_PROJECT);
  nextPrompt = '   ';
  await di.ensureRemotePassword(UV, KISS_PROJECT);
  assert.strictEqual(prompts.length, 3);
  assert.strictEqual(infos.length, 2);
  assert.match(infos[0], /set the remote access password later/);
  cfg = readConfig();
  assert.strictEqual(cfg.remote_password, undefined);
  assertBaseKeysPreserved(cfg, 'dismissed');

  // The save fails (uv missing): nothing is written, and the user gets
  // an error pointing at the settings panel's Remote password field.
  resetConfig();
  const before = fs.readFileSync(CONFIG, 'utf-8');
  nextPrompt = 'doomed';
  await di.ensureRemotePassword(null, KISS_PROJECT);
  assert.strictEqual(prompts.length, 4);
  assert.strictEqual(
    fs.readFileSync(CONFIG, 'utf-8'),
    before,
    'a failed save must leave config.json untouched',
  );
  assert.strictEqual(errors.length, 1, 'one error notification expected');
  assert.match(errors[0], /could not save the remote access password/i);
  assert.match(errors[0], /settings panel[\s\S]*Remote password/);
  assert.match(errors[0], /uv not found/);
  assert.ok(
    /ensureRemotePassword: saving the password failed[\s\S]*uv not found/.test(
      fs.readFileSync(logFile, 'utf-8'),
    ),
    'the failure must be logged',
  );
  console.log(
    '  ✓ ensureRemotePassword(): set / found on retry / prompted / dismissed / save failed',
  );
}

async function main() {
  try {
    await singleSave();
    await raceWithDaemon();
    await noFallback();
    await eventLoopNotBlocked();
    await promptFlow();
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
  console.log(
    'audit0902_vscode_ext2_remote_password_single_writer: all tests passed',
  );
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
