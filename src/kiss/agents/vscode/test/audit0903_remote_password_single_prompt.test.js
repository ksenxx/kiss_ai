// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-03 (vscode-installer): ensureRemotePassword() in
// src/DependencyInstaller.ts had no cross-window guard around its input
// box.  Two VS Code windows finalizing together (both reach
// runFinalization after the update marker) both read an empty
// config.json, both slept the same 2 s daemon-grace retry, and both
// prompted the user for a remote-access password — two stacked password
// boxes, and whichever save landed last silently overwrote the other
// (the 0902 fix serialized the WRITES through save_config, not the
// prompts).
//
// The follow-up review found the first lock reuse incomplete at both
// ends of the holder's lifetime, fixed here and covered below:
//  - a waiter only polled fs.existsSync(lockFile), so a holder that
//    CRASHED mid-prompt wedged every waiter for the full 10-minute
//    deadline (scenario: holder crash);
//  - the reused daemon-restart policy force-broke a lock held by a LIVE
//    pid after 10 minutes, evicting a legitimately open password prompt
//    and stacking a second one (scenario: live holder is never evicted);
//  - a waiter could not tell a holder that deliberately skipped from
//    one that crashed: the holder now records a terminal outcome
//    (saved / skipped / failed) under its lock token (scenarios: skip,
//    dead owner with recorded outcome).
//
// Every scenario drives the real compiled ensureRemotePassword() — in
// two real node worker processes where cross-process behavior is the
// point, in this process for the state-machine branches.  Passwords are
// saved through the real `uv run python` save_config path.

/* global require, process, console, __dirname, setTimeout */

'use strict';

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OUT = path.join(__dirname, '..', 'out');
const DI = path.join(OUT, 'DependencyInstaller.js');
const STUB = path.join(__dirname, '_vscode-stub.js');
const KISS_PROJECT = path.resolve(__dirname, '..', '..', '..', '..', '..');
assert.ok(
  fs.existsSync(path.join(KISS_PROJECT, 'pyproject.toml')),
  `kiss project not found at ${KISS_PROJECT}`,
);

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit-pwlock-'));

// Each worker: prompt marker on showInputBox, then (mode 'save'/'skip')
// block until the test releases it and answer, or (mode 'hang') block
// forever so the test can SIGKILL it mid-prompt.
const WORKER_SRC = `
'use strict';
const Module = require('module');
const fs = require('fs');
const path = require('path');
const marks = process.argv[3];
const release = process.argv[4];
const mode = process.argv[6] || 'save';
global.__kissVscodeStub = {
  ProgressLocation: {Notification: 15},
  window: {
    showInputBox: async () => {
      fs.writeFileSync(path.join(marks, 'prompt-' + process.pid), '');
      if (mode === 'hang') {
        // A timer keeps the event loop (and so this process) alive
        // until the test SIGKILLs it mid-prompt.
        setInterval(() => {}, 1000);
        await new Promise(() => {});
      }
      while (!fs.existsSync(release)) {
        await new Promise(r => setTimeout(r, 25));
      }
      if (mode === 'skip') return undefined;
      return 'pw-' + process.pid;
    },
    showInformationMessage: msg => {
      fs.writeFileSync(path.join(marks, 'info-' + process.pid), String(msg));
      return Promise.resolve(undefined);
    },
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: msg => {
      fs.writeFileSync(path.join(marks, 'error-' + process.pid), String(msg));
      return Promise.resolve(undefined);
    },
  },
  commands: {executeCommand: () => Promise.resolve()},
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') {
    return origResolve.call(this, process.argv[2], ...rest);
  }
  return origResolve.call(this, request, ...rest);
};
const di = require(process.argv[1]);
const {findUvPath} = require(path.join(path.dirname(process.argv[1]), 'kissPaths.js'));
const uv = findUvPath();
if (!uv) { console.error('no uv'); process.exit(3); }
di.ensureRemotePassword(uv, process.argv[5])
  .then(() => process.exit(0))
  .catch(err => { console.error(err); process.exit(1); });
`;

/** Create an isolated $KISS_HOME + marks dir + release file for one scenario. */
function makeCtx(name) {
  const dir = path.join(root, name);
  const kissHome = path.join(dir, 'kiss-home');
  const marks = path.join(dir, 'marks');
  fs.mkdirSync(kissHome, {recursive: true});
  fs.mkdirSync(marks);
  fs.writeFileSync(
    path.join(kissHome, 'config.json'),
    JSON.stringify({email: 'user@example.com', seq: 1}, null, 2) + '\n',
  );
  return {
    kissHome,
    marks,
    release: path.join(dir, 'release'),
    config: path.join(kissHome, 'config.json'),
    lock: path.join(kissHome, '.remote-password.lock'),
  };
}

// Every spawned worker is registered so a failing assertion cannot
// leave a hung child holding this process's stdio pipes open.
const liveWorkers = new Set();

function startWorker(ctx, mode) {
  const child = spawn(
    process.execPath,
    ['-e', WORKER_SRC, DI, STUB, ctx.marks, ctx.release, KISS_PROJECT, mode],
    {
      stdio: ['ignore', 'inherit', 'inherit'],
      env: {...process.env, KISS_HOME: ctx.kissHome},
    },
  );
  liveWorkers.add(child);
  child.on('exit', () => liveWorkers.delete(child));
  return child;
}

function promptMarks(ctx) {
  return fs.readdirSync(ctx.marks).filter(f => f.startsWith('prompt-'));
}

function waitFor(pred, what, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const poll = () => {
      if (pred()) return resolve(undefined);
      if (Date.now() > deadline) {
        return reject(new Error('timed out waiting for ' + what));
      }
      setTimeout(poll, 50);
    };
    poll();
  });
}

function exitOf(child) {
  return new Promise((resolve, reject) => {
    child.on('error', reject);
    child.on('exit', code => resolve(code));
  });
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ─── Scenario 1: normal contention — one prompt, waiter honors the save ───

async function scenarioSinglePrompt() {
  const ctx = makeCtx('normal');
  const a = startWorker(ctx, 'save');
  const b = startWorker(ctx, 'save');
  const exits = Promise.all([exitOf(a), exitOf(b)]);

  // Both processes pass their 2 s retry at about the same time; give the
  // loser ample time to have prompted too if it was going to (the old
  // code prompted immediately after the retry).
  await waitFor(() => promptMarks(ctx).length >= 1, 'first prompt', 30000);
  await sleep(3000);
  const prompted = promptMarks(ctx);
  assert.strictEqual(
    prompted.length,
    1,
    `both windows showed the remote-password prompt: ${prompted}`,
  );

  // Release the prompt; the holder saves through real uv/save_config,
  // the waiter sees the recorded outcome and returns without prompting.
  fs.writeFileSync(ctx.release, '');
  const [codeA, codeB] = await exits;
  assert.strictEqual(codeA, 0, 'worker A failed');
  assert.strictEqual(codeB, 0, 'worker B failed');

  const finalPrompts = promptMarks(ctx);
  assert.strictEqual(
    finalPrompts.length,
    1,
    `a second prompt appeared after the first finished: ${finalPrompts}`,
  );
  const winnerPid = finalPrompts[0].slice('prompt-'.length);
  const cfg = JSON.parse(fs.readFileSync(ctx.config, 'utf-8'));
  assert.strictEqual(
    cfg.remote_password,
    'pw-' + winnerPid,
    'the saved password is not the prompting window\u2019s answer',
  );
  assert.strictEqual(cfg.email, 'user@example.com', 'config keys lost');
  const errors = fs.readdirSync(ctx.marks).filter(f => f.startsWith('error-'));
  assert.deepStrictEqual(errors, [], `error notifications: ${errors}`);
  assert.ok(!fs.existsSync(ctx.lock), 'the prompt lock was not released');
  console.log('  ✓ normal contention: one prompt, one saved password');
}

// ─── Scenario 2: the prompting window CRASHES — a waiter takes over ───
//
// The old waiter only polled fs.existsSync(lockFile): a SIGKILLed
// holder left its lock behind and the waiter sat out the full
// 10-minute deadline with no prompt anywhere.  A waiter must detect
// the dead owner pid and take the prompt over within seconds.

async function scenarioHolderCrash() {
  const ctx = makeCtx('crash');
  const holder = startWorker(ctx, 'hang');
  await waitFor(() => promptMarks(ctx).length >= 1, 'holder prompt', 30000);

  const waiter = startWorker(ctx, 'save');
  const waiterExit = exitOf(waiter);
  // Let the waiter pass its 2 s retry and observe the held lock.
  await sleep(3500);
  assert.strictEqual(
    promptMarks(ctx).length,
    1,
    'the waiter prompted while the holder was alive',
  );

  holder.kill('SIGKILL');
  await exitOf(holder);
  const killedAt = Date.now();

  // The recovery must take seconds, not the 10-minute deadline.
  await waitFor(
    () => promptMarks(ctx).length === 2,
    'the waiter to take over the dead holder\u2019s prompt',
    15000,
  );
  const tookMs = Date.now() - killedAt;
  console.log(`  (takeover after ${tookMs} ms)`);

  fs.writeFileSync(ctx.release, '');
  assert.strictEqual(await waiterExit, 0, 'waiter failed');
  assert.strictEqual(
    promptMarks(ctx).length,
    2,
    'more prompts appeared after the takeover',
  );
  const cfg = JSON.parse(fs.readFileSync(ctx.config, 'utf-8'));
  assert.strictEqual(
    cfg.remote_password,
    'pw-' + waiter.pid,
    'the takeover window\u2019s password was not saved',
  );
  assert.ok(!fs.existsSync(ctx.lock), 'the prompt lock was not released');
  console.log('  ✓ crashed holder: waiter recovered in seconds, one new prompt');
}

// ─── Scenario 3: the holder deliberately SKIPS (Esc) — no second prompt ───

async function scenarioHolderSkips() {
  const ctx = makeCtx('skip');
  const a = startWorker(ctx, 'skip');
  const b = startWorker(ctx, 'skip');
  const exits = Promise.all([exitOf(a), exitOf(b)]);
  await waitFor(() => promptMarks(ctx).length >= 1, 'first prompt', 30000);
  fs.writeFileSync(ctx.release, '');
  const [codeA, codeB] = await exits;
  assert.strictEqual(codeA, 0, 'worker A failed');
  assert.strictEqual(codeB, 0, 'worker B failed');
  assert.strictEqual(
    promptMarks(ctx).length,
    1,
    'a deliberate skip must not make the other window re-prompt',
  );
  const cfg = JSON.parse(fs.readFileSync(ctx.config, 'utf-8'));
  assert.strictEqual(cfg.remote_password, undefined, 'skip saved a password');
  console.log('  ✓ deliberate skip: respected, no second prompt');
}

// ─── In-process branch scenarios ───
//
// These drive the same compiled module in this process, with the
// vscode stub's showInputBox switched per scenario.  KISS_HOME is
// fixed at module load, so they share one home and reset config.json
// between scenarios; each uses its own lock path.

const inHome = path.join(root, 'in-process-home');
fs.mkdirSync(inHome);
const inConfig = path.join(inHome, 'config.json');

function resetInConfig(extra) {
  fs.writeFileSync(
    inConfig,
    JSON.stringify({email: 'user@example.com', ...extra}, null, 2) + '\n',
  );
}

let promptedHere = 0;
let inAnswer = undefined;
const stub = {
  ProgressLocation: {Notification: 15},
  window: {
    showInputBox: async () => {
      promptedHere += 1;
      return inAnswer;
    },
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
  },
  commands: {executeCommand: () => Promise.resolve()},
};

function loadInProcess() {
  const Module = require('module');
  global.__kissVscodeStub = stub;
  const origResolve = Module._resolveFilename;
  Module._resolveFilename = function (request, ...rest) {
    if (request === 'vscode') {
      return origResolve.call(this, STUB, ...rest);
    }
    return origResolve.call(this, request, ...rest);
  };
  process.env.KISS_HOME = inHome;
  return require(DI);
}

function deadPid() {
  // A pid that provably does not exist: spawn-and-reap would still be
  // racy against reuse, so probe downward from a huge pid instead.
  for (let pid = 2 ** 21 - 1; ; pid -= 1) {
    try {
      process.kill(pid, 0);
    } catch (err) {
      if (err.code === 'ESRCH') return pid;
    }
  }
}

async function inProcessScenarios() {
  const di = loadInProcess();

  // Stored password: short-circuits before any prompt or lock.
  resetInConfig({remote_password: 'kept'});
  promptedHere = 0;
  await di.ensureRemotePassword(null, KISS_PROJECT, path.join(inHome, 'l0'));
  assert.strictEqual(promptedHere, 0, 'prompted despite a saved password');
  console.log('  ✓ stored password short-circuits');

  // Password appears during the 2 s daemon-grace retry: no prompt.
  resetInConfig({});
  promptedHere = 0;
  const retryCall = di.ensureRemotePassword(
    null,
    KISS_PROJECT,
    path.join(inHome, 'l1'),
  );
  await sleep(500);
  resetInConfig({remote_password: 'from-daemon'});
  await retryCall;
  assert.strictEqual(promptedHere, 0, 'prompted despite the retry finding one');
  console.log('  ✓ password found on the 2 s retry: no prompt');

  const dead = deadPid();

  // A dead owner with a recorded outcome, seen by a FRESH caller: an
  // outcome binds only the waiters of its session, so this caller
  // prompts again — the same behavior as any activation after a
  // normally released skip.  The dead lock is cleaned up.
  resetInConfig({});
  promptedHere = 0;
  inAnswer = undefined;
  const l2 = path.join(inHome, 'l2');
  fs.writeFileSync(l2, JSON.stringify({pid: dead, token: 'tok-dead'}));
  fs.writeFileSync(
    l2 + '.outcome',
    JSON.stringify({token: 'tok-dead', outcome: 'skipped'}),
  );
  await di.ensureRemotePassword(null, KISS_PROJECT, l2, 15000, 50);
  assert.strictEqual(promptedHere, 1, 'a stale outcome bound a fresh caller');
  assert.ok(!fs.existsSync(l2), 'the dead holder\u2019s lock was not cleaned');
  console.log('  ✓ dead owner + old outcome, fresh caller: prompts again');

  // The same state reached WHILE WAITING: a live holder records its
  // terminal outcome and dies before unlinking its lock.  The waiter
  // honours the outcome (a deliberate skip must never stack a second
  // prompt) and cleans the lock up.
  resetInConfig({});
  promptedHere = 0;
  const holderProc = spawn(process.execPath, [
    '-e',
    'setInterval(() => {}, 1000);',
  ]);
  try {
    const l2b = path.join(inHome, 'l2b');
    fs.writeFileSync(
      l2b,
      JSON.stringify({pid: holderProc.pid, token: 'tok-live'}),
    );
    const waiterCall = di.ensureRemotePassword(null, KISS_PROJECT, l2b, 20000, 50);
    await sleep(2600); // past the 2 s retry, waiting on the live holder
    // The holder finishes (outcome recorded) and then dies before its
    // release could unlink the lock.
    fs.writeFileSync(
      l2b + '.outcome',
      JSON.stringify({token: 'tok-live', outcome: 'skipped'}),
    );
    holderProc.kill('SIGKILL');
    await waiterCall;
    assert.strictEqual(promptedHere, 0, 'a waited-on skip was not honored');
    assert.ok(!fs.existsSync(l2b), 'the finished holder\u2019s lock remains');
  } finally {
    try {
      holderProc.kill('SIGKILL');
    } catch {}
  }
  console.log('  ✓ dead owner + outcome, waiter: skip honored, lock cleaned');

  // A dead owner with NO outcome for its token (a stale one from an
  // older session does not count): the prompt is taken over.
  resetInConfig({});
  promptedHere = 0;
  inAnswer = undefined; // Esc
  const l3 = path.join(inHome, 'l3');
  fs.writeFileSync(l3, JSON.stringify({pid: dead, token: 'tok-new'}));
  fs.writeFileSync(
    l3 + '.outcome',
    JSON.stringify({token: 'tok-older-session', outcome: 'saved'}),
  );
  await di.ensureRemotePassword(null, KISS_PROJECT, l3, 15000, 50);
  assert.strictEqual(promptedHere, 1, 'the dead holder was not taken over');
  assert.ok(!fs.existsSync(l3), 'the takeover lock was not released');
  const oc3 = JSON.parse(fs.readFileSync(l3 + '.outcome', 'utf-8'));
  assert.strictEqual(oc3.outcome, 'skipped', 'takeover outcome not recorded');
  console.log('  ✓ dead owner without an outcome: prompt taken over');

  // Malformed outcome records are ignored (treated as none).
  resetInConfig({});
  promptedHere = 0;
  const l4 = path.join(inHome, 'l4');
  fs.writeFileSync(l4, JSON.stringify({pid: dead, token: 't4'}));
  fs.writeFileSync(l4 + '.outcome', JSON.stringify({token: 5, outcome: 'saved'}));
  await di.ensureRemotePassword(null, KISS_PROJECT, l4, 15000, 50);
  assert.strictEqual(promptedHere, 1, 'a malformed outcome suppressed takeover');

  resetInConfig({});
  promptedHere = 0;
  const l5 = path.join(inHome, 'l5');
  fs.writeFileSync(l5, JSON.stringify({pid: dead, token: 't5'}));
  fs.writeFileSync(
    l5 + '.outcome',
    JSON.stringify({token: 't5', outcome: 'not-a-real-outcome'}),
  );
  await di.ensureRemotePassword(null, KISS_PROJECT, l5, 15000, 50);
  assert.strictEqual(promptedHere, 1, 'a bad outcome value suppressed takeover');
  console.log('  ✓ malformed outcome records are ignored');

  // A LIVE holder is never evicted, no matter how old its lock is:
  // the waiter runs out its deadline and leaves the lock alone.
  resetInConfig({});
  promptedHere = 0;
  const l6 = path.join(inHome, 'l6');
  fs.writeFileSync(l6, JSON.stringify({pid: process.pid, token: 'live'}));
  const old = new Date(Date.now() - 20 * 60_000);
  fs.utimesSync(l6, old, old);
  await di.ensureRemotePassword(null, KISS_PROJECT, l6, 1200, 100);
  assert.strictEqual(promptedHere, 0, 'prompted around a live holder');
  assert.ok(
    fs.existsSync(l6),
    'a LIVE holder\u2019s 20-minute-old lock was evicted',
  );
  console.log('  ✓ live holder: never age-evicted, waiter times out quietly');

  // An unreadable lock too young to be stale is honored as live.
  resetInConfig({});
  promptedHere = 0;
  const l7 = path.join(inHome, 'l7');
  fs.writeFileSync(l7, '');
  await di.ensureRemotePassword(null, KISS_PROJECT, l7, 900, 100);
  assert.strictEqual(promptedHere, 0, 'prompted over a fresh unreadable lock');
  assert.ok(fs.existsSync(l7), 'a fresh unreadable lock was broken');
  console.log('  ✓ fresh unreadable lock honored');

  // An unreadable lock past the stale window is broken and the prompt
  // proceeds.
  resetInConfig({});
  promptedHere = 0;
  inAnswer = undefined;
  const l8 = path.join(inHome, 'l8');
  fs.writeFileSync(l8, '');
  fs.utimesSync(l8, old, old);
  await di.ensureRemotePassword(null, KISS_PROJECT, l8, 15000, 50);
  assert.strictEqual(promptedHere, 1, 'a stale unreadable lock was honored');
  console.log('  ✓ stale unreadable lock broken, prompt proceeds');

  // The under-lock re-check: while this process waits on a live
  // holder, the password gets saved and the lock vanishes WITHOUT an
  // outcome (crash cleanup).  The waiter takes the lock, re-checks the
  // config under it and returns without prompting.
  resetInConfig({});
  promptedHere = 0;
  const l9 = path.join(inHome, 'l9');
  fs.writeFileSync(l9, JSON.stringify({pid: process.pid, token: 'l9t'}));
  const recheck = di.ensureRemotePassword(null, KISS_PROJECT, l9, 20000, 50);
  await sleep(2600); // past the 2 s retry, inside the waiter loop
  resetInConfig({remote_password: 'saved-elsewhere'});
  fs.unlinkSync(l9);
  await recheck;
  assert.strictEqual(promptedHere, 0, 'prompted despite the under-lock save');
  const oc9 = JSON.parse(fs.readFileSync(l9 + '.outcome', 'utf-8'));
  assert.strictEqual(oc9.outcome, 'saved', 'under-lock re-check outcome');
  console.log('  ✓ under-lock re-check: saved password respected');

  // A failing save records a "failed" outcome (uv missing here) and
  // does not crash activation.
  resetInConfig({});
  promptedHere = 0;
  inAnswer = 'pw-will-fail';
  const l10 = path.join(inHome, 'l10');
  await di.ensureRemotePassword(null, KISS_PROJECT, l10, 15000, 50);
  assert.strictEqual(promptedHere, 1, 'no prompt in the failed-save scenario');
  const oc10 = JSON.parse(fs.readFileSync(l10 + '.outcome', 'utf-8'));
  assert.strictEqual(oc10.outcome, 'failed', 'failed save outcome missing');
  console.log('  ✓ failed save: recorded as failed, no crash');

  // An unwritable lock directory: the lock can never be taken, the
  // waiter loop runs out its deadline without prompting or throwing.
  resetInConfig({});
  promptedHere = 0;
  const roDir = path.join(inHome, 'ro');
  fs.mkdirSync(roDir);
  fs.chmodSync(roDir, 0o500);
  try {
    await di.ensureRemotePassword(
      null,
      KISS_PROJECT,
      path.join(roDir, 'lock'),
      900,
      100,
    );
  } finally {
    fs.chmodSync(roDir, 0o755);
  }
  assert.strictEqual(promptedHere, 0, 'prompted with an untakeable lock');
  console.log('  ✓ untakeable lock dir: quiet deadline, no prompt');

  // The outcome write itself failing (lock dir turned read-only while
  // the prompt was open) is swallowed: the prompt result stands.
  resetInConfig({});
  promptedHere = 0;
  const ocDir = path.join(inHome, 'oc');
  fs.mkdirSync(ocDir);
  const l11 = path.join(ocDir, 'lock');
  stub.window.showInputBox = async () => {
    promptedHere += 1;
    fs.chmodSync(ocDir, 0o500);
    return undefined;
  };
  try {
    await di.ensureRemotePassword(null, KISS_PROJECT, l11, 15000, 50);
  } finally {
    fs.chmodSync(ocDir, 0o755);
  }
  assert.strictEqual(promptedHere, 1, 'no prompt in the outcome-failure run');
  assert.ok(!fs.existsSync(l11 + '.outcome'), 'outcome written to a ro dir?');
  console.log('  ✓ unwritable outcome: swallowed, activation survives');
}

async function main() {
  await scenarioHolderCrash();
  await scenarioSinglePrompt();
  await scenarioHolderSkips();
  await inProcessScenarios();
  fs.rmSync(root, {recursive: true, force: true});
  console.log('audit0903_remote_password_single_prompt: all scenarios passed');
}

main().catch(err => {
  console.error(err);
  for (const child of liveWorkers) {
    try {
      child.kill('SIGKILL');
    } catch {}
  }
  process.exit(1);
});
