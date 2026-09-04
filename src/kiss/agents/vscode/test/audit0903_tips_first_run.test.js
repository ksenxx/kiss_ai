// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-03 (vscode-main partition): consumeTipsFirstRun TOCTOU.
//
// The first-run tips marker was claimed with `existsSync` followed by
// `writeFileSync`: two VS Code windows activating at the same time (two
// extension-host processes) could both pass the existence check before
// either wrote, so BOTH returned true and the one-time tips popup opened
// in both windows.  The fix claims the marker atomically with the 'wx'
// open flag: exactly one writer wins, the loser gets EEXIST.
//
// Reproduction: two real child node processes, released by a busy-wait
// file barrier so they hit the marker within microseconds of each other,
// each round on a fresh KISS_HOME.  On the broken code at least one
// round ends with two winners; the invariant checked is "exactly one
// winner per round, every round".

/* global require, process, console, __dirname */

'use strict';

const assert = require('assert');
const {spawn} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OUT_SORCAR_TAB = path.join(__dirname, '..', 'out', 'SorcarTab.js');
if (!fs.existsSync(OUT_SORCAR_TAB)) {
  console.log(`SKIP: ${OUT_SORCAR_TAB} missing — run \`npm run compile\``);
  process.exit(0);
}

const ROUNDS = 20;

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit-tips-'));

// The child: hooks the vscode stub, requires the compiled SorcarTab,
// signals readiness, busy-waits for the shared go-file, then races
// consumeTipsFirstRun and reports the result on stdout.
const childScript = path.join(tmpRoot, 'race-child.js');
fs.writeFileSync(
  childScript,
  `
'use strict';
const fs = require('fs');
const Module = require('module');
global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => undefined}),
  },
};
const realResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return ${JSON.stringify(
    require.resolve('./_vscode-stub.js'),
  )};
  return realResolve.call(this, request, ...rest);
};
const {consumeTipsFirstRun} = require(${JSON.stringify(OUT_SORCAR_TAB)});
const [readyFile, goFile] = process.argv.slice(2);
fs.writeFileSync(readyFile, 'ready');
// Busy-wait (no sleep): both children see the go-file within
// microseconds of each other, tighter than the check-then-write window.
for (;;) {
  if (fs.existsSync(goFile)) break;
}
const won = consumeTipsFirstRun();
// A second call in the same process must always lose: the marker is
// there now (whoever wrote it).
const second = consumeTipsFirstRun();
process.stdout.write(JSON.stringify({won, second}));
`,
);

function runChild(kissHome, readyFile, goFile) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [childScript, readyFile, goFile], {
      env: Object.assign({}, process.env, {KISS_HOME: kissHome}),
    });
    let out = '';
    let errOut = '';
    child.stdout.on('data', c => (out += c));
    child.stderr.on('data', c => (errOut += c));
    child.on('error', reject);
    child.on('exit', code => {
      if (code !== 0) {
        reject(new Error(`child exited ${code}: ${errOut}`));
        return;
      }
      try {
        resolve(JSON.parse(out));
      } catch (err) {
        reject(new Error(`bad child output ${JSON.stringify(out)}: ${err}`));
      }
    });
  });
}

function waitForFile(file, timeoutMs) {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const poll = () => {
      if (fs.existsSync(file)) {
        resolve();
        return;
      }
      if (Date.now() - startedAt > timeoutMs) {
        reject(new Error(`timed out waiting for ${file}`));
        return;
      }
      setTimeout(poll, 5);
    };
    poll();
  });
}

async function runRound(round) {
  const dir = path.join(tmpRoot, `round-${round}`);
  const kissHome = path.join(dir, 'kiss-home');
  fs.mkdirSync(dir, {recursive: true});
  const goFile = path.join(dir, 'go');
  const ready1 = path.join(dir, 'ready-1');
  const ready2 = path.join(dir, 'ready-2');
  const p1 = runChild(kissHome, ready1, goFile);
  const p2 = runChild(kissHome, ready2, goFile);
  await waitForFile(ready1, 10_000);
  await waitForFile(ready2, 10_000);
  fs.writeFileSync(goFile, 'go');
  const [r1, r2] = await Promise.all([p1, p2]);
  return {r1, r2};
}

// ─── Post-update reset election ───
//
// Every window that saw the `.extension-updated` marker independently
// removed TIPS_SHOWN before consuming it, so window A could
// reset-and-claim and window B could then reset A's fresh claim and
// claim again: two "what's new" popups for one update.  The reset is
// now an ELECTION: an atomically created (`wx`) claim file named by the
// update stamp picks exactly one window that may remove TIPS_SHOWN;
// every other window's reset is a no-op.
//
// The reproduction is the review's exact interleaving, run as two REAL
// sequential extension-host processes against one home: each does
// reset-then-consume; the second process's reset must NOT clear the
// first one's claim.

const updateChildScript = path.join(tmpRoot, 'update-child.js');
fs.writeFileSync(
  updateChildScript,
  `
'use strict';
const fs = require('fs');
const Module = require('module');
global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => undefined}),
  },
};
const realResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return ${JSON.stringify(
    require.resolve('./_vscode-stub.js'),
  )};
  return realResolve.call(this, request, ...rest);
};
const tab = require(${JSON.stringify(OUT_SORCAR_TAB)});
const goFile = process.argv[2];
if (goFile && goFile !== '-') {
  // Barrier mode: race the reset+consume pair against the sibling.
  fs.writeFileSync(process.argv[3], 'ready');
  for (;;) {
    if (fs.existsSync(goFile)) break;
  }
}
tab.resetTipsOnExtensionUpdate();
const won = tab.consumeTipsFirstRun();
process.stdout.write(JSON.stringify({won}));
`,
);

function runUpdateChild(kissHome, goFile, readyFile) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      [updateChildScript, goFile || '-', readyFile || '-'],
      {env: Object.assign({}, process.env, {KISS_HOME: kissHome})},
    );
    let out = '';
    let errOut = '';
    child.stdout.on('data', c => (out += c));
    child.stderr.on('data', c => (errOut += c));
    child.on('error', reject);
    child.on('exit', code => {
      if (code !== 0) reject(new Error(`update child exited ${code}: ${errOut}`));
      else resolve(JSON.parse(out));
    });
  });
}

function seedUpdatedHome(name, stamp) {
  const kissHome = path.join(tmpRoot, name);
  fs.mkdirSync(kissHome, {recursive: true});
  // A previous run already showed the tips once...
  fs.writeFileSync(path.join(kissHome, 'TIPS_SHOWN'), 'old-claim\n');
  // ...and then the installer wrote the update marker.
  fs.writeFileSync(path.join(kissHome, '.extension-updated'), stamp);
  return kissHome;
}

async function updateResetScenarios() {
  // The review's interleaving, sequential and therefore deterministic:
  // A resets and claims; B then resets and claims too on the old code.
  const seqHome = seedUpdatedHome('update-seq', '2026-09-03T10:00:00Z\n');
  const a = await runUpdateChild(seqHome);
  const b = await runUpdateChild(seqHome);
  const winners = (a.won ? 1 : 0) + (b.won ? 1 : 0);
  assert.strictEqual(
    winners,
    1,
    `post-update: ${winners} windows claimed the tips popup (want 1): ` +
      JSON.stringify({a, b}),
  );
  console.log('  ✓ sequential post-update reset: one winner');

  // The same pair raced through the barrier, several rounds.
  for (let round = 0; round < 10; round++) {
    const home = seedUpdatedHome(`update-race-${round}`, 'stamp-x\n');
    const goFile = path.join(tmpRoot, `update-go-${round}`);
    const ready1 = path.join(tmpRoot, `update-ready-${round}-1`);
    const ready2 = path.join(tmpRoot, `update-ready-${round}-2`);
    const p1 = runUpdateChild(home, goFile, ready1);
    const p2 = runUpdateChild(home, goFile, ready2);
    await waitForFile(ready1, 10_000);
    await waitForFile(ready2, 10_000);
    fs.writeFileSync(goFile, 'go');
    const [r1, r2] = await Promise.all([p1, p2]);
    const raceWinners = (r1.won ? 1 : 0) + (r2.won ? 1 : 0);
    assert.strictEqual(
      raceWinners,
      1,
      `update race round ${round}: ${raceWinners} winners: ` +
        JSON.stringify({r1, r2}),
    );
  }
  console.log('  ✓ 10 racing post-update rounds: one winner each');

  // Branch scenarios, in-process (kissHomeDir reads $KISS_HOME per call).
  const Module = require('module');
  global.__kissVscodeStub = {
    workspace: {
      isTrusted: true,
      workspaceFolders: [],
      getConfiguration: () => ({get: () => undefined}),
    },
  };
  const realResolve = Module._resolveFilename;
  Module._resolveFilename = function (request, ...rest) {
    if (request === 'vscode') return require.resolve('./_vscode-stub.js');
    return realResolve.call(this, request, ...rest);
  };
  const tab = require(OUT_SORCAR_TAB);

  // No update marker: the reset never touches TIPS_SHOWN.
  const plainHome = path.join(tmpRoot, 'no-marker');
  fs.mkdirSync(plainHome);
  fs.writeFileSync(path.join(plainHome, 'TIPS_SHOWN'), 'claimed\n');
  process.env.KISS_HOME = plainHome;
  tab.resetTipsOnExtensionUpdate();
  assert.ok(
    fs.existsSync(path.join(plainHome, 'TIPS_SHOWN')),
    'reset removed TIPS_SHOWN without an update marker',
  );
  assert.strictEqual(tab.consumeTipsFirstRun(), false);

  // An EMPTY update marker (interrupted install) still elects one
  // resetter instead of crashing or resetting forever.
  const emptyHome = seedUpdatedHome('empty-marker', '');
  process.env.KISS_HOME = emptyHome;
  tab.resetTipsOnExtensionUpdate();
  assert.strictEqual(tab.consumeTipsFirstRun(), true, 'empty-stamp reset lost');
  tab.resetTipsOnExtensionUpdate(); // same stamp: must be a no-op now
  assert.strictEqual(
    tab.consumeTipsFirstRun(),
    false,
    'a repeat reset for the SAME update stamp re-cleared the claim',
  );

  // Claim files from older updates are cleaned up when a new stamp
  // arrives, and the new stamp still resets exactly once.
  const cleanHome = seedUpdatedHome('claim-cleanup', 'stamp-NEW\n');
  fs.writeFileSync(path.join(cleanHome, '.tips-reset-stamp-OLD'), 'x');
  process.env.KISS_HOME = cleanHome;
  tab.resetTipsOnExtensionUpdate();
  assert.ok(
    !fs.existsSync(path.join(cleanHome, '.tips-reset-stamp-OLD')),
    'an old update\u2019s reset claim was not cleaned up',
  );
  assert.strictEqual(tab.consumeTipsFirstRun(), true);
  console.log('  ✓ reset branches: no marker, empty stamp, claim cleanup');
}

async function main() {
  try {
    await updateResetScenarios();
    for (let round = 0; round < ROUNDS; round++) {
      const {r1, r2} = await runRound(round);
      const winners = (r1.won ? 1 : 0) + (r2.won ? 1 : 0);
      assert.strictEqual(
        winners,
        1,
        `round ${round}: ${winners} windows claimed the first-run tips ` +
          `popup (want exactly 1): ${JSON.stringify({r1, r2})}`,
      );
      assert.strictEqual(
        r1.second,
        false,
        `round ${round}: a repeat call re-claimed the tips popup`,
      );
      assert.strictEqual(
        r2.second,
        false,
        `round ${round}: a repeat call re-claimed the tips popup`,
      );
    }
  } finally {
    fs.rmSync(tmpRoot, {recursive: true, force: true});
  }
  console.log(
    `  ✓ ${ROUNDS} simultaneous-activation rounds: one tips winner each`,
  );
  console.log('audit0903_tips_first_run: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
