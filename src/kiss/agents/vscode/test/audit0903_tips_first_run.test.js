// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-03 (vscode-main partition): claimTipsPopup TOCTOU.
//
// The tips marker was claimed with `existsSync` followed by
// `writeFileSync`: two VS Code windows activating at the same time (two
// extension-host processes) could both pass the existence check before
// either wrote, so BOTH returned true and the tips popup opened
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
// claimTipsPopup and reports the result on stdout.
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
const {claimTipsPopup} = require(${JSON.stringify(OUT_SORCAR_TAB)});
const [readyFile, goFile] = process.argv.slice(2);
fs.writeFileSync(readyFile, 'ready');
// Busy-wait (no sleep): both children see the go-file within
// microseconds of each other, tighter than the check-then-write window.
for (;;) {
  if (fs.existsSync(goFile)) break;
}
const won = claimTipsPopup();
// A second call in the same process must always lose: the marker is
// there now (whoever wrote it).
const second = claimTipsPopup();
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

// ─── An update re-arms the popup exactly once ───
//
// The claim is per version (`TIPS_SHOWN-<version>`,
// tipsClaimPerVersion.test.js): a home that showed the tips for an
// earlier version shows them again once for the running one.  Two REAL
// sequential extension-host processes against one updated home: the
// first claims the popup, the second must not.

const updateChildScript = path.join(tmpRoot, 'update-child.js');
fs.writeFileSync(
  updateChildScript,
  `
'use strict';
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
const won = tab.claimTipsPopup();
process.stdout.write(JSON.stringify({won, version: tab.getVersion()}));
`,
);

function runUpdateChild(kissHome) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [updateChildScript], {
      env: Object.assign({}, process.env, {KISS_HOME: kissHome}),
    });
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

async function updateScenario() {
  const home = path.join(tmpRoot, 'updated-home');
  fs.mkdirSync(home, {recursive: true});
  // A previous version already showed the tips once (the marker of a
  // version that is not the running one, plus the legacy unversioned
  // marker of pre-2026.10 installs)...
  fs.writeFileSync(path.join(home, 'TIPS_SHOWN-0.0.1'), 'old-claim\n');
  fs.writeFileSync(path.join(home, 'TIPS_SHOWN'), 'legacy-claim\n');
  // ...and the running version is new: the first window after the
  // update opens the tips, the second stays quiet.
  const a = await runUpdateChild(home);
  const b = await runUpdateChild(home);
  assert.deepStrictEqual(
    [a.won, b.won],
    [true, false],
    `post-update: exactly the first window opens the tips: ${JSON.stringify({a, b})}`,
  );
  const safe = (a.version || 'unknown').replace(/[^A-Za-z0-9.]/g, '_');
  assert.deepStrictEqual(
    fs.readdirSync(home).filter(n => n.startsWith('TIPS_SHOWN')).sort(),
    ['TIPS_SHOWN-0.0.1', 'TIPS_SHOWN-' + safe],
    'the legacy marker is retired; versioned claims are kept',
  );
  console.log('  ✓ an update reopens the tips popup exactly once per home');
}

async function main() {
  try {
    await updateScenario();
    for (let round = 0; round < ROUNDS; round++) {
      const {r1, r2} = await runRound(round);
      const winners = (r1.won ? 1 : 0) + (r2.won ? 1 : 0);
      assert.strictEqual(
        winners,
        1,
        `round ${round}: ${winners} windows claimed the tips ` +
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
