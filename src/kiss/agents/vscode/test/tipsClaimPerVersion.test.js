// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The tips popup opens once per version of KISS in every $KISS_HOME:
// on the first run and again after each update.
//  - claimTipsPopup() claims `$KISS_HOME/TIPS_SHOWN-<version>` on disk
//    ('wx'), hands the popup to exactly one caller for that version
//    and stays quiet afterwards;
//  - bumping the version (an update) re-arms the popup once; the
//    legacy unversioned `TIPS_SHOWN` of pre-2026.10 installs is retired
//    while versioned claims stay, so two installations of different
//    versions sharing one home never erase each other's claim;
//  - recordTipsOptOut() persists TIPS_DISABLED and the popup never
//    opens again, not even for a new version.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT_DIR = path.join(__dirname, '..', 'out');

const vscodeStub = {
  Uri: {
    file: p => ({fsPath: p}),
    joinPath: (base, ...parts) => ({fsPath: path.join(base.fsPath, ...parts)}),
  },
  workspace: {
    isTrusted: true,
    getConfiguration: () => ({get: () => undefined}),
  },
};
const origLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return vscodeStub;
  return origLoad.call(this, request, parent, isMain);
};

function freshHome(tag) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), `kiss-tips-ver-${tag}-`));
  process.env.KISS_HOME = home;
  return home;
}

// A minimal KISS checkout whose version the test controls: getVersion()
// reads `src/kiss/core/_version.py` of $KISS_PROJECT_PATH.
const project = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-tips-ver-proj-'));
fs.writeFileSync(path.join(project, 'pyproject.toml'), 'name = "kiss"\n');
fs.mkdirSync(path.join(project, 'src', 'kiss', 'core'), {recursive: true});
function setVersion(version) {
  fs.writeFileSync(
    path.join(project, 'src', 'kiss', 'core', '_version.py'),
    `__version__ = "${version}"\n`,
  );
}
process.env.KISS_PROJECT_PATH = project;

const tabPath = path.join(OUT_DIR, 'SorcarTab.js');
assert.ok(fs.existsSync(tabPath), 'run `npm run compile` first');
const tab = require(tabPath);

assert.strictEqual(typeof tab.claimTipsPopup, 'function');
assert.strictEqual(typeof tab.recordTipsOptOut, 'function');
assert.strictEqual(typeof tab.tipsDisabled, 'function');

const markers = home =>
  fs.readdirSync(home).filter(n => n.startsWith('TIPS_SHOWN')).sort();

// --- first run: exactly one claim for the running version, then silence ---
setVersion('2026.9.25');
const home1 = freshHome('first');
assert.strictEqual(tab.tipsDisabled(), false);
assert.strictEqual(tab.claimTipsPopup(), true, 'first run opens the tips');
assert.deepStrictEqual(markers(home1), ['TIPS_SHOWN-2026.9.25']);
assert.strictEqual(tab.claimTipsPopup(), false, 'second run stays quiet');
assert.strictEqual(tab.claimTipsPopup(), false, 'and so does the third');

// --- an update re-arms the popup once --------------------------------------
setVersion('2026.10.1');
assert.strictEqual(tab.claimTipsPopup(), true, 'an update reopens the tips');
assert.deepStrictEqual(
  markers(home1),
  ['TIPS_SHOWN-2026.10.1', 'TIPS_SHOWN-2026.9.25'],
  'the claim of the earlier version stays',
);
assert.strictEqual(tab.claimTipsPopup(), false, 'once per version only');

// --- two versions sharing the home (mixed installations) -------------------
// The older installation renders again: its own claim is still there,
// so it stays quiet, and the newer claim is untouched.
setVersion('2026.9.25');
assert.strictEqual(
  tab.claimTipsPopup(),
  false,
  'the earlier version must not reopen the tips after a newer one claimed',
);
setVersion('2026.10.1');
assert.strictEqual(tab.claimTipsPopup(), false, 'nor the newer one again');
assert.deepStrictEqual(markers(home1), [
  'TIPS_SHOWN-2026.10.1',
  'TIPS_SHOWN-2026.9.25',
]);

// --- upgrading from the unversioned era: the legacy marker is retired -----
const home2 = freshHome('legacy');
fs.writeFileSync(path.join(home2, 'TIPS_SHOWN'), 'legacy\n');
assert.strictEqual(
  tab.claimTipsPopup(),
  true,
  'a home with only the legacy TIPS_SHOWN sees the tips after updating',
);
assert.deepStrictEqual(markers(home2), ['TIPS_SHOWN-2026.10.1']);
assert.strictEqual(tab.claimTipsPopup(), false);
assert.deepStrictEqual(markers(home2), ['TIPS_SHOWN-2026.10.1']);

// --- a version with odd characters gets a safe file name -------------------
setVersion('1.0.0-rc/1');
const home3 = freshHome('odd');
assert.strictEqual(tab.claimTipsPopup(), true);
assert.deepStrictEqual(markers(home3), ['TIPS_SHOWN-1.0.0_rc_1']);

// --- no detectable version: still once per home ---------------------------
fs.rmSync(path.join(project, 'src', 'kiss', 'core', '_version.py'));
const home4 = freshHome('nover');
assert.strictEqual(tab.claimTipsPopup(), true);
assert.deepStrictEqual(markers(home4), ['TIPS_SHOWN-unknown']);
assert.strictEqual(tab.claimTipsPopup(), false);
setVersion('2026.10.1');

// --- opt-out after a run: nothing reopens, not even a new version ---------
process.env.KISS_HOME = home1;
tab.recordTipsOptOut();
assert.ok(fs.existsSync(path.join(home1, 'TIPS_DISABLED')), 'opt-out persisted');
assert.strictEqual(tab.tipsDisabled(), true);
setVersion('2026.11.1');
assert.strictEqual(
  tab.claimTipsPopup(),
  false,
  'a persisted opt-out keeps the popup closed across updates',
);
assert.deepStrictEqual(
  markers(home1),
  ['TIPS_SHOWN-2026.10.1', 'TIPS_SHOWN-2026.9.25'],
  'no claim is written while opted out',
);
tab.recordTipsOptOut(); // idempotent
assert.strictEqual(tab.tipsDisabled(), true);

// --- opt-out recorded BEFORE any run wins over the first run ---------------
const home5 = freshHome('optout');
tab.recordTipsOptOut();
assert.strictEqual(tab.claimTipsPopup(), false);
assert.deepStrictEqual(markers(home5), []);

// --- an unwritable home: the opt-out is ignored, nothing throws ------------
process.env.KISS_HOME = path.join(home5, 'TIPS_DISABLED', 'not-a-dir');
assert.doesNotThrow(() => tab.recordTipsOptOut());
assert.strictEqual(tab.tipsDisabled(), false);
assert.strictEqual(
  tab.claimTipsPopup(),
  false,
  'an unwritable home never opens the popup',
);

for (const home of [home1, home2, home3, home4, home5]) {
  fs.rmSync(home, {recursive: true, force: true});
}
fs.rmSync(project, {recursive: true, force: true});
console.log('\nAll tipsClaimPerVersion tests passed');
