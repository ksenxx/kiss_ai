// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A6/A12 tips window (audit-extension H2): the tips popup is a
// first-run affordance, not a per-update announcement.
//  - out/SorcarTab.js no longer exports resetTipsOnExtensionUpdate, and
//    an `.extension-updated` marker does NOT re-arm the popup;
//  - consumeTipsFirstRun() hands the popup to exactly one caller per
//    $KISS_HOME (real 'wx' claim on disk) and stays quiet afterwards;
//  - recordTipsOptOut() — the host side of the webview's
//    {type:'tipsOptOut'} message — persists TIPS_DISABLED, and the popup
//    never opens again, even on a first run in that home.

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
  workspace: {getConfiguration: () => ({get: () => undefined})},
};
const origLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return vscodeStub;
  return origLoad.call(this, request, parent, isMain);
};

function freshHome(tag) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), `kiss-uap-tips-${tag}-`));
  process.env.KISS_HOME = home;
  return home;
}

const tabPath = path.join(OUT_DIR, 'SorcarTab.js');
assert.ok(fs.existsSync(tabPath), 'run `npm run compile` first');
const tab = require(tabPath);

// --- the update-reset entry point is gone --------------------------------
assert.strictEqual(
  tab.resetTipsOnExtensionUpdate,
  undefined,
  'resetTipsOnExtensionUpdate must no longer exist: tips are first-run only',
);
assert.strictEqual(typeof tab.consumeTipsFirstRun, 'function');
assert.strictEqual(typeof tab.recordTipsOptOut, 'function');
assert.strictEqual(typeof tab.tipsDisabled, 'function');

// --- first run: exactly one claim, then silence --------------------------
const home1 = freshHome('first');
assert.strictEqual(tab.tipsDisabled(), false);
assert.strictEqual(tab.consumeTipsFirstRun(), true, 'first run opens the tips');
assert.ok(fs.existsSync(path.join(home1, 'TIPS_SHOWN')), 'claim written');
assert.strictEqual(tab.consumeTipsFirstRun(), false, 'second run stays quiet');

// --- an extension update does not re-arm the popup ------------------------
fs.writeFileSync(
  path.join(home1, '.extension-updated'),
  new Date().toISOString() + '\n',
);
assert.strictEqual(
  tab.consumeTipsFirstRun(),
  false,
  'an .extension-updated marker must not reopen the tips window',
);
assert.ok(
  fs.existsSync(path.join(home1, 'TIPS_SHOWN')),
  'the first-run claim survives an update',
);
assert.deepStrictEqual(
  fs.readdirSync(home1).filter(n => n.startsWith('.tips-reset-')),
  [],
  'no per-update election files are created any more',
);

// --- opt-out after the first run --------------------------------------------
tab.recordTipsOptOut();
assert.ok(fs.existsSync(path.join(home1, 'TIPS_DISABLED')), 'opt-out persisted');
assert.strictEqual(tab.tipsDisabled(), true);
assert.strictEqual(tab.consumeTipsFirstRun(), false);
tab.recordTipsOptOut(); // idempotent
assert.strictEqual(tab.tipsDisabled(), true);

// --- opt-out recorded BEFORE any first run wins over the first run ----------
const home2 = freshHome('optout');
tab.recordTipsOptOut();
assert.strictEqual(tab.tipsDisabled(), true);
assert.strictEqual(
  tab.consumeTipsFirstRun(),
  false,
  'a persisted opt-out keeps the popup closed even on a first run',
);
assert.ok(
  !fs.existsSync(path.join(home2, 'TIPS_SHOWN')),
  'no claim is written while opted out',
);

// --- an unwritable home: the opt-out is ignored, nothing throws ------------
process.env.KISS_HOME = path.join(home2, 'TIPS_DISABLED', 'not-a-dir');
assert.doesNotThrow(() => tab.recordTipsOptOut());
assert.strictEqual(tab.tipsDisabled(), false);
assert.strictEqual(
  tab.consumeTipsFirstRun(),
  false,
  'an unwritable home never opens the popup',
);

fs.rmSync(home1, {recursive: true, force: true});
fs.rmSync(home2, {recursive: true, force: true});
console.log('\nAll ui_antipattern_tips_reset tests passed');
