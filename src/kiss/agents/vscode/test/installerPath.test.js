// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for ``installerPath.js`` — the helper backing the
// Settings → Update button in the KISS Sorcar VS Code extension.
//
// Regression locked in:
//
//   The Update button used to compute ``install.sh`` from the user's
//   workspace / PWD, so anyone whose VS Code workspace wasn't
//   ``~/kiss_ai`` saw::
//
//     Cannot update KISS Sorcar: install.sh not found in <their PWD>.
//
//   ``scripts/install.sh`` always clones the repo to ``~/kiss_ai`` and
//   the updater script lives at its root, so the lookup must be rooted
//   there — independent of the workspace folder.
//
// Runs against the real ``installerPath.js`` and the real filesystem
// (a throwaway temp dir), exercised directly with ``node`` — no VS
// Code extension host required:
//
//     node test/installerPath.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {kissAiRoot, findInstallScript} = require('../src/installerPath');

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  FAIL - ${name}: ${err && err.message}`);
  }
}

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-installerpath-'));
const fakeRoot = path.join(tmpHome, 'kiss_ai');

function clearFakeRoot() {
  fs.rmSync(fakeRoot, {recursive: true, force: true});
}

// ---------------------------------------------------------------------------
// kissAiRoot — production default
// ---------------------------------------------------------------------------

test('kissAiRoot resolves to ~/kiss_ai under the real home directory', () => {
  assert.strictEqual(kissAiRoot(), path.join(os.homedir(), 'kiss_ai'));
});

test('kissAiRoot is an absolute path', () => {
  assert.ok(path.isAbsolute(kissAiRoot()));
});

// ---------------------------------------------------------------------------
// findInstallScript — explicit root override (the test seam)
// ---------------------------------------------------------------------------

test('findInstallScript returns null when the root directory is missing', () => {
  clearFakeRoot();
  assert.strictEqual(findInstallScript(fakeRoot), null);
});

test('findInstallScript returns null when install.sh is absent', () => {
  clearFakeRoot();
  fs.mkdirSync(fakeRoot, {recursive: true});
  // Decoy: a different file at the same level must not satisfy the
  // probe.  The Update button must look specifically for install.sh.
  fs.writeFileSync(path.join(fakeRoot, 'README.md'), '# decoy\n');
  assert.strictEqual(findInstallScript(fakeRoot), null);
});

test('findInstallScript returns the absolute install.sh path when present', () => {
  clearFakeRoot();
  fs.mkdirSync(fakeRoot, {recursive: true});
  const script = path.join(fakeRoot, 'install.sh');
  fs.writeFileSync(script, '#!/bin/bash\necho hi\n');
  const found = findInstallScript(fakeRoot);
  assert.strictEqual(found, script);
  assert.ok(path.isAbsolute(found));
  // The string the extension passes to ``bash '<path>'`` must point at
  // a real, readable file.
  assert.ok(fs.statSync(found).isFile());
});

test('findInstallScript with no argument probes the real ~/kiss_ai root', () => {
  // We don't assert presence (the dev box may or may not have the
  // checkout) — we only assert the *path probed* is the production
  // default, not the caller's CWD.  This pins the bug fix: the lookup
  // is no longer rooted at process.cwd().
  const realCandidate = path.join(kissAiRoot(), 'install.sh');
  const probed = findInstallScript();
  if (probed === null) {
    assert.ok(
      !fs.existsSync(realCandidate),
      'probed null but ~/kiss_ai/install.sh exists on disk',
    );
  } else {
    assert.strictEqual(probed, realCandidate);
  }
});

test('findInstallScript ignores process.cwd() — workspace-independent', () => {
  // Reproduces the original failure mode: the user's workspace
  // contains a stray install.sh, but the production lookup must
  // still target ~/kiss_ai, not the workspace.
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-cwd-'));
  fs.writeFileSync(
    path.join(workspace, 'install.sh'),
    '#!/bin/bash\necho stray\n',
  );
  const origCwd = process.cwd();
  process.chdir(workspace);
  try {
    const probed = findInstallScript();
    // Whatever the result is, it must NOT be the stray script in cwd.
    assert.notStrictEqual(probed, path.join(workspace, 'install.sh'));
    if (probed !== null) {
      assert.strictEqual(probed, path.join(kissAiRoot(), 'install.sh'));
    }
  } finally {
    process.chdir(origCwd);
    fs.rmSync(workspace, {recursive: true, force: true});
  }
});

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

fs.rmSync(tmpHome, {recursive: true, force: true});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`\n${f.name}:\n`, f.err);
  }
  process.exit(1);
}
