// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {
  kissAiRoot,
  findInstallScript,
  bootstrapInstallUrl,
} = require('../src/installerPath');

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
const fakeRoot = path.join(tmpHome, '.kiss', 'kiss_ai');

function clearFakeRoot() {
  fs.rmSync(fakeRoot, {recursive: true, force: true});
}

test('kissAiRoot resolves to ~/.kiss/kiss_ai under the real home directory', () => {
  assert.strictEqual(kissAiRoot(), path.join(os.homedir(), '.kiss', 'kiss_ai'));
});

test('kissAiRoot is an absolute path', () => {
  assert.ok(path.isAbsolute(kissAiRoot()));
});

test('kissAiRoot never resolves to the legacy ~/kiss_ai location', () => {
  assert.notStrictEqual(kissAiRoot(), path.join(os.homedir(), 'kiss_ai'));
  assert.strictEqual(path.basename(path.dirname(kissAiRoot())), '.kiss');
});

test('findInstallScript returns null when the root directory is missing', () => {
  clearFakeRoot();
  assert.strictEqual(findInstallScript(fakeRoot), null);
});

test('findInstallScript returns null when install.sh is absent', () => {
  clearFakeRoot();
  fs.mkdirSync(fakeRoot, {recursive: true});
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
  assert.ok(fs.statSync(found).isFile());
});

test('findInstallScript with no argument probes the real ~/.kiss/kiss_ai root', () => {
  const realCandidate = path.join(kissAiRoot(), 'install.sh');
  const probed = findInstallScript();
  if (probed === null) {
    assert.ok(
      !fs.existsSync(realCandidate),
      'probed null but ~/.kiss/kiss_ai/install.sh exists on disk',
    );
  } else {
    assert.strictEqual(probed, realCandidate);
  }
});

test('findInstallScript ignores process.cwd() — workspace-independent', () => {
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-cwd-'));
  fs.writeFileSync(
    path.join(workspace, 'install.sh'),
    '#!/bin/bash\necho stray\n',
  );
  const origCwd = process.cwd();
  process.chdir(workspace);
  try {
    const probed = findInstallScript();
    assert.notStrictEqual(probed, path.join(workspace, 'install.sh'));
    if (probed !== null) {
      assert.strictEqual(probed, path.join(kissAiRoot(), 'install.sh'));
    }
  } finally {
    process.chdir(origCwd);
    fs.rmSync(workspace, {recursive: true, force: true});
  }
});

test('bootstrapInstallUrl defaults to the public scripts/install.sh raw URL', () => {
  const saved = process.env.KISS_UPDATE_BOOTSTRAP_URL;
  delete process.env.KISS_UPDATE_BOOTSTRAP_URL;
  try {
    assert.strictEqual(
      bootstrapInstallUrl(),
      'https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh',
    );
  } finally {
    if (saved !== undefined) process.env.KISS_UPDATE_BOOTSTRAP_URL = saved;
  }
});

test('bootstrapInstallUrl honours $KISS_UPDATE_BOOTSTRAP_URL', () => {
  const saved = process.env.KISS_UPDATE_BOOTSTRAP_URL;
  process.env.KISS_UPDATE_BOOTSTRAP_URL = 'file:///tmp/fake-install.sh';
  try {
    assert.strictEqual(bootstrapInstallUrl(), 'file:///tmp/fake-install.sh');
  } finally {
    if (saved === undefined) delete process.env.KISS_UPDATE_BOOTSTRAP_URL;
    else process.env.KISS_UPDATE_BOOTSTRAP_URL = saved;
  }
});

fs.rmSync(tmpHome, {recursive: true, force: true});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`\n${f.name}:\n`, f.err);
  }
  process.exit(1);
}
