// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// H-R3: one _version.py parser. UpdateChecker exports readVersionPy and
// SorcarTab.getVersion delegates to it, so both agree on every input.
// H-R5: one sleep() helper, exported from daemonHealth and used by
// daemonRestartVerify and macLaunchd.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => undefined}),
  },
  ProgressLocation: {Notification: 15},
  window: {},
  commands: {executeCommand: () => Promise.resolve()},
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

const OUT = path.join(__dirname, '..', 'out');
const SRC = path.join(__dirname, '..', 'src');

async function main() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rrhi-ver-'));

  // A minimal kiss project with a version file.
  const proj = path.join(tmp, 'proj');
  fs.mkdirSync(path.join(proj, 'src', 'kiss', 'core'), {recursive: true});
  fs.writeFileSync(path.join(proj, 'pyproject.toml'), 'name = "kiss"\n');
  const versionPy = path.join(proj, 'src', 'kiss', 'core', '_version.py');
  fs.writeFileSync(versionPy, '__version__ = "2099.1.2"\n');
  process.env.KISS_PROJECT_PATH = proj;

  const {readVersionPy} = require(path.join(OUT, 'UpdateChecker.js'));
  assert.strictEqual(readVersionPy(versionPy), '2099.1.2');
  assert.strictEqual(
    readVersionPy(path.join(tmp, 'missing.py')),
    null,
    'missing file must read as null',
  );
  console.log('  ok - UpdateChecker.readVersionPy parses _version.py');

  const {getVersion} = require(path.join(OUT, 'SorcarTab.js'));
  assert.strictEqual(
    getVersion(),
    '2099.1.2',
    'SorcarTab.getVersion must agree with readVersionPy',
  );

  fs.writeFileSync(versionPy, "__version__ = '2099.9.9'\n");
  assert.strictEqual(getVersion(), '2099.9.9', 'single quotes must parse');

  fs.writeFileSync(versionPy, 'not_a_version = 1\n');
  assert.strictEqual(
    getVersion(),
    '',
    'unparseable version file must yield the empty string',
  );
  console.log('  ok - SorcarTab.getVersion delegates to the shared parser');

  // The single sleep(): exported from daemonHealth, awaited for real.
  const {sleep} = require(path.join(SRC, 'daemonHealth.js'));
  const t0 = Date.now();
  await sleep(60);
  assert.ok(Date.now() - t0 >= 50, 'sleep must actually wait');
  // Its two consumers still load and run with the shared helper.
  const {verifyDaemonStartup} = require(path.join(SRC, 'daemonRestartVerify.js'));
  assert.strictEqual(typeof verifyDaemonStartup, 'function');
  const macLaunchd = require(path.join(SRC, 'macLaunchd.js'));
  assert.strictEqual(typeof macLaunchd.restartLaunchAgent, 'function');
  console.log('  ok - shared sleep() works and both consumers load');

  fs.rmSync(tmp, {recursive: true, force: true});
  console.log('rr_area_hi_version_and_sleep: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
