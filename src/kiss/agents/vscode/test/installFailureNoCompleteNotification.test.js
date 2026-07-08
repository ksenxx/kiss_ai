// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test: a FAILED slow-path dependency install must never be
// followed by a "KISS Sorcar: Installation complete ..." notification.
//
// Bug locked in: ``ensureDependenciesImpl`` consumed the
// ``~/.kiss/.extension-updated`` marker up front (arming
// ``showRestartNotification``), and after the slow path it computed
//
//     showRestartNotification = showRestartNotification || !!result.success;
//
// so when the install FAILED (e.g. uv missing and ``curl`` unavailable)
// while the marker existed, the user saw the specific error
// notification immediately followed by "Installation complete, but ...
// API key required" — contradictory and hiding the failure.
//
// This test drives the REAL compiled ``out/DependencyInstaller.js``
// (only ``vscode`` and the notification sink are stubbed) in a sandbox:
// fresh HOME, a PATH with no uv/curl/tar, a fake kiss project, and the
// update marker present.  It asserts the curl error IS shown and no
// "Installation complete" notification follows.
//
// Run directly with ``node`` (after ``npm run compile``):
//
//     node src/kiss/agents/vscode/test/installFailureNoCompleteNotification.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');

// --- Sandbox: fresh HOME, empty PATH dir, fake kiss project -------------
const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-instfail-'));
const emptyBin = path.join(tmpHome, 'bin');
fs.mkdirSync(emptyBin, {recursive: true});
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
// The update marker that armed the completion notification.
fs.writeFileSync(path.join(tmpHome, '.kiss', '.extension-updated'), 'x\n');

const fakeProject = path.join(tmpHome, 'kiss_project');
fs.mkdirSync(fakeProject, {recursive: true});
fs.writeFileSync(
  path.join(fakeProject, 'pyproject.toml'),
  '[project]\nname = "kiss-agent-framework"\n',
);

process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_PROJECT_PATH = fakeProject;
// No uv, no curl, no tar anywhere on PATH (the shell for `which` lookups
// is spawned via an absolute path, so an empty PATH dir suffices).
process.env.PATH = emptyBin;

// --- vscode stub ---------------------------------------------------------
const vscodeStub = {
  workspace: {
    isTrusted: true,
    getConfiguration: () => ({get: () => undefined}),
  },
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
  },
  ProgressLocation: {Notification: 15},
  Uri: {
    file: p => ({fsPath: p}),
    joinPath: (base, ...parts) => ({fsPath: path.join(base.fsPath, ...parts)}),
  },
  commands: {executeCommand: () => Promise.resolve()},
  EventEmitter: class {
    constructor() {
      this.event = () => ({dispose: () => {}});
    }
    fire() {}
    dispose() {}
  },
};

const origLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return vscodeStub;
  return origLoad.call(this, request, parent, isMain);
};

// --- Notification sink (replaces the real WebviewNotifications) ---------
const notifications = [];

function stubModule(filePath, exports) {
  const fakeMod = new Module(filePath);
  fakeMod.filename = filePath;
  fakeMod.loaded = true;
  fakeMod.exports = exports;
  require.cache[filePath] = fakeMod;
}

stubModule(path.join(OUT_DIR, 'WebviewNotifications.js'), {
  setWebviewNotificationPoster: () => {},
  resolveWebviewNotificationAction: () => {},
  showInformationNotification: (message, ...actions) => {
    notifications.push({kind: 'info', message, actions});
    return Promise.resolve(undefined);
  },
  showWarningNotification: (message, ...actions) => {
    notifications.push({kind: 'warning', message, actions});
    return Promise.resolve(undefined);
  },
  showErrorNotification: (message, ...actions) => {
    notifications.push({kind: 'error', message, actions});
    return Promise.resolve(undefined);
  },
  withWebviewNotificationProgress: (_opts, task) =>
    Promise.resolve(
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    ),
});

const sourcePath = path.join(OUT_DIR, 'DependencyInstaller.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`npm run compile\` first`,
);
delete require.cache[require.resolve(sourcePath)];
const {ensureDependencies} = require(sourcePath);
assert.strictEqual(typeof ensureDependencies, 'function');

async function runTest() {
  await ensureDependencies();

  const errors = notifications.filter(n => n.kind === 'error');
  assert.ok(
    errors.some(n => /['"]?curl['"]? is required/.test(n.message)),
    'sanity: the specific install-failure error notification must be ' +
      `shown; got: ${JSON.stringify(notifications)}`,
  );

  const completions = notifications.filter(n =>
    n.message.includes('Installation complete'),
  );
  assert.deepStrictEqual(
    completions,
    [],
    'BUG: a FAILED install must not be followed by an "Installation ' +
      `complete" notification; got: ${JSON.stringify(completions)}`,
  );

  console.log('\nAll install-failure notification tests passed');
}

runTest()
  .then(
    () => {
      fs.rmSync(tmpHome, {recursive: true, force: true});
      process.exit(0);
    },
    err => {
      console.error('FAIL:', err);
      fs.rmSync(tmpHome, {recursive: true, force: true});
      process.exit(1);
    },
  )
  .finally(() => {
    Module._load = origLoad;
  });
