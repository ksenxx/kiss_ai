// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// While ./install.sh runs it publishes its current step to
// $KISS_HOME/.install-progress ("<pid>\n<step>") and removes the file on
// exit.  The extension must mirror that file as a native, non-blocking
// VS Code progress notification whose message follows the step and which
// closes when the install ends.  These tests drive the real compiled
// extension through the ui-antipattern host with a temp $KISS_HOME and a
// recording `window.withProgress`; the only stubbed pieces are the VS Code
// API itself and the sidebar view the host always replaces.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');
const {OUT_DIR, loadExtension, waitFor} = require('./_ui_antipattern_host');

const tmpHome = fs.mkdtempSync(
  path.join(os.tmpdir(), 'kiss-install-progress-'),
);
process.env.KISS_HOME = tmpHome;
const progressPath = path.join(tmpHome, '.install-progress');

/** Attach a recording `withProgress` to the stub; `delayTask` defers the
 * task callback like the real extension host does. */
function recordProgress(vscodeStub, delayTask = 0) {
  const toasts = [];
  vscodeStub.window.withProgress = (options, task) => {
    const toast = {options, reports: [], done: false};
    toasts.push(toast);
    const progress = {report: v => toast.reports.push(v.message)};
    const token = {isCancellationRequested: false};
    const start = () =>
      Promise.resolve(task(progress, token)).then(() => {
        toast.done = true;
      });
    return delayTask
      ? new Promise(r => setTimeout(() => r(start()), delayTask))
      : start();
  };
  return toasts;
}

function publish(message, pid = process.pid) {
  fs.writeFileSync(progressPath, `${pid}\n${message}\n`);
}

/** A pid no process has: spawn a short-lived child and wait for it. */
function deadPid() {
  const child = spawnSync('true');
  return child.pid;
}

const {PRODUCT_NAME} = require(path.join(OUT_DIR, 'brand.js'));

// One host for every test: installProgress.js binds the `vscode` stub of
// the load that first required it, so each test re-attaches its recorder
// to that same stub instead of loading the extension again.
const h = loadExtension();
const {InstallProgressWatcher, readInstallProgress, isInstallerAlive} = require(
  path.join(OUT_DIR, 'installProgress.js'),
);

async function testActivationMirrorsInstallProgress() {
  const toasts = recordProgress(h.vscodeStub);
  const ctx = h.makeContext();
  try {
    await h.extension.activate(ctx);
    assert.strictEqual(toasts.length, 0, 'no toast without a progress file');

    publish('[1/5] Checking git...');
    await waitFor(() => toasts.length === 1, 'toast opened for step 1', 200);
    assert.strictEqual(
      toasts[0].options.location,
      15,
      'ProgressLocation.Notification',
    );
    assert.strictEqual(toasts[0].options.title, `Installing ${PRODUCT_NAME}`);
    assert.strictEqual(toasts[0].options.cancellable, false);
    assert.deepStrictEqual(toasts[0].reports, ['[1/5] Checking git...']);

    publish('[4/5] Building VS Code extension...');
    await waitFor(
      () => toasts[0].reports.length === 2,
      'toast message follows the step',
      200,
    );
    assert.strictEqual(
      toasts[0].reports[1],
      '[4/5] Building VS Code extension...',
    );
    assert.strictEqual(
      toasts[0].done,
      false,
      'toast stays open while install.sh runs',
    );

    fs.rmSync(progressPath);
    await waitFor(
      () => toasts[0].done,
      'toast closes when install.sh exits',
      200,
    );
    assert.strictEqual(toasts.length, 1, 'one toast per install');
  } finally {
    h.disposeContext(ctx);
  }
}

async function testFileLeftByDeadInstallerIsDroppedWithoutToast() {
  const toasts = recordProgress(h.vscodeStub);
  publish('[2/5] Checking Node.js...', deadPid());
  const ctx = h.makeContext();
  try {
    await h.extension.activate(ctx);
    await new Promise(r => setTimeout(r, 1200)); // past one poll
    assert.strictEqual(toasts.length, 0, 'no toast for a dead installer');
    // The stale file is left alone: a new install reuses the path, and a
    // delayed removal could take that install's first step with it.
    assert.ok(
      fs.existsSync(progressPath),
      'stale file is ignored, not deleted',
    );
    publish('[1/5] Checking git...');
    await waitFor(() => toasts.length === 1, 'next install opens a toast', 200);
    fs.rmSync(progressPath);
    await waitFor(() => toasts[0].done, 'toast closes', 200);
  } finally {
    h.disposeContext(ctx);
  }
}

async function testToastOpensForFilePresentAtActivation() {
  // The window reloads in step [5/5] while install.sh is still running:
  // the fresh extension host must pick the toast up straight away.
  const toasts = recordProgress(h.vscodeStub);
  publish('[5/5] Installing VS Code extension...');
  const ctx = h.makeContext();
  try {
    await h.extension.activate(ctx);
    assert.strictEqual(toasts.length, 1, 'toast opened on activation');
    assert.deepStrictEqual(toasts[0].reports, [
      '[5/5] Installing VS Code extension...',
    ]);
    h.disposeContext(ctx);
    await waitFor(() => toasts[0].done, 'deactivation closes the toast', 50);
  } finally {
    fs.rmSync(progressPath, {force: true});
  }
}

async function testWatcherEdgeCases() {
  // Malformed files publish nothing.
  fs.writeFileSync(progressPath, '');
  assert.strictEqual(
    readInstallProgress(progressPath),
    undefined,
    'empty file',
  );
  fs.writeFileSync(progressPath, 'abc\n[1/5] x\n');
  assert.strictEqual(
    readInstallProgress(progressPath),
    undefined,
    'pid not a number',
  );
  fs.writeFileSync(progressPath, `${process.pid}\n`);
  assert.strictEqual(
    readInstallProgress(progressPath),
    undefined,
    'no step yet',
  );
  fs.rmSync(progressPath);
  assert.strictEqual(
    readInstallProgress(progressPath),
    undefined,
    'missing file',
  );

  assert.strictEqual(isInstallerAlive(process.pid), true, 'own pid is alive');
  if (process.platform !== 'win32') {
    // On Windows the MSYS `$$` is not a Windows pid, so the liveness
    // check is skipped there by design (isInstallerAlive is always true).
    assert.strictEqual(isInstallerAlive(deadPid()), false, 'exited pid is dead');
  }
  if (process.getuid && process.getuid() !== 0) {
    // kill(1, 0) fails with EPERM for a non-root user: the process exists.
    assert.strictEqual(isInstallerAlive(1), true, 'EPERM counts as alive');
  }

  // A malformed file never opens a toast; the same step twice reports once.
  const toasts = recordProgress(h.vscodeStub);
  fs.writeFileSync(progressPath, 'garbage');
  const watcher = new InstallProgressWatcher(progressPath, 10);
  await new Promise(r => setTimeout(r, 40));
  assert.strictEqual(toasts.length, 0, 'garbage file opens no toast');
  publish('[3/5] Checking VS Code CLI...');
  await waitFor(() => toasts.length === 1, 'toast opened', 100);
  publish('[3/5] Checking VS Code CLI...');
  await new Promise(r => setTimeout(r, 40));
  assert.deepStrictEqual(toasts[0].reports, ['[3/5] Checking VS Code CLI...']);
  // The installer dies without cleaning up: the toast closes.
  publish('[3/5] Checking VS Code CLI...', deadPid());
  await waitFor(() => toasts[0].done, 'dead installer', 100);
  watcher.dispose();
  fs.rmSync(progressPath);
}

async function testCloseBeforeReporterArrivesLeavesNoSpinner() {
  // The real extension host hands the progress reporter to the task
  // asynchronously.  If install.sh finishes in that window the task must
  // still resolve, or a spinner would stay open forever.
  const toasts = recordProgress(h.vscodeStub, 60);
  publish('[5/5] Installing VS Code extension...');
  const watcher = new InstallProgressWatcher(progressPath, 10);
  assert.strictEqual(toasts.length, 1, 'toast requested');
  publish('[5/5] Copying model catalog...');
  fs.rmSync(progressPath);
  await waitFor(
    () => toasts[0].done,
    'task resolves once the reporter arrives',
    100,
  );
  assert.deepStrictEqual(toasts[0].reports, [], 'nothing reported after close');
  watcher.dispose();
}

(async () => {
  await testActivationMirrorsInstallProgress();
  await testFileLeftByDeadInstallerIsDroppedWithoutToast();
  await testToastOpensForFilePresentAtActivation();
  await testWatcherEdgeCases();
  await testCloseBeforeReporterArrivesLeavesNoSpinner();
  fs.rmSync(tmpHome, {recursive: true, force: true});
  console.log('install_progress_notification: all tests passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
