// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A3 setup-failure toasts and A9 cancellable setup (audit-extension H4,
// H5), end to end against the REAL out/DependencyInstaller.js and
// out/WebviewNotifications.js with a fake `uv` on disk:
//  - a failing `uv sync` rejects ensureDependencies() with a ONE-LINE
//    cause (the first error line of its output, <= 200 chars) while the
//    full output lands in $KISS_HOME/install.log;
//  - the relayed error toast is sticky; info toasts without actions
//    are not;
//  - the first-run progress toast carries a 'Cancel' action; clicking
//    it cancels the token, kills the running `uv sync`, and ends in an
//    info toast that names "KISS: Retry Setup" with a 'Retry setup'
//    action that runs the command;
//  - Cancel pressed during finalization ("Checking cloudflared...")
//    stops the setup before the daemon restart and the credential
//    prompts, with no "Installation complete" (R3-2);
//  - summarizeFailure() picks the meaningful line and truncates.
// Then, against out/extension.js: the failure toast reads
// 'Setup failed. <cause>' with 'Open log' (opens install.log in an
// editor) and 'Retry' (re-runs the setup); kissSorcar.retrySetup does
// the same.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

if (process.platform === 'win32') {
  console.log(
    'ui_antipattern_setup_errors: POSIX-only (fake uv is a shell script)',
  );
  process.exit(0);
}

const OUT_DIR = path.join(__dirname, '..', 'out');
const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-uap-setup-'));
const kissHome = path.join(tmpHome, '.kiss');
const fakeBin = path.join(tmpHome, '.local', 'bin');
fs.mkdirSync(kissHome, {recursive: true});
fs.mkdirSync(fakeBin, {recursive: true});

const fakeUv = path.join(fakeBin, 'uv');
fs.writeFileSync(
  fakeUv,
  [
    '#!/bin/sh',
    'if [ "$FAKE_UV_MODE" = "hang" ]; then sleep 30; exit 0; fi',
    // "ok": every uv command succeeds so the setup reaches finalization.
    'if [ "$FAKE_UV_MODE" = "ok" ]; then',
    '  case "$*" in *"python --version"*) echo "Python 3.14.0";; esac',
    '  exit 0',
    'fi',
    'echo "Resolved 12 packages in 0.5s"',
    'echo "warning: a deprecation notice" >&2',
    'echo "error: No solution found when resolving dependencies" >&2',
    'echo "  because package-x depends on package-y>=9 and only 1.0 is available" >&2',
    'exit 1',
  ].join('\n') + '\n',
  {mode: 0o755},
);
// A `code` CLI so the setup does not try to install one.
fs.writeFileSync(path.join(fakeBin, 'code'), '#!/bin/sh\nexit 0\n', {
  mode: 0o755,
});

const fakeProject = path.join(tmpHome, 'kiss_project');
fs.mkdirSync(fakeProject, {recursive: true});
fs.writeFileSync(
  path.join(fakeProject, 'pyproject.toml'),
  '[project]\nname = "kiss-agent-framework"\n',
);

process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_HOME = kissHome;
process.env.KISS_PROJECT_PATH = fakeProject;
process.env.PATH = `${fakeBin}${path.delimiter}${process.env.PATH || ''}`;

const executedCommands = [];
const vscodeStub = {
  workspace: {
    isTrusted: true,
    getConfiguration: () => ({get: () => undefined}),
  },
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {
          isCancellationRequested: false,
          onCancellationRequested: () => ({dispose: () => {}}),
        },
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
  commands: {
    executeCommand: (cmd, ...args) => {
      executedCommands.push({cmd, args});
      return Promise.resolve();
    },
  },
  CancellationTokenSource: class {
    constructor() {
      this._listeners = [];
      const self = this;
      this.token = {
        get isCancellationRequested() {
          return self._cancelled === true;
        },
        onCancellationRequested: cb => {
          self._listeners.push(cb);
          return {dispose: () => {}};
        },
      };
    }
    cancel() {
      if (this._cancelled) return;
      this._cancelled = true;
      for (const cb of this._listeners.slice()) cb();
    }
    dispose() {}
  },
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

const notif = require(path.join(OUT_DIR, 'WebviewNotifications.js'));
const posted = [];
// When set, the poster presses Cancel the moment the progress toast shows
// this step message — the user cancelling while that step is announced.
let cancelOnProgressMessage = null;
notif.setWebviewNotificationPoster(msg => {
  posted.push(msg);
  if (
    cancelOnProgressMessage &&
    msg.progress === true &&
    msg.progressMessage === cancelOnProgressMessage
  ) {
    cancelOnProgressMessage = null;
    notif.resolveWebviewNotificationAction(msg.id, 'Cancel');
  }
});

const di = require(path.join(OUT_DIR, 'DependencyInstaller.js'));
const logPath = path.join(kissHome, 'install.log');

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitFor(predicate, message, tries = 300) {
  for (let i = 0; i < tries; i++) {
    if (predicate()) return;
    await sleep(20);
  }
  throw new Error(message || 'waitFor timed out');
}

function summarizeFailureTests() {
  assert.strictEqual(
    di.summarizeFailure('Resolved 3 packages\nwarning: x\nerror: boom\nmore'),
    'error: boom',
    'the first error line wins over earlier noise',
  );
  assert.strictEqual(
    di.summarizeFailure('\n  just one line  \n'),
    'just one line',
    'falls back to the first non-empty line',
  );
  assert.strictEqual(di.summarizeFailure(''), 'no output');
  const long = 'error: ' + 'x'.repeat(400);
  const cause = di.summarizeFailure(long);
  assert.strictEqual(cause.length, 200, 'capped at 200 characters');
  assert.ok(cause.endsWith('…'));
  console.log('summarizeFailure tests passed');
}

async function failureTest() {
  delete process.env.FAKE_UV_MODE;
  let rejected;
  await di.ensureDependencies().catch(err => {
    rejected = err;
  });
  assert.ok(rejected instanceof Error, 'a failing uv sync rejects setup');
  assert.strictEqual(
    rejected.message,
    `${fakeUv} sync failed (exit code 1): error: No solution found when resolving dependencies`,
    'the message is the command plus its one-line cause',
  );
  const log = fs.readFileSync(logPath, 'utf-8');
  assert.ok(
    log.includes('because package-x depends on package-y>=9'),
    'the full output is in install.log',
  );
  assert.ok(
    !posted.some(
      m =>
        m.type === 'notification' &&
        m.close !== true &&
        (m.message || '').includes('Installation complete'),
    ),
    'a failed install never reports completion',
  );
  console.log('one-line cause / full log tests passed');
}

function stickyTests() {
  posted.length = 0;
  notif.showErrorNotification('setup broke');
  notif.showInformationNotification('all good');
  const err = posted.find(m => m.message === 'setup broke');
  const info = posted.find(m => m.message === 'all good');
  assert.strictEqual(err.severity, 'error');
  assert.strictEqual(err.sticky, true, 'relayed errors never auto-dismiss');
  assert.strictEqual(
    info.sticky,
    false,
    'plain info toasts still auto-dismiss',
  );
  console.log('sticky error toast tests passed');
}

async function cancelTest() {
  process.env.FAKE_UV_MODE = 'hang';
  posted.length = 0;
  const started = Date.now();
  const done = di.ensureDependencies();
  let progressToast;
  await waitFor(() => {
    progressToast = posted.find(
      m =>
        m.type === 'notification' &&
        m.progress === true &&
        (m.message || '').includes('Setting up'),
    );
    return !!progressToast;
  }, 'the first-run progress toast must appear');
  assert.deepStrictEqual(
    progressToast.actions,
    ['Cancel'],
    'the progress toast offers Cancel',
  );
  // Every progress re-post keeps the Cancel action.
  await waitFor(
    () =>
      posted.some(
        m =>
          m.id === progressToast.id &&
          (m.progressMessage || '').includes('Python environment'),
      ),
    'uv sync step reported',
  );
  for (const m of posted.filter(
    m => m.id === progressToast.id && m.close !== true,
  )) {
    assert.deepStrictEqual(m.actions, ['Cancel']);
  }

  // Click Cancel while `uv sync` (sleep 30) is running.
  notif.resolveWebviewNotificationAction(progressToast.id, 'Cancel');
  await done; // resolves — a cancel is not a failure
  assert.ok(
    Date.now() - started < 20_000,
    'cancelling must kill the running command instead of waiting for it',
  );
  assert.ok(
    posted.some(m => m.id === progressToast.id && m.close === true),
    'the progress toast is closed',
  );
  const cancelled = posted.find(
    m => m.type === 'notification' && (m.message || '').includes('cancelled'),
  );
  assert.ok(cancelled, 'a "setup was cancelled" toast is shown');
  assert.strictEqual(cancelled.severity, 'info');
  assert.ok(
    cancelled.message.includes('"KISS: Retry Setup"'),
    'the toast names the command that restarts setup',
  );
  assert.deepStrictEqual(cancelled.actions, ['Retry setup']);
  notif.resolveWebviewNotificationAction(cancelled.id, 'Retry setup');
  await waitFor(
    () => executedCommands.some(c => c.cmd === 'kissSorcar.retrySetup'),
    "'Retry setup' must run kissSorcar.retrySetup",
  );
  assert.ok(
    fs.readFileSync(logPath, 'utf-8').includes('Setup cancelled by the user'),
    'the cancel is logged',
  );
  console.log('cancellable setup tests passed');
}

// R3-2: Cancel keeps working after `uv sync`: pressed while the toast
// says "Checking cloudflared...", the setup stops right there — no
// cloudflared install, no kiss-web restart, no API-key or remote-password
// prompt, no "Installation complete" — and shows the cancelled toast.
async function cancelDuringFinalizationTest() {
  process.env.FAKE_UV_MODE = 'ok';
  posted.length = 0;
  executedCommands.length = 0;
  const logBefore = fs.existsSync(logPath)
    ? fs.readFileSync(logPath, 'utf-8').length
    : 0;
  cancelOnProgressMessage = 'Checking cloudflared...';
  await di.ensureDependencies(); // resolves — a cancel is not a failure
  assert.strictEqual(cancelOnProgressMessage, null, 'Cancel was pressed');
  const steps = posted
    .filter(m => m.progress === true && m.progressMessage)
    .map(m => m.progressMessage);
  assert.ok(steps.includes('Finalizing setup...'), 'finalization started');
  assert.strictEqual(
    steps[steps.length - 1],
    'Checking cloudflared...',
    `no step runs after the cancelled one, got: ${steps.join(' | ')}`,
  );
  for (const later of [
    'Restarting kiss-web daemon...',
    'Updating shell PATH...',
    'Checking API keys...',
    'Checking remote password...',
  ]) {
    assert.ok(!steps.includes(later), `${later} must not run after Cancel`);
  }
  const logAfter = fs.readFileSync(logPath, 'utf-8').slice(logBefore);
  // restartKissWebDaemon's first log line in this fake project would be
  // "kiss-web binary not found at ... — skipping daemon setup".
  assert.ok(
    !logAfter.includes('kiss-web binary not found'),
    'the daemon restart never started',
  );
  assert.ok(
    logAfter.includes('Setup cancelled by the user'),
    'the cancel is logged',
  );
  assert.ok(
    !posted.some(m => (m.message || '').includes('Installation complete')),
    'a cancelled setup never reports completion',
  );
  assert.ok(
    !posted.some(m => (m.message || '').includes('API key')),
    'no API-key prompt after Cancel',
  );
  assert.deepStrictEqual(
    executedCommands,
    [],
    'no command (API key entry, settings) is run after Cancel',
  );
  const cancelled = posted.find(
    m => m.type === 'notification' && (m.message || '').includes('cancelled'),
  );
  assert.ok(cancelled, 'the "setup was cancelled" toast is shown');
  assert.deepStrictEqual(cancelled.actions, ['Retry setup']);
  console.log('cancel during finalization tests passed');
}

async function hostTests() {
  // Fresh module graph for the extension: its own vscode stub and a
  // DependencyInstaller stub that fails twice, then succeeds.
  Module._load = origLoad;
  for (const key of Object.keys(require.cache)) {
    if (key.startsWith(OUT_DIR)) delete require.cache[key];
  }
  const {
    loadExtension,
    waitFor: hostWaitFor,
  } = require('./_ui_antipattern_host');
  let attempts = 0;
  const h = loadExtension({
    config: {'kissSorcar.editorTabsMode': false},
    modules: {
      'UpdateChecker.js': {
        checkForExtensionUpdate: () => Promise.resolve({checked: false}),
        snoozeUpdateNotification: () => ({}),
        skipUpdateVersion: () => ({}),
      },
      'DependencyInstaller.js': {
        ensureLocalBinInPath: () => {},
        ensureDependencies: () => {
          attempts += 1;
          return attempts <= 2
            ? Promise.reject(
                new Error(
                  'uv sync failed (exit code 1): error: No solution found',
                ),
              )
            : Promise.resolve();
        },
        promptApiKeysNow: () => Promise.resolve(true),
      },
    },
  });
  const ctx = h.makeContext();
  h.extension.activate(ctx);
  await hostWaitFor(
    () => h.notifications.some(n => n.kind === 'error'),
    'a setup failure shows an error toast',
  );
  const toast = h.notifications.find(n => n.kind === 'error');
  assert.strictEqual(
    toast.message,
    'KISS Sorcar: Setup failed. uv sync failed (exit code 1): error: No solution found',
  );
  assert.deepStrictEqual(toast.actions, ['Open log', 'Retry']);
  assert.ok(
    !toast.message.includes('install.log'),
    'the toast no longer dumps a file path; Open log does that',
  );

  // 'Open log' opens install.log in an editor tab.
  fs.writeFileSync(logPath, 'full output\n');
  toast.resolve('Open log');
  await hostWaitFor(
    () => h.shownDocuments.length === 1,
    'Open log shows a document',
  );
  assert.strictEqual(h.openedDocuments[0].fsPath, logPath);

  // 'Retry' re-runs the setup; a second failure offers the same actions.
  const second = () => h.notifications.filter(n => n.kind === 'error')[1];
  assert.strictEqual(attempts, 1);
  const retry = h.registeredCommands.get('kissSorcar.retrySetup');
  assert.strictEqual(typeof retry, 'function', 'kissSorcar.retrySetup exists');
  retry();
  await hostWaitFor(() => !!second(), 'the retry failure shows a toast');
  assert.strictEqual(attempts, 2);
  second().resolve('Retry');
  await hostWaitFor(() => attempts === 3, "'Retry' re-runs ensureDependencies");
  await sleep(100);
  assert.strictEqual(
    h.notifications.filter(n => n.kind === 'error').length,
    2,
    'a successful retry shows no further error',
  );
  assert.strictEqual(
    typeof h.registeredCommands.get('kissSorcar.enterApiKey'),
    'function',
    'kissSorcar.enterApiKey is registered',
  );
  h.extension.deactivate();
  h.disposeContext(ctx);
  console.log('host failure toast tests passed');
}

async function runTest() {
  summarizeFailureTests();
  await failureTest();
  stickyTests();
  await cancelTest();
  await cancelDuringFinalizationTest();
  await hostTests();
  fs.rmSync(tmpHome, {recursive: true, force: true});
  console.log('\nAll ui_antipattern_setup_errors tests passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
