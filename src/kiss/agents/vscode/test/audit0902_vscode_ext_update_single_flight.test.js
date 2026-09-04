// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-02 (vscode-ext partition): SorcarSidebarView.runUpdate()
// is the extension twin of the daemon's `runUpdate` handler.  "Update now"
// clicked twice -- or the notification action plus the settings button --
// used to open TWO terminals each running install.sh against the same
// ~/.kiss/kiss_ai tree at once (concurrent git reset, concurrent `uv
// sync`, concurrent daemon restarts).
//
// Real SorcarSidebarView from out/, real install.sh on disk, a stub
// vscode.window that records terminals and lets the test close them.
//
// Review follow-up (review-vscode.md #1/#2): two VS Code WINDOWS are two
// extension hosts, so the root fix is the cross-process lock inside
// scripts/install.sh (the curl bootstrap: sync the clone, then hand over
// to ./install.sh); the terminal runs THAT script, with
// KISS_NONINTERACTIVE=1, whenever the clone has it, and a second run
// prints "another KISS update is already running (pid N)" and exits 1.
// A clone without scripts/install.sh (an old checkout being updated)
// keeps the unlocked preflight + install.sh.
//
// Round 2 (review2-vscode.md #3): the per-window guard that reused the
// "KISS Sorcar Update" terminal followed the integrated shell's lifetime,
// not the installer's -- after a finished update every later click was
// refused until the terminal was closed.  The guard is gone: each click
// opens a terminal running the locked installer, and the lock alone
// decides whether it may proceed.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT_SIDEBAR = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
if (!fs.existsSync(OUT_SIDEBAR)) {
  console.log(`SKIP: ${OUT_SIDEBAR} missing — run \`npm run compile\``);
  process.exit(0);
}

class StubEventEmitter {
  constructor() {
    this._listeners = [];
    this.event = cb => {
      this._listeners.push(cb);
      return {
        dispose: () => {
          const idx = this._listeners.indexOf(cb);
          if (idx >= 0) this._listeners.splice(idx, 1);
        },
      };
    };
  }
  fire(arg) {
    for (const cb of this._listeners.slice()) cb(arg);
  }
  dispose() {
    this._listeners = [];
  }
}

const terminals = [];
const closeTerminalEmitter = new StubEventEmitter();
const nativeMessages = [];

global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => ({fsPath: p, scheme: 'file'}),
    joinPath: (uri, ...parts) => ({
      fsPath: path.join(uri.fsPath, ...parts),
      scheme: uri.scheme || 'file',
    }),
  },
  ProgressLocation: {Notification: 15},
  window: {
    createTerminal(opts) {
      const t = {
        name: opts && opts.name,
        cwd: opts && opts.cwd,
        sent: [],
        shows: 0,
        disposed: false,
        show() {
          this.shows += 1;
        },
        sendText(text) {
          this.sent.push(text);
        },
        dispose() {
          this.disposed = true;
          closeTerminalEmitter.fire(this);
        },
      };
      terminals.push(t);
      return t;
    },
    onDidCloseTerminal: closeTerminalEmitter.event,
    showInformationMessage: (m, ...items) => {
      nativeMessages.push({severity: 'info', m, items});
      return Promise.resolve(undefined);
    },
    showWarningMessage: (m, ...items) => {
      nativeMessages.push({severity: 'warning', m, items});
      return Promise.resolve(undefined);
    },
    showErrorMessage: (m, ...items) => {
      nativeMessages.push({severity: 'error', m, items});
      return Promise.resolve(undefined);
    },
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
  },
  commands: {executeCommand: () => Promise.resolve()},
};
const realResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return realResolve.call(this, request, ...rest);
};

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit-upd-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_SORCAR_SOCK = path.join(tmpHome, 'no-daemon.sock');
// findInstallScript() looks in os.homedir()/.kiss/kiss_ai; os.homedir()
// honours $HOME on POSIX.
const installRoot = path.join(tmpHome, '.kiss', 'kiss_ai');
fs.mkdirSync(installRoot, {recursive: true});
fs.writeFileSync(path.join(installRoot, 'install.sh'), '#!/bin/sh\nexit 0\n');

const {SorcarSidebarView} = require(OUT_SIDEBAR);

function makeSidebar() {
  const view = new SorcarSidebarView({
    fsPath: path.resolve(__dirname, '..'),
    scheme: 'file',
  });
  const posted = [];
  const listeners = [];
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-webview://stub',
    asWebviewUri: uri => ({toString: () => `vscode-webview://${uri.fsPath}`}),
    postMessage: m => {
      posted.push(m);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => {
      listeners.push(cb);
      return {dispose: () => {}};
    },
  };
  view.resolveWebviewView(
    {
      webview,
      visible: true,
      show: () => {},
      onDidDispose: () => ({dispose: () => {}}),
      onDidChangeVisibility: () => ({dispose: () => {}}),
    },
    {state: undefined},
    {
      isCancellationRequested: false,
      onCancellationRequested: () => ({dispose: () => {}}),
    },
  );
  return {
    view,
    posted,
    fire: m => {
      for (const cb of listeners.slice()) cb(m);
    },
  };
}

function toasts(posted) {
  return posted.filter(m => m.type === 'notification' && !m.close);
}

async function testSecondClickRunsTheInstallerAgain() {
  const {view, posted, fire} = makeSidebar();
  view.runUpdate();
  assert.strictEqual(terminals.length, 1, 'first click opened no terminal');
  assert.strictEqual(terminals[0].name, 'KISS Sorcar Update');
  assert.strictEqual(terminals[0].cwd, installRoot);
  assert.strictEqual(terminals[0].sent.length, 1);
  assert.ok(
    terminals[0].sent[0].includes('install.sh') &&
      terminals[0].sent[0].includes('--non-interactive'),
    `terminal did not run install.sh: ${terminals[0].sent[0]}`,
  );
  assert.strictEqual(terminals[0].shows, 1);

  // The installer has finished but its shell is still open (the command
  // has no trailing `exit`).  A second click -- notification action or
  // the webview's Update button -- must simply run the installer again
  // in a fresh terminal: whether another installer is STILL running is
  // decided by install.sh's own cross-process lock, which prints the
  // refusal itself.  The old per-window guard keyed on the terminal's
  // lifetime and wrongly refused every later click until the terminal
  // was closed.
  view.runUpdate();
  fire({type: 'runUpdate'});
  await new Promise(resolve => setImmediate(resolve));
  assert.strictEqual(
    terminals.length,
    3,
    `later clicks must run install.sh again (${terminals.length} terminals)`,
  );
  for (const t of terminals) {
    assert.strictEqual(t.sent.length, 1);
    assert.strictEqual(t.sent[0], terminals[0].sent[0]);
    assert.strictEqual(t.shows, 1);
  }
  const already = toasts(posted).filter(m =>
    /already running/i.test(m.message || ''),
  );
  assert.strictEqual(
    already.length,
    0,
    'the extension must not guess that an update is still running',
  );
  const started = toasts(posted).filter(m =>
    /is getting installed/.test(m.message || ''),
  );
  assert.strictEqual(started.length, 3);

  // Closing terminals is irrelevant to later clicks, and the view holds
  // no terminal bookkeeping: disposing it leaves the terminals alone.
  terminals[0].dispose();
  view.runUpdate();
  assert.strictEqual(terminals.length, 4);
  view.dispose();
  assert.strictEqual(terminals[1].disposed, false);
  assert.strictEqual(terminals[3].disposed, false);
  assert.strictEqual(closeTerminalEmitter._listeners.length, 0);
  console.log('  ✓ every Update click runs the locked installer again');
}

async function testMissingInstallScriptStillErrors() {
  const {view, posted} = makeSidebar();
  fs.rmSync(path.join(installRoot, 'install.sh'));
  try {
    const before = terminals.length;
    view.runUpdate();
    assert.strictEqual(terminals.length, before, 'terminal opened w/o script');
    const errs = toasts(posted).filter(m => m.severity === 'error');
    assert.strictEqual(errs.length, 1);
    assert.ok(/install\.sh not found/.test(errs[0].message));
  } finally {
    fs.writeFileSync(path.join(installRoot, 'install.sh'), '#!/bin/sh\n');
    view.dispose();
  }
  console.log('  ✓ missing install.sh still reports an error');
}

function testLockedBootstrapIsPreferred() {
  // Without scripts/install.sh: the legacy preflight + install.sh.
  const legacy = makeSidebar();
  legacy.view.runUpdate();
  const legacyCmd = terminals[terminals.length - 1].sent[0];
  assert.ok(
    /git reset --hard/.test(legacyCmd) &&
      /bash '[^']*\/install\.sh' --non-interactive$/.test(legacyCmd),
    `old clone must keep the preflight + install.sh: ${legacyCmd}`,
  );
  // install.sh writes the .extension-updated marker into $KISS_HOME and
  // extension.ts watches the extension host's $KISS_HOME; the command must
  // pin that value so a shell rc exporting a different KISS_HOME cannot
  // send the marker where no watcher looks.
  assert.ok(
    /KISS_HOME='[^']*' bash '[^']*\/install\.sh' --non-interactive$/.test(
      legacyCmd,
    ),
    `legacy preflight must pin the extension host's KISS_HOME: ${legacyCmd}`,
  );
  legacy.view.dispose();

  // With scripts/install.sh: one command, the locked bootstrap, and no
  // unlocked git preflight in front of it.
  const scriptsDir = path.join(installRoot, 'scripts');
  fs.mkdirSync(scriptsDir, {recursive: true});
  const bootstrap = path.join(scriptsDir, 'install.sh');
  fs.writeFileSync(bootstrap, '#!/bin/bash\nexit 0\n');
  try {
    const {view} = makeSidebar();
    view.runUpdate();
    const cmd = terminals[terminals.length - 1].sent[0];
    assert.ok(
      !/git reset --hard|git fetch|git stash/.test(cmd),
      `no unlocked git preflight may run before the locked bootstrap: ${cmd}`,
    );
    assert.ok(
      cmd.includes(`KISS_NONINTERACTIVE=1 bash '${bootstrap}'`),
      `terminal must run the locked bootstrap non-interactively: ${cmd}`,
    );
    assert.ok(
      /KISS_HOME='[^']*' KISS_NONINTERACTIVE=1 bash /.test(cmd),
      `bootstrap must pin the extension host's KISS_HOME: ${cmd}`,
    );
    assert.strictEqual(
      terminals[terminals.length - 1].cwd,
      installRoot,
      'the terminal still opens in the clone',
    );
    view.dispose();
  } finally {
    fs.rmSync(scriptsDir, {recursive: true, force: true});
  }
  console.log('  ✓ the locked scripts/install.sh is preferred when present');
}

async function main() {
  try {
    await testSecondClickRunsTheInstallerAgain();
    await testMissingInstallScriptStillErrors();
    testLockedBootstrapIsPreferred();
  } finally {
    fs.rmSync(tmpHome, {recursive: true, force: true});
  }
  console.log('audit0902_vscode_ext_update_single_flight: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
