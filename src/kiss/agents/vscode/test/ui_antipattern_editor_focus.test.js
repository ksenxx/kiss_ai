// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A7 focus stealing and A7/M5 silent setting rewrite (audit-extension
// M3, M4, M5), against the REAL out/SorcarPanelManager.js (with its
// SorcarSidebarView controllers talking to a fake daemon socket) and
// out/editorActionsLocation.js:
//  - a finished task's `revealPanel` reveals the chat tab with
//    preserveFocus while no text editor is active, and does NOTHING
//    while the user is reading a file (the tab decoration suffices);
//  - closing the last chat tab (editor X, or the root chat closed inside
//    the webview) reopens the one-tab replacement in the BACKGROUND:
//    preserveFocus true, no focusInput — closing is not a request for a
//    new chat; only switching the mode on focuses a fresh chat;
//  - `workbench.editor.editorActionsLocation` is never rewritten
//    silently: the first sync asks once ('Move actions to title bar' /
//    'Keep my setting'), a second sync while the question is open asks
//    nothing more, 'Keep' (and dismissal) is remembered across
//    activations, 'Move' moves right away and is remembered too.

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');
const {createFakeDaemon} = require('./fakeDaemon');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');

class StubEventEmitter {
  constructor() {
    this._listeners = [];
    this.event = cb => {
      this._listeners.push(cb);
      return {
        dispose: () => {
          const i = this._listeners.indexOf(cb);
          if (i >= 0) this._listeners.splice(i, 1);
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

function makeUri(fsPath) {
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

const createdPanels = [];

function makeFakePanel(viewType, title, showOptions) {
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const viewStateEmitter = new StubEventEmitter();
  const posted = [];
  const panel = {
    viewType,
    title,
    iconPath: undefined,
    visible: true,
    active: true,
    revealCalls: [],
    disposed: false,
    showOptions,
    webview: {
      options: {},
      html: '',
      cspSource: 'vscode-resource:',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: msg => {
        posted.push(msg);
        return Promise.resolve(true);
      },
      onDidReceiveMessage: cb => recvEmitter.event(cb),
    },
    reveal: (column, preserveFocus) => {
      panel.revealCalls.push({column, preserveFocus});
    },
    onDidChangeViewState: cb => viewStateEmitter.event(cb),
    onDidDispose: cb => disposeEmitter.event(cb),
    dispose: () => {
      if (panel.disposed) return;
      panel.disposed = true;
      disposeEmitter.fire();
    },
    _posted: posted,
    _recv: recvEmitter,
  };
  return panel;
}

// The live configuration behind getConfiguration(): the panel manager
// reads kissSorcar.editorTabsMode; editorActionsLocation reads/writes
// workbench.editor.editorActionsLocation and window.customTitleBarVisibility.
const config = {'kissSorcar.editorTabsMode': true};
const configUpdates = [];
const infoToasts = [];

const vscodeStub = {
  workspace: {
    workspaceFolders: [],
    getConfiguration: section => ({
      get: (key, def) => {
        const full = section ? `${section}.${key}` : key;
        return full in config ? config[full] : def;
      },
      inspect: key => {
        const full = section ? `${section}.${key}` : key;
        return {globalValue: config[full]};
      },
      update: (key, value, target) => {
        const full = section ? `${section}.${key}` : key;
        configUpdates.push({key: full, value, target});
        if (value === undefined) delete config[full];
        else config[full] = value;
        return Promise.resolve();
      },
    }),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  ProgressLocation: {Notification: 15},
  ViewColumn: {Active: -1, One: 1},
  ConfigurationTarget: {Global: 1},
  window: {
    createWebviewPanel: (viewType, title, showOptions) => {
      const panel = makeFakePanel(viewType, title, showOptions);
      createdPanels.push(panel);
      return panel;
    },
    registerWebviewPanelSerializer: () => ({dispose: () => {}}),
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: (message, _opts, ...actions) =>
      new Promise(resolve => infoToasts.push({message, actions, resolve})),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {executeCommand: () => Promise.resolve()},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-uap-focus-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const endpointPath = path.join(tmpHome, '.kiss', 'sorcar-local.json');
const server = createFakeDaemon(sock => {
  sock.on('data', () => {});
  sock.on('error', () => {});
});

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function waitFor(predicate, message) {
  for (let i = 0; i < 150; i++) {
    if (predicate()) return;
    await sleep(20);
  }
  throw new Error(message || 'waitFor timed out');
}

const {SorcarPanelManager} = require(path.join(OUT_DIR, 'SorcarPanelManager.js'));
const {
  LOCATION_CONSENT_KEY,
  PRIOR_LOCATION_KEY,
  syncEditorActionsLocation,
} = require(path.join(OUT_DIR, 'editorActionsLocation.js'));

function lastPanel() {
  return createdPanels[createdPanels.length - 1];
}

async function assertBackgroundReplacement(panel, why) {
  assert.strictEqual(
    panel.showOptions.preserveFocus,
    true,
    `${why}: the replacement opens without taking focus`,
  );
  await sleep(150);
  assert.ok(
    !panel._posted.some(m => m.type === 'focusInput'),
    `${why}: the replacement composer is not focused`,
  );
}

async function panelTests() {
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));

  // --- a finished task reveals only while no file is being read ---------
  const controller = manager.openNewChat();
  assert.ok(controller);
  const panel = createdPanels[0];
  panel._recv.fire({type: 'revealPanel'});
  await waitFor(() => panel.revealCalls.length === 1, 'reveal with no editor');
  assert.strictEqual(panel.revealCalls[0].preserveFocus, true);

  vscodeStub.window.activeTextEditor = {document: {uri: makeUri('/ws/a.py')}};
  panel._recv.fire({type: 'revealPanel'});
  await sleep(150);
  assert.strictEqual(
    panel.revealCalls.length,
    1,
    'a finished task must not reveal its tab while the user is in a file',
  );
  vscodeStub.window.activeTextEditor = undefined;
  panel._recv.fire({type: 'revealPanel'});
  await waitFor(() => panel.revealCalls.length === 2, 'reveal resumes');

  // --- editor-tab X on the last chat: background replacement ------------
  panel.dispose();
  assert.strictEqual(createdPanels.length, 2, 'the one-tab invariant holds');
  await assertBackgroundReplacement(lastPanel(), 'tab X');

  // --- root chat closed inside the webview (retire): background too -----
  const panelB = lastPanel();
  panelB._recv.fire({type: 'closePanel', retire: true});
  await waitFor(() => panelB.disposed, 'closePanel(retire) disposes');
  assert.strictEqual(createdPanels.length, 3);
  await assertBackgroundReplacement(lastPanel(), 'webview close');

  // --- a remote close (no retire): background as before ------------------
  const panelC = lastPanel();
  panelC._recv.fire({type: 'closePanel'});
  await waitFor(() => panelC.disposed, 'closePanel disposes');
  assert.strictEqual(createdPanels.length, 4);
  await assertBackgroundReplacement(lastPanel(), 'remote close');

  // --- the user ASKING for a chat still gets focus -----------------------
  lastPanel().dispose();
  await sleep(50);
  const fresh = manager.openNewChat();
  assert.ok(fresh);
  assert.strictEqual(
    lastPanel().showOptions.preserveFocus,
    false,
    'an explicit new chat takes focus',
  );
  manager.dispose();
  console.log('panel focus tests passed');
}

function makeMemento() {
  const store = new Map();
  return {
    get: (key, def) => (store.has(key) ? store.get(key) : def),
    update: (key, value) => {
      if (value === undefined) store.delete(key);
      else store.set(key, value);
      return Promise.resolve();
    },
  };
}

function actionUpdates() {
  return configUpdates.filter(
    u => u.key === 'workbench.editor.editorActionsLocation',
  );
}

async function consentTests() {
  const EDITOR_ACTIONS = 'workbench.editor.editorActionsLocation';
  // --- first sync: a question, no write ---------------------------------
  const ctx = {globalState: makeMemento()};
  await syncEditorActionsLocation(ctx, true);
  assert.deepStrictEqual(actionUpdates(), [], 'no silent rewrite');
  assert.strictEqual(infoToasts.length, 1, 'the user is asked');
  assert.ok(/title bar/.test(infoToasts[0].message));
  assert.deepStrictEqual(infoToasts[0].actions, [
    'Move actions to title bar',
    'Keep my setting',
  ]);
  // A second sync while the question is open asks nothing more.
  await syncEditorActionsLocation(ctx, true);
  assert.strictEqual(infoToasts.length, 1, 'one open question at a time');

  // --- 'Keep my setting': remembered, never asked again ------------------
  infoToasts[0].resolve('Keep my setting');
  await waitFor(() => ctx.globalState.get(LOCATION_CONSENT_KEY) === 'keep');
  await syncEditorActionsLocation(ctx, true);
  assert.strictEqual(infoToasts.length, 1, 'keep: not asked again');
  assert.deepStrictEqual(actionUpdates(), [], 'keep: setting untouched');
  assert.strictEqual(config[EDITOR_ACTIONS], undefined);

  // --- dismissing the question counts as keep ----------------------------
  const ctx2 = {globalState: makeMemento()};
  await syncEditorActionsLocation(ctx2, true);
  assert.strictEqual(infoToasts.length, 2);
  infoToasts[1].resolve(undefined);
  await waitFor(() => ctx2.globalState.get(LOCATION_CONSENT_KEY) === 'keep');
  assert.deepStrictEqual(actionUpdates(), []);

  // --- 'Move actions to title bar': moved now, remembered ----------------
  const ctx3 = {globalState: makeMemento()};
  await syncEditorActionsLocation(ctx3, true);
  assert.strictEqual(infoToasts.length, 3);
  infoToasts[2].resolve('Move actions to title bar');
  await waitFor(
    () => actionUpdates().length === 1,
    'agreeing moves the actions right away',
  );
  assert.deepStrictEqual(actionUpdates()[0], {
    key: EDITOR_ACTIONS,
    value: 'titleBar',
    target: 1,
  });
  assert.strictEqual(ctx3.globalState.get(LOCATION_CONSENT_KEY), 'move');
  assert.deepStrictEqual(ctx3.globalState.get(PRIOR_LOCATION_KEY), {
    prior: null,
  });
  // The next session (same globalState) moves without asking; the
  // mode-off restore still works as before.
  delete config[EDITOR_ACTIONS];
  await ctx3.globalState.update(PRIOR_LOCATION_KEY, undefined);
  await syncEditorActionsLocation(ctx3, true);
  assert.strictEqual(infoToasts.length, 3, 'consent remembered: no question');
  assert.strictEqual(config[EDITOR_ACTIONS], 'titleBar');
  await syncEditorActionsLocation(ctx3, false);
  assert.strictEqual(config[EDITOR_ACTIONS], undefined, 'mode off restores');
  assert.strictEqual(ctx3.globalState.get(PRIOR_LOCATION_KEY), undefined);

  // --- agreeing after the mode was switched off moves nothing ------------
  const ctx4 = {globalState: makeMemento()};
  await syncEditorActionsLocation(ctx4, true);
  assert.strictEqual(infoToasts.length, 4);
  config['kissSorcar.editorTabsMode'] = false;
  infoToasts[3].resolve('Move actions to title bar');
  await waitFor(() => ctx4.globalState.get(LOCATION_CONSENT_KEY) === 'move');
  await sleep(100);
  assert.strictEqual(
    config[EDITOR_ACTIONS],
    undefined,
    'a stale agreement never moves the actions while the mode is off',
  );
  console.log('editor-actions consent tests passed');
}

async function runTest() {
  server.listen(endpointPath);
  try {
    await panelTests();
    await consentTests();
  } finally {
    server.close();
    fs.rmSync(tmpHome, {recursive: true, force: true});
  }
  console.log('\nAll ui_antipattern_editor_focus tests passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
