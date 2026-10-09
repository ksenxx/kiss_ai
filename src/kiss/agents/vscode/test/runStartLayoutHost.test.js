// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The VS Code host side of "a task starts: hide the editor window, show
// the right sidebar". The webview posts `taskStarted` for a task that
// just started in the chat tab it shows (runStartLayout.test.js) and:
//
//   * the secondary-sidebar chat view, while on screen, maximizes the
//     secondary side bar (workbench.action.maximizeAuxiliaryBar: VS Code
//     hides the editor area, the primary side bar and the panel); a
//     hidden view (a task launched from another surface) does nothing,
//     and a host without the command (VS Code < 1.102) is tolerated
//     quietly;
//   * an editor-tab chat raises a `taskStarted` panel event, and the
//     panel manager brings the Task Info view up in the secondary side
//     bar with the keyboard focus left in the chat
//     (kissSorcar.metaViewSecondary.focus {preserveFocus: true}) — for
//     the ACTIVE panel on screen only, since the view describes that
//     one; a failing reveal is logged, not thrown — and the panel
//     manager's onTaskStarted hook fires for that same start, through
//     which extension.ts has the Task Info view arrange its sections
//     for the run (postShowForRun: `showForRun` to a ready webview, or
//     on its `ready` when the view is still loading).

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const {createFakeDaemon} = require('./fakeDaemon');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
assert.ok(
  fs.existsSync(path.join(OUT_DIR, 'SorcarPanelManager.js')),
  'compiled extension missing — run `npm run compile` first',
);

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

function makeFakePanel(viewType, title) {
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const viewStateEmitter = new StubEventEmitter();
  const panel = {
    viewType,
    title,
    iconPath: undefined,
    visible: true,
    active: true,
    disposed: false,
    webview: {
      options: {},
      html: '',
      cspSource: 'vscode-resource:',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: () => Promise.resolve(true),
      onDidReceiveMessage: cb => recvEmitter.event(cb),
    },
    reveal: () => {},
    onDidChangeViewState: cb => viewStateEmitter.event(cb),
    onDidDispose: cb => disposeEmitter.event(cb),
    dispose: () => {
      if (panel.disposed) return;
      panel.disposed = true;
      disposeEmitter.fire();
    },
    _recv: recvEmitter,
  };
  return panel;
}

// Every command the host ran, and the outcome the stub answers with.
const commands = [];
let commandOutcome = () => Promise.resolve();

const vscodeStub = {
  workspace: {
    workspaceFolders: [],
    getConfiguration: () => ({
      get: key => (key === 'editorTabsMode' ? true : ''),
      update: () => Promise.resolve(),
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
  window: {
    createWebviewPanel: (viewType, title) => {
      const panel = makeFakePanel(viewType, title);
      createdPanels.push(panel);
      return panel;
    },
    registerWebviewPanelSerializer: () => ({dispose: () => {}}),
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showWarningMessage: () => {},
    showErrorMessage: () => {},
    showTextDocument: () => Promise.resolve({}),
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {
    executeCommand: (id, ...args) => {
      commands.push({id, args});
      return commandOutcome();
    },
  },
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-runstart-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const endpointPath = path.join(tmpHome, '.kiss', 'sorcar-local.json');

const server = createFakeDaemon(sock => {
  sock.on('data', () => {});
  sock.on('error', () => {});
});

const unhandled = [];
process.on('unhandledRejection', err => unhandled.push(err));

const tick = () => new Promise(r => setTimeout(r, 30));

function takeCommands() {
  return commands.splice(0);
}

const {SorcarPanelManager} = require(
  path.join(OUT_DIR, 'SorcarPanelManager.js'),
);
const {SorcarSidebarView} = require(
  path.join(OUT_DIR, 'SorcarSidebarView.js'),
);

function makeWebviewHost() {
  const recvEmitter = new StubEventEmitter();
  // Every message the host posted into the webview.
  const posted = [];
  const host = {
    webview: {
      options: {},
      html: '',
      cspSource: 'vscode-resource:',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: m => {
        posted.push(m);
        return Promise.resolve(true);
      },
      onDidReceiveMessage: cb => recvEmitter.event(cb),
    },
    visible: true,
    show: () => {},
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };
  return {host, posted, fireMessage: m => recvEmitter.fire(m)};
}

async function testTaskInfoViewShowForRunWaitsForReady() {
  const view = new SorcarSidebarView(makeUri(EXT_ROOT), {
    rootTabId: 'meta-panel',
    bodyAttrs: ' class="editor-tab-mode meta-panel-mode"',
    onEvent: () => {},
  });
  const wv = makeWebviewHost();
  view.resolveWebviewView(wv.host, {}, {});
  const showForRun = () => wv.posted.filter(m => m.type === 'showForRun');

  // The view revealed for the start resolves after the relay: the
  // message waits for the webview's `ready` (once, however many
  // starts queued) instead of being lost on a page still loading.
  view.postShowForRun();
  view.postShowForRun();
  assert.strictEqual(showForRun().length, 0, 'nothing posted before ready');
  wv.fireMessage({type: 'ready', tabId: 'meta-panel'});
  await tick();
  assert.strictEqual(showForRun().length, 1, 'posted once on ready');

  // A ready webview gets it at once; a later ready replays nothing.
  view.postShowForRun();
  assert.strictEqual(showForRun().length, 2, 'posted at once when ready');
  wv.fireMessage({type: 'ready', tabId: 'meta-panel'});
  await tick();
  assert.strictEqual(showForRun().length, 2, 'nothing pending to replay');

  view.dispose();
  console.log('  ok - Task Info view: showForRun posted when ready, else on ready');
}

async function testSidebarViewMaximizesTheBar() {
  const view = new SorcarSidebarView(makeUri(EXT_ROOT));
  const wv = makeWebviewHost();
  view.resolveWebviewView(wv.host, {}, {});
  takeCommands();

  wv.fireMessage({type: 'taskStarted'});
  await tick();
  assert.deepStrictEqual(
    takeCommands(),
    [{id: 'workbench.action.maximizeAuxiliaryBar', args: []}],
    'the sidebar chat view maximizes the secondary side bar',
  );

  // The bar closed (a task launched from another surface): nothing
  // covers what the user is working on.
  wv.host.visible = false;
  wv.fireMessage({type: 'taskStarted'});
  await tick();
  assert.deepStrictEqual(takeCommands(), [], 'a hidden view maximizes nothing');
  wv.host.visible = true;

  // A host without the command: nothing thrown, nothing unhandled.
  commandOutcome = () => Promise.reject(new Error("command 'x' not found"));
  wv.fireMessage({type: 'taskStarted'});
  await tick();
  commandOutcome = () => Promise.resolve();
  assert.strictEqual(takeCommands().length, 1, 'the command was still tried');
  assert.deepStrictEqual(unhandled, [], 'the missing command is swallowed');

  view.dispose();
  console.log('  ok - sidebar chat view: maximizeAuxiliaryBar, missing command tolerated');
}

async function testEditorPanelRevealsTaskInfoForTheActivePanel() {
  const manager = new SorcarPanelManager(makeUri(EXT_ROOT));
  let starts = 0;
  manager.onTaskStarted = () => starts++;
  manager.openNewChat();
  const panelA = createdPanels[0];
  manager.openNewChat();
  const panelB = createdPanels[1];
  panelA.active = false;
  await tick();
  takeCommands();

  // The ACTIVE panel's start brings the Task Info view up, focus kept,
  // and tells the hook (extension.ts has the view arrange its sections).
  panelB._recv.fire({type: 'taskStarted'});
  await tick();
  assert.deepStrictEqual(
    takeCommands(),
    [{id: 'kissSorcar.metaViewSecondary.focus', args: [{preserveFocus: true}]}],
    'the active editor-tab chat reveals the Task Info view without taking focus',
  );
  assert.strictEqual(starts, 1, 'the start reaches onTaskStarted');

  // A background panel's start (a task launched from another surface)
  // must not bring the view up for the wrong chat.
  panelA._recv.fire({type: 'taskStarted'});
  await tick();
  assert.deepStrictEqual(takeCommands(), [], 'a background panel reveals nothing');
  assert.strictEqual(starts, 1, 'a background panel reaches no hook');

  // The last active chat hidden behind a text editor: still the panel
  // the Task Info view describes, but not on screen.
  panelB.active = false;
  panelB.visible = false;
  panelB._recv.fire({type: 'taskStarted'});
  await tick();
  assert.deepStrictEqual(takeCommands(), [], 'a chat behind an editor reveals nothing');
  assert.strictEqual(starts, 1, 'a chat behind an editor reaches no hook');
  panelB.active = true;
  panelB.visible = true;

  // No hook installed: the reveal alone.
  manager.onTaskStarted = undefined;
  panelB._recv.fire({type: 'taskStarted'});
  await tick();
  assert.strictEqual(takeCommands().length, 1, 'the reveal without a hook');
  manager.onTaskStarted = () => starts++;

  // A failing reveal is logged, never thrown.
  const errors = [];
  const origError = console.error;
  console.error = (...args) => errors.push(args);
  commandOutcome = () => Promise.reject(new Error('no such view'));
  panelB._recv.fire({type: 'taskStarted'});
  await tick();
  commandOutcome = () => Promise.resolve();
  console.error = origError;
  assert.strictEqual(takeCommands().length, 1);
  assert.ok(
    errors.some(a => String(a[0]).includes('Task Info reveal failed')),
    'the failure is logged: ' + JSON.stringify(errors),
  );
  assert.deepStrictEqual(unhandled, [], 'the failure is not unhandled');

  manager.dispose();
  console.log('  ok - editor-tab chat: Task Info view for the active panel only');
}

async function runTests() {
  await new Promise(r => server.listen(endpointPath, r));
  await testSidebarViewMaximizesTheBar();
  await testEditorPanelRevealsTaskInfoForTheActivePanel();
  await testTaskInfoViewShowForRunWaitsForReady();
}

runTests().then(
  () => {
    server.close();
    fs.rmSync(tmpHome, {recursive: true, force: true});
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err);
    server.close();
    fs.rmSync(tmpHome, {recursive: true, force: true});
    process.exit(1);
  },
);
