// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end: the webview's `{type: 'tipsOptOut'}` and `{type:
// 'getWelcomeInfo'}` messages, driven through SorcarSidebarView's
// onDidReceiveMessage handler, are forwarded to the daemon verbatim.  The
// daemon owns the $KISS_HOME/TIPS_DISABLED marker (the persisted "Don't
// show tips again" choice, `tips_opt_out` in sorcar.py) and the remote
// URL (`getWelcomeInfo` answers with the `remote_url` event), so the host
// writes and reads neither itself: one implementation serves the VS Code
// webview and the remote webapp alike.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

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

let workspaceFolders = [];

const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
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
  ViewColumn: {One: 1},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showErrorMessage: () => {},
    showTextDocument: () => Promise.resolve({}),
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-tipsoptout-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_HOME = path.join(tmpHome, 'kiss-home');
const marker = path.join(process.env.KISS_HOME, 'TIPS_DISABLED');

function makeWebviewView() {
  const recvEmitter = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: () => Promise.resolve(true),
    onDidReceiveMessage: cb => recvEmitter.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };
  return {webviewView, fireMessage: m => recvEmitter.fire(m)};
}

function runTests() {
  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
  );
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-tipsoptout-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const forwarded = [];
  view._send = cmd => forwarded.push(cmd);
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});

  wv.fireMessage({type: 'tipsOptOut'});
  wv.fireMessage({type: 'tipsOptOut', optOut: true});
  wv.fireMessage({type: 'tipsOptOut', optOut: false});
  assert.deepStrictEqual(
    forwarded.filter(c => c.type === 'tipsOptOut'),
    [
      {type: 'tipsOptOut', optOut: undefined},
      {type: 'tipsOptOut', optOut: true},
      {type: 'tipsOptOut', optOut: false},
    ],
    'every tipsOptOut reaches the daemon with its optOut flag intact',
  );
  assert.ok(
    !fs.existsSync(marker),
    'the host must not write $KISS_HOME/TIPS_DISABLED itself',
  );
  console.log('  ok - tipsOptOut is forwarded to the daemon, not written locally');

  wv.fireMessage({type: 'getWelcomeInfo'});
  assert.deepStrictEqual(
    forwarded.filter(c => c.type === 'getWelcomeInfo'),
    [{type: 'getWelcomeInfo'}],
    'getWelcomeInfo reaches the daemon, which answers with remote_url',
  );
  console.log('  ok - getWelcomeInfo is forwarded to the daemon');

  if (typeof view.dispose === 'function') view.dispose();
  fs.rmSync(ws, {recursive: true, force: true});
}

try {
  runTests();
  fs.rmSync(tmpHome, {recursive: true, force: true});
  console.log('\nAll tests passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err);
  fs.rmSync(tmpHome, {recursive: true, force: true});
  process.exit(1);
}
