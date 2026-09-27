// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end: the webview's `{type: 'tipsOptOut'}` message, driven through
// SorcarSidebarView's onDidReceiveMessage handler, creates the
// $KISS_HOME/TIPS_DISABLED marker (the persisted "Don't show tips again"
// choice) and `{type: 'tipsOptOut', optOut: false}` removes it again.  On
// the old code the message fell through the switch and no marker was ever
// written, so the tips window kept reopening.

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
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});

  assert.ok(!fs.existsSync(marker), 'no marker before the user opts out');

  wv.fireMessage({type: 'tipsOptOut'});
  assert.ok(
    fs.existsSync(marker),
    'tipsOptOut (no optOut field) must write $KISS_HOME/TIPS_DISABLED',
  );
  console.log('  ok - tipsOptOut writes the TIPS_DISABLED marker');

  wv.fireMessage({type: 'tipsOptOut', optOut: true});
  assert.ok(fs.existsSync(marker), 'tipsOptOut is idempotent');

  wv.fireMessage({type: 'tipsOptOut', optOut: false});
  assert.ok(
    !fs.existsSync(marker),
    'tipsOptOut with optOut:false must remove the marker',
  );
  console.log('  ok - tipsOptOut optOut:false removes the marker');

  wv.fireMessage({type: 'tipsOptOut', optOut: false});
  assert.ok(!fs.existsSync(marker), 'removing a missing marker is a no-op');

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
