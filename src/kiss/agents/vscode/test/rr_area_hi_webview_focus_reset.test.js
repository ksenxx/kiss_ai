// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// H-RC3: a webview disposed (or replaced) while it held focus must not
// leave SorcarSidebarView.hasFocus stuck at true — the toggleFocus
// keybinding reads it to decide between focusing the chat and focusing
// the editor, so a stale true means the chat can never be refocused.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

'use strict';

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

const vscodeStub = {
  EventEmitter: StubEventEmitter,
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => 'stub-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
  },
  Uri: {
    file: p => ({fsPath: p, scheme: 'file'}),
    joinPath: (uri, ...parts) => ({
      fsPath: path.join(uri.fsPath, ...parts),
      scheme: uri.scheme || 'file',
    }),
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
global.__kissVscodeStub = vscodeStub;

// Keep the AgentClient pointed at a socket nothing listens on so the
// view under test never talks to a real daemon.
const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rrhi-focus-'));
process.env.KISS_SORCAR_SOCK = path.join(tmpHome, 'no-daemon.sock');
process.env.KISS_HOME = path.join(tmpHome, '.kiss');

const projectRoot = path.resolve(__dirname, '..');
const {SorcarSidebarView} = require(
  path.join(projectRoot, 'out', 'SorcarSidebarView.js'),
);

function makeStubWebviewView() {
  const messageListeners = [];
  const disposeListeners = [];
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-webview://stub',
    asWebviewUri: uri => ({toString: () => `vscode-webview://${uri.fsPath}`}),
    postMessage: () => Promise.resolve(true),
    onDidReceiveMessage: cb => {
      messageListeners.push(cb);
      return {dispose: () => {}};
    },
  };
  return {
    view: {
      webview,
      visible: true,
      show: () => {},
      onDidDispose: cb => {
        disposeListeners.push(cb);
        return {dispose: () => {}};
      },
      onDidChangeVisibility: () => ({dispose: () => {}}),
    },
    fireMessage: m => {
      for (const cb of messageListeners.slice()) cb(m);
    },
    fireDispose: () => {
      for (const cb of disposeListeners.slice()) cb();
    },
  };
}

async function settle() {
  await new Promise(resolve => setTimeout(resolve, 20));
}

async function main() {
  const extensionUri = {fsPath: projectRoot, scheme: 'file'};
  const view = new SorcarSidebarView(extensionUri);
  const resolveArgs = [
    {state: undefined},
    {
      isCancellationRequested: false,
      onCancellationRequested: () => ({dispose: () => {}}),
    },
  ];

  const stub1 = makeStubWebviewView();
  view.resolveWebviewView(stub1.view, ...resolveArgs);
  assert.strictEqual(view.hasFocus, false, 'fresh webview reports no focus');

  stub1.fireMessage({type: 'webviewFocusChanged', focused: true});
  await settle();
  assert.strictEqual(view.hasFocus, true, 'focus report is tracked');

  // Disposed while focused: the flag must drop with the webview.
  stub1.fireDispose();
  assert.strictEqual(
    view.hasFocus,
    false,
    'hasFocus must reset when the focused webview is disposed',
  );
  console.log('  ok - onDidDispose clears a stuck focus flag');

  // Replaced while focused (sidebar re-resolved without a dispose event
  // for the old webview first): same requirement.
  const stub2 = makeStubWebviewView();
  view.resolveWebviewView(stub2.view, ...resolveArgs);
  stub2.fireMessage({type: 'webviewFocusChanged', focused: true});
  await settle();
  assert.strictEqual(view.hasFocus, true);
  const stub3 = makeStubWebviewView();
  view.resolveWebviewView(stub3.view, ...resolveArgs);
  assert.strictEqual(
    view.hasFocus,
    false,
    'hasFocus must reset when a new webview is resolved',
  );
  console.log('  ok - resolveWebviewView starts from an unfocused state');

  console.log('rr_area_hi_webview_focus_reset: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
