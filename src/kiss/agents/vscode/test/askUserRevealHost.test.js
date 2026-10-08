// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The VS Code host side of an ask_user_question: the webview posts
// `revealForQuestion` when a question reaches one of its tabs, and the
// host only brings the surface forward (sidebar view: show(true); editor
// panel: a forced `reveal` panel event). No native notification of any
// kind is raised, whether the webview was visible or hidden: the
// Question panel in the webview is the whole notice.

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

function makeUri(fsPath) {
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

let workspaceFolders = [];

// Every native notice the host tried to raise, by API name. A question
// must leave this empty.
const nativeNotices = [];

function recordNotice(api) {
  return (...args) => {
    nativeNotices.push({api, args});
    return Promise.resolve(undefined);
  };
}

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
    withProgress: recordNotice('withProgress'),
    showInformationMessage: recordNotice('showInformationMessage'),
    showWarningMessage: recordNotice('showWarningMessage'),
    showErrorMessage: recordNotice('showErrorMessage'),
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-askreveal-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});

function makeWebviewHost(visible) {
  const posted = [];
  const shows = [];
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const visEmitter = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => recvEmitter.event(cb),
  };
  const host = {
    webview,
    visible,
    show: preserveFocus => shows.push(preserveFocus),
    onDidChangeVisibility: cb => visEmitter.event(cb),
    onDidDispose: cb => disposeEmitter.event(cb),
  };
  return {
    host,
    posted,
    shows,
    fireMessage: m => recvEmitter.fire(m),
    disposeWebview: () => disposeEmitter.fire(),
  };
}

const tick = () => new Promise(r => setTimeout(r, 20));

function loadView() {
  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
  );
  delete require.cache[require.resolve(sourcePath)];
  return require(sourcePath).SorcarSidebarView;
}

async function testSidebarIsShownWithoutNativeNotice(SorcarSidebarView) {
  for (const visible of [true, false]) {
    const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
    const wv = makeWebviewHost(visible);
    view.resolveWebviewView(wv.host, {}, {});
    const before = nativeNotices.length;

    wv.fireMessage({type: 'revealForQuestion'});
    await tick();

    assert.deepStrictEqual(
      wv.shows,
      [true],
      `visible=${visible}: the sidebar view is brought forward with focus preserved`,
    );
    assert.strictEqual(
      nativeNotices.length,
      before,
      `visible=${visible}: no native notification: ` +
        JSON.stringify(nativeNotices.slice(before)),
    );

    // The retired protocol is gone: its messages fall through unhandled
    // and raise nothing either.
    wv.fireMessage({type: 'askWaiting', tabId: 't1', question: 'Deploy?'});
    wv.fireMessage({type: 'askWaitingDone', tabId: 't1'});
    await tick();
    assert.deepStrictEqual(wv.shows, [true], 'no extra reveal');
    assert.strictEqual(nativeNotices.length, before);

    wv.disposeWebview();
    view.dispose();
    await tick();
    assert.strictEqual(nativeNotices.length, before, 'nothing on teardown');
  }
  console.log('  ok - sidebar (visible and hidden): shown, no native notification');
}

async function testEditorPanelIsForceRevealed(SorcarSidebarView) {
  const events = [];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')), {
    rootTabId: 'root',
    onEvent: e => events.push(e),
  });
  const wv = makeWebviewHost(false);
  view.attachWebviewHost(wv.host, 'class="editor-tab-mode"');
  const before = nativeNotices.length;

  wv.fireMessage({type: 'revealForQuestion'});
  await tick();
  assert.deepStrictEqual(
    events.filter(e => e.kind === 'reveal'),
    [{kind: 'reveal', force: true}],
    'editor-tabs mode reveals the panel even while a text editor is active',
  );
  assert.deepStrictEqual(wv.shows, [], 'the panel path does not call show()');
  assert.strictEqual(nativeNotices.length, before, 'no native notification');
  view.dispose();
  console.log('  ok - editor panel: forced reveal, no native notification');
}

async function runTests() {
  const SorcarSidebarView = loadView();
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-askreveal-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];
  try {
    await testSidebarIsShownWithoutNativeNotice(SorcarSidebarView);
    await testEditorPanelIsForceRevealed(SorcarSidebarView);
  } finally {
    fs.rmSync(ws, {recursive: true, force: true});
  }
}

runTests().then(
  () => {
    fs.rmSync(tmpHome, {recursive: true, force: true});
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err);
    fs.rmSync(tmpHome, {recursive: true, force: true});
    process.exit(1);
  },
);
