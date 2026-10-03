// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The VS Code host side of "the agent is waiting for your answer":
// the webview posts `askWaiting` when an ask_user_question reaches one
// of its tabs, and the host
//  * brings the surface forward (sidebar view: show(true); editor
//    panel: a forced `reveal` panel event), and
//  * when the webview was hidden, raises a native cancellable progress
//    notification that stays until the user cancels it, the question is
//    retired (`askWaitingDone`) or the controller is disposed.
// A webview the user can already see shows its own sticky toast, so no
// native notification doubles it.

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

// Every native progress notification the host raised: its options, a
// `settled` flag flipped when the task promise resolves, and `cancel()`
// standing in for the user's click on the notification's X.
const progressNotices = [];

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
    withProgress: (opts, task) => {
      const cancelEmitter = new StubEventEmitter();
      const notice = {
        opts,
        settled: false,
        cancel: () => cancelEmitter.fire(),
      };
      progressNotices.push(notice);
      const done = task(
        {report: () => {}},
        {onCancellationRequested: cb => cancelEmitter.event(cb)},
      );
      return Promise.resolve(done).then(() => {
        notice.settled = true;
      });
    },
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-askwait-'));
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

function settledCount() {
  return progressNotices.filter(n => n.settled).length;
}

async function testVisibleSidebarIsShownWithoutNativeNotice(SorcarSidebarView) {
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewHost(true);
  view.resolveWebviewView(wv.host, {}, {});
  const before = progressNotices.length;

  wv.fireMessage({type: 'askWaiting', tabId: 't1', question: 'Deploy?'});
  await tick();

  assert.deepStrictEqual(
    wv.shows,
    [true],
    'the sidebar view is brought forward with focus preserved',
  );
  assert.strictEqual(
    progressNotices.length,
    before,
    'a visible webview shows its own toast: no native notification',
  );
  view.dispose();
  console.log('  ok - visible sidebar: shown, no native notification');
}

async function testHiddenSidebarRaisesNoticeUntilAnswered(SorcarSidebarView) {
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewHost(false);
  view.resolveWebviewView(wv.host, {}, {});
  const before = progressNotices.length;

  wv.fireMessage({
    type: 'askWaiting',
    tabId: 't1',
    question: 'Which branch should I push?',
  });
  await tick();

  assert.deepStrictEqual(wv.shows, [true], 'the hidden view is revealed');
  assert.strictEqual(progressNotices.length, before + 1);
  const notice = progressNotices[before];
  assert.strictEqual(notice.opts.location, 15, 'a Notification-area toast');
  assert.strictEqual(notice.opts.cancellable, true, 'the user can clear it');
  assert.ok(
    /waiting for your answer: Which branch should I push\?$/.test(
      notice.opts.title,
    ),
    'the notification names the question: ' + notice.opts.title,
  );
  assert.strictEqual(notice.settled, false, 'it stays while unanswered');

  // A second question on the same tab (the first was answered without
  // the host hearing of it) replaces the notice instead of stacking.
  wv.fireMessage({type: 'askWaiting', tabId: 't1', question: 'And now?'});
  await tick();
  assert.strictEqual(notice.settled, true, 'the stale notice is closed');
  assert.strictEqual(progressNotices.length, before + 2);
  const second = progressNotices[before + 1];
  assert.strictEqual(second.settled, false);

  // Answered: the notice goes away by itself.
  wv.fireMessage({type: 'askWaitingDone', tabId: 't1'});
  await tick();
  assert.strictEqual(second.settled, true, 'askWaitingDone closes the notice');

  // A done for a tab without a notice is a no-op.
  const settledBefore = settledCount();
  wv.fireMessage({type: 'askWaitingDone', tabId: 'no-such-tab'});
  await tick();
  assert.strictEqual(settledCount(), settledBefore);
  view.dispose();
  console.log('  ok - hidden sidebar: native notice until the answer');
}

async function testUserCancelClearsNotice(SorcarSidebarView) {
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewHost(false);
  view.resolveWebviewView(wv.host, {}, {});
  const before = progressNotices.length;

  wv.fireMessage({type: 'askWaiting', tabId: 't2', question: 'Proceed?'});
  await tick();
  const notice = progressNotices[before];
  notice.cancel();
  await tick();
  assert.strictEqual(notice.settled, true, 'cancelling clears the notice');

  // The host forgot it: a later done is a no-op and a new question
  // raises a fresh notice.
  wv.fireMessage({type: 'askWaitingDone', tabId: 't2'});
  wv.fireMessage({type: 'askWaiting', tabId: 't2', question: 'Again?'});
  await tick();
  assert.strictEqual(progressNotices.length, before + 2);
  assert.strictEqual(progressNotices[before + 1].settled, false);
  view.dispose();
  await tick();
  assert.strictEqual(
    progressNotices[before + 1].settled,
    true,
    'disposing the controller closes every open notice',
  );
  console.log('  ok - user cancel and dispose clear the notice');
}

// The sidebar webview going away (its container closed) means no
// askWaitingDone can ever arrive: the native notice closes with it.
async function testWebviewDisposalClearsNotice(SorcarSidebarView) {
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewHost(false);
  view.resolveWebviewView(wv.host, {}, {});
  const before = progressNotices.length;
  wv.fireMessage({type: 'askWaiting', tabId: 't3', question: 'Ship it?'});
  await tick();
  assert.strictEqual(progressNotices[before].settled, false);
  wv.disposeWebview();
  await tick();
  assert.strictEqual(
    progressNotices[before].settled,
    true,
    'disposing the webview closes its waiting notice',
  );
  view.dispose();
  console.log('  ok - webview disposal clears the notice');
}

async function testEditorPanelIsForceRevealed(SorcarSidebarView) {
  const events = [];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')), {
    rootTabId: 'root',
    onEvent: e => events.push(e),
  });
  const wv = makeWebviewHost(true);
  view.attachWebviewHost(wv.host, 'class="editor-tab-mode"');

  wv.fireMessage({type: 'askWaiting', tabId: 'root', question: 'Merge?'});
  await tick();
  assert.deepStrictEqual(
    events.filter(e => e.kind === 'reveal'),
    [{kind: 'reveal', force: true}],
    'editor-tabs mode reveals the panel even while a text editor is active',
  );
  assert.deepStrictEqual(wv.shows, [], 'the panel path does not call show()');
  view.dispose();
  console.log('  ok - editor panel: forced reveal');
}

async function runTests() {
  const SorcarSidebarView = loadView();
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-askwait-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];
  try {
    await testVisibleSidebarIsShownWithoutNativeNotice(SorcarSidebarView);
    await testHiddenSidebarRaisesNoticeUntilAnswered(SorcarSidebarView);
    await testUserCancelClearsNotice(SorcarSidebarView);
    await testWebviewDisposalClearsNotice(SorcarSidebarView);
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
