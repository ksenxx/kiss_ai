// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for KISS Sorcar chat-webview notifications.
//
// Bug reproduced: extension-side events such as worktree_result used
// VS Code's native showInformationMessage/showErrorMessage APIs, so the
// user saw notifications in VS Code's workbench notification area rather
// than at the top-right corner of the KISS chat webview.
//
// This test drives the real compiled SorcarSidebarView + AgentClient
// against a real Unix-domain socket daemon stub.  Only the `vscode`
// module is stubbed.  A daemon worktree_result event must be forwarded
// to the resolved webview as a `notification` message and must not call
// native VS Code notification APIs.

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const vm = require('vm');
const Module = require('module');
const {JSDOM} = require('jsdom');

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

let cancellationDisposeCount = 0;
class StubCancellationTokenSource {
  constructor() {
    this.token = {onCancellationRequested: () => ({dispose: () => {}})};
  }
  dispose() {
    cancellationDisposeCount++;
  }
}

function makeUri(fsPath) {
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

let nativeNotificationCount = 0;
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
  CancellationTokenSource: StubCancellationTokenSource,
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
    showInformationMessage: () => {
      nativeNotificationCount++;
      return Promise.resolve(undefined);
    },
    showWarningMessage: () => {
      nativeNotificationCount++;
      return Promise.resolve(undefined);
    },
    showErrorMessage: () => {
      nativeNotificationCount++;
      return Promise.resolve(undefined);
    },
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
// ``_vscode-stub.js`` is a git-tracked fixture shared by several tests
// that run in parallel; it already contains
// ``module.exports = global.__kissVscodeStub;`` — never write or delete
// it here or concurrent tests lose their ``vscode`` module mid-run.
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-webview-ntf-'));
const tmpDirs = [tmpHome];
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  for (const dir of tmpDirs.slice().reverse()) {
    fs.rmSync(dir, {recursive: true, force: true});
  }
  process.exit(0);
}

let lastServerSock = null;
const server = net.createServer(sock => {
  lastServerSock = sock;
  sock.on('data', () => {});
});

function daemonSend(msg) {
  assert.ok(lastServerSock, 'daemon has no connected client socket');
  lastServerSock.write(JSON.stringify(msg) + '\n');
  return new Promise(r => setTimeout(r, 80));
}

async function waitForClient() {
  for (let i = 0; i < 100 && !lastServerSock; i++) {
    await new Promise(r => setTimeout(r, 20));
  }
  assert.ok(lastServerSock, 'client never connected to daemon');
}

function makeWebviewView() {
  const recv = new StubEventEmitter();
  const posted = [];
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => recv.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };
  return {webviewView, posted, fireMessage: m => recv.fire(m)};
}

function makeDomWebview() {
  const mediaDir = path.join(__dirname, '..', 'media');
  let html = fs.readFileSync(path.join(mediaDir, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => undefined,
      setState: () => {},
    };
  };
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'panelCopy.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  vm.runInContext(
    fs.readFileSync(path.join(mediaDir, 'main.js'), 'utf8'),
    dom.getInternalVMContext(),
  );
  return {win, posted, close: () => win.close()};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(el) {
  el.dispatchEvent(new el.ownerDocument.defaultView.MouseEvent('click', {bubbles: true}));
}

async function waitFor(predicate, message) {
  for (let i = 0; i < 100; i++) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(fs.existsSync(sourcePath), `compiled extension missing: ${sourcePath}`);
  const notificationPath = path.join(
    __dirname,
    '..',
    'out',
    'WebviewNotifications.js',
  );
  assert.ok(
    fs.existsSync(notificationPath),
    `compiled notification helper missing: ${notificationPath}`,
  );
  delete require.cache[require.resolve(notificationPath)];
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);
  const notificationsApi = require(notificationPath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-webview-ntf-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  const TAB = 'tab-notify';

  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: TAB, restoredTabs: []});
  await waitForClient();

  await daemonSend({
    type: 'worktree_result',
    success: true,
    message: 'Worktree merged successfully.',
    tabId: TAB,
  });

  const notifications = wv.posted.filter(m => m.type === 'notification');
  assert.strictEqual(
    notifications.length,
    1,
    'worktree_result should render through chat-webview notification system',
  );
  assert.strictEqual(notifications[0].severity, 'info');
  assert.strictEqual(notifications[0].message, 'Worktree merged successfully.');
  assert.strictEqual(
    nativeNotificationCount,
    0,
    'native VS Code notification APIs must not be used when the webview is active',
  );

  const domWebview = makeDomWebview();
  try {
    const ready = domWebview.posted.find(m => m.type === 'ready');
    assert.ok(ready && ready.tabId, 'DOM webview must post ready with tabId');
    send(domWebview.win, {
      type: 'notification',
      id: 'choose-1',
      severity: 'warning',
      message: 'Choose an API key action.',
      actions: ['Enter Key', 'Skip'],
      sticky: true,
    });
    let toast = await waitFor(
      () => domWebview.win.document.querySelector('.kiss-notification'),
      'notification toast was not rendered in the DOM webview',
    );
    assert.ok(toast, 'notification toast should exist');
    const closeButton = toast.querySelector('.kiss-notification-close');
    assert.ok(closeButton, 'notification close button should exist');
    click(closeButton);
    await waitFor(
      () => domWebview.posted.some(m => m.type === 'notificationAction' && m.id === 'choose-1'),
      'closing an action notification must notify extension so the promise resolves undefined',
    );
    const dismissed = domWebview.posted.find(
      m => m.type === 'notificationAction' && m.id === 'choose-1',
    );
    assert.strictEqual(
      Object.prototype.hasOwnProperty.call(dismissed, 'action'),
      true,
      'dismissal message should explicitly carry action: undefined',
    );
    assert.strictEqual(dismissed.action, undefined);

    send(domWebview.win, {
      type: 'notification',
      id: 'choose-2',
      severity: 'info',
      message: 'Choose another action.',
      actions: ['Apply'],
      sticky: true,
    });
    toast = await waitFor(
      () => domWebview.win.document.querySelector(
        '.kiss-notification[data-notification-id="choose-2"]',
      ),
      'second notification toast was not rendered',
    );
    const actionButton = toast.querySelector('.kiss-notification-action');
    assert.ok(actionButton, 'notification action button should exist');
    click(actionButton);
    await waitFor(
      () => domWebview.posted.some(
        m => m.type === 'notificationAction' && m.id === 'choose-2' && m.action === 'Apply',
      ),
      'clicking an action notification button must notify extension with the selected action',
    );
  } finally {
    domWebview.close();
  }

  const pendingActionPromise = notificationsApi.showWarningNotification(
    'Choose a pending action.',
    'Continue',
  );
  const pendingBeforeDispose = await Promise.race([
    pendingActionPromise.then(value => ({settled: true, value})),
    new Promise(resolve => setTimeout(() => resolve({settled: false}), 80)),
  ]);
  assert.deepStrictEqual(
    pendingBeforeDispose,
    {settled: false},
    'action notification promise should remain pending before webview disposal',
  );
  view.dispose();
  const pendingAfterDispose = await Promise.race([
    pendingActionPromise.then(value => ({settled: true, value})),
    new Promise(resolve => setTimeout(() => resolve({settled: false}), 200)),
  ]);
  assert.deepStrictEqual(
    pendingAfterDispose,
    {settled: true, value: undefined},
    'disposing the active webview must resolve pending action notifications undefined',
  );

  const progressPosts = [];
  notificationsApi.setWebviewNotificationPoster(message => {
    progressPosts.push(message);
  });
  cancellationDisposeCount = 0;
  const thrown = new Error('synchronous progress failure');
  let rejected;
  try {
    await notificationsApi.withWebviewNotificationProgress(
      {
        location: vscodeStub.ProgressLocation.Notification,
        title: 'Sync failing progress',
      },
      () => {
        throw thrown;
      },
    );
  } catch (err) {
    rejected = err;
  }
  assert.strictEqual(
    rejected,
    thrown,
    'sync progress task errors must still reject with the original error',
  );
  assert.strictEqual(
    cancellationDisposeCount,
    1,
    'sync progress task errors must dispose the cancellation source',
  );
  assert.ok(
    progressPosts.some(m => m.type === 'notification' && m.progress === true),
    'sync progress test must post the progress notification before the failure',
  );
  assert.ok(
    progressPosts.some(m => m.type === 'notification' && m.close === true),
    'sync progress task errors must close the webview progress notification',
  );
  notificationsApi.setWebviewNotificationPoster(undefined);

  // Bug C: replacing the poster (e.g. webview re-resolved while the old
  // webview's onDidDispose has not yet fired) must resolve pending action
  // promises tied to the OLD poster — otherwise their notifications can
  // never be acknowledged (the new webview does not know those IDs).
  const postedA = [];
  notificationsApi.setWebviewNotificationPoster(message => postedA.push(message));
  const oldPosterPending = notificationsApi.showWarningNotification(
    'Pending from old webview poster.',
    'Continue',
  );
  const oldPosterBeforeSwap = await Promise.race([
    oldPosterPending.then(value => ({settled: true, value})),
    new Promise(resolve => setTimeout(() => resolve({settled: false}), 80)),
  ]);
  assert.deepStrictEqual(
    oldPosterBeforeSwap,
    {settled: false},
    'pending action should remain pending while old poster is active',
  );
  // Now swap to a NEW poster WITHOUT clearing first.  This mirrors the
  // VS Code ordering where a fresh webview is resolved BEFORE the stale
  // old webview's onDidDispose fires.
  const postedB = [];
  notificationsApi.setWebviewNotificationPoster(message => postedB.push(message));
  const oldPosterAfterSwap = await Promise.race([
    oldPosterPending.then(value => ({settled: true, value})),
    new Promise(resolve => setTimeout(() => resolve({settled: false}), 200)),
  ]);
  assert.deepStrictEqual(
    oldPosterAfterSwap,
    {settled: true, value: undefined},
    'replacing the poster must resolve pending action promises from the old poster as undefined',
  );
  notificationsApi.setWebviewNotificationPoster(undefined);
}

function cleanup() {
  try {
    if (lastServerSock) lastServerSock.destroy();
  } catch {}
  try {
    server.close();
  } catch {}
  try {
    fs.unlinkSync(sockPath);
  } catch {}
  for (const dir of tmpDirs.slice().reverse()) {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

runTests().then(
  () => {
    cleanup();
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    cleanup();
    process.exit(1);
  },
);
