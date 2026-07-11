// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: discarding a worktree branch must be
// SILENT — no notification toast in the chat webview and no printed
// line in the chat transcript.
//
// Bug reproduced: when the user clicked "Discard" on the worktree
// action bar the extension (SorcarSidebarView) opened a
// "Discarding worktree…" progress toast, and when the daemon
// broadcast the successful ``worktree_result`` with the message
// ``Discarded branch '<name>'.`` it unconditionally called
// ``showInformationNotification`` — popping pointless toasts over
// the chat webview for an action whose visible effect (the worktree
// bar disappearing) is already obvious.
//
// This test drives the real compiled SorcarSidebarView + AgentClient
// against a real Unix-domain-socket daemon stub (only the `vscode`
// module is stubbed), plus the real media/main.js chat webview in a
// JSDOM.  It asserts:
//   1. The REAL discard click path (webview worktreeAction command +
//      daemon worktree_result) produces NO notification of any kind —
//      no progress toast, no result toast, no native VS Code
//      notification.
//   2. The merge click path still shows a progress toast and a result
//      notification, and closes the toast on completion (regression
//      guard).
//   3. A discard result carrying a warning still produces a
//      notification (the user must see warnings).
//   4. A FAILED discard still produces an error notification.
//   5. The chat webview (main.js) does not append a transcript line
//      for a plain successful discard — neither on the active tab nor
//      buffered into a background tab's restored transcript — while
//      merge results, warning-carrying discards, and failures are
//      still printed.

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

class StubCancellationTokenSource {
  constructor() {
    this.token = {onCancellationRequested: () => ({dispose: () => {}})};
  }
  dispose() {}
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
// that run in parallel; never write or delete it here.
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wt-discard-'));
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
  return new Promise(r => setTimeout(r, 120));
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

function notificationsPosted(posted) {
  return posted.filter(
    m => m.type === 'notification' && !m.close && !m.progress,
  );
}

function progressPosted(posted) {
  return posted.filter(m => m.type === 'notification' && m.progress === true);
}

async function testExtensionSide() {
  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath}`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wt-discard-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  const TAB = 'tab-discard-silent';

  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: TAB, restoredTabs: []});
  await waitForClient();

  // 1. The REAL discard click path: webview posts a worktreeAction
  //    discard command, then the daemon replies with the result.
  //    Neither step may open any notification — not even a
  //    "Discarding worktree…" progress toast.
  wv.fireMessage({type: 'worktreeAction', action: 'discard', tabId: TAB});
  await new Promise(r => setTimeout(r, 120));
  assert.strictEqual(
    progressPosted(wv.posted).length,
    0,
    'clicking Discard must NOT open a progress notification',
  );
  assert.strictEqual(
    notificationsPosted(wv.posted).length,
    0,
    'clicking Discard must NOT open any notification',
  );
  await daemonSend({
    type: 'worktree_result',
    success: true,
    message: "Discarded branch 'kiss/wt-20260101-120000'.",
    tabId: TAB,
  });
  assert.strictEqual(
    progressPosted(wv.posted).length,
    0,
    'a plain successful discard must NOT show any progress notification',
  );
  assert.strictEqual(
    notificationsPosted(wv.posted).length,
    0,
    'a plain successful discard must NOT show a chat-webview notification',
  );
  assert.strictEqual(
    nativeNotificationCount,
    0,
    'a plain successful discard must NOT show a native VS Code notification',
  );

  // 2. The REAL merge click path → progress toast AND result
  //    notification still shown (regression guard).
  wv.fireMessage({type: 'worktreeAction', action: 'merge', tabId: TAB});
  await new Promise(r => setTimeout(r, 120));
  assert.strictEqual(
    progressPosted(wv.posted).length,
    1,
    'clicking Merge must still open a progress notification',
  );
  await daemonSend({
    type: 'worktree_result',
    success: true,
    message: "Successfully merged branch 'kiss/wt-1' into 'main'.",
    tabId: TAB,
  });
  let toasts = notificationsPosted(wv.posted);
  assert.strictEqual(
    toasts.length,
    1,
    'a successful merge must still show a notification',
  );
  assert.strictEqual(toasts[0].severity, 'info');
  assert.ok(
    wv.posted.some(m => m.type === 'notification' && m.close === true),
    'the merge progress toast must be closed when the result arrives',
  );

  // 3. Discard with a warning → notification still shown so the user
  //    sees the warning.
  await daemonSend({
    type: 'worktree_result',
    success: true,
    message:
      "Discarded branch 'kiss/wt-20260101-120000'.\n" +
      "⚠️  Could not checkout 'main': dirty tree",
    tabId: TAB,
  });
  toasts = notificationsPosted(wv.posted);
  assert.strictEqual(
    toasts.length,
    2,
    'a discard result carrying a warning must still show a notification',
  );

  // 3b. Partial discard (branch could not be deleted) → still shown.
  await daemonSend({
    type: 'worktree_result',
    success: true,
    message:
      "Partially discarded branch 'kiss/wt-20260101-120000'.\n" +
      "⚠️  Branch 'kiss/wt-20260101-120000' could not be deleted",
    tabId: TAB,
  });
  toasts = notificationsPosted(wv.posted);
  assert.strictEqual(
    toasts.length,
    3,
    'a partial-discard result must still show a notification',
  );

  // 4. Failed discard → error notification still shown.
  await daemonSend({
    type: 'worktree_result',
    success: false,
    message: 'No pending worktree changes to act on',
    tabId: TAB,
  });
  toasts = notificationsPosted(wv.posted);
  assert.strictEqual(
    toasts.length,
    4,
    'a failed worktree action must still show an error notification',
  );
  assert.strictEqual(toasts[3].severity, 'error');
  assert.strictEqual(
    nativeNotificationCount,
    0,
    'native VS Code notification APIs must never fire while webview is active',
  );

  view.dispose();
}

async function testWebviewSide() {
  const domWebview = makeDomWebview();
  try {
    const win = domWebview.win;
    const ready = domWebview.posted.find(m => m.type === 'ready');
    assert.ok(ready && ready.tabId, 'DOM webview must post ready with tabId');
    const activeTab = ready.tabId;

    // Active tab: plain successful discard → nothing printed in chat.
    send(win, {
      type: 'worktree_result',
      success: true,
      message: "Discarded branch 'kiss/wt-20260101-120000'.",
      tabId: activeTab,
    });
    assert.strictEqual(
      win.document.querySelectorAll('.wt-result-ok, .wt-result-err').length,
      0,
      'a plain successful discard must not print a line in the chat webview',
    );

    // Active tab: merge result → printed (regression guard).
    send(win, {
      type: 'worktree_result',
      success: true,
      message: "Successfully merged branch 'kiss/wt-1' into 'main'.",
      tabId: activeTab,
    });
    assert.strictEqual(
      win.document.querySelectorAll('.wt-result-ok').length,
      1,
      'a merge result must still be printed in the chat webview',
    );

    // Active tab: discard with warning → printed so the warning shows.
    send(win, {
      type: 'worktree_result',
      success: true,
      message:
        "Discarded branch 'kiss/wt-2'.\n⚠️  Could not checkout 'main': dirty",
      tabId: activeTab,
    });
    assert.strictEqual(
      win.document.querySelectorAll('.wt-result-ok').length,
      2,
      'a discard result with a warning must still be printed',
    );

    // Background tab: make the current tab a real background tab by
    // opening a second tab, deliver a plain silent discard plus a
    // failed discard for the backgrounded tab, then switch back and
    // inspect the restored transcript.
    const api = win._demoApi;
    assert.ok(api, '_demoApi must be exposed by main.js');
    const tab1 = api.getActiveTabId();
    api.createNewTab();
    const tab2 = api.getActiveTabId();
    assert.ok(tab2 && tab2 !== tab1, 'a fresh second tab must be active');
    send(win, {
      type: 'worktree_result',
      success: true,
      message: "Discarded branch 'kiss/wt-bg'.",
      tabId: tab1,
    });
    send(win, {
      type: 'worktree_result',
      success: false,
      message: 'discard failed: worktree locked BGERR1',
      tabId: tab1,
    });
    // Neither event may leak into the ACTIVE (tab2) transcript.
    const activeText = win.document.getElementById('output').textContent;
    assert.ok(
      !activeText.includes("Discarded branch 'kiss/wt-bg'.") &&
        !activeText.includes('BGERR1'),
      'background-tab worktree results must not render in the active tab',
    );
    // Switch back to tab1 (the real user gesture).
    const tabEl = win.document.querySelector(
      '.chat-tab[data-tab-id="' + tab1 + '"]',
    );
    assert.ok(tabEl, 'tab1 element must exist in the tab bar');
    tabEl.click();
    assert.strictEqual(api.getActiveTabId(), tab1, 'tab1 must be active now');
    const restored = win.document.getElementById('output').textContent;
    assert.ok(
      !restored.includes("Discarded branch 'kiss/wt-bg'."),
      'a silent discard must not be buffered into a background tab transcript',
    );
    assert.ok(
      restored.includes('BGERR1'),
      'a FAILED discard for a background tab must still be buffered and shown',
    );
  } finally {
    domWebview.close();
  }
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );
  await testExtensionSide();
  await testWebviewSide();
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
    cleanup();
    console.error(err);
    process.exit(1);
  },
);
