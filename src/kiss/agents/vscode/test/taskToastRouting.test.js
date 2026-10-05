// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A task's outcome toast belongs to the task's chat, never to another one.
//
// The extension host turns the daemon's `autocommit_done`, `worktree_result`
// and `main_tree_result` events into webview toasts. Those events carry the
// task's tab id, but the toasts used to be posted untagged to whichever chat
// panel was active, so task A's commit message popped over task B's chat.
//
// This file pins down, end to end through the compiled extension and the
// real media/main.js:
//  - the host stamps each such toast with the originating tab id;
//  - in editor-tabs mode the toast is posted to the panel that owns the
//    tab, not the active panel, and is dropped when no panel owns it;
//  - a window-level (untagged) toast still goes to the active panel;
//  - in the sidebar webview the tagged toast shows over its own tab only.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const vm = require('vm');
const Module = require('module');
const {JSDOM} = require('jsdom');
const {createFakeDaemon} = require('./fakeDaemon');

const EXT_ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(EXT_ROOT, 'out');
const MEDIA = path.join(EXT_ROOT, 'media');
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
  const posted = [];
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
      postMessage: msg => {
        posted.push(msg);
        return Promise.resolve(true);
      },
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
    _posted: posted,
    _recv: recvEmitter,
    _viewState: viewStateEmitter,
  };
  return panel;
}

let nativeNotificationCount = 0;
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
  CancellationTokenSource: class {
    constructor() {
      this.token = {onCancellationRequested: () => ({dispose: () => {}})};
    }
    dispose() {}
  },
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
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-task-toast-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const endpointPath = path.join(tmpHome, '.kiss', 'sorcar-local.json');

const serverSockets = [];
const server = createFakeDaemon(sock => {
  serverSockets.push(sock);
  sock.on('data', () => {});
  sock.on('error', () => {});
});

function daemonBroadcast(msg) {
  const line = JSON.stringify(msg) + '\n';
  for (const sock of serverSockets) {
    if (!sock.destroyed) sock.write(line);
  }
  return new Promise(r => setTimeout(r, 120));
}

async function waitFor(predicate, message) {
  for (let i = 0; i < 150; i++) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

function tabIdOf(panel) {
  const m = /data-kiss-tab-id="([^"]+)"/.exec(panel.webview.html);
  assert.ok(m, 'panel html must carry data-kiss-tab-id');
  return m[1];
}

function toastsOf(panel) {
  return panel._posted.filter(m => m.type === 'notification');
}

function makeDomWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.acquireVsCodeApi = function () {
    return {postMessage: () => {}, getState: () => undefined, setState: () => {}};
  };
  for (const file of ['panelCopy.js', 'api.js', 'main.js']) {
    vm.runInContext(
      fs.readFileSync(path.join(MEDIA, file), 'utf8'),
      dom.getInternalVMContext(),
    );
  }
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function toastText(win) {
  const container = win.document.getElementById('kiss-notification-container');
  return container ? container.textContent : '';
}

async function runTest() {
  await new Promise((res, rej) =>
    server.listen(endpointPath, err => (err ? rej(err) : res())),
  );
  const {SorcarPanelManager} = require(path.join(OUT_DIR, 'SorcarPanelManager.js'));
  const notifications = require(path.join(OUT_DIR, 'WebviewNotifications.js'));
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));

  // Two chat panels for two different tasks; B is the one the user is on.
  manager.openNewChat();
  manager.openNewChat();
  assert.strictEqual(createdPanels.length, 2);
  const [panelA, panelB] = createdPanels;
  const tabA = tabIdOf(panelA);
  const tabB = tabIdOf(panelB);
  assert.notStrictEqual(tabA, tabB);
  await waitFor(
    () => serverSockets.length >= 2,
    'both panel controllers must connect to the daemon',
  );
  panelA.active = false;
  panelB._viewState.fire({webviewPanel: panelB});

  // --- task A's auto-commit toast lands in A's panel, not the active B ---
  await daemonBroadcast({
    type: 'autocommit_done',
    success: true,
    message: 'Committed task A: TOAST_A_COMMIT',
    tabId: tabA,
  });
  let aToasts = toastsOf(panelA);
  assert.strictEqual(aToasts.length, 1, 'A must get exactly one toast');
  assert.strictEqual(aToasts[0].message, 'Committed task A: TOAST_A_COMMIT');
  assert.strictEqual(aToasts[0].tabId, tabA, 'the toast names its tab');
  assert.strictEqual(
    toastsOf(panelB).length,
    0,
    "task A's commit outcome must not be posted to task B's chat panel",
  );

  // --- worktree_result and main_tree_result follow the same rule ---------
  await daemonBroadcast({
    type: 'worktree_result',
    success: false,
    message: 'Merge of task A failed: TOAST_A_MERGE',
    tabId: tabA,
  });
  await daemonBroadcast({
    type: 'main_tree_result',
    success: true,
    message: 'Main tree of task A: TOAST_A_MAIN',
    tabId: tabA,
  });
  aToasts = toastsOf(panelA);
  assert.deepStrictEqual(
    aToasts.map(t => [t.message, t.tabId, t.severity]),
    [
      ['Committed task A: TOAST_A_COMMIT', tabA, 'info'],
      ['Merge of task A failed: TOAST_A_MERGE', tabA, 'error'],
      ['Main tree of task A: TOAST_A_MAIN', tabA, 'info'],
    ],
  );
  assert.strictEqual(toastsOf(panelB).length, 0);

  // --- B's own outcome reaches B --------------------------------------------
  await daemonBroadcast({
    type: 'autocommit_done',
    success: false,
    message: 'Commit of task B failed: TOAST_B_COMMIT',
    tabId: tabB,
  });
  const bToasts = toastsOf(panelB);
  assert.strictEqual(bToasts.length, 1);
  assert.strictEqual(bToasts[0].tabId, tabB);
  assert.strictEqual(bToasts[0].severity, 'error');
  assert.strictEqual(toastsOf(panelA).length, 3, 'A gets nothing of B');

  // --- a manual commit is toasted by the daemon itself, not the host ------
  await daemonBroadcast({
    type: 'autocommit_done',
    success: true,
    manual: true,
    message: 'manual commit',
    tabId: tabA,
  });
  assert.strictEqual(toastsOf(panelA).length, 3);

  // --- an untagged, window-level toast still goes to the active panel ------
  notifications.showInformationNotification('TOAST_WINDOW_LEVEL');
  const windowToast = toastsOf(panelB).find(
    t => t.message === 'TOAST_WINDOW_LEVEL',
  );
  assert.ok(windowToast, 'a window-level toast goes to the active panel');
  assert.strictEqual(windowToast.tabId, undefined);
  assert.ok(!toastsOf(panelA).some(t => t.message === 'TOAST_WINDOW_LEVEL'));

  // --- a toast for a tab no panel holds is dropped, not shown elsewhere ----
  notifications.showErrorNotification('TOAST_ORPHAN', {tabId: 'tab-gone'});
  assert.ok(!toastsOf(panelA).some(t => t.message === 'TOAST_ORPHAN'));
  assert.ok(!toastsOf(panelB).some(t => t.message === 'TOAST_ORPHAN'));
  assert.strictEqual(
    nativeNotificationCount,
    0,
    'no toast may fall back to a native VS Code notification',
  );

  // --- a merge's progress toast lives in A's panel from open to close ---
  // The webview of panel A asks for the merge; B stays the active panel.
  panelA._recv.fire({type: 'worktreeAction', action: 'merge', tabId: tabA});
  await waitFor(
    () => toastsOf(panelA).some(t => t.progress && t.tabId === tabA),
    "the merge progress toast must open in A's panel",
  );
  const progressToast = toastsOf(panelA).find(t => t.progress);
  await daemonBroadcast({
    type: 'worktree_progress',
    message: 'TOAST_A_PROGRESS',
    tabId: tabA,
  });
  assert.ok(
    toastsOf(panelA).some(
      t =>
        t.id === progressToast.id &&
        t.tabId === tabA &&
        t.progressMessage === 'TOAST_A_PROGRESS',
    ),
    "a progress update must reach A's panel, tagged with A",
  );
  await daemonBroadcast({
    type: 'worktree_result',
    success: true,
    message: 'Merged: TOAST_A_MERGED',
    tabId: tabA,
  });
  await waitFor(
    () =>
      toastsOf(panelA).some(
        t => t.id === progressToast.id && t.close && t.tabId === tabA,
      ),
    "the terminal result must close the progress toast in A's panel",
  );
  assert.ok(
    !toastsOf(panelB).some(t => t.progress || t.close),
    "nothing of A's merge progress may be posted to B's panel",
  );

  // --- a dropped submit warns the tab that submitted ---------------------
  const controllerA = manager._panels.get(tabA).controller;
  controllerA._handleDroppedCommand(
    {type: 'submit', tabId: tabA, prompt: 'x'},
    'expired',
  );
  const droppedToast = toastsOf(panelA).find(
    t => t.severity === 'warning' && /not started/.test(t.message || ''),
  );
  assert.ok(droppedToast, "the dropped-submit warning goes to A's panel");
  assert.strictEqual(droppedToast.tabId, tabA);
  assert.ok(
    !toastsOf(panelB).some(t => /not started/.test(t.message || '')),
    "a dropped submit of A must not warn in B's chat",
  );

  // --- the webview honours the tag: the toast shows over its own tab only -
  // The sidebar runs ONE webview holding both tabs; the host posts the
  // tagged toast there and media/main.js decides whether to show it.
  const win = makeDomWebview();
  const api = win._testApi;
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  assert.notStrictEqual(first, second);
  const hostToast = {...aToasts[0], tabId: first};
  send(win, hostToast);
  assert.ok(
    !toastText(win).includes('TOAST_A_COMMIT'),
    "a background tab's commit toast must not show over the tab on screen",
  );
  const firstTabEl = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(first)}]`,
  );
  assert.ok(firstTabEl, 'the first tab must be in the tab bar');
  firstTabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(api.getActiveTabId(), first);
  send(win, {...hostToast, id: hostToast.id + '-again'});
  assert.ok(
    toastText(win).includes('TOAST_A_COMMIT'),
    'the same toast shows once its own tab is on screen',
  );

  // A sticky progress toast opened while its tab was on screen must still
  // close after the user switches tabs: the close is tagged with the
  // (now background) tab and shows nothing, so it is never gated.
  send(win, {
    type: 'notification',
    id: 'progress-first',
    tabId: first,
    severity: 'info',
    message: 'TOAST_FIRST_PROGRESS',
    progress: true,
    sticky: true,
  });
  assert.ok(toastText(win).includes('TOAST_FIRST_PROGRESS'));
  const secondTabEl = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(second)}]`,
  );
  secondTabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(api.getActiveTabId(), second);
  send(win, {type: 'notification', id: 'progress-first', tabId: first, close: true});
  assert.ok(
    !toastText(win).includes('TOAST_FIRST_PROGRESS'),
    'a tagged close must retire the toast even while another tab is active',
  );
  win.close();

  manager.dispose();
}

runTest()
  .then(() => {
    console.log('taskToastRouting: OK');
    server.close();
    for (const s of serverSockets) s.destroy();
    fs.rmSync(tmpHome, {recursive: true, force: true});
    process.exit(0);
  })
  .catch(err => {
    console.error(err);
    server.close();
    for (const s of serverSockets) s.destroy();
    process.exit(1);
  });
