// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A task's outcome toast belongs to the task's chat, never to another one.
//
// The extension host turns the daemon's `worktree_result` events into
// webview toasts. Those events carry the task's tab id; the toasts used
// to be posted untagged, so task A's merge message popped over task B's
// chat.
//
// Editor-tabs mode runs ONE chat editor tab whose webview mirrors the
// whole registry, so the host no longer routes toasts between panels:
// the panel's poster posts every toast to that webview, tag intact, and
// media/main.js shows a tagged toast only over its own chat.
//
// This file pins down, end to end through the compiled extension and the
// real media/main.js:
//  - the host stamps each such toast with the originating tab id;
//  - every toast (tagged or window-level) reaches the panel webview, none
//    falls back to a native VS Code notification;
//  - a merge's progress toast carries its tab from open to close, and a
//    dropped submit warns tagged with the tab that submitted;
//  - the poster is released when the panel closes and installed again
//    when it reopens;
//  - in the webview a tagged toast shows over its own tab only.

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
    return {
      postMessage: () => {},
      getState: () => undefined,
      setState: () => {},
    };
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
  const {SorcarPanelManager} = require(
    path.join(OUT_DIR, 'SorcarPanelManager.js'),
  );
  const notifications = require(path.join(OUT_DIR, 'WebviewNotifications.js'));
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));

  // One chat panel holding two chats, A and B; B is the one on screen.
  // The webview reports each chat it shows (activeTabChanged), which
  // is how the host learns the tabs this surface holds.
  manager.open();
  assert.strictEqual(createdPanels.length, 1);
  const panel = createdPanels[0];
  const tabA = 'tab-a';
  const tabB = 'tab-b';
  await waitFor(
    () => serverSockets.length >= 1,
    'the panel controller must connect to the daemon',
  );
  panel._recv.fire({type: 'activeTabChanged', tabId: tabA});
  panel._recv.fire({type: 'activeTabChanged', tabId: tabB});

  // --- task A's merge toast reaches the panel, tagged with A's tab --------
  // (The agent's commit_and_merge tool reports through worktree_result
  // while the task still runs; the event carries the task's tab.)
  await daemonBroadcast({
    type: 'worktree_result',
    action: 'merge',
    branch: 'kiss/wt-a',
    success: true,
    message: 'Merged task A: TOAST_A_COMMIT',
    tabId: tabA,
  });
  let toasts = toastsOf(panel);
  assert.strictEqual(toasts.length, 1, 'the panel must get exactly one toast');
  assert.strictEqual(toasts[0].message, 'Merged task A: TOAST_A_COMMIT');
  assert.strictEqual(toasts[0].tabId, tabA, 'the toast names its tab');

  // --- a failed worktree_result follows the same rule ---------------------
  await daemonBroadcast({
    type: 'worktree_result',
    success: false,
    message: 'Merge of task A failed: TOAST_A_MERGE',
    tabId: tabA,
  });
  toasts = toastsOf(panel);
  assert.deepStrictEqual(
    toasts.map(t => [t.message, t.tabId, t.severity]),
    [
      ['Merged task A: TOAST_A_COMMIT', tabA, 'info'],
      ['Merge of task A failed: TOAST_A_MERGE', tabA, 'error'],
    ],
  );

  // --- B's own outcome reaches the panel tagged with B --------------------
  await daemonBroadcast({
    type: 'worktree_result',
    success: false,
    message: 'Merge of task B failed: TOAST_B_MERGE',
    tabId: tabB,
  });
  toasts = toastsOf(panel);
  assert.strictEqual(toasts.length, 3);
  assert.strictEqual(toasts[2].tabId, tabB);
  assert.strictEqual(toasts[2].severity, 'error');

  // --- a manual Git Commit is toasted by the daemon itself, not the host --
  await daemonBroadcast({
    type: 'autocommit_done',
    success: true,
    manual: true,
    message: 'manual commit',
    tabId: tabA,
  });
  assert.strictEqual(toastsOf(panel).length, 3);

  // --- an untagged, window-level toast reaches the panel untagged ---------
  notifications.showInformationNotification('TOAST_WINDOW_LEVEL');
  const windowToast = toastsOf(panel).find(
    t => t.message === 'TOAST_WINDOW_LEVEL',
  );
  assert.ok(windowToast, 'a window-level toast goes to the panel');
  assert.strictEqual(windowToast.tabId, undefined);

  // --- a toast for a tab the webview does not show is still posted -------
  // The host does no per-tab routing any more: the webview holds the
  // whole registry and gates the toast by its tag (see the webview part
  // below). Nothing falls back to a native notification.
  notifications.showErrorNotification('TOAST_OTHER_TAB', {tabId: 'tab-gone'});
  const otherToast = toastsOf(panel).find(t => t.message === 'TOAST_OTHER_TAB');
  assert.ok(otherToast, 'a tagged toast is posted to the one panel');
  assert.strictEqual(otherToast.tabId, 'tab-gone', 'the tag is kept');
  assert.strictEqual(
    nativeNotificationCount,
    0,
    'no toast may fall back to a native VS Code notification',
  );

  // --- a merge's progress toast carries A's tab from open to close -------
  // The webview asks for the merge of A's worktree while B is on screen.
  panel._recv.fire({type: 'worktreeAction', action: 'merge', tabId: tabA});
  await waitFor(
    () => toastsOf(panel).some(t => t.progress && t.tabId === tabA),
    'the merge progress toast must open tagged with A',
  );
  const progressToast = toastsOf(panel).find(t => t.progress);
  await daemonBroadcast({
    type: 'worktree_progress',
    message: 'TOAST_A_PROGRESS',
    tabId: tabA,
  });
  assert.ok(
    toastsOf(panel).some(
      t =>
        t.id === progressToast.id &&
        t.tabId === tabA &&
        t.progressMessage === 'TOAST_A_PROGRESS',
    ),
    'a progress update must reach the panel, tagged with A',
  );
  await daemonBroadcast({
    type: 'worktree_result',
    success: true,
    message: 'Merged: TOAST_A_MERGED',
    tabId: tabA,
  });
  await waitFor(
    () =>
      toastsOf(panel).some(
        t => t.id === progressToast.id && t.close && t.tabId === tabA,
      ),
    'the terminal result must close the progress toast, tagged with A',
  );
  assert.ok(
    !toastsOf(panel).some(t => (t.progress || t.close) && t.tabId !== tabA),
    "every post of A's merge progress carries A's tab",
  );

  // --- a dropped submit warns tagged with the tab that submitted ---------
  manager
    .controller()
    ._handleDroppedCommand(
      {type: 'submit', tabId: tabA, prompt: 'x'},
      'expired',
    );
  const droppedToast = toastsOf(panel).find(
    t => t.severity === 'warning' && /not started/.test(t.message || ''),
  );
  assert.ok(droppedToast, 'the dropped-submit warning goes to the panel');
  assert.strictEqual(droppedToast.tabId, tabA);

  // --- the poster is released when the panel closes ----------------------
  const postedBeforeClose = panel._posted.length;
  panel.dispose();
  assert.strictEqual(manager.controller(), undefined);
  notifications.showInformationNotification('TOAST_AFTER_CLOSE');
  assert.strictEqual(
    nativeNotificationCount,
    1,
    'with no panel the toast falls back to a native notification',
  );
  assert.strictEqual(
    panel._posted.length,
    postedBeforeClose,
    'nothing is posted to a closed panel',
  );

  // --- ... and installed again when the panel reopens --------------------
  manager.open();
  assert.strictEqual(createdPanels.length, 2);
  const reopened = createdPanels[1];
  notifications.showInformationNotification('TOAST_AFTER_REOPEN');
  assert.ok(
    toastsOf(reopened).some(t => t.message === 'TOAST_AFTER_REOPEN'),
    'the reopened panel receives the toasts',
  );
  assert.strictEqual(nativeNotificationCount, 1);

  // --- the webview honours the tag: the toast shows over its own tab only -
  // The chat webview (sidebar view or editor tab alike) holds both tabs;
  // the host posts the tagged toast there and media/main.js decides
  // whether to show it.
  const win = makeDomWebview();
  const api = win._testApi;
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  assert.notStrictEqual(first, second);
  const hostToast = {...toasts[0], tabId: first};
  send(win, hostToast);
  assert.ok(
    !toastText(win).includes('TOAST_A_COMMIT'),
    "a background tab's commit toast must not show over the tab on screen",
  );
  // A toast for a tab this webview does not hold at all shows nowhere.
  send(win, {...otherToast, id: otherToast.id + '-webview'});
  assert.ok(
    !toastText(win).includes('TOAST_OTHER_TAB'),
    'a toast tagged with an unknown tab is dropped by the webview',
  );
  // Chats have no tab row of their own any more: switch as a Chats-panel
  // pick would.
  assert.ok(
    api.openTabs().some(t => t.id === first),
    'the first chat must still be open',
  );
  api.switchToTab(first);
  assert.strictEqual(api.getActiveTabId(), first);
  send(win, {...hostToast, id: hostToast.id + '-again'});
  assert.ok(
    toastText(win).includes('TOAST_A_COMMIT'),
    'the same toast shows once its own tab is on screen',
  );
  // A window-level toast shows whichever tab is on screen.
  send(win, {...windowToast, id: windowToast.id + '-webview'});
  assert.ok(
    toastText(win).includes('TOAST_WINDOW_LEVEL'),
    'an untagged toast is window-level and always shows',
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
  api.switchToTab(second);
  assert.strictEqual(api.getActiveTabId(), second);
  send(win, {
    type: 'notification',
    id: 'progress-first',
    tabId: first,
    close: true,
  });
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
