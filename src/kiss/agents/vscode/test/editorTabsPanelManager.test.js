// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for editor-tabs mode's host side
// (out/SorcarPanelManager.js + out/SorcarSidebarView.js) against a
// REAL local-WSS daemon stub. The mode keeps ONE chat editor tab that
// mirrors the whole daemon registry, like the remote web app:
//  - open() creates one WebviewPanel (viewType kissSorcar.chatTab) whose
//    chat HTML has the editor-tab-mode body class and NO pinned tab id;
//    a second open() reveals that panel instead of creating another;
//  - the webview's `panelTitle {title, state}` retitles the editor tab:
//    the spinner icon while running, a ✅ / ❌ title prefix after;
//  - openChat({chatId, taskId, title}) posts `openChatFromHistory` to
//    the webview once it is `ready` and never opens a second panel;
//  - openChat({pendingText, autoSubmit}) posts `clearChat` and then
//    `insertAndSubmit` — at once when the webview is ready, else both
//    after its `ready`, clearChat first, and a later `ready` repeats
//    neither; without autoSubmit it posts `clearChat` and then
//    `appendToInput` (the text in the new chat's composer, unsent);
//  - a USER panel close and close() (mode switched off) both drop the
//    controller and send NO `closeTab` to the daemon;
//  - ensureOpen() opens a background panel only when no panel exists
//    and no chat editor tab (serialized placeholder) is listed in
//    vscode.window.tabGroups;
//  - the serializer re-adopts a revived panel (stripping a leftover
//    status prefix from its title) and disposes a revived husk when
//    the mode is off or a panel already exists;
//  - terminal dispose() leaves the panel standing for the serializer.

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

let editorTabsMode = true;
const createdPanels = [];
let registeredSerializer = null;

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
    reveals: 0,
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
    reveal: () => {
      panel.reveals += 1;
    },
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

const vscodeStub = {
  workspace: {
    workspaceFolders: [],
    getConfiguration: () => ({
      get: key => (key === 'editorTabsMode' ? editorTabsMode : ''),
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
    createWebviewPanel: (viewType, title, showOptions, _options) => {
      const panel = makeFakePanel(viewType, title);
      panel._showOptions = showOptions;
      createdPanels.push(panel);
      return panel;
    },
    registerWebviewPanelSerializer: (viewType, serializer) => {
      registeredSerializer = {viewType, serializer};
      return {dispose: () => {}};
    },
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showWarningMessage: () => {},
    showErrorMessage: () => {},
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-edtabs-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const endpointPath = path.join(tmpHome, '.kiss', 'sorcar-local.json');

const serverSockets = [];
const daemonCommands = [];
const server = createFakeDaemon(sock => {
  serverSockets.push(sock);
  let buf = '';
  sock.on('data', chunk => {
    buf += chunk.toString('utf8');
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (line.trim()) daemonCommands.push(JSON.parse(line));
    }
  });
  sock.on('error', () => {});
});

async function waitFor(predicate, message) {
  for (let i = 0; i < 150; i++) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

const {SorcarPanelManager, CHAT_PANEL_VIEW_TYPE} = require(
  path.join(OUT_DIR, 'SorcarPanelManager.js'),
);
const {PRODUCT_NAME} = require(path.join(OUT_DIR, 'brand.js'));

function iconName(panel) {
  assert.ok(panel.iconPath, 'panel must carry an icon');
  return path.basename(panel.iconPath.fsPath);
}

function assertNoCloseTab(why) {
  assert.ok(
    !daemonCommands.some(c => c.type === 'closeTab'),
    why || 'no closeTab must ever reach the daemon',
  );
}

function assertChatHtml(panel) {
  const html = panel.webview.html;
  assert.ok(
    html.includes('<body class="editor-tab-mode"'),
    'chat html must be in editor-tab mode',
  );
  assert.ok(
    !html.includes('data-kiss-tab-id'),
    'the single panel is pinned to no root tab id',
  );
}

/** A `tabGroups.all` entry whose one tab has *viewType* as its input. */
function tabGroupWith(viewType) {
  return {tabs: [{input: viewType === null ? null : {viewType}}]};
}

async function runTest() {
  server.listen(endpointPath);
  const manager = new SorcarPanelManager(vscodeStub.Uri.file(EXT_ROOT));
  vscodeStub._ignored = manager.registerSerializer();
  assert.strictEqual(registeredSerializer.viewType, CHAT_PANEL_VIEW_TYPE);
  assert.strictEqual(manager.controller(), undefined, 'no panel yet');

  // --- open() creates the one chat panel --------------------------------
  const controllerA = manager.open();
  assert.strictEqual(createdPanels.length, 1);
  const panelA = createdPanels[0];
  assert.strictEqual(panelA.viewType, CHAT_PANEL_VIEW_TYPE);
  assert.strictEqual(panelA.title, PRODUCT_NAME, 'initial title');
  assert.strictEqual(
    panelA._showOptions.preserveFocus,
    false,
    'a plain open takes the focus like any new editor',
  );
  assertChatHtml(panelA);
  assert.strictEqual(iconName(panelA), 'kiss-icon.svg');
  assert.strictEqual(manager.controller(), controllerA);
  await waitFor(
    () => serverSockets.length >= 1,
    'panel controller must connect to the daemon',
  );

  // --- a second open() reveals the same panel ----------------------------
  const controllerAgain = manager.open();
  assert.strictEqual(controllerAgain, controllerA, 'same controller');
  assert.strictEqual(createdPanels.length, 1, 'no second panel');
  assert.strictEqual(panelA.reveals, 1, 'the existing panel is revealed');

  // --- panelTitle retitles the editor tab --------------------------------
  panelA._recv.fire({type: 'panelTitle', title: '  fix the bug  ', state: ''});
  await waitFor(
    () => panelA.title === 'fix the bug',
    'panelTitle must retitle the panel (trimmed)',
  );
  assert.strictEqual(iconName(panelA), 'kiss-icon.svg');
  panelA._recv.fire({
    type: 'panelTitle',
    title: 'fix the bug',
    state: 'running',
  });
  await waitFor(
    () => iconName(panelA) === 'spinner-running.svg',
    'a running task turns the tab icon into the spinner',
  );
  assert.strictEqual(panelA.title, 'fix the bug', 'running adds no prefix');
  panelA._recv.fire({type: 'panelTitle', title: 'fix the bug', state: 'ok'});
  await waitFor(
    () => panelA.title === '\u2705 fix the bug',
    'a finished task prefixes the title with a tick',
  );
  assert.strictEqual(iconName(panelA), 'kiss-icon.svg', 'icon back to logo');
  panelA._recv.fire({type: 'panelTitle', title: 'fix the bug', state: 'fail'});
  await waitFor(
    () => panelA.title === '\u274C fix the bug',
    'a failed task prefixes the title with a cross',
  );
  panelA._recv.fire({type: 'panelTitle', title: '', state: ''});
  await waitFor(
    () => panelA.title === PRODUCT_NAME,
    'an empty title falls back to the product name',
  );

  // --- openChat({chatId}) waits for the webview's ready -----------------
  manager.openChat({chatId: 'chat-B', taskId: 7, title: 'resumed chat'});
  assert.strictEqual(createdPanels.length, 1, 'openChat opens no panel');
  await sleep(400);
  assert.ok(
    !panelA._posted.some(m => m.type === 'openChatFromHistory'),
    'nothing is posted before the webview is ready',
  );
  panelA._recv.fire({type: 'ready'});
  await waitFor(
    () => panelA._posted.some(m => m.type === 'openChatFromHistory'),
    'the chat is opened once the webview is ready',
  );
  const opened = panelA._posted.filter(m => m.type === 'openChatFromHistory');
  assert.deepStrictEqual(opened, [
    {
      type: 'openChatFromHistory',
      chatId: 'chat-B',
      taskId: 7,
      title: 'resumed chat',
      machine: '',
    },
  ]);
  assert.strictEqual(createdPanels.length, 1, 'still the one panel');
  await waitFor(
    () => daemonCommands.some(c => c.type === 'ready'),
    'the webview ready is forwarded to the daemon',
  );

  // --- a ready webview gets the history open at once ---------------------
  manager.openChat({chatId: 'chat-C'});
  await waitFor(
    () =>
      panelA._posted.some(
        m => m.type === 'openChatFromHistory' && m.chatId === 'chat-C',
      ),
    'a second history open reaches the ready webview',
  );
  const openedC = panelA._posted.find(
    m => m.type === 'openChatFromHistory' && m.chatId === 'chat-C',
  );
  assert.strictEqual(openedC.taskId, null, 'no task id -> null');
  assert.strictEqual(openedC.title, '', 'no title -> empty');

  // --- a draft WITHOUT autoSubmit: a new chat with the text in the
  // composer (a Recommendations row), nothing submitted ------------------
  manager.openChat({pendingText: 'only a draft'});
  await waitFor(
    () => panelA._posted.some(m => m.type === 'appendToInput'),
    'the draft reaches the new chat composer',
  );
  const draftClearIdx = panelA._posted.findIndex(m => m.type === 'clearChat');
  const appendIdx = panelA._posted.findIndex(m => m.type === 'appendToInput');
  assert.ok(draftClearIdx >= 0, 'a draft opens a new chat (clearChat)');
  assert.ok(appendIdx > draftClearIdx, 'the draft is appended after clearChat');
  assert.strictEqual(panelA._posted[appendIdx].text, 'only a draft');
  assert.strictEqual(
    panelA._posted[appendIdx].replace,
    true,
    'the prompt replaces the draft "+" carried over, not under it',
  );
  assert.ok(
    !panelA._posted.some(m => m.type === 'insertAndSubmit'),
    'nothing is submitted without autoSubmit',
  );
  assert.ok(
    !panelA._posted.some(
      m => m.type === 'openChatFromHistory' && m.chatId === '',
    ),
    'a draft is not a history open',
  );

  // --- autoSubmit on a ready webview: clearChat, then insertAndSubmit ---
  manager.openChat({pendingText: 'Connect my Slack app', autoSubmit: true});
  const clearIdx = panelA._posted.findIndex(m => m.type === 'clearChat');
  const submitIdx = panelA._posted.findIndex(m => m.type === 'insertAndSubmit');
  assert.ok(clearIdx >= 0, 'autoSubmit opens a new chat (clearChat)');
  assert.ok(submitIdx > clearIdx, 'the prompt is submitted after clearChat');
  assert.strictEqual(panelA._posted[submitIdx].text, 'Connect my Slack app');
  assert.strictEqual(createdPanels.length, 1, 'autoSubmit opens no panel');

  // --- the active-task relay follows the chat on screen -----------------
  const activeTasks = [];
  manager.setActiveTaskSink((chatId, taskId) =>
    activeTasks.push([chatId, taskId]),
  );
  assert.deepStrictEqual(activeTasks, [['', '']], 'pushed on set');
  panelA._recv.fire({type: 'activeTask', chatId: 'chat-B', taskId: 'task-1'});
  await waitFor(
    () => activeTasks.length === 2,
    'the panel activeTask reaches the sink',
  );
  assert.deepStrictEqual(activeTasks[1], ['chat-B', 'task-1']);

  // --- a USER close keeps the chats registered --------------------------
  panelA.dispose();
  assert.strictEqual(manager.controller(), undefined, 'controller dropped');
  assert.deepStrictEqual(
    activeTasks[activeTasks.length - 1],
    ['', ''],
    'the sink is told the panel is gone',
  );
  await sleep(150);
  assertNoCloseTab('a user panel close must NOT retire any chat tab');

  // --- open() after a close creates a fresh panel ------------------------
  manager.open({preserveFocus: true});
  assert.strictEqual(createdPanels.length, 2, 'a fresh panel B');
  const panelB = createdPanels[1];
  assert.strictEqual(
    panelB._showOptions.preserveFocus,
    true,
    'preserveFocus is passed through',
  );
  assertChatHtml(panelB);
  assert.strictEqual(panelB.title, PRODUCT_NAME, 'a fresh panel starts clean');
  assert.ok(manager.controller(), 'panel B has a controller');

  // --- autoSubmit before ready: both messages wait, clearChat first ----
  manager.openChat({pendingText: 'Connect GitHub', autoSubmit: true});
  assert.strictEqual(createdPanels.length, 2, 'the open reuses panel B');
  await sleep(100);
  assert.ok(
    !panelB._posted.some(
      m => m.type === 'clearChat' || m.type === 'insertAndSubmit',
    ),
    'nothing is posted into a webview that is not ready',
  );
  panelB._recv.fire({type: 'ready'});
  await waitFor(
    () => panelB._posted.some(m => m.type === 'insertAndSubmit'),
    'the prompt is submitted once the webview is ready',
  );
  const clearB = panelB._posted.findIndex(m => m.type === 'clearChat');
  const submitB = panelB._posted.findIndex(m => m.type === 'insertAndSubmit');
  assert.ok(clearB >= 0, 'the held clearChat is posted on ready');
  assert.ok(submitB > clearB, 'clearChat arrives before insertAndSubmit');
  assert.strictEqual(panelB._posted[submitB].text, 'Connect GitHub');
  // A later `ready` (webview reload, daemon reconnect) repeats neither.
  panelB._recv.fire({type: 'ready'});
  await sleep(50);
  assert.strictEqual(
    panelB._posted.filter(m => m.type === 'clearChat').length,
    1,
    'a second ready posts no clearChat',
  );
  assert.strictEqual(
    panelB._posted.filter(m => m.type === 'insertAndSubmit').length,
    1,
    'a second ready submits nothing',
  );

  // --- openSettings reveals the panel and opens its settings ------------
  await manager.openSettings();
  assert.ok(
    panelB._posted.some(m => m.type === 'openSettings'),
    'openSettings posts to the panel webview',
  );
  assert.strictEqual(createdPanels.length, 2);

  // --- close() (mode switched off) keeps the chats registered -----------
  manager.close();
  assert.ok(panelB.disposed, 'close() disposes the panel');
  assert.strictEqual(manager.controller(), undefined);
  await sleep(150);
  assertNoCloseTab('close() must NOT retire any chat tab');
  manager.close();
  assert.strictEqual(
    createdPanels.length,
    2,
    'close() without a panel is a no-op',
  );

  // --- ensureOpen(): background panel only when no chat tab exists ------
  editorTabsMode = false;
  manager.ensureOpen();
  assert.strictEqual(
    createdPanels.length,
    2,
    'mode off: ensureOpen is a no-op',
  );
  editorTabsMode = true;
  vscodeStub.window.tabGroups.all = [
    tabGroupWith('mainThreadWebview-' + CHAT_PANEL_VIEW_TYPE),
  ];
  manager.ensureOpen();
  assert.strictEqual(
    createdPanels.length,
    2,
    'a serialized chat tab placeholder means no new panel',
  );
  vscodeStub.window.tabGroups.all = [
    tabGroupWith('mainThreadWebview-other.view'),
    tabGroupWith(null),
    {tabs: []},
  ];
  manager.ensureOpen();
  assert.strictEqual(createdPanels.length, 3, 'no chat tab: a panel opens');
  const panelC = createdPanels[2];
  assert.strictEqual(
    panelC._showOptions.preserveFocus,
    true,
    'ensureOpen opens in the background',
  );
  assertChatHtml(panelC);
  manager.ensureOpen();
  assert.strictEqual(createdPanels.length, 3, 'a live panel: no second one');

  // --- a draft before ready: the held text keeps its replace flag ------
  manager.openChat({pendingText: 'held draft'});
  await sleep(50);
  assert.ok(
    !panelC._posted.some(m => m.type === 'appendToInput'),
    'nothing is posted into a webview that is not ready',
  );
  panelC._recv.fire({type: 'ready'});
  await waitFor(
    () => panelC._posted.some(m => m.type === 'appendToInput'),
    'the held draft reaches the composer on ready',
  );
  const heldC = panelC._posted.find(m => m.type === 'appendToInput');
  assert.strictEqual(heldC.text, 'held draft');
  assert.strictEqual(heldC.replace, true, 'the held draft still replaces');
  panelC._recv.fire({type: 'ready'});
  await sleep(50);
  assert.strictEqual(
    panelC._posted.filter(m => m.type === 'appendToInput').length,
    1,
    'a second ready repeats no draft',
  );

  vscodeStub.window.tabGroups.all = [];
  panelC.dispose();
  assert.strictEqual(manager.controller(), undefined);

  // --- serializer: a revived panel is re-adopted, prefix stripped ------
  const revived = makeFakePanel(CHAT_PANEL_VIEW_TYPE, '\u2705 deploy status');
  await registeredSerializer.serializer.deserializeWebviewPanel(revived, {});
  assert.ok(!revived.disposed, 'the revived panel stays');
  assert.strictEqual(
    revived.title,
    'deploy status',
    'the previous session status prefix is stripped',
  );
  assert.strictEqual(iconName(revived), 'kiss-icon.svg');
  assertChatHtml(revived);
  const revivedController = manager.controller();
  assert.ok(revivedController, 'the revived panel gets a controller');
  assert.strictEqual(
    manager.open(),
    revivedController,
    'open() reuses the revived panel',
  );
  assert.strictEqual(createdPanels.length, 3, 'no panel created next to it');
  assert.strictEqual(revived.reveals, 1, 'the revived panel is revealed');
  // The webview's own title is authoritative: a chat really titled
  // "✅ deploy status" keeps its tick (nothing strips it twice).
  revived._recv.fire({
    type: 'panelTitle',
    title: '\u2705 deploy status',
    state: 'ok',
  });
  await waitFor(
    () => revived.title === '\u2705 \u2705 deploy status',
    'only the serializer strips a prefix',
  );

  // --- serializer: a second revival while a panel exists is dropped ----
  const duplicate = makeFakePanel(CHAT_PANEL_VIEW_TYPE, PRODUCT_NAME);
  await registeredSerializer.serializer.deserializeWebviewPanel(duplicate, {});
  assert.ok(duplicate.disposed, 'one panel only: the duplicate is dropped');
  assert.ok(!revived.disposed, 'the standing panel is untouched');
  assert.strictEqual(manager.controller(), revivedController);

  // --- serializer: mode off disposes revived panels --------------------
  editorTabsMode = false;
  revived.dispose();
  assert.strictEqual(manager.controller(), undefined);
  const staleRevive = makeFakePanel(CHAT_PANEL_VIEW_TYPE, PRODUCT_NAME);
  await registeredSerializer.serializer.deserializeWebviewPanel(
    staleRevive,
    {},
  );
  assert.ok(
    staleRevive.disposed,
    'a panel revived with the mode off must be disposed',
  );
  assert.strictEqual(manager.controller(), undefined);
  editorTabsMode = true;
  await sleep(150);
  assertNoCloseTab('serializer drops never retire a chat');

  // --- terminal dispose leaves the panel standing for revival ----------
  manager.open();
  assert.strictEqual(createdPanels.length, 4, 'a fresh panel D');
  const panelD = createdPanels[3];
  manager.dispose();
  assert.ok(
    !panelD.disposed,
    'teardown must NOT close the editor tab (the workbench persists it ' +
      'for the serializer)',
  );
  assert.strictEqual(manager.controller(), undefined, 'controller torn down');
  manager.ensureOpen();
  assert.strictEqual(
    createdPanels.length,
    4,
    'ensureOpen after dispose is a no-op',
  );
  await sleep(150);
  assertNoCloseTab('teardown must not retire any chat');

  console.log('editorTabsPanelManager: all tests passed');
}

runTest()
  .then(() => {
    server.close();
    fs.rmSync(tmpHome, {recursive: true, force: true});
    process.exit(0);
  })
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
