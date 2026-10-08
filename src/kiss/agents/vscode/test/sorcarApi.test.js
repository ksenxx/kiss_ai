// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

const assert = require('assert');
const {test} = require('node:test');
const path = require('path');

const {createSorcarApi, SORCAR_API_COMMANDS} = require('../media/api.js');
const {makeWebview} = require('./simplify2_harness.js');

test('every api.js method posts its catalog command', () => {
  const posted = [];
  const api = createSorcarApi(msg => posted.push(msg));
  SORCAR_API_COMMANDS.forEach(name => {
    assert.strictEqual(typeof api[name], 'function', name);
    api[name]();
    assert.deepStrictEqual(posted.pop(), {type: name});
    api[name]({tabId: 't1', extra: 42});
    assert.deepStrictEqual(posted.pop(), {type: name, tabId: 't1', extra: 42});
  });
  assert.strictEqual(posted.length, 0);
});

test('api.send forwards catalog commands and rejects the rest', () => {
  const posted = [];
  const api = createSorcarApi(msg => posted.push(msg));
  api.send({type: 'stop', tabId: 'tab-9'});
  assert.deepStrictEqual(posted, [{type: 'stop', tabId: 'tab-9'}]);
  assert.throws(() => api.send({type: 'notAThing'}), /unknown command/);
  assert.throws(() => api.send(null), /unknown command/);
  assert.strictEqual(posted.length, 1);
});

test('catalog contains the core task-lifecycle commands', () => {
  ['run', 'submit', 'stop', 'userAnswer', 'newChat', 'closeTab',
   'getHistory', 'getConfig', 'setWorkDir', 'auth',
  ].forEach(name => assert.ok(SORCAR_API_COMMANDS.includes(name), name));
});

test('the real chat webview only sends API commands', async () => {
  const {win, posted} = makeWebview();
  const inp = win.document.getElementById('task-input');
  inp.value = 'do something';
  win.document.getElementById('send-btn').click();
  await new Promise(r => setTimeout(r, 50));
  assert.ok(posted.length > 0, 'webview posted no messages');
  for (const msg of posted) {
    assert.ok(
      SORCAR_API_COMMANDS.includes(msg.type),
      `non-API message sent by webview: ${JSON.stringify(msg)}`,
    );
  }
  assert.ok(
    posted.some(m => m.type === 'submit' && m.prompt === 'do something'),
    'submit command missing',
  );
});

// The extension host has no wrapper class: SorcarSidebarView sends the
// wire command itself (`_send`).  Drive its webview message handler and
// check the exact command each message turns into.
test('SorcarSidebarView turns webview messages into wire commands', async () => {
  const fs = require('fs');
  const Module = require('module');
  const origResolve = Module._resolveFilename;
  Module._resolveFilename = function (request, parent, ...rest) {
    if (request === 'vscode') return require.resolve('./_vscode-stub.js');
    return origResolve.call(this, request, parent, ...rest);
  };
  global.__kissVscodeStub = {
    Uri: {file: p => ({fsPath: p, scheme: 'file'})},
    EventEmitter: class {
      _subs = [];
      event = cb => {
        this._subs.push(cb);
        return {dispose() {}};
      };
      fire(v) {
        for (const cb of this._subs) cb(v);
      }
      dispose() {}
    },
    TabInputText: class {},
    window: {
      visibleTextEditors: [],
      activeTextEditor: undefined,
      tabGroups: {all: []},
      onDidChangeVisibleTextEditors: () => ({dispose() {}}),
    },
    workspace: {
      isTrusted: true,
      workspaceFolders: [{uri: {fsPath: '/w', scheme: 'file'}}],
      getConfiguration: () => ({get: () => undefined}),
      onDidChangeWorkspaceFolders: () => ({dispose() {}}),
      textDocuments: [],
    },
    commands: {executeCommand: () => Promise.resolve()},
  };
  const outDir = path.join(__dirname, '..', 'out');
  assert.ok(
    fs.existsSync(path.join(outDir, 'SorcarSidebarView.js')),
    'compiled extension missing — run `npm run compile` first',
  );
  const {SorcarSidebarView} = require(
    path.join(outDir, 'SorcarSidebarView.js'),
  );
  const view = new SorcarSidebarView({fsPath: '/ext'});
  const sent = [];
  view._send = cmd => sent.push(cmd);
  view._getWorkDir = () => '/w';
  view._selectedModel = 'm';
  view._runningTabs.add('r1');
  view._runningTabs.add('r2');

  const messages = [
    {type: 'submit', prompt: 'p', model: 'm', attachments: [], tabId: 't'},
    {type: 'stop', tabId: 't'},
    {type: 'stop'},
    {type: 'userAnswer', answer: 'yes', tabId: 't'},
    {type: 'resumeSession', chatId: 'c1', taskId: 'task1', tabId: 't'},
    {type: 'selectModel', model: 'm2', tabId: 't'},
    {type: 'complete', query: 'q', tabId: 't'},
    {type: 'recordFileUsage', path: '/f', workDir: '/w'},
    {type: 'worktreeAction', action: 'discard', tabId: 't'},
    {type: 'mainTreeAction', action: 'discard', tabId: 't'},
    {type: 'autocommitAction', tabId: 't'},
    {type: 'serverReset'},
    {type: 'getHistory', query: 'x'},
  ];
  for (const m of messages) await view._handleMessage(m);
  view.generateCommitMessage(undefined, 't', '/repo');
  view.dispose();

  assert.deepStrictEqual(sent, [
    {type: 'submit', prompt: 'p', model: 'm', attachments: [], tabId: 't',
     activeFile: undefined},
    {type: 'stop', tabId: 't'},
    {type: 'stop', tabId: 'r1'},
    {type: 'stop', tabId: 'r2'},
    {type: 'userAnswer', answer: 'yes', tabId: 't'},
    {type: 'resumeSession', chatId: 'c1', taskId: 'task1', tabId: 't'},
    {type: 'selectModel', model: 'm2', tabId: 't'},
    {type: 'complete', query: 'q', tabId: 't', activeFile: undefined,
     activeFileContent: undefined},
    {type: 'recordFileUsage', path: '/f', workDir: '/w'},
    {type: 'worktreeAction', action: 'discard', tabId: 't'},
    {type: 'mainTreeAction', action: 'discard', tabId: 't', workDir: '/w'},
    {type: 'autocommitAction', tabId: 't', workDir: '/w'},
    {type: 'serverReset'},
    {type: 'getHistory', query: 'x', tag: undefined, offset: undefined,
     generation: undefined},
    {type: 'generateCommitMessage', model: 'm2', tabId: 't', workDir: '/repo'},
  ]);
  assert.strictEqual(view._selectedModel, 'm2', 'selectModel is remembered');
});
