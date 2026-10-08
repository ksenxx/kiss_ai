// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// SorcarSidebarView.dispose() must settle every pending
// generateCommitMessage() wait.  Before the fix dispose() disposed the
// commit-message emitter without firing it, so a request still in
// flight at teardown could only end on its 30 s safety timer: the
// promise (and the command awaiting it) hung for half a minute after
// the view was gone, and the timer kept the extension host alive.
//
// Drives the compiled view with the daemon send path stubbed out, so
// no answer ever arrives, and asserts the wait ends with dispose().

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const Module = require('module');

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

class EventEmitterLite {
  constructor() {
    this._subs = [];
  }
  event = cb => {
    this._subs.push(cb);
    return {dispose: () => this._subs.splice(this._subs.indexOf(cb), 1)};
  };
  fire(v) {
    for (const cb of [...this._subs]) cb(v);
  }
  dispose() {
    this._subs = [];
  }
}

global.__kissVscodeStub = {
  Uri: {file: p => ({fsPath: p, scheme: 'file'})},
  EventEmitter: EventEmitterLite,
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
const {SorcarSidebarView} = require(path.join(outDir, 'SorcarSidebarView.js'));

async function main() {
  const view = new SorcarSidebarView({fsPath: '/ext'});
  const sent = [];
  view._send = cmd => sent.push(cmd);
  const events = [];
  view.onCommitMessage(ev => events.push(ev));

  const waits = [
    view.generateCommitMessage(undefined, 'tab-a', '/repo-a'),
    view.generateCommitMessage(undefined, 'tab-b', '/repo-b'),
  ];
  assert.strictEqual(
    sent.filter(c => c.type === 'generateCommitMessage').length,
    2,
    'both requests went to the daemon',
  );
  assert.deepStrictEqual(
    [...view._commitPendingTabs].sort(),
    ['tab-a', 'tab-b'],
    'both waits are pending',
  );

  view.dispose();

  let settled = false;
  const timeout = new Promise(resolve => setTimeout(resolve, 500, 'timeout'));
  const outcome = await Promise.race([
    Promise.all(waits).then(() => {
      settled = true;
      return 'settled';
    }),
    timeout,
  ]);
  assert.strictEqual(outcome, 'settled', 'dispose() settles pending waits');
  assert.ok(settled);
  assert.deepStrictEqual(
    [...view._commitPendingTabs],
    [],
    'no wait is left pending after dispose()',
  );
  assert.deepStrictEqual(
    events.map(ev => [ev.tabId, ev.message, ev.error]),
    [
      ['tab-a', '', undefined],
      ['tab-b', '', undefined],
    ],
    'each wait is settled with an empty answer and no error (no toast)',
  );

  // A late answer after dispose() must be inert.
  view._onCommitMessage.fire({message: 'late', tabId: 'tab-a'});
  assert.strictEqual(events.length, 2, 'the disposed emitter delivers nothing');

  console.log('commitWaitSettledOnDispose: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
