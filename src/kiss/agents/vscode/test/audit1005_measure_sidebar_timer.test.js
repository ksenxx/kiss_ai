// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-10-05 (scope I): every sidebar measurement
// (SorcarSidebarView._measureSidebar, driven by widenToOneThird) armed a
// 1.5 s safety-net timer that was never cleared once the webview's
// sizeReport arrived, so a prompt reply still left a referenced timer
// behind that kept the extension host alive.  The probe below runs the
// widening to completion in a child process whose event loop then has
// nothing left to do: it must exit at once, not after the leaked timers
// fire.  The time from "widening resolved" to the child's `exit` event
// is measured INSIDE the child so process start-up does not count.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {execFileSync} = require('child_process');

const OUT = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
assert.ok(fs.existsSync(OUT), 'run `npm run compile` first');

const probe = `
  const path = require('path');
  const Module = require('module');
  const stub = {
    workspace: {
      workspaceFolders: [],
      getConfiguration: () => ({get: () => 'stub-default-model'}),
      onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
      textDocuments: [],
    },
    EventEmitter: class {
      constructor() { this._l = []; this.event = cb => { this._l.push(cb); return {dispose: () => {}}; }; }
      fire(a) { for (const cb of this._l) cb(a); }
      dispose() {}
    },
    Uri: {
      file: p => ({fsPath: p, toString: () => 'file://' + p}),
      joinPath: (b, ...p) => ({fsPath: path.join(b.fsPath, ...p), toString: () => 'file://' + path.join(b.fsPath, ...p)}),
    },
    ProgressLocation: {Notification: 15},
    ViewColumn: {One: 1},
    window: {activeTextEditor: undefined, tabGroups: {all: []}},
    commands: {executeCommand: () => Promise.resolve()},
  };
  const origResolve = Module._resolveFilename;
  Module._resolveFilename = function (request, parent, ...rest) {
    if (request === 'vscode') return ${JSON.stringify(path.join(__dirname, '_vscode-stub.js'))};
    return origResolve.call(this, request, parent, ...rest);
  };
  global.__kissVscodeStub = stub;
  const {SorcarSidebarView} = require(${JSON.stringify(OUT)});
  const view = new SorcarSidebarView(stub.Uri.file(${JSON.stringify(path.join(__dirname, '..'))}));
  let receive;
  const host = {
    webview: {
      options: {}, html: '', cspSource: 'vscode-resource:',
      asWebviewUri: u => u,
      // The webview answers every measurement with a width already at
      // one third of the screen, so the widening ends after two
      // measurements without resizing anything.
      postMessage: msg => {
        if (msg.type === 'measureSize') {
          setImmediate(() => receive({type: 'sizeReport', innerWidth: 400, screenWidth: 1200}));
        }
        return Promise.resolve(true);
      },
      onDidReceiveMessage: cb => { receive = cb; return {dispose: () => {}}; },
    },
    visible: true,
    show: () => {},
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };
  view.resolveWebviewView(host, {}, {});
  let resolvedAt = 0;
  process.on('exit', () => { console.log('EXIT ' + (Date.now() - resolvedAt)); });
  view.widenToOneThird().then(() => {
    resolvedAt = Date.now();
    view.dispose();
    console.log('RESOLVED');
  });
`;

const out = execFileSync(process.execPath, ['-e', probe], {
  encoding: 'utf-8',
  env: {
    ...process.env,
    KISS_HOME: path.join(__dirname, '..', 'tmp-nonexistent-home'),
  },
  timeout: 20_000,
});
const lines = out.trim().split('\n');
assert.strictEqual(lines[0], 'RESOLVED', out);
const idleMs = Number(/^EXIT (\d+)$/.exec(lines[1] || '')?.[1]);
assert.ok(Number.isFinite(idleMs), 'child must report its idle time: ' + out);
// The leaked timers held the process for 1.5 s; a clean loop exits
// within the same tick (generous bound for a loaded machine).
assert.ok(
  idleMs < 1000,
  `the extension host stayed alive ${idleMs} ms after the widening ` +
    'finished: a measurement timer outlived its report',
);
console.log(
  `  ok - no measurement timer outlives its sizeReport (idle ${idleMs} ms)`,
);
console.log('audit1005_measure_sidebar_timer.test.js: all passed');
