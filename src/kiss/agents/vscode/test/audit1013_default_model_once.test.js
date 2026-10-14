// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-10-13 (scope J, finding 2): every SorcarSidebarView
// construction (activation, then each chat panel opened in editor-tabs
// mode) called resolveDefaultModel(), which spawned a fresh
// `uv run … get_default_model()` once the previous lookup had finished,
// although the daemon's `models` reply overwrites the answer as soon as
// the panel connects.  The lookup now runs once per session: a fake
// `~/.local/bin/uv` counts its invocations across three sequential
// resolutions, and the fallback answer "No model" is not cached.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

if (process.platform === 'win32') {
  console.log('SKIP: POSIX shell shim only');
  process.exit(0);
}

const OUT = path.join(__dirname, '..', 'out');
assert.ok(
  fs.existsSync(path.join(OUT, 'DependencyInstaller.js')),
  'run `npm run compile` first',
);

const fakeHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-dm-'));
const binDir = path.join(fakeHome, '.local', 'bin');
fs.mkdirSync(binDir, {recursive: true});
const countFile = path.join(fakeHome, 'uv-calls');
const fakeUv = path.join(binDir, 'uv');
// Fails (no output) while `fail` exists so the fallback answer is
// produced; prints a model otherwise.
fs.writeFileSync(
  fakeUv,
  `#!/bin/sh
echo run >> "${countFile}"
[ -e "${fakeHome}/fail" ] && exit 1
echo catalog/model
`,
);
fs.chmodSync(fakeUv, 0o755);
const project = path.join(fakeHome, 'kiss_project');
fs.mkdirSync(project);
fs.writeFileSync(path.join(project, 'pyproject.toml'), 'name = "kiss"\n');

process.env.HOME = fakeHome;
process.env.KISS_PROJECT_PATH = project;
// No API keys, no `claude`/`codex` on PATH: the fallback is "No model".
for (const key of Object.keys(process.env)) {
  if (/API_KEY$/.test(key)) delete process.env[key];
}
process.env.PATH = binDir;

global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    getConfiguration: () => ({get: () => undefined}),
  },
  window: {},
  ProgressLocation: {Notification: 15},
  Uri: {file: p => ({fsPath: p})},
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return path.join(__dirname, '_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
const {resolveDefaultModel, provisionalDefaultModel} = require(
  path.join(OUT, 'DependencyInstaller.js'),
);

function uvCalls() {
  return fs.existsSync(countFile)
    ? fs.readFileSync(countFile, 'utf8').split('\n').filter(Boolean).length
    : 0;
}

(async () => {
  // The catalog is unreachable: the fallback answer is not kept, so
  // the next caller tries the catalog again.
  fs.writeFileSync(path.join(fakeHome, 'fail'), '');
  assert.strictEqual(await resolveDefaultModel(), 'No model');
  assert.strictEqual(uvCalls(), 1);
  assert.strictEqual(provisionalDefaultModel(), 'No model');
  fs.unlinkSync(path.join(fakeHome, 'fail'));
  assert.strictEqual(await resolveDefaultModel(), 'catalog/model');
  assert.strictEqual(uvCalls(), 2, 'a fallback answer is retried');

  // Known answer: later panels reuse it without spawning uv.
  const [a, b] = await Promise.all([
    resolveDefaultModel(),
    resolveDefaultModel(),
  ]);
  assert.strictEqual(a, 'catalog/model');
  assert.strictEqual(b, 'catalog/model');
  assert.strictEqual(await resolveDefaultModel(), 'catalog/model');
  assert.strictEqual(uvCalls(), 2, 'the catalog lookup runs once per session');
  assert.strictEqual(provisionalDefaultModel(), 'catalog/model');
  fs.rmSync(fakeHome, {recursive: true, force: true});
  console.log('audit1013_default_model_once: passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
