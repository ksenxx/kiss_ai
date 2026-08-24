// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// H-RC1: every daemon-shared path the extension host touches must live
// under the SAME root the daemon uses: $KISS_HOME (fallback ~/.kiss),
// with the socket overridable via $KISS_SORCAR_SOCK. DependencyInstaller
// used to hardcode $HOME/.kiss for its lock/marker/config files and
// $HOME/.kiss/sorcar.sock for the daemon probe, so with KISS_HOME set it
// probed a socket the daemon never binds, killed it mid-task, and then
// polled the wrong path for 180s.
//
// Each scenario runs in a child process because the roots are resolved
// when the module loads.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');

const OUT = path.join(__dirname, '..', 'out');

function runNode(code, env) {
  const res = spawnSync(process.execPath, ['-e', code], {
    env: {...process.env, ...env},
    encoding: 'utf-8',
  });
  if (res.status !== 0) {
    throw new Error(`child failed: ${res.stderr || res.stdout}`);
  }
  return res.stdout.trim();
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rrhi-home-'));
const fakeHome = path.join(tmp, 'home');
const kissHome = path.join(tmp, 'kiss-home');
fs.mkdirSync(fakeHome, {recursive: true});
fs.mkdirSync(kissHome, {recursive: true});

// userAssets.sorcarSockPath(): KISS_SORCAR_SOCK wins, then $KISS_HOME,
// then ~/.kiss.
const sockCode = `
  const {sorcarSockPath} = require(${JSON.stringify(
    path.join(OUT, 'userAssets.js'),
  )});
  console.log(sorcarSockPath());
`;
assert.strictEqual(
  runNode(sockCode, {
    HOME: fakeHome,
    KISS_HOME: kissHome,
    KISS_SORCAR_SOCK: '',
  }),
  path.join(kissHome, 'sorcar.sock'),
  'sorcarSockPath must honor KISS_HOME',
);
assert.strictEqual(
  runNode(sockCode, {
    HOME: fakeHome,
    KISS_HOME: '',
    KISS_SORCAR_SOCK: '',
  }),
  path.join(fakeHome, '.kiss', 'sorcar.sock'),
  'sorcarSockPath must fall back to ~/.kiss',
);
assert.strictEqual(
  runNode(sockCode, {
    HOME: fakeHome,
    KISS_HOME: kissHome,
    KISS_SORCAR_SOCK: path.join(tmp, 'override.sock'),
  }),
  path.join(tmp, 'override.sock'),
  'KISS_SORCAR_SOCK must override everything',
);
console.log('  ok - sorcarSockPath honors KISS_SORCAR_SOCK / KISS_HOME');

// DependencyInstaller's cross-window locks (the daemon restart lock and
// the API-keys prompt lock share LOG_DIR with config.json and every
// daemon marker) must land under KISS_HOME, not $HOME/.kiss.
const lockCode = `
  global.__kissVscodeStub = {
    workspace: {
      isTrusted: true,
      workspaceFolders: [],
      getConfiguration: () => ({get: () => undefined}),
    },
    window: {},
    ProgressLocation: {Notification: 15},
  };
  const Module = require('module');
  const orig = Module._resolveFilename;
  Module._resolveFilename = function (request, ...rest) {
    if (request === 'vscode') {
      return require.resolve(${JSON.stringify(
        path.join(__dirname, '_vscode-stub.js'),
      )});
    }
    return orig.call(this, request, ...rest);
  };
  const di = require(${JSON.stringify(path.join(OUT, 'DependencyInstaller.js'))});
  const release = di.acquireDaemonRestartLock();
  if (!release) throw new Error('lock not acquired');
  const fs = require('fs');
  const path = require('path');
  const expected = path.join(process.env.KISS_HOME, '.kiss-web.restart.lock');
  if (!fs.existsSync(expected)) {
    throw new Error('restart lock not under KISS_HOME: ' + expected);
  }
  release();
  console.log('ok');
`;
assert.strictEqual(
  runNode(lockCode, {HOME: fakeHome, KISS_HOME: kissHome}),
  'ok',
  'restart lock must live under KISS_HOME',
);
assert.strictEqual(
  fs.existsSync(path.join(fakeHome, '.kiss')),
  false,
  'nothing may be created under $HOME/.kiss when KISS_HOME is set',
);
console.log('  ok - DependencyInstaller shared state lives under KISS_HOME');

fs.rmSync(tmp, {recursive: true, force: true});
console.log('rr_area_hi_kiss_home_paths: all tests passed');
