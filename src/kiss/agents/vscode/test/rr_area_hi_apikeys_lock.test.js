// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// H-RC4: two windows activating together must not both run the API-key
// prompt flow. Each saved key is a read-modify-write of the SAME shell
// rc file, so parallel prompts can lose a key line; and the user gets
// asked twice. ensureApiKeys now serializes across processes with the
// restart-lock pattern: the second window waits for the first and then
// reuses whatever it saved, without prompting at all.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawn} = require('child_process');

if (process.platform === 'win32') {
  console.log('SKIP: POSIX-only child process orchestration');
  process.exit(0);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rrhi-keys-'));
const home = path.join(tmp, 'home');
fs.mkdirSync(home, {recursive: true});
const markerDir = path.join(tmp, 'markers');
fs.mkdirSync(markerDir, {recursive: true});
const lockFile = path.join(home, '.kiss', '.api-keys.lock');
const rcPath = path.join(home, '.bashrc');

const driverPath = path.join(tmp, 'driver.js');
fs.writeFileSync(
  driverPath,
  `
'use strict';
const fs = require('fs');
const path = require('path');
const Module = require('module');

const role = process.env.KISS_TEST_ROLE;
const markerDir = process.env.KISS_TEST_MARKER_DIR;
const delayMs = Number(process.env.KISS_TEST_DELAY_MS || '0');

global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => undefined}),
  },
  ProgressLocation: {Notification: 15},
  window: {
    withProgress: (_opts, task) =>
      task({report: () => {}}, {
        onCancellationRequested: () => ({dispose: () => {}}),
      }),
    showInputBox: async opts => {
      fs.appendFileSync(
        path.join(markerDir, 'prompted-' + role),
        (opts && opts.title ? opts.title : '?') + '\\n',
      );
      if (opts && opts.title && opts.title.includes('OpenAI')) {
        await new Promise(r => setTimeout(r, delayMs));
        return 'sk-test-' + role;
      }
      return undefined; // skip every other key prompt
    },
    showWarningMessage: () => Promise.resolve(undefined),
    showInformationMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
  },
  commands: {executeCommand: () => Promise.resolve()},
};
const orig = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') {
    return orig.call(this, process.env.KISS_TEST_VSCODE_STUB, ...rest);
  }
  return orig.call(this, request, ...rest);
};

const di = require(process.env.KISS_TEST_DI_PATH);
di.ensureApiKeys().then(ready => {
  console.log('RESULT ' + role + ' ' + ready);
  process.exit(0);
}, err => {
  console.error(err);
  process.exit(1);
});
`,
);

function runDriver(role, delayMs) {
  const env = {
    ...process.env,
    HOME: home,
    USERPROFILE: home,
    SHELL: '/bin/bash',
    PATH: path.join(tmp, 'empty-bin'), // no `claude`, no `which`
    KISS_TEST_ROLE: role,
    KISS_TEST_MARKER_DIR: markerDir,
    KISS_TEST_DELAY_MS: String(delayMs),
    KISS_TEST_VSCODE_STUB: path.join(__dirname, '_vscode-stub.js'),
    KISS_TEST_DI_PATH: path.join(__dirname, '..', 'out', 'DependencyInstaller.js'),
  };
  delete env.KISS_HOME;
  delete env.KISS_SORCAR_SOCK;
  delete env.ANTHROPIC_API_KEY;
  delete env.OPENAI_API_KEY;
  const child = spawn(process.execPath, [driverPath], {
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let out = '';
  let errOut = '';
  child.stdout.on('data', d => (out += d.toString()));
  child.stderr.on('data', d => (errOut += d.toString()));
  return new Promise(resolve => {
    child.on('exit', code => resolve({code, out, errOut}));
  });
}

async function waitFor(predicate, message, timeoutMs = 10000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt <= timeoutMs) {
    if (predicate()) return;
    await new Promise(r => setTimeout(r, 25));
  }
  throw new Error(message);
}

async function main() {
  // Window A starts first and sits in the OpenAI prompt for 2s.
  const aDone = runDriver('A', 2000);
  await waitFor(
    () => fs.existsSync(lockFile),
    'window A never took the api-keys lock',
  );

  // Window B activates while A is mid-prompt.
  const bDone = runDriver('B', 0);
  const [a, b] = await Promise.all([aDone, bDone]);

  assert.strictEqual(a.code, 0, `A failed: ${a.errOut}`);
  assert.strictEqual(b.code, 0, `B failed: ${b.errOut}`);
  assert.ok(a.out.includes('RESULT A true'), `A not ready: ${a.out}`);
  assert.ok(b.out.includes('RESULT B true'), `B not ready: ${b.out}`);

  // Only the lock holder may prompt.
  assert.ok(
    fs.existsSync(path.join(markerDir, 'prompted-A')),
    'window A should have prompted',
  );
  assert.strictEqual(
    fs.existsSync(path.join(markerDir, 'prompted-B')),
    false,
    'window B prompted despite window A holding the api-keys lock',
  );

  // Exactly one saved key line; nothing lost, nothing duplicated.
  const rc = fs.readFileSync(rcPath, 'utf-8');
  const keyLines = rc
    .split('\n')
    .filter(line => line.startsWith('export OPENAI_API_KEY='));
  assert.deepStrictEqual(
    keyLines,
    ['export OPENAI_API_KEY="sk-test-A"'],
    `unexpected rc contents: ${rc}`,
  );

  assert.strictEqual(
    fs.existsSync(lockFile),
    false,
    'api-keys lock must be released',
  );

  fs.rmSync(tmp, {recursive: true, force: true});
  console.log('rr_area_hi_apikeys_lock: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
