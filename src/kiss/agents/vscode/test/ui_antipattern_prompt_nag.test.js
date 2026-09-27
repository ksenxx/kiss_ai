// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A6 repeated prompts (audit-extension M1/M2), against the REAL
// out/DependencyInstaller.js with a temp $KISS_HOME and a scripted
// vscode.window.showInputBox:
//  - Esc on the remote-access-password prompt persists a 'declined'
//    marker; the follow-up toast offers 'Open settings' (runs
//    kissSorcar.openSettings); no later session prompts again;
//  - Skip on the API-key prompt persists a marker, the follow-up names
//    "KISS: Enter API Key", later sessions stay silent, and
//    promptApiKeysNow() (the command) forgets the skip and asks again;
//  - promptForApiKey() opens a password box, re-opens it with the typed
//    value after 'Try again', tells a rejected key apart from a provider
//    that could not be reached (and lets the user keep the key then).

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT_DIR = path.join(__dirname, '..', 'out');
const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-uap-prompt-'));
const kissHome = path.join(tmpHome, '.kiss');
fs.mkdirSync(kissHome, {recursive: true});
const fakeProject = path.join(tmpHome, 'kiss_project');
fs.mkdirSync(fakeProject, {recursive: true});
fs.writeFileSync(
  path.join(fakeProject, 'pyproject.toml'),
  '[project]\nname = "kiss-agent-framework"\n',
);

process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_HOME = kissHome;
process.env.KISS_PROJECT_PATH = fakeProject;
process.env.SHELL = '/bin/bash';
// No `claude` CLI and no keys: the prompts must run.
process.env.PATH = '/usr/bin:/bin';
delete process.env.ANTHROPIC_API_KEY;
delete process.env.OPENAI_API_KEY;

const inputBoxCalls = [];
const inputAnswers = [];
const warnings = [];
const infos = [];
const warningAnswers = [];
const executedCommands = [];

const vscodeStub = {
  workspace: {
    isTrusted: true,
    getConfiguration: () => ({get: () => undefined}),
  },
  window: {
    showInputBox: opts => {
      inputBoxCalls.push(opts);
      return Promise.resolve(inputAnswers.shift());
    },
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {
          isCancellationRequested: false,
          onCancellationRequested: () => ({dispose: () => {}}),
        },
      ),
    showInformationMessage: (message, _opts, ...actions) =>
      new Promise(resolve => infos.push({message, actions, resolve})),
    showWarningMessage: (message, _opts, ...actions) => {
      warnings.push({message, actions});
      return Promise.resolve(warningAnswers.shift());
    },
    showErrorMessage: () => Promise.resolve(undefined),
  },
  ProgressLocation: {Notification: 15},
  Uri: {
    file: p => ({fsPath: p}),
    joinPath: (base, ...parts) => ({fsPath: path.join(base.fsPath, ...parts)}),
  },
  commands: {
    executeCommand: (cmd, ...args) => {
      executedCommands.push({cmd, args});
      return Promise.resolve();
    },
  },
  CancellationTokenSource: class {
    constructor() {
      this.token = {
        isCancellationRequested: false,
        onCancellationRequested: () => ({dispose: () => {}}),
      };
    }
    cancel() {}
    dispose() {}
  },
  EventEmitter: class {
    constructor() {
      this.event = () => ({dispose: () => {}});
    }
    fire() {}
    dispose() {}
  },
};

const origLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === 'vscode') return vscodeStub;
  return origLoad.call(this, request, parent, isMain);
};

const di = require(path.join(OUT_DIR, 'DependencyInstaller.js'));

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitFor(predicate, message, tries = 200) {
  for (let i = 0; i < tries; i++) {
    if (predicate()) return;
    await sleep(20);
  }
  throw new Error(message || 'waitFor timed out');
}

async function remotePasswordTests() {
  const lockFile = path.join(kissHome, 'rp.lock');
  const declined = path.join(kissHome, 'rp-declined');
  inputAnswers.push(undefined); // Esc
  await di.ensureRemotePassword(null, fakeProject, lockFile, 5000, 50, declined);
  assert.strictEqual(inputBoxCalls.length, 1, 'the password box opened once');
  assert.strictEqual(inputBoxCalls[0].password, true);
  assert.ok(fs.existsSync(declined), 'Esc persists the declined marker');
  const followUp = infos.find(i => i.message.includes('remote access password'));
  assert.ok(followUp, 'a follow-up says where to set it later');
  assert.deepStrictEqual(followUp.actions, ['Open settings']);
  followUp.resolve('Open settings');
  await waitFor(
    () => executedCommands.some(c => c.cmd === 'kissSorcar.openSettings'),
    "'Open settings' opens the settings panel",
  );

  // Next session: silent and immediate.
  const t0 = Date.now();
  await di.ensureRemotePassword(null, fakeProject, lockFile, 5000, 50, declined);
  assert.strictEqual(inputBoxCalls.length, 1, 'no second password prompt');
  assert.ok(Date.now() - t0 < 1500, 'the declined path skips the 2 s retry');
  console.log('remote password declined-marker tests passed');
}

async function apiKeyTests() {
  const lockFile = path.join(kissHome, 'ak.lock');
  const declined = path.join(kissHome, 'ak-declined');
  inputBoxCalls.length = 0;
  // Esc on both optional key prompts, then Skip on the "required" toast.
  inputAnswers.push(undefined, undefined);
  warningAnswers.push('Skip');
  let ready = await di.ensureApiKeys(lockFile, declined);
  assert.strictEqual(ready, false);
  assert.strictEqual(inputBoxCalls.length, 2, 'both keys were asked once');
  for (const call of inputBoxCalls) {
    assert.strictEqual(call.password, true, 'API keys are entered masked');
  }
  assert.ok(fs.existsSync(declined), 'Skip persists the declined marker');
  const followUp = infos.find(i => i.message.includes('API key'));
  assert.ok(followUp, 'a follow-up says how to add a key later');
  assert.ok(
    followUp.message.includes('"KISS: Enter API Key"'),
    'the follow-up names the command',
  );

  // Next session: silent.
  warnings.length = 0;
  ready = await di.ensureApiKeys(lockFile, declined);
  assert.strictEqual(ready, false);
  assert.strictEqual(inputBoxCalls.length, 2, 'no repeated key prompt');
  assert.strictEqual(warnings.length, 0, 'no repeated "required" toast');

  // Closing the "required" toast (undefined) counts as Skip too.
  fs.rmSync(declined, {force: true});
  inputAnswers.push(undefined, undefined);
  warningAnswers.push(undefined);
  await di.ensureApiKeys(lockFile, declined);
  assert.ok(fs.existsSync(declined), 'closing the toast also stops the nag');
  console.log('API key declined-marker tests passed');
}

async function promptApiKeysNowTest() {
  // The command path clears the DEFAULT marker, so exercise it with the
  // default file: skip once, then promptApiKeysNow() must ask again.
  const defaultDeclined = path.join(kissHome, '.api-keys-declined');
  fs.writeFileSync(defaultDeclined, 'x\n');
  inputBoxCalls.length = 0;
  await di.ensureApiKeys();
  assert.strictEqual(inputBoxCalls.length, 0, 'default marker honoured');
  // Order: Anthropic first (validated), then OpenAI. Skip Anthropic, give
  // an OpenAI key (no validator) so the run ends with a key.
  inputAnswers.push(undefined, 'sk-openai-123');
  const ready = await di.promptApiKeysNow();
  assert.strictEqual(ready, true, 'the command prompts and saves the key');
  assert.strictEqual(inputBoxCalls.length, 2, 'promptApiKeysNow asks again');
  assert.ok(!fs.existsSync(defaultDeclined), 'the skip is forgotten');
  assert.strictEqual(process.env.OPENAI_API_KEY, 'sk-openai-123');
  const rc = fs.readFileSync(path.join(tmpHome, '.bashrc'), 'utf-8');
  assert.ok(rc.includes('OPENAI_API_KEY'), 'the key is saved to the shell rc');
  console.log('promptApiKeysNow tests passed');
}

async function promptForApiKeyTests() {
  inputBoxCalls.length = 0;
  // 1. rejected key -> 'Try again' -> box reopens WITH the typed value ->
  //    unreachable -> 'Save without validating' keeps the key.
  const outcomes = ['rejected', 'unreachable'];
  const probed = [];
  const validate = key => {
    probed.push(key);
    return Promise.resolve(outcomes.shift());
  };
  inputAnswers.push('sk-ant-typo', 'sk-ant-fixed');
  warningAnswers.push('Try again', 'Save without validating');
  warnings.length = 0;
  const key = await di.promptForApiKey(
    'Anthropic API Key',
    'sk-ant-...',
    validate,
    true,
    'api.anthropic.com',
  );
  assert.strictEqual(key, 'sk-ant-fixed');
  assert.deepStrictEqual(probed, ['sk-ant-typo', 'sk-ant-fixed']);
  assert.strictEqual(inputBoxCalls.length, 2);
  assert.strictEqual(inputBoxCalls[0].password, true);
  assert.strictEqual(inputBoxCalls[0].value, '', 'first box is empty');
  assert.strictEqual(
    inputBoxCalls[1].value,
    'sk-ant-typo',
    "'Try again' keeps the typed value",
  );
  assert.strictEqual(warnings.length, 2);
  assert.ok(
    /api\.anthropic\.com rejected this Anthropic API Key/.test(warnings[0].message),
    `rejected message names the provider: ${warnings[0].message}`,
  );
  assert.deepStrictEqual(warnings[0].actions, ['Try again', 'Cancel']);
  assert.ok(
    /Could not reach api\.anthropic\.com to validate the key; check your connection/.test(
      warnings[1].message,
    ),
    `unreachable message is distinct: ${warnings[1].message}`,
  );
  assert.deepStrictEqual(warnings[1].actions, [
    'Try again',
    'Save without validating',
    'Cancel',
  ]);

  // 2. unreachable -> Cancel returns undefined; rejected -> Cancel too.
  outcomes.push('unreachable');
  inputAnswers.push('sk-x');
  warningAnswers.push('Cancel');
  assert.strictEqual(
    await di.promptForApiKey('K', 'p', validate, true, 'h'),
    undefined,
  );
  outcomes.push('rejected');
  inputAnswers.push('sk-y');
  warningAnswers.push(undefined);
  assert.strictEqual(
    await di.promptForApiKey('K', 'p', validate, true, 'h'),
    undefined,
  );

  // 3. ok -> returned; empty input re-asks; a required key's Esc offers
  //    'Enter Key' then returns after Skip.
  outcomes.push('ok');
  inputAnswers.push('   ', 'sk-ok');
  assert.strictEqual(await di.promptForApiKey('K', 'p', validate, true), 'sk-ok');
  inputAnswers.push(undefined, undefined);
  warningAnswers.push('Enter Key', 'Skip');
  assert.strictEqual(await di.promptForApiKey('K', 'p', undefined, false), undefined);
  // Without a validator the key is taken as typed; the default host
  // wording is used when none is given.
  outcomes.push('rejected');
  inputAnswers.push('sk-z');
  warningAnswers.push('Cancel');
  warnings.length = 0;
  await di.promptForApiKey('K', 'p', validate, true);
  assert.ok(warnings[0].message.startsWith('The provider rejected this K'));
  console.log('promptForApiKey tests passed');
}

async function runTest() {
  await remotePasswordTests();
  await apiKeyTests();
  await promptApiKeysNowTest();
  await promptForApiKeyTests();
  fs.rmSync(tmpHome, {recursive: true, force: true});
  console.log('\nAll ui_antipattern_prompt_nag tests passed');
}

runTest()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });
