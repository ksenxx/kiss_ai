// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: the "Working on it." voice-dictation ack plays
// NATIVELY on the extension host machine — never via the webview.
//
// Chain under test (all real, compiled code):
//
//   host `voiceSpeech` → media/voice.js (jsdom) inserts the dictated
//   text and raises `kiss-voice-submit` → media/main.js submits the
//   task AND forwards voice.js's `{type: 'voiceAck'}` bridge post to
//   the extension host → compiled out/SorcarSidebarView.js handles
//   'voiceAck' → out/voiceAckPlayer.js spawns a REAL audio-player
//   child process (KISS_SORCAR_PLAY_CMD recorder script — the same
//   override cli_talk honours) with media/working-on-it.mp3 as its
//   last argument.
//
// This is the fix for the "alien voice": the webview's Audio.play()
// is rejected by Chromium's autoplay policy (no recent click during
// dictation, microsoft/vscode#197937) and its old Web Speech fallback
// spoke every ack with the loud robotic system voice.
//
// Run with:  node test/voiceAckHostPlayback.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');
const {JSDOM} = require('jsdom');

const EXT_ROOT = path.join(__dirname, '..');
const MEDIA = path.join(EXT_ROOT, 'media');
const OUT_SIDEBAR = path.join(EXT_ROOT, 'out', 'SorcarSidebarView.js');
const OUT_PLAYER = path.join(EXT_ROOT, 'out', 'voiceAckPlayer.js');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS + sh test)');
  process.exit(0);
}
for (const compiled of [OUT_SIDEBAR, OUT_PLAYER]) {
  assert.ok(
    fs.existsSync(compiled),
    `compiled extension missing: ${compiled} — run \`npm run compile\` first`,
  );
}

// ---- vscode module stub (for the compiled extension) ----------------
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

let workspaceFolders = [];

global.__kissVscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
    textDocuments: [],
    isTrusted: true,
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  ProgressLocation: {Notification: 15},
  ViewColumn: {One: 1},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showErrorMessage: () => {},
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

// ---- sandbox HOME + recorder player ---------------------------------
const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ack-e2e-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

// A REAL scripted child process standing in for afplay: records its
// argv (the mp3 path) — the KISS_SORCAR_PLAY_CMD override contract
// shared with kiss.agents.sorcar.cli_talk.
const recordFile = path.join(tmpHome, 'played.txt');
const recorder = path.join(tmpHome, 'recorder.sh');
fs.writeFileSync(
  recorder,
  `#!/bin/sh\nprintf '%s\\n' "$@" >> "${recordFile}"\n`,
  {mode: 0o755},
);
process.env.KISS_SORCAR_PLAY_CMD = `sh "${recorder}"`;

// ---- stub daemon (UDS) so main.js can submit the dictated task ------
const server = net.createServer(sock => {
  let buf = '';
  sock.on('data', chunk => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      let cmd;
      try {
        cmd = JSON.parse(line);
      } catch {
        continue;
      }
      if (cmd.type === 'run') {
        sock.write(
          JSON.stringify({type: 'clear', chat_id: 'c1', tabId: cmd.tabId}) +
            '\n',
        );
      }
    }
  });
});

// ---- jsdom webview running the REAL main.js + voice.js --------------
function buildWebviewWindow() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  // Keep the wake-word listener OFF: this test dictates via a direct
  // voiceSpeech message, so the host must not spawn a real listener.
  win.localStorage.setItem('kissVoiceEnabled', '0');
  win.__VOICE__ = {mode: 'webview'};
  const ctx = {toExtension: null};
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => {
        if (ctx.toExtension) ctx.toExtension(msg);
      },
      getState: () => undefined,
      setState: () => {},
    };
  };
  ctx.win = win;
  return ctx;
}

function makeWebviewView(ctx) {
  const recv = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => uri,
    postMessage: msg => {
      ctx.win.dispatchEvent(new ctx.win.MessageEvent('message', {data: msg}));
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => recv.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };
  return {
    webviewView,
    wire: () => {
      ctx.toExtension = msg => recv.fire(msg);
    },
  };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function waitFor(cond, deadlineMs) {
  const end = Date.now() + deadlineMs;
  while (Date.now() < end) {
    if (cond()) return true;
    await sleep(50);
  }
  return cond();
}

async function main() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  delete require.cache[require.resolve(OUT_SIDEBAR)];
  const {SorcarSidebarView} = require(OUT_SIDEBAR);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ack-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];
  const view = new SorcarSidebarView(makeUri(EXT_ROOT));

  const ctx = buildWebviewWindow();
  const wvv = makeWebviewView(ctx);
  view.resolveWebviewView(wvv.webviewView, {}, {});
  wvv.wire();
  ctx.win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  ctx.win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  ctx.win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  await sleep(80);

  // The host delivers a dictated, translated task (wake word already
  // consumed) exactly like VoiceWakeService does.
  wvv.webviewView.webview.postMessage({
    type: 'voiceSpeech',
    text: 'Fix the flaky test',
    speaker: 1,
  });

  const played = await waitFor(
    () => fs.existsSync(recordFile) && fs.readFileSync(recordFile, 'utf8').trim(),
    10000,
  );
  assert.ok(
    played,
    'ack player child process was never spawned for the dictated task',
  );
  const args = fs.readFileSync(recordFile, 'utf8').trim().split('\n');
  const clip = args[args.length - 1];
  assert.strictEqual(
    clip,
    path.join(EXT_ROOT, 'media', 'working-on-it.mp3'),
    `player must receive the bundled ack clip, got: ${JSON.stringify(args)}`,
  );
  assert.ok(fs.existsSync(clip), 'bundled working-on-it.mp3 must exist');
  console.log('  \u2713 dictated task plays the ack natively on the host');

  // The webview never received a playable ack URL in webview mode, and
  // exactly one player process ran for exactly one dictation.
  assert.strictEqual(args.length, 1, `expected one playback, got ${args.length}`);
  console.log('  \u2713 exactly one native playback per dictated task');

  server.close();
  console.log('\n2 passed, 0 failed');
  process.exit(0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
