// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test: the VS Code webview HTML's Content-Security-Policy
// must allow `media-src data:` so the GPT-synthesized talk audio (a
// base64 MP3 data: URI produced by speech_synthesis.py and shipped in
// the 'talk' event) can actually play inside the webview.  Without it
// the webview silently rejected Audio.play() and fell back to the
// robotic Web Speech API voice, so the natural GPT voice was never
// heard in the VS Code interface (browser tab and iOS webapp have no
// CSP and were unaffected).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const Module = require('module');

const projectRoot = path.resolve(__dirname, '..');
const sourcePath = path.join(projectRoot, 'out', 'SorcarTab.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`npm run compile\` first`,
);

global.__kissVscodeStub = {
  Uri: {
    joinPath(base, ...parts) {
      return {fsPath: path.join(base.fsPath, ...parts)};
    },
  },
  workspace: {
    isTrusted: true,
    workspaceFolders: [{uri: {fsPath: path.resolve(projectRoot, '../../..')}}],
    getConfiguration() {
      return {get: () => undefined};
    },
  },
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

const {buildChatHtml} = require(sourcePath);

const webview = {
  cspSource: 'vscode-resource://test-source',
  asWebviewUri(uri) {
    return {
      toString() {
        return 'vscode-resource://test-source/' + path.basename(uri.fsPath);
      },
    };
  },
};

const html = buildChatHtml(webview, {fsPath: projectRoot}, 'test-model');

const cspMatch = html.match(
  /<meta http-equiv="Content-Security-Policy" content="([^"]*)"/,
);
assert.ok(cspMatch, 'webview HTML must carry a CSP meta tag');
const csp = cspMatch[1];

// The CSP starts from default-src 'none', so audio playback is denied
// unless media-src explicitly allows the data: scheme used by the
// GPT-synthesized talk MP3s.
assert.ok(
  /media-src[^;]*\bdata:/.test(csp),
  `CSP must allow media-src data: for GPT talk audio, got: ${csp}`,
);
// The webview resource origin must also be allowed so voice.js can
// play the bundled "Working on it" ack clip (media/working-on-it.mp3).
assert.ok(
  new RegExp(`media-src[^;]*${webview.cspSource}`).test(csp),
  `CSP must allow media-src ${webview.cspSource} for the ack clip, got: ${csp}`,
);
// The lockdown of everything else must remain intact.
assert.ok(csp.includes("default-src 'none'"), 'default-src stays none');
assert.ok(csp.includes("object-src 'none'"), 'object-src stays none');

// The voice config must hand voice.js the URL of the ack clip.
const voiceCfgMatch = html.match(/window\.__VOICE__ = (\{[^\n]*\});/);
assert.ok(voiceCfgMatch, 'webview HTML must inject window.__VOICE__');
const voiceCfg = JSON.parse(voiceCfgMatch[1]);
assert.strictEqual(voiceCfg.mode, 'webview');
assert.ok(
  typeof voiceCfg.ackAudioUrl === 'string' &&
    voiceCfg.ackAudioUrl.includes('working-on-it.mp3'),
  `voice config must carry the working-on-it.mp3 ack URL, got: ${voiceCfgMatch[1]}`,
);

console.log('webviewTalkAudioCsp.test.js: all assertions passed');
