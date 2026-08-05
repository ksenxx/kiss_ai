// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

assert.ok(
  /media-src[^;]*\bdata:/.test(csp),
  `CSP must allow media-src data: for GPT talk audio, got: ${csp}`,
);
assert.ok(
  new RegExp(`media-src[^;]*${webview.cspSource}`).test(csp),
  `CSP must allow media-src ${webview.cspSource} for the ack clip, got: ${csp}`,
);
assert.ok(csp.includes("default-src 'none'"), 'default-src stays none');
assert.ok(csp.includes("object-src 'none'"), 'object-src stays none');

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
