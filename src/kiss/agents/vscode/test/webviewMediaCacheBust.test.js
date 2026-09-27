// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
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

function hashFor(name) {
  const bytes = fs.readFileSync(path.join(projectRoot, 'media', name));
  return crypto.createHash('sha256').update(bytes).digest('hex').slice(0, 16);
}

// Every media URL in the page: href / src attributes and the quoted
// URLs handed to main.js in the inline shim (the theme highlight sheets
// in window.__HLJS_THEME_CSS__).
function mediaUrls(html) {
  const values = [];
  const re = /"([^"]*\/media\/[^"]+)"/g;
  let m;
  while ((m = re.exec(html)) !== null) values.push(m[1]);
  return values;
}

function assertAssetUrl(html, name) {
  const expectedVersion = hashFor(name);
  const matching = mediaUrls(html).filter(u => u.includes('/media/' + name));
  assert.ok(matching.length >= 1, `expected a generated URL for ${name}`);
  for (const u of matching) {
    const url = new URL(u, 'https://webview.invalid/');
    assert.strictEqual(
      url.searchParams.get('v'),
      expectedVersion,
      `${name} must carry a content hash cache-buster`,
    );
  }
}

function testBuildChatHtmlUsesContentVersionedMediaUrls() {
  const extensionUri = {fsPath: projectRoot};
  const webview = {
    cspSource: 'vscode-webview://stub',
    asWebviewUri(uri) {
      // Like the real API, answer a URI (forward slashes even on Windows).
      const urlPath = uri.fsPath.split(path.sep).join('/');
      return {toString: () => 'vscode-webview://' + urlPath};
    },
  };
  const html = buildChatHtml(webview, extensionUri, 'test-model');

  [
    'main.css',
    'highlight-vscode-dark.css',
    'highlight-vscode-light.css',
    'welcome-logo.png',
    'welcome-logo-dark.png',
    'highlight.min.js',
    'marked.min.js',
    'api.js',
    'panelCopy.js',
    'contentContextMenu.js',
    'treeContextMenu.js',
    'main.js',
  ].forEach(name => assertAssetUrl(html, name));

  console.log('  ok - buildChatHtml content-versions every media URL');
}

testBuildChatHtmlUsesContentVersionedMediaUrls();
