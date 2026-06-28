// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test: the VS Code webview HTML must use content-versioned
// media URLs.  Without a cache-busting query string, VS Code can reuse
// an older cached main.js/main.css after an extension update, causing
// the History sidebar to run stale code that never renders running-task
// rows/dots even though the source tree is fixed.

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

function hrefsAndSrcs(html) {
  const values = [];
  const re = /(?:href|src)="([^"]+)"/g;
  let m;
  while ((m = re.exec(html)) !== null) values.push(m[1]);
  return values;
}

function assertAssetUrl(html, name) {
  const expectedVersion = hashFor(name);
  const matching = hrefsAndSrcs(html).filter(u => u.includes('/media/' + name));
  assert.strictEqual(
    matching.length,
    1,
    `expected exactly one generated URL for ${name}; got ${matching}`,
  );
  const url = new URL(matching[0], 'https://webview.invalid/');
  assert.strictEqual(
    url.searchParams.get('v'),
    expectedVersion,
    `${name} must carry a content hash cache-buster`,
  );
}

function testBuildChatHtmlUsesContentVersionedMediaUrls() {
  const extensionUri = {fsPath: projectRoot};
  const webview = {
    cspSource: 'vscode-webview://stub',
    asWebviewUri(uri) {
      return {toString: () => 'vscode-webview://' + uri.fsPath};
    },
  };
  const html = buildChatHtml(webview, extensionUri, 'test-model');

  [
    'main.css',
    'highlight-github-dark.min.css',
    'highlight.min.js',
    'marked.min.js',
    'panelCopy.js',
    'main.js',
    'demo.js',
  ].forEach(name => assertAssetUrl(html, name));

  console.log('  ok - buildChatHtml content-versions every media URL');
}

testBuildChatHtmlUsesContentVersionedMediaUrls();
