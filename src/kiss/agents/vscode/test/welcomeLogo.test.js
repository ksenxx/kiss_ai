// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The welcome page shows the KISS Sorcar logo and no suggested-prompt
// chips.  Checks the HTML the extension builds (buildChatHtml) and the
// live webview (chat.html + main.js in JSDOM): the logo is present with a
// content-versioned URL, the #suggestions container and its chips are
// gone, and opening a new chat asks the host only for the welcome info
// (remote URL), never for suggestions.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const Module = require('module');
const {JSDOM} = require('jsdom');

const projectRoot = path.resolve(__dirname, '..');
const MEDIA = path.join(projectRoot, 'media');
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

function testBuiltHtmlHasLogoAndNoSuggestions() {
  const webview = {
    cspSource: 'vscode-webview://stub',
    asWebviewUri(uri) {
      const urlPath = uri.fsPath.split(path.sep).join('/');
      return {toString: () => 'vscode-webview://' + urlPath};
    },
  };
  const html = buildChatHtml(webview, {fsPath: projectRoot}, 'test-model');
  const doc = new JSDOM(html).window.document;

  const logo = doc.querySelector('#welcome > img#welcome-logo');
  assert.ok(logo, 'the welcome page must show the logo image');
  const url = new URL(logo.getAttribute('src'));
  assert.ok(
    url.pathname.endsWith('/media/welcome-logo.png'),
    `logo src must point at media/welcome-logo.png, got ${url}`,
  );
  const bytes = fs.readFileSync(path.join(MEDIA, 'welcome-logo.png'));
  assert.strictEqual(
    url.searchParams.get('v'),
    crypto.createHash('sha256').update(bytes).digest('hex').slice(0, 16),
    'the logo URL must carry a content-hash cache-buster',
  );
  assert.strictEqual(
    logo.getAttribute('alt'),
    '',
    'the logo is decorative: the heading next to it names the product',
  );
  assert.strictEqual(
    bytes.subarray(1, 4).toString('latin1'),
    'PNG',
    'media/welcome-logo.png must be a PNG image',
  );
  assert.ok(
    doc.querySelector('#welcome h2').textContent.includes('Welcome to'),
    'the greeting heading stays under the logo',
  );
  assert.strictEqual(doc.getElementById('suggestions'), null);
  assert.ok(!html.includes('{{'), 'every template placeholder is filled');
  console.log('  ok - buildChatHtml renders the logo and no suggestions');
}

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
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
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function testNewChatRequestsWelcomeInfoOnly() {
  const {win, posted} = makeWebview();
  const welcome = win.document.getElementById('welcome');
  assert.ok(welcome.querySelector('#welcome-logo'), 'logo is in the webview');

  posted.length = 0;
  win.document
    .getElementById('new-chat-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const types = posted.map(m => m.type);
  assert.ok(types.includes('newChat'), `new chat posts newChat: ${types}`);
  assert.ok(
    types.includes('getWelcomeInfo'),
    `new chat asks for the welcome info (remote URL): ${types}`,
  );
  assert.ok(!types.includes('getWelcomeSuggestions'), `${types}`);

  // A daemon that still speaks the old protocol must not bring the
  // chips back.
  win.dispatchEvent(
    new win.MessageEvent('message', {
      data: {type: 'welcome_suggestions', suggestions: [{text: 'old chip'}]},
    }),
  );
  const shown = win.document.getElementById('welcome');
  assert.ok(!shown.textContent.includes('old chip'));
  assert.ok(!shown.textContent.includes('Suggested prompt'));
  assert.strictEqual(win.document.querySelector('.suggestion-chip'), null);
  assert.ok(shown.querySelector('#welcome-logo'), 'logo survives a new chat');
  win.close();
  console.log('  ok - new chat requests getWelcomeInfo and shows no chips');
}

testBuiltHtmlHasLogoAndNoSuggestions();
testNewChatRequestsWelcomeInfoOnly();
