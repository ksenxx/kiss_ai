// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The welcome page is the same as the remote webapp's: the KISS Sorcar
// logo (transparent art, a light-theme and a dark-theme colouring), the
// greeting and the tagline; no suggested-prompt chips and no remote
// URL / password block.  Checks the HTML the extension builds
// (buildChatHtml) and the live webview (chat.html + main.js in JSDOM):
// both logos are present with content-versioned URLs, the highlight.js
// sheet follows the editor theme (light sheet on body.vscode-light,
// swapped live when VS Code rewrites the body classes), the
// #suggestions and #welcome-config containers are gone, and opening a
// new chat asks the host only for the welcome info (remote URL).

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

  for (const [id, file] of [
    ['welcome-logo', 'welcome-logo.png'],
    ['welcome-logo-dark', 'welcome-logo-dark.png'],
  ]) {
    const logo = doc.querySelector(`#welcome > img#${id}.welcome-logo`);
    assert.ok(logo, `the welcome page must show the ${id} image`);
    const url = new URL(logo.getAttribute('src'));
    assert.ok(
      url.pathname.endsWith('/media/' + file),
      `${id} src must point at media/${file}, got ${url}`,
    );
    const bytes = fs.readFileSync(path.join(MEDIA, file));
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
      `media/${file} must be a PNG image`,
    );
    // Transparent art: an RGBA PNG (colour type 6 in the IHDR chunk)
    // whose corner pixel is fully transparent, so the logo sits directly
    // on the theme background instead of on a white card.
    assert.strictEqual(bytes[25], 6, `media/${file} must be an RGBA PNG`);
  }
  assert.ok(
    doc.querySelector('#welcome h2').textContent.includes('Welcome to'),
    'the greeting heading stays under the logo',
  );
  assert.strictEqual(doc.getElementById('suggestions'), null);
  assert.strictEqual(
    doc.getElementById('welcome-config'),
    null,
    'the remote URL / password block is not on the welcome page',
  );
  assert.ok(!html.includes('{{'), 'every template placeholder is filled');

  // The highlight.js sheet: the dark VS Code sheet is linked, and both
  // theme sheets are handed to main.js so it can follow the editor theme.
  const sheet = doc.getElementById('hljs-theme').getAttribute('href');
  assert.ok(
    new URL(sheet).pathname.endsWith('/media/highlight-vscode-dark.css'),
    `the linked highlight sheet is the VS Code dark one, got ${sheet}`,
  );
  const shim = /window\.__HLJS_THEME_CSS__ = (\{.*?\});<\/script>/.exec(html);
  assert.ok(shim, 'buildChatHtml provides window.__HLJS_THEME_CSS__');
  const urls = JSON.parse(shim[1]);
  assert.ok(
    new URL(urls.dark).pathname.endsWith('/media/highlight-vscode-dark.css'),
  );
  assert.ok(
    new URL(urls.light).pathname.endsWith('/media/highlight-vscode-light.css'),
  );
  assert.ok(!html.includes('highlight-github'), 'the github sheets are gone');
  console.log(
    '  ok - buildChatHtml renders both logos, the theme sheets and no config',
  );
}

// The webview follows the editor theme: a light body class selects the
// light highlight sheet at start-up, and a theme change (VS Code
// rewriting the body classes) swaps it live, both ways.
// A MutationObserver callback runs as a microtask; yielding one macrotask
// on the page's own timer queue lets it fire before the next assertion.
function nextTick(win) {
  return new Promise(resolve => win.setTimeout(resolve, 0));
}

async function testHighlightSheetFollowsTheEditorTheme() {
  const {win} = makeWebview({
    bodyClass: 'vscode-light',
    hljs: {dark: '/m/dark.css', light: '/m/light.css'},
  });
  try {
    const link = win.document.getElementById('hljs-theme');
    assert.strictEqual(
      link.getAttribute('href'),
      '/m/light.css',
      'light at start',
    );
    win.document.body.className = 'vscode-dark';
    await nextTick(win);
    assert.strictEqual(
      link.getAttribute('href'),
      '/m/dark.css',
      'dark after switch',
    );
    win.document.body.className =
      'vscode-high-contrast vscode-high-contrast-light';
    await nextTick(win);
    assert.strictEqual(
      link.getAttribute('href'),
      '/m/light.css',
      'high-contrast light is a light theme',
    );
    win.document.body.className = 'vscode-high-contrast';
    await nextTick(win);
    assert.strictEqual(
      link.getAttribute('href'),
      '/m/dark.css',
      'high contrast is dark',
    );
  } finally {
    win.close();
  }
  console.log('  ok - the highlight sheet follows the editor theme');
}

function makeWebview(opts) {
  const {bodyClass, hljs} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{HLJS_CSS_HREF}}', (hljs && hljs.dark) || '');
  html = html.replace(
    ' {{BODY_CLASS_ATTR}}',
    bodyClass ? ` class="${bodyClass}"` : '',
  );
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  if (hljs) win.__HLJS_THEME_CSS__ = hljs;
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
testHighlightSheetFollowsTheEditorTheme().catch(err => {
  console.error(err);
  process.exit(1);
});
