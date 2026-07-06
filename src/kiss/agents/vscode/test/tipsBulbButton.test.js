// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the Tips bulb button in the chat webview.
//
// Contract locked in here:
//
//   * The chat HTML rendered by ``buildChatHtml`` (compiled
//     ``out/SorcarTab.js``) contains a ``#tips-btn`` button placed
//     strictly between the "Inject promptlet" button (``#tricks-btn``)
//     and the mic button (``#voice-btn``), carrying a lightbulb SVG
//     icon, a tooltip, and an accessible label.
//   * ``media/tips.js`` wires a click handler on ``#tips-btn``:
//     clicking it mounts the ``<kiss-tips-panel>`` tips window showing
//     the tips from ``window.__TIPS__.tips`` — regardless of the
//     ``show`` (fresh-install) flag.
//   * Only one panel instance may exist at a time: clicking the bulb
//     while the panel is open does not stack a second panel; after
//     Close the bulb reopens it.
//   * The handler is robust: it works when ``window.__TIPS__`` is
//     missing or has no tips (the panel opens empty and closeable),
//     and tips.js does not throw when ``#tips-btn`` is absent.
//
// Runs against the compiled extension under ``out/`` and the real
// ``media/chat.html`` + ``media/tips.js`` + ``media/marked.min.js``
// in a jsdom DOM:
//
//     tsc -p . && node test/tipsBulbButton.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const projectRoot = path.resolve(__dirname, '..');

// ---------------------------------------------------------------------------
// Stub ``vscode`` before the compiled extension loads it.
// ---------------------------------------------------------------------------

const stubPath = path.join(__dirname, '_vscode-stub.js');
fs.writeFileSync(
  stubPath,
  `'use strict';\nmodule.exports = global.__kissVscodeStub || {};\n`,
);
global.__kissVscodeStub = {
  Uri: {
    joinPath(base, ...parts) {
      return {fsPath: path.join(base.fsPath, ...parts)};
    },
  },
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
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

const sourcePath = path.join(projectRoot, 'out', 'SorcarTab.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
);
delete require.cache[require.resolve(sourcePath)];
const {buildChatHtml} = require(sourcePath);

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  FAIL - ${name}: ${err && err.message}`);
  }
}

function mkTmp(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/**
 * Run ``fn`` with KISS_HOME and KISS_TIPS_PATH pointing at fresh temp
 * locations.  ``setTips(content)`` writes the tips markdown file.
 */
function withSandbox(fn) {
  const kissHome = path.join(mkTmp('kiss-bulb-home-'), 'kisshome');
  const tipsFile = path.join(mkTmp('kiss-bulb-md-'), 'TIPS.md');
  const prevHome = process.env.KISS_HOME;
  const prevTips = process.env.KISS_TIPS_PATH;
  process.env.KISS_HOME = kissHome;
  process.env.KISS_TIPS_PATH = tipsFile;
  const setTips = content => fs.writeFileSync(tipsFile, content);
  try {
    fn({kissHome, tipsFile, setTips});
  } finally {
    if (prevHome === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prevHome;
    if (prevTips === undefined) delete process.env.KISS_TIPS_PATH;
    else process.env.KISS_TIPS_PATH = prevTips;
  }
}

function renderChatHtml() {
  const extensionUri = {fsPath: projectRoot};
  const webview = {
    cspSource: 'vscode-webview://stub',
    asWebviewUri(uri) {
      return {toString: () => 'vscode-webview://' + uri.fsPath};
    },
  };
  return buildChatHtml(webview, extensionUri, 'test-model');
}

// ---------------------------------------------------------------------------
// jsdom harness: real rendered chat HTML + real media/tips.js
// ---------------------------------------------------------------------------

const {JSDOM} = require(path.join(projectRoot, 'node_modules', 'jsdom'));

/**
 * Load the real rendered chat HTML into jsdom (scripts inert), then
 * eval marked.min.js, ``window.__TIPS__ = cfg`` and media/tips.js —
 * mirroring the webview script order.  Returns the jsdom window.
 */
function loadChatDom(cfg, {withConfig = true, withMarked = true} = {}) {
  const html = renderChatHtml();
  const dom = new JSDOM(html, {runScripts: 'outside-only'});
  const run = file =>
    dom.window.eval(
      fs.readFileSync(path.join(projectRoot, 'media', file), 'utf-8'),
    );
  if (withMarked) run('marked.min.js');
  if (withConfig) dom.window.eval(`window.__TIPS__ = ${JSON.stringify(cfg)};`);
  run('tips.js');
  return dom.window;
}

function panel(win) {
  return win.document.body.querySelector('kiss-tips-panel');
}

function panelParts(win) {
  const host = panel(win);
  assert.ok(host, 'kiss-tips-panel must be mounted on document.body');
  const root = host.shadowRoot;
  assert.ok(root, 'kiss-tips-panel must use shadow DOM');
  return {
    host,
    root,
    body: root.querySelector('.tips-body'),
    counter: root.querySelector('.tips-counter'),
    prev: root.querySelector('.tips-prev'),
    next: root.querySelector('.tips-next'),
    close: root.querySelector('.tips-close'),
  };
}

const THREE_TIPS = [
  '## First tip\n\n- alpha\n- beta',
  'Second tip with **bold** text.',
  '### Third tip\n\n`code` here.',
];

// ---------------------------------------------------------------------------
// Button placement and appearance in the rendered chat HTML
// ---------------------------------------------------------------------------

test('bulb button sits strictly between #tricks-btn and #voice-btn', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom({tips: ['Hello.'], show: false});
    const doc = win.document;
    const tips = doc.getElementById('tips-btn');
    const tricks = doc.getElementById('tricks-btn');
    const voice = doc.getElementById('voice-btn');
    assert.ok(tips, '#tips-btn must exist in the chat HTML');
    assert.ok(tricks && voice, 'neighbour buttons must exist');
    assert.strictEqual(
      tips.parentElement,
      tricks.parentElement,
      'bulb button must live in the same container as #tricks-btn',
    );
    const kids = Array.from(tips.parentElement.children);
    const iTricks = kids.indexOf(tricks);
    const iTips = kids.indexOf(tips);
    const iVoice = kids.indexOf(voice);
    assert.ok(
      iTricks < iTips && iTips < iVoice,
      `expected tricks(${iTricks}) < tips(${iTips}) < voice(${iVoice})`,
    );
    assert.strictEqual(iTips, iTricks + 1, 'bulb directly after promptlet');
    assert.strictEqual(iVoice, iTips + 1, 'mic directly after bulb');
  });
});

test('bulb button carries a lightbulb icon, tooltip, and aria-label', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom({tips: ['Hello.'], show: false});
    const btn = win.document.getElementById('tips-btn');
    assert.ok(btn, '#tips-btn must exist');
    assert.strictEqual(btn.tagName, 'BUTTON');
    assert.ok(btn.querySelector('svg'), 'bulb button must contain an SVG');
    const paths = Array.from(btn.querySelectorAll('svg path')).map(p =>
      p.getAttribute('d'),
    );
    assert.ok(
      paths.some(d => d && d.includes('A6 6 0 0 0 6 8')),
      `SVG must draw a lightbulb, got paths: ${JSON.stringify(paths)}`,
    );
    const tooltip = btn.getAttribute('data-tooltip') || '';
    assert.ok(/tip/i.test(tooltip), `tooltip must mention tips: "${tooltip}"`);
    const label = btn.getAttribute('aria-label') || '';
    assert.ok(/tip/i.test(label), `aria-label must mention tips: "${label}"`);
    assert.ok(
      btn.classList.contains('toggle-btn'),
      'bulb button must share the inline toolbar button styling',
    );
  });
});

test('rendered HTML contains exactly one #tips-btn', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom({tips: ['Hello.'], show: false});
    assert.strictEqual(
      win.document.querySelectorAll('#tips-btn').length,
      1,
      'the bulb button id must be unique',
    );
  });
});

// ---------------------------------------------------------------------------
// Click behaviour — the bulb shows the tips window
// ---------------------------------------------------------------------------

test('clicking the bulb shows the tips window with the tips', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom({tips: THREE_TIPS, show: false});
    assert.strictEqual(panel(win), null, 'no panel before the click');
    win.document.getElementById('tips-btn').click();
    const {body, counter, prev, next} = panelParts(win);
    assert.strictEqual(counter.textContent, '1 / 3');
    assert.ok(
      body.innerHTML.includes('<h2') && body.innerHTML.includes('<li>'),
      `first tip markdown must render to HTML, got: ${body.innerHTML}`,
    );
    assert.strictEqual(prev.disabled, true, 'Previous disabled on first tip');
    assert.strictEqual(next.disabled, false, 'Next enabled on first tip');
  });
});

test('bulb-opened panel navigates and closes with the usual semantics', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom({tips: THREE_TIPS, show: false});
    win.document.getElementById('tips-btn').click();
    const {body, counter, next, close} = panelParts(win);
    next.click();
    assert.strictEqual(counter.textContent, '2 / 3');
    assert.ok(body.innerHTML.includes('<strong>bold</strong>'));
    next.click();
    assert.strictEqual(counter.textContent, '3 / 3');
    assert.strictEqual(next.disabled, true, 'Next disabled on last tip');
    close.click();
    assert.strictEqual(panel(win), null, 'Close removes the panel');
  });
});

test('clicking the bulb twice does not stack a second panel', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom({tips: THREE_TIPS, show: false});
    const btn = win.document.getElementById('tips-btn');
    btn.click();
    btn.click();
    assert.strictEqual(
      win.document.querySelectorAll('kiss-tips-panel').length,
      1,
      'only one tips panel instance may exist at a time',
    );
  });
});

test('after Close the bulb reopens the tips window from the first tip', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom({tips: THREE_TIPS, show: false});
    const btn = win.document.getElementById('tips-btn');
    btn.click();
    const first = panelParts(win);
    first.next.click();
    assert.strictEqual(first.counter.textContent, '2 / 3');
    first.close.click();
    assert.strictEqual(panel(win), null);
    btn.click();
    const second = panelParts(win);
    assert.strictEqual(second.counter.textContent, '1 / 3');
  });
});

test('bulb also dismisses-and-reopens after a fresh-install auto-show', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom({tips: THREE_TIPS, show: true});
    // Auto-shown on fresh install; the bulb must not stack a duplicate.
    const btn = win.document.getElementById('tips-btn');
    btn.click();
    assert.strictEqual(
      win.document.querySelectorAll('kiss-tips-panel').length,
      1,
    );
    panelParts(win).close.click();
    btn.click();
    assert.strictEqual(panelParts(win).counter.textContent, '1 / 3');
  });
});

test('bulb opens an empty, closeable panel when there are no tips', () => {
  withSandbox(({setTips}) => {
    setTips('no tip markers here\n');
    const win = loadChatDom({tips: [], show: false});
    win.document.getElementById('tips-btn').click();
    const {counter, prev, next, close} = panelParts(win);
    assert.strictEqual(counter.textContent, '');
    assert.strictEqual(prev.disabled, true);
    assert.strictEqual(next.disabled, true);
    close.click();
    assert.strictEqual(panel(win), null);
  });
});

test('bulb click does not throw when window.__TIPS__ is missing', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom(null, {withConfig: false});
    win.document.getElementById('tips-btn').click();
    const {counter} = panelParts(win);
    assert.strictEqual(counter.textContent, '');
  });
});

test('bulb click falls back to plain text when marked is unavailable', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    const win = loadChatDom(
      {tips: ['**not rendered**'], show: false},
      {withMarked: false},
    );
    win.document.getElementById('tips-btn').click();
    const {body} = panelParts(win);
    assert.ok(body.textContent.includes('**not rendered**'));
    assert.ok(!body.innerHTML.includes('<strong>'));
  });
});

test('tips.js tolerates a DOM without #tips-btn', () => {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'outside-only',
  });
  dom.window.eval(
    `window.__TIPS__ = ${JSON.stringify({tips: THREE_TIPS, show: false})};`,
  );
  dom.window.eval(
    fs.readFileSync(path.join(projectRoot, 'media', 'tips.js'), 'utf-8'),
  );
  assert.strictEqual(dom.window.document.querySelector('kiss-tips-panel'), null);
});

// ---------------------------------------------------------------------------

if (failures.length > 0) {
  console.error(`\n${failures.length} test(s) failed, ${passed} passed`);
  process.exit(1);
}
console.log(`\nAll ${passed} tipsBulbButton tests passed`);
