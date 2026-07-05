// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the fresh-install Tips window.
//
// Contract locked in here:
//
//   * ``getTips()`` (compiled ``out/SorcarTab.js``) parses the bundled
//     ``src/kiss/TIPS.md``: every line starting with ``# Tip`` starts a
//     new tip whose body is the markdown text up to the next such line
//     (or EOF), trimmed.  Empty bodies are skipped.  A missing or
//     unreadable file yields ``[]``.  The ``KISS_TIPS_PATH`` env var
//     overrides the bundled file location for testability.
//   * ``consumeTipsFirstRun()`` returns ``true`` exactly once per
//     installation: the first call creates the ``~/.kiss/TIPS_SHOWN``
//     marker (honouring ``KISS_HOME``) and returns ``true``; every
//     later call returns ``false``.  When the marker cannot be
//     written it returns ``false`` (never spam the user).
//   * ``buildChatHtml`` injects ``window.__TIPS__ = {tips, show}``
//     into the chat webview HTML and loads ``media/tips.js`` with a
//     content-hash cache-buster.  No ``{{TIPS...}}`` placeholder may
//     survive substitution.
//   * ``media/tips.js`` defines the ``<kiss-tips-panel>`` web
//     component: a centered overlay panel that renders the current
//     tip's markdown as HTML (via ``window.marked``), with Previous /
//     Next / Close buttons following the usual semantics (Previous
//     disabled on the first tip, Next disabled on the last, Close
//     removes the panel).  It auto-mounts only when
//     ``window.__TIPS__.show`` is true and at least one tip exists.
//
// Runs against the compiled extension under ``out/`` and the real
// ``media/tips.js`` + ``media/marked.min.js`` in a jsdom DOM:
//
//     tsc -p . && node test/tipsWindow.test.js

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
const {getTips, consumeTipsFirstRun, buildChatHtml} = require(sourcePath);

assert.strictEqual(typeof getTips, 'function', 'getTips must be exported');
assert.strictEqual(
  typeof consumeTipsFirstRun,
  'function',
  'consumeTipsFirstRun must be exported',
);

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
  const kissHome = path.join(mkTmp('kiss-tips-home-'), 'kisshome');
  const tipsFile = path.join(mkTmp('kiss-tips-md-'), 'TIPS.md');
  const prevHome = process.env.KISS_HOME;
  const prevTips = process.env.KISS_TIPS_PATH;
  const prevProject = process.env.KISS_PROJECT_PATH;
  process.env.KISS_HOME = kissHome;
  process.env.KISS_TIPS_PATH = tipsFile;
  delete process.env.KISS_PROJECT_PATH;
  const setTips = content => fs.writeFileSync(tipsFile, content);
  try {
    fn({kissHome, tipsFile, setTips});
  } finally {
    if (prevHome === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prevHome;
    if (prevTips === undefined) delete process.env.KISS_TIPS_PATH;
    else process.env.KISS_TIPS_PATH = prevTips;
    if (prevProject === undefined) delete process.env.KISS_PROJECT_PATH;
    else process.env.KISS_PROJECT_PATH = prevProject;
  }
}

// ---------------------------------------------------------------------------
// getTips — TIPS.md parsing
// ---------------------------------------------------------------------------

test('getTips parses the body after every "# Tip" line', () => {
  withSandbox(({setTips}) => {
    setTips(
      '# Tip\n\n## First tip\n- bullet one\n\n# Tip \n\n## Second tip\n\n' +
        'Some **bold** text.\n\n# Tip\n\nThird tip body.\n',
    );
    assert.deepStrictEqual(getTips(), [
      '## First tip\n- bullet one',
      '## Second tip\n\nSome **bold** text.',
      'Third tip body.',
    ]);
  });
});

test('getTips ignores content before the first "# Tip" line', () => {
  withSandbox(({setTips}) => {
    setTips('Preamble to ignore\n\n# Tip\n\nOnly tip.\n');
    assert.deepStrictEqual(getTips(), ['Only tip.']);
  });
});

test('getTips skips tips with empty bodies', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\n   \n\n# Tip\n\nReal tip.\n\n# Tip\n');
    assert.deepStrictEqual(getTips(), ['Real tip.']);
  });
});

test('getTips does not split on "## Tip" or indented "# Tip" lines', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nBody with\n## Tip heading\nand  # Tip inline\nend.\n');
    assert.deepStrictEqual(getTips(), [
      'Body with\n## Tip heading\nand  # Tip inline\nend.',
    ]);
  });
});

test('getTips returns [] when the tips file is missing', () => {
  withSandbox(() => {
    assert.deepStrictEqual(getTips(), []);
  });
});

test('getTips reads <kissRoot>/src/kiss/TIPS.md without the env override', () => {
  withSandbox(() => {
    delete process.env.KISS_TIPS_PATH;
    const root = mkTmp('kiss-tips-root-');
    fs.writeFileSync(
      path.join(root, 'pyproject.toml'),
      '[project]\nname = "kiss"\n',
    );
    fs.mkdirSync(path.join(root, 'src', 'kiss'), {recursive: true});
    fs.writeFileSync(
      path.join(root, 'src', 'kiss', 'TIPS.md'),
      '# Tip\n\nBundled tip.\n',
    );
    process.env.KISS_PROJECT_PATH = root;
    assert.deepStrictEqual(getTips(), ['Bundled tip.']);
  });
});

test('getTips returns [] when no KISS project root can be found', () => {
  // Only meaningful when no embedded kiss_project/ directory exists
  // next to the extension sources (true for dev checkouts).
  if (fs.existsSync(path.join(projectRoot, 'kiss_project'))) return;
  withSandbox(() => {
    delete process.env.KISS_TIPS_PATH;
    assert.deepStrictEqual(getTips(), []);
  });
});

test('the bundled src/kiss/TIPS.md yields at least one non-empty tip', () => {
  const bundled = path.resolve(projectRoot, '..', '..', 'TIPS.md');
  const prev = process.env.KISS_TIPS_PATH;
  process.env.KISS_TIPS_PATH = bundled;
  try {
    const tips = getTips();
    assert.ok(tips.length > 0, 'bundled TIPS.md must produce tips');
    for (const tip of tips) {
      assert.ok(tip.trim().length > 0, 'no empty tips');
    }
  } finally {
    if (prev === undefined) delete process.env.KISS_TIPS_PATH;
    else process.env.KISS_TIPS_PATH = prev;
  }
});

// ---------------------------------------------------------------------------
// consumeTipsFirstRun — fresh-install marker
// ---------------------------------------------------------------------------

test('consumeTipsFirstRun is true exactly once per installation', () => {
  withSandbox(({kissHome}) => {
    assert.strictEqual(consumeTipsFirstRun(), true);
    assert.ok(
      fs.existsSync(path.join(kissHome, 'TIPS_SHOWN')),
      'first call must create the TIPS_SHOWN marker',
    );
    assert.strictEqual(consumeTipsFirstRun(), false);
    assert.strictEqual(consumeTipsFirstRun(), false);
  });
});

test('consumeTipsFirstRun is false when the marker cannot be written', () => {
  withSandbox(() => {
    // Point KISS_HOME *inside a regular file* so mkdir/write must fail.
    const blocker = path.join(mkTmp('kiss-tips-blocked-'), 'file');
    fs.writeFileSync(blocker, 'not a directory');
    process.env.KISS_HOME = path.join(blocker, 'kiss');
    assert.strictEqual(consumeTipsFirstRun(), false);
  });
});

// ---------------------------------------------------------------------------
// buildChatHtml — webview injection
// ---------------------------------------------------------------------------

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

test('buildChatHtml injects window.__TIPS__ with show:true on fresh install', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello **tips**.\n');
    const html = renderChatHtml();
    assert.ok(
      html.includes('window.__TIPS__'),
      'chat HTML must define window.__TIPS__',
    );
    const m = html.match(/window\.__TIPS__\s*=\s*(\{.*?\});<\/script>/);
    assert.ok(m, 'window.__TIPS__ must be assigned a JSON object literal');
    const cfg = JSON.parse(m[1].replace(/<\\\//g, '</'));
    assert.deepStrictEqual(cfg, {tips: ['Hello **tips**.'], show: true});
    assert.ok(
      /src="[^"]*\/media\/tips\.js\?v=[0-9a-f]{16}"/.test(html),
      'tips.js must be loaded with a content-hash cache-buster',
    );
    assert.ok(!html.includes('{{TIPS'), 'no TIPS placeholder may survive');
  });
});

test('buildChatHtml injects show:false after the first render', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nHello.\n');
    renderChatHtml();
    const m = renderChatHtml().match(
      /window\.__TIPS__\s*=\s*(\{.*?\});<\/script>/,
    );
    assert.ok(m, 'window.__TIPS__ must be assigned on every render');
    const cfg = JSON.parse(m[1].replace(/<\\\//g, '</'));
    assert.deepStrictEqual(cfg, {tips: ['Hello.'], show: false});
  });
});

test('buildChatHtml does not consume first-run marker when no tips exist', () => {
  withSandbox(({kissHome}) => {
    const html = renderChatHtml();
    const m = html.match(/window\.__TIPS__\s*=\s*(\{.*?\});<\/script>/);
    assert.ok(m, 'window.__TIPS__ must be assigned even with no tips');
    const cfg = JSON.parse(m[1].replace(/<\\\//g, '</'));
    assert.deepStrictEqual(cfg, {tips: [], show: false});
    assert.strictEqual(
      fs.existsSync(path.join(kissHome, 'TIPS_SHOWN')),
      false,
      'empty or missing tips should not consume the fresh-install marker',
    );
  });
});

test('buildChatHtml escapes </script> inside tip bodies', () => {
  withSandbox(({setTips}) => {
    setTips('# Tip\n\nUse `</script>` carefully.\n');
    const html = renderChatHtml();
    const m = html.match(/window\.__TIPS__\s*=\s*(\{.*?\});<\/script>/);
    assert.ok(m, 'window.__TIPS__ must still parse');
    assert.ok(
      !m[1].includes('</script>'),
      'raw </script> must not appear inside the injected JSON',
    );
    const cfg = JSON.parse(m[1].replace(/<\\\//g, '</'));
    assert.deepStrictEqual(cfg.tips, ['Use `</script>` carefully.']);
  });
});

// ---------------------------------------------------------------------------
// <kiss-tips-panel> — web component in a real DOM (jsdom)
// ---------------------------------------------------------------------------

const {JSDOM} = require(path.join(projectRoot, 'node_modules', 'jsdom'));

/**
 * Load marked.min.js (optionally) + media/tips.js into a fresh jsdom
 * window with ``window.__TIPS__`` set to ``cfg`` and return the window.
 */
function loadTipsDom(cfg, {withMarked = true, withConfig = true} = {}) {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'outside-only',
  });
  const run = file =>
    dom.window.eval(
      fs.readFileSync(path.join(projectRoot, 'media', file), 'utf-8'),
    );
  if (withMarked) run('marked.min.js');
  if (withConfig) dom.window.eval(`window.__TIPS__ = ${JSON.stringify(cfg)};`);
  run('tips.js');
  return dom.window;
}

function panelParts(win) {
  const host = win.document.querySelector('kiss-tips-panel');
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

test('tips panel auto-mounts centered and renders markdown as HTML', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {root, body, counter, prev, next} = panelParts(win);
  const css = root.querySelector('style').textContent;
  assert.ok(css.includes('position: fixed'), 'overlay must be fixed');
  assert.ok(css.includes('justify-content: center'), 'centered horizontally');
  assert.ok(css.includes('align-items: center'), 'centered vertically');
  assert.ok(
    body.innerHTML.includes('<h2') && body.innerHTML.includes('<li>'),
    `tip markdown must render to HTML, got: ${body.innerHTML}`,
  );
  assert.strictEqual(counter.textContent, '1 / 3');
  assert.strictEqual(prev.disabled, true, 'Previous disabled on first tip');
  assert.strictEqual(next.disabled, false, 'Next enabled on first tip');
});

test('next/previous navigate tips with the usual boundary semantics', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {body, counter, prev, next} = panelParts(win);

  next.click();
  assert.strictEqual(counter.textContent, '2 / 3');
  assert.ok(body.innerHTML.includes('<strong>bold</strong>'));
  assert.strictEqual(prev.disabled, false);
  assert.strictEqual(next.disabled, false);

  next.click();
  assert.strictEqual(counter.textContent, '3 / 3');
  assert.ok(body.innerHTML.includes('<code>code</code>'));
  assert.strictEqual(next.disabled, true, 'Next disabled on last tip');

  next.click(); // no-op past the end
  assert.strictEqual(counter.textContent, '3 / 3');

  prev.click();
  assert.strictEqual(counter.textContent, '2 / 3');
  prev.click();
  assert.strictEqual(counter.textContent, '1 / 3');
  assert.strictEqual(prev.disabled, true);
  prev.click(); // no-op before the start
  assert.strictEqual(counter.textContent, '1 / 3');
});

test('close removes the tips panel from the DOM', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {close} = panelParts(win);
  close.click();
  assert.strictEqual(win.document.querySelector('kiss-tips-panel'), null);
});

test('a single tip disables both Previous and Next', () => {
  const win = loadTipsDom({tips: ['Only one.'], show: true});
  const {counter, prev, next} = panelParts(win);
  assert.strictEqual(counter.textContent, '1 / 1');
  assert.strictEqual(prev.disabled, true);
  assert.strictEqual(next.disabled, true);
});

test('tips panel does not mount when show is false', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: false});
  assert.strictEqual(win.document.querySelector('kiss-tips-panel'), null);
});

test('tips panel does not mount when there are no tips', () => {
  const win = loadTipsDom({tips: [], show: true});
  assert.strictEqual(win.document.querySelector('kiss-tips-panel'), null);
});

test('tips panel does not mount when window.__TIPS__ is missing', () => {
  const win = loadTipsDom(null, {withConfig: false});
  assert.strictEqual(win.document.querySelector('kiss-tips-panel'), null);
});

test('tips fall back to plain text when marked is unavailable', () => {
  const win = loadTipsDom(
    {tips: ['**not rendered**'], show: true},
    {withMarked: false},
  );
  const {body} = panelParts(win);
  assert.ok(
    body.textContent.includes('**not rendered**'),
    'raw markdown text must be shown when marked is missing',
  );
  assert.ok(
    !body.innerHTML.includes('<strong>'),
    'no HTML rendering without marked',
  );
});

// ---------------------------------------------------------------------------

if (failures.length > 0) {
  console.error(`\n${failures.length} test(s) failed, ${passed} passed`);
  process.exit(1);
}
console.log(`\nAll ${passed} tipsWindow tests passed`);
