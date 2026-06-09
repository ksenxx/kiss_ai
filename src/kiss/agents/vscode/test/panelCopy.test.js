// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the chat-webview panel Copy button.
//
// Locks in the rule: when the user clicks a Copy button on any chat
// panel (Result, Prompt, System Prompt, streamed Thoughts text, ...)
// the clipboard must receive the agent's raw markdown / plain text —
// NOT the rendered HTML's markdown-stripped textContent.
//
// Runs the real ``media/panelCopy.js`` against a real jsdom document
// (no mocks, no fakes for the code under test).  Exercised directly
// with ``node``:
//
//     node test/panelCopy.test.js

'use strict';

const assert = require('assert');
const path = require('path');
const {JSDOM} = require('jsdom');

const PANEL_COPY_PATH = path.join(__dirname, '..', 'media', 'panelCopy.js');

// ---------------------------------------------------------------------------
// Test harness — same shape as reloadGuard.test.js / installerPath.test.js.
// ---------------------------------------------------------------------------

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
}

/**
 * Build a fresh jsdom window with ``panelCopy.js`` loaded as a real
 * <script> tag and a writable ``navigator.clipboard.writeText`` stub
 * that records the most recent payload.  Returning the window + a
 * ``getClipboard()`` getter mirrors how the webview runs in production
 * (browser globals only — no Node-specific shortcuts).
 */
function makeWindow() {
  // about:blank gives us a working document.body + element factory.
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true},
  );
  const win = dom.window;
  // Plant a clipboard recorder before panelCopy.js runs so the click
  // handler picks it up via the standard navigator.clipboard surface.
  let clipboardText = null;
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: text => {
        clipboardText = String(text);
        return Promise.resolve();
      },
    },
  });
  // Load panelCopy.js as a real script tag so it registers on window.
  const fs = require('fs');
  const src = fs.readFileSync(PANEL_COPY_PATH, 'utf-8');
  const scriptEl = win.document.createElement('script');
  scriptEl.textContent = src;
  win.document.head.appendChild(scriptEl);
  assert.ok(win.PanelCopy, 'panelCopy.js must expose window.PanelCopy');
  return {
    window: win,
    document: win.document,
    PanelCopy: win.PanelCopy,
    getClipboard: () => clipboardText,
  };
}

function clickFirst(el, selector) {
  const btn = el.querySelector(selector);
  assert.ok(btn, `expected to find ${selector} inside panel`);
  btn.dispatchEvent(new btn.ownerDocument.defaultView.MouseEvent('click', {
    bubbles: true,
    cancelable: true,
  }));
}

// Synchronously flush the microtask queue so the awaited
// ``clipboard.writeText`` resolves before assertions.  jsdom's
// ``setImmediate`` is enough because the click handler only chains a
// single ``.then(done)`` off the writeText promise.
function flushMicrotasks() {
  return new Promise(resolve => setImmediate(resolve));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

async function run() {
  // --- API surface ----------------------------------------------------------
  test('exports getRawText + addCopyButton on window.PanelCopy', () => {
    const {PanelCopy} = makeWindow();
    assert.strictEqual(typeof PanelCopy.getRawText, 'function');
    assert.strictEqual(typeof PanelCopy.addCopyButton, 'function');
  });

  // --- getRawText -----------------------------------------------------------
  test('getRawText honours data-raw-text override', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    panel.innerHTML =
      '<h3>Heading</h3><p><strong>bold</strong> text</p>';
    // Simulate a Result panel that stashed the source markdown so the
    // walker returns the markdown instead of recursing into the HTML.
    panel.dataset.rawText = '# Heading\n\n**bold** text';
    assert.strictEqual(
      PanelCopy.getRawText(panel),
      '# Heading\n\n**bold** text',
    );
  });

  test('getRawText skips copy-btn / chevron / collapse-preview chrome', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    panel.innerHTML =
      '<button class="panel-copy-btn">copy</button>' +
      '<span class="collapse-chv">\u25BE</span>' +
      '<span class="collapse-preview">preview</span>' +
      '<div>real content</div>';
    assert.strictEqual(PanelCopy.getRawText(panel), 'real content');
  });

  test('getRawText recurses through children when no override is set', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    const a = document.createElement('div');
    a.textContent = 'alpha';
    const b = document.createElement('div');
    b.textContent = 'beta';
    panel.appendChild(a);
    panel.appendChild(b);
    // Block children join with a newline.
    assert.strictEqual(PanelCopy.getRawText(panel), 'alpha\nbeta');
  });

  test('getRawText prefers nested data-raw-text over child markup', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    const header = document.createElement('div');
    header.textContent = 'Result';
    const body = document.createElement('div');
    body.innerHTML = '<h1>Hello</h1><p>World</p>';
    body.dataset.rawText = '# Hello\n\nWorld';
    panel.appendChild(header);
    panel.appendChild(body);
    assert.strictEqual(
      PanelCopy.getRawText(panel),
      'Result\n# Hello\n\nWorld',
    );
  });

  // --- addCopyButton (full integration with the click handler) -------------
  test('addCopyButton attaches a single button (idempotent)', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    document.body.appendChild(panel);
    PanelCopy.addCopyButton(panel);
    PanelCopy.addCopyButton(panel);
    const btns = panel.querySelectorAll(':scope > .panel-copy-btn');
    assert.strictEqual(btns.length, 1);
    assert.ok(panel.classList.contains('copyable'));
  });

  test(
    'clicking copy on a Result panel copies the raw markdown',
    async () => {
      const {document, PanelCopy, getClipboard} = makeWindow();
      // Replicate the DOM shape main.js builds for a ``result`` event
      // with a markdown summary.  The Copy button must pull the raw
      // markdown from data-raw-text — the rendered HTML's textContent
      // is "Hello World" with the #/`*` markers stripped.
      const rc = document.createElement('div');
      rc.className = 'ev rc';
      rc.innerHTML =
        '<div class="rc-h"><h3>Result</h3><div class="rs">' +
        '<span>Tokens <b>123</b></span><span>Cost <b>$0.01</b></span>' +
        '</div></div>' +
        '<div class="rc-body md-body">' +
        '<h1>Hello</h1><p><strong>World</strong></p>' +
        '</div>';
      const rawMd = '# Hello\n\n**World**';
      rc.dataset.rawText = rawMd;
      document.body.appendChild(rc);

      PanelCopy.addCopyButton(rc);
      clickFirst(rc, '.panel-copy-btn');
      await flushMicrotasks();

      assert.strictEqual(getClipboard(), rawMd);
      // The textContent path would have copied this — assert we did
      // NOT take it.
      assert.notStrictEqual(getClipboard(), 'Result Tokens 123 Cost $0.01 Hello World');
    },
  );

  test(
    'clicking copy on a Prompt panel copies the raw markdown',
    async () => {
      const {document, PanelCopy, getClipboard} = makeWindow();
      const el = document.createElement('div');
      el.className = 'ev prompt';
      el.innerHTML =
        '<div class="prompt-h">Prompt</div>' +
        '<div class="prompt-body md-body"><h2>Hi</h2></div>';
      const raw = '## Hi';
      el.dataset.rawText = raw;
      document.body.appendChild(el);

      PanelCopy.addCopyButton(el);
      clickFirst(el, '.panel-copy-btn');
      await flushMicrotasks();

      assert.strictEqual(getClipboard(), raw);
    },
  );

  test(
    'clicking copy on a bash output panel preserves whitespace',
    async () => {
      const {document, PanelCopy, getClipboard} = makeWindow();
      // bash panels carry plain-text output in a nested
      // .bash-panel-content child — no data-raw-text override is set,
      // so getRawText must fall back to walking the textContent
      // verbatim (preserving columns, indentation, blank lines).
      const op = document.createElement('div');
      op.className = 'bash-panel';
      const opContent = document.createElement('div');
      opContent.className = 'bash-panel-content';
      const bashOut = 'line one\nline two with    spaces\n  indented';
      opContent.textContent = bashOut;
      op.appendChild(opContent);
      document.body.appendChild(op);

      PanelCopy.addCopyButton(op);
      clickFirst(op, '.panel-copy-btn');
      await flushMicrotasks();

      assert.strictEqual(getClipboard(), bashOut);
    },
  );

  test(
    'clicking copy on a Thoughts container with a nested text block ' +
      'returns the raw markdown of that text block',
    async () => {
      const {document, PanelCopy, getClipboard} = makeWindow();
      // Thoughts panels do not set their own data-raw-text — the
      // walker descends to nested children, picks up the raw markdown
      // stashed by ``text_end`` on the .txt block, and concatenates
      // it with neighbouring text content.
      const llmPanel = document.createElement('div');
      llmPanel.className = 'llm-panel';
      const hdr = document.createElement('div');
      hdr.className = 'llm-panel-hdr';
      hdr.textContent = 'Thoughts';
      const txt = document.createElement('div');
      txt.className = 'txt md-body';
      txt.innerHTML = '<h3>Plan</h3><ol><li>step</li></ol>';
      const rawMd = '### Plan\n\n1. step';
      txt.dataset.rawText = rawMd;
      llmPanel.appendChild(hdr);
      llmPanel.appendChild(txt);
      document.body.appendChild(llmPanel);

      PanelCopy.addCopyButton(llmPanel);
      clickFirst(llmPanel, '.panel-copy-btn');
      await flushMicrotasks();

      assert.strictEqual(getClipboard(), 'Thoughts\n' + rawMd);
    },
  );

  test(
    'clicking copy on a Result panel WITHOUT a data-raw-text ' +
      'override falls back to a textContent walk',
    async () => {
      const {document, PanelCopy, getClipboard} = makeWindow();
      // Older replayed events or plain-text results may not carry
      // data-raw-text.  The copy must still produce something useful
      // (not the empty string).
      const rc = document.createElement('div');
      rc.className = 'ev rc';
      rc.innerHTML =
        '<div class="rc-h"><h3>Result</h3></div>' +
        '<div class="rc-body md-body pre">no markdown here</div>';
      document.body.appendChild(rc);

      PanelCopy.addCopyButton(rc);
      clickFirst(rc, '.panel-copy-btn');
      await flushMicrotasks();

      const got = getClipboard();
      assert.ok(got && got.includes('no markdown here'),
        `expected fallback walk to include the body text, got: ${JSON.stringify(got)}`);
    },
  );

  test('click handler does not collapse / propagate to ancestors', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    panel.dataset.rawText = 'x';
    document.body.appendChild(panel);
    let ancestorClicks = 0;
    document.body.addEventListener('click', () => {
      ancestorClicks++;
    });
    PanelCopy.addCopyButton(panel);
    clickFirst(panel, '.panel-copy-btn');
    assert.strictEqual(ancestorClicks, 0);
  });

  // ---------------------------------------------------------------------
  // Summary
  // ---------------------------------------------------------------------
  console.log('');
  if (failures.length === 0) {
    console.log(`All ${passed} panelCopy tests passed.`);
    process.exit(0);
  } else {
    console.log(`${failures.length} test(s) failed (${passed} passed):`);
    for (const f of failures) {
      console.log(`  - ${f.name}: ${f.error.stack || f.error.message}`);
    }
    process.exit(1);
  }
}

run().catch(e => {
  console.error(e);
  process.exit(1);
});
