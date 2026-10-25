// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const PANEL_COPY_PATH = path.join(__dirname, '..', 'media', 'panelCopy.js');

const {test, report} = require('./ui_antipattern_harness').makeRunner();

// Every window made, closed once the run is over so node can exit.
const openWindows = [];

function makeWindow() {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
  });
  const win = dom.window;
  openWindows.push(win);
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
  const src = fs.readFileSync(PANEL_COPY_PATH, 'utf-8');
  const scriptEl = win.document.createElement('script');
  scriptEl.textContent = src;
  win.document.head.appendChild(scriptEl);
  assert.ok(win.PanelCopy, 'panelCopy.js must expose window.PanelCopy');
  return {
    document: win.document,
    PanelCopy: win.PanelCopy,
    getClipboard: () => clipboardText,
  };
}

function clickFirst(el, selector) {
  const btn = el.querySelector(selector);
  assert.ok(btn, `expected to find ${selector} inside panel`);
  btn.dispatchEvent(
    new btn.ownerDocument.defaultView.MouseEvent('click', {
      bubbles: true,
      cancelable: true,
    }),
  );
}

function flushMicrotasks() {
  return new Promise(resolve => setImmediate(resolve));
}

async function run() {
  await test('exports getRawText + addCopyButton on window.PanelCopy', () => {
    const {PanelCopy} = makeWindow();
    assert.strictEqual(typeof PanelCopy.getRawText, 'function');
    assert.strictEqual(typeof PanelCopy.addCopyButton, 'function');
  });

  await test('getRawText honours data-raw-text override', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    panel.innerHTML = '<h3>Heading</h3><p><strong>bold</strong> text</p>';
    panel.dataset.rawText = '# Heading\n\n**bold** text';
    assert.strictEqual(
      PanelCopy.getRawText(panel),
      '# Heading\n\n**bold** text',
    );
  });

  await test('getRawText skips copy-btn / collapse-preview chrome', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    panel.innerHTML =
      '<button class="panel-copy-btn">copy</button>' +
      '<span class="collapse-preview">preview</span>' +
      '<div>real content</div>';
    assert.strictEqual(PanelCopy.getRawText(panel), 'real content');
  });

  await test('getRawText recurses through children when no override is set', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    const a = document.createElement('div');
    a.textContent = 'alpha';
    const b = document.createElement('div');
    b.textContent = 'beta';
    panel.appendChild(a);
    panel.appendChild(b);
    assert.strictEqual(PanelCopy.getRawText(panel), 'alpha\nbeta');
  });

  await test('getRawText prefers nested data-raw-text over child markup', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    const header = document.createElement('div');
    header.textContent = 'Result';
    const body = document.createElement('div');
    body.innerHTML = '<h1>Hello</h1><p>World</p>';
    body.dataset.rawText = '# Hello\n\nWorld';
    panel.appendChild(header);
    panel.appendChild(body);
    assert.strictEqual(PanelCopy.getRawText(panel), 'Result\n# Hello\n\nWorld');
  });

  await test('addCopyButton attaches a single button (idempotent)', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    document.body.appendChild(panel);
    PanelCopy.addCopyButton(panel);
    PanelCopy.addCopyButton(panel);
    const btns = panel.querySelectorAll(':scope > .panel-copy-btn');
    assert.strictEqual(btns.length, 1);
    assert.ok(panel.classList.contains('copyable'));
  });

  await test('clicking copy on a Result panel copies the raw markdown', async () => {
    const {document, PanelCopy, getClipboard} = makeWindow();
    const rc = document.createElement('div');
    rc.className = 'ev rc';
    rc.innerHTML =
      '<div class="rc-h"><h3>Result</h3></div>' +
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
    assert.notStrictEqual(getClipboard(), 'Result Hello World');
  });

  await test('clicking copy on a Prompt panel copies the raw markdown', async () => {
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
  });

  await test('clicking copy on a bash output panel preserves whitespace', async () => {
    const {document, PanelCopy, getClipboard} = makeWindow();
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
  });

  await test(
    'clicking copy on a Thoughts container with a nested text block ' +
      'returns the raw markdown of that text block',
    async () => {
      const {document, PanelCopy, getClipboard} = makeWindow();
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

  await test(
    'clicking copy on a Result panel WITHOUT a data-raw-text ' +
      'override falls back to a textContent walk',
    async () => {
      const {document, PanelCopy, getClipboard} = makeWindow();
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
      assert.ok(
        got && got.includes('no markdown here'),
        `expected fallback walk to include the body text, got: ${JSON.stringify(got)}`,
      );
    },
  );

  await test('click handler does not collapse / propagate to ancestors', () => {
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

  for (const win of openWindows) win.close();
  report('panelCopy');
}

run().catch(e => {
  console.error(e);
  process.exit(1);
});
