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
const {stubClipboard, flushMicrotasks, click} = require('./simplify2_harness');
const {makePage, send, queueRaf} = require('./simplify2_harness');

// Every window made, closed once the run is over so node can exit.
const openWindows = [];

function makeWindow() {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
  });
  const win = dom.window;
  openWindows.push(win);
  const clip = stubClipboard(win);
  const src = fs.readFileSync(PANEL_COPY_PATH, 'utf-8');
  const scriptEl = win.document.createElement('script');
  scriptEl.textContent = src;
  win.document.head.appendChild(scriptEl);
  assert.ok(win.PanelCopy, 'panelCopy.js must expose window.PanelCopy');
  return {
    document: win.document,
    PanelCopy: win.PanelCopy,
    getClipboard: clip.last,
  };
}

/**
 * Replace *document*'s `execCommand` with a recorder that reports the
 * text the fallback textarea held when 'copy' ran, then returns *ok*.
 */
function recordExecCommand(document, ok) {
  const calls = [];
  document.execCommand = cmd => {
    const ta = document.querySelector('textarea');
    calls.push({cmd, text: ta ? ta.value : null});
    return ok;
  };
  return calls;
}

function clickFirst(el, selector) {
  const btn = el.querySelector(selector);
  assert.ok(btn, `expected to find ${selector} inside panel`);
  click(btn.ownerDocument.defaultView, btn);
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

  // The one chrome predicate every text walker shares (getRawText here,
  // main.js's collapsed-preview text): the panel's copy and stop
  // buttons, its collapsed preview and its timestamp / elapsed labels
  // are chrome; anything else, and a non-element, is content.
  await test('shouldSkip is exported and names exactly the panel chrome', () => {
    const {document, PanelCopy} = makeWindow();
    assert.strictEqual(typeof PanelCopy.shouldSkip, 'function');
    [
      'panel-copy-btn',
      'panel-stop-btn',
      'collapse-preview',
      'panel-ts',
      'panel-time',
    ].forEach(cls => {
      const el = document.createElement('span');
      el.className = 'other ' + cls;
      assert.strictEqual(PanelCopy.shouldSkip(el), true, cls + ' is chrome');
    });
    const content = document.createElement('div');
    content.className = 'tc-body panel-text';
    assert.strictEqual(PanelCopy.shouldSkip(content), false);
    assert.strictEqual(PanelCopy.shouldSkip(document.createTextNode('t')), false);
    assert.strictEqual(PanelCopy.shouldSkip(null), false);
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

  // Audit 2026-11-21 (L4): a file link, a highlighted token, a diff
  // highlight and an argument's value are inline: each stays on the
  // line of the text around it.  A block still starts a line, and ends
  // it before an inline element that follows, without a second break
  // where its text or its neighbour's already has one.
  await test('getRawText keeps an inline element on its line', () => {
    const {document, PanelCopy} = makeWindow();
    const panel = document.createElement('div');
    panel.innerHTML =
      '<div><span>path:</span> <span data-path="/ws/a.py">/ws/a.py</span></div>' +
      '<pre><code><span>cat</span> /ws/a.py | <span>head</span> -3</code></pre>' +
      '<div>- f(<span>old</span>) <a href="#">link</a> <code>x</code> ' +
      '<strong>y</strong> <em>z</em></div>' +
      '<span>after a block</span><span>, on its line</span>' +
      '<div>ends\n</div><div>a line</div><div>\nstarts</div>';
    assert.strictEqual(
      PanelCopy.getRawText(panel),
      'path: /ws/a.py\ncat /ws/a.py | head -3\n' +
        '- f(old) link x y z\nafter a block, on its line\n' +
        'ends\na line\nstarts',
    );
    // The one inline predicate the text walkers share (getRawText
    // here, main.js's collapsed-preview text).
    assert.strictEqual(PanelCopy.isInline(document.createElement('ins')), true);
    assert.strictEqual(
      PanelCopy.isInline(document.createElement('div')),
      false,
    );
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

  // Audit 2026-11-21 (L4), on the real page: a streamed Bash line that
  // names a file is copied as the panel shows it, while the path waits
  // for the daemon's verdict and once it is a link; so is the call's
  // panel, whose command is highlighted token by token.  Here the
  // frames, the clipboard and the verdict are the harness's.  With a
  // real browser's, and the daemon's own verdict on a real file:
  // tests/webapp/test_result_ls_listing_file_links.py
  // (test_copy_keeps_a_streamed_file_path_on_its_line).
  await test('copy of a Bash output panel keeps a file path on its line', () => {
    const {win} = makePage({
      extraScripts: ['highlight.min.js'],
      beforeScripts: w => (w.flushRaf = queueRaf(w).run),
    });
    openWindows.push(win);
    const copied = stubClipboard(win).writes;
    const tabId = win._testApi.getActiveTabId();
    const workDir = '/ws/proj';
    const command = 'cat /ws/proj/out/x.log | head -3';
    const line = 'Log: /ws/proj/out/x.log done';
    send(win, {type: 'configData', config: {work_dir: workDir}, apiKeys: {}});
    const bash = {type: 'tool_call', name: 'Bash', command, description: 'd'};
    send(win, {...bash, tabId});
    send(win, {type: 'system_output', text: line + '\n', tabId});
    win.flushRaf();
    const call = win.document.querySelector('#output .ev.tc');
    const output = call.querySelector('.bash-panel');
    clickFirst(output, '.panel-copy-btn');
    const results = {'/ws/proj/out/x.log': true};
    send(win, {type: 'pathsExist', results, tabId, workDir});
    assert.ok(output.querySelector('.kiss-filelink'), 'the path is a link');
    clickFirst(output, '.panel-copy-btn');
    assert.ok(call.querySelector('pre code span'), 'the command has tokens');
    clickFirst(call, ':scope > .panel-copy-btn');
    const whole = 'Bash\ndescription: d\n' + command + '\n' + line;
    assert.deepStrictEqual(copied, [line, line, whole]);
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

  await test('copyText falls back to execCommand when clipboard has no writeText', async () => {
    const {document, PanelCopy} = makeWindow();
    Object.defineProperty(document.defaultView.navigator, 'clipboard', {
      configurable: true,
      value: {},
    });
    const calls = recordExecCommand(document, true);
    assert.strictEqual(await PanelCopy.copyText('plain text', document), true);
    assert.deepStrictEqual(calls, [{cmd: 'copy', text: 'plain text'}]);
    assert.strictEqual(
      document.querySelector('textarea'),
      null,
      'the helper textarea is removed after the copy',
    );
  });

  await test('copyText falls back to execCommand when writeText throws', async () => {
    const {document, PanelCopy} = makeWindow();
    Object.defineProperty(document.defaultView.navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: () => {
          throw new Error('not allowed');
        },
      },
    });
    const calls = recordExecCommand(document, true);
    assert.strictEqual(await PanelCopy.copyText('thrown', document), true);
    assert.deepStrictEqual(calls, [{cmd: 'copy', text: 'thrown'}]);
    assert.strictEqual(document.querySelector('textarea'), null);
  });

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
