// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end webview tests: the Result panel's copy button copies the
// FORMATTED text of the rendered summary — never the summary's raw
// HTML markup.
//
// Reproduces the issue where createResultPanel() stored the raw HTML
// summary string in dataset.rawText, so pressing the panel's copy
// button put "<h2>Report</h2><ul><li>..." on the clipboard.
//
// The same media/main.js + media/panelCopy.js are served to BOTH the
// VS Code webview and the remote webapp (src/kiss/server/
// web_server.py), so this covers both surfaces.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
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
  win.HTMLElement.prototype.scrollTo = function () {};

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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage() {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, getClipboard: () => clipboardText};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickCopy(win) {
  const rc = win.document.querySelector('#output > .rc');
  assert.ok(rc, 'expected a Result panel');
  const btn = rc.querySelector('.panel-copy-btn');
  assert.ok(btn, 'Result panel must have a copy button');
  btn.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

function flushMicrotasks() {
  return new Promise(resolve => setImmediate(resolve));
}

async function testHtmlSummaryCopiesFormattedText() {
  const {win, getClipboard} = makeWebview();
  send(win, {
    type: 'result',
    success: true,
    is_continue: false,
    summary:
      '<h2>Report</h2>' +
      '<p>All <b>good</b>.</p>' +
      '<ul><li>alpha</li><li>beta</li></ul>' +
      '<ol start="3"><li>three</li><li>four</li></ol>' +
      '<pre><code>x = 1\ny = 2</code></pre>' +
      '<table><tr><th>k</th><th>v</th></tr>' +
      '<tr><td>a</td><td>b</td></tr></table>',
    total_tokens: 20,
    cost: '$0.02',
  });
  clickCopy(win);
  await flushMicrotasks();
  const text = getClipboard();
  assert.ok(text, 'copy button must write to the clipboard');
  assert.ok(
    !/<\/?(h2|p|ul|li|pre|code|table|b)\b/i.test(text),
    'BUG: clipboard must not contain raw HTML tags; got: ' + text,
  );
  assert.ok(text.includes('Report'), 'heading text must survive');
  assert.ok(text.includes('All good.'), 'inline markup must flatten');
  assert.ok(text.includes('- alpha'), 'ul items get "- " markers');
  assert.ok(text.includes('- beta'), 'every ul item gets a marker');
  assert.ok(text.includes('3. three'), 'ol honours its start number');
  assert.ok(text.includes('4. four'), 'ol numbering increments');
  assert.ok(
    text.includes('x = 1\ny = 2'),
    'pre content keeps its own line breaks',
  );
  assert.ok(text.includes('k | v'), 'table header cells join with " | "');
  assert.ok(text.includes('a | b'), 'table row cells join with " | "');
  assert.ok(
    /Report\n\nAll good\./.test(text),
    'paragraph blocks separate with a blank line; got: ' + text,
  );
  win.close();
}

async function testPlainTextResultStillCopiesItsText() {
  const {win, getClipboard} = makeWebview();
  send(win, {
    type: 'result',
    success: true,
    is_continue: false,
    text: 'plain output line 1\nline 2 <not-a-tag>',
    total_tokens: 5,
    cost: '$0.01',
  });
  clickCopy(win);
  await flushMicrotasks();
  const text = getClipboard();
  assert.ok(
    text.includes('plain output line 1\nline 2 <not-a-tag>'),
    'a plain-text result copies its text unchanged; got: ' + text,
  );
  win.close();
}

async function testFailedStatusLineIsCopiedTooAndChromeIsNot() {
  const {win, getClipboard} = makeWebview();
  send(win, {
    type: 'result',
    success: false,
    is_continue: false,
    summary: '<p>it broke</p>',
    total_tokens: 5,
    cost: '$0.01',
  });
  clickCopy(win);
  await flushMicrotasks();
  const text = getClipboard();
  assert.ok(
    text.includes('Status: FAILED'),
    'the status line is part of the copied text; got: ' + text,
  );
  assert.ok(text.includes('it broke'));
  assert.ok(
    !/<\/?p\b/.test(text),
    'BUG: clipboard must not contain raw HTML tags; got: ' + text,
  );
  win.close();
}

async function testEmptyAndImageOnlySummariesNeverCopyRawHtml() {
  // An HTML summary whose formatted text is empty (or reduces to an
  // image's alt text) must still never fall back to the raw markup.
  const cases = [
    {summary: '<p></p>', expect: ''},
    {summary: '<br>', expect: ''},
    {summary: '<img src="x" alt="diagram">', expect: 'diagram'},
  ];
  for (const c of cases) {
    const {win, getClipboard} = makeWebview();
    send(win, {
      type: 'result',
      success: true,
      is_continue: false,
      summary: c.summary,
      total_tokens: 1,
      cost: '$0.01',
    });
    clickCopy(win);
    await flushMicrotasks();
    const text = getClipboard();
    assert.ok(
      !/<[a-z][^>]*>/i.test(text),
      'BUG: raw HTML must never reach the clipboard for ' +
        JSON.stringify(c.summary) +
        '; got: ' +
        text,
    );
    assert.strictEqual(
      text,
      c.expect,
      'formatted text for ' + JSON.stringify(c.summary),
    );
    win.close();
  }
}

function testFormattedTextFromNodeBranches() {
  const {win} = makeWebview();
  const fmt = win.PanelCopy.formattedTextFromNode;
  const doc = win.document;
  const el = doc.createElement('div');
  el.innerHTML =
    '<p>a<br>b</p><hr>' +
    '<ul><li>top<ul><li>nested</li></ul></li></ul>' +
    '<script>bad()</script>' +
    '<div class="panel-copy-btn">chrome</div>' +
    '<ol start="x"><li>fallback-one</li></ol>';
  const text = fmt(el).replace(/^\n+|\n+$/g, '');
  assert.ok(text.includes('a\nb'), '<br> becomes a newline; got: ' + text);
  assert.ok(text.includes('---'), '<hr> becomes a rule line');
  assert.ok(text.includes('- top'), 'outer li gets a marker');
  assert.ok(text.includes('  - nested'), 'nested li indents two spaces');
  assert.ok(!text.includes('bad()'), 'script content is skipped');
  assert.ok(!text.includes('chrome'), 'panel chrome (copy button) skipped');
  assert.ok(
    text.includes('1. fallback-one'),
    'a non-numeric ol start falls back to 1; got: ' + text,
  );
  win.close();
}

async function main() {
  await testHtmlSummaryCopiesFormattedText();
  console.log('  ok - HTML summary copies as formatted text, not raw HTML');
  await testPlainTextResultStillCopiesItsText();
  console.log('  ok - plain-text result still copies its text unchanged');
  await testFailedStatusLineIsCopiedTooAndChromeIsNot();
  console.log('  ok - status line copied, HTML tags still absent');
  await testEmptyAndImageOnlySummariesNeverCopyRawHtml();
  console.log('  ok - empty/image-only summaries never copy raw HTML');
  testFormattedTextFromNodeBranches();
  console.log('  ok - formattedTextFromNode branch behaviours');
  console.log('resultPanelFormattedCopy.test.js: all tests passed');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
