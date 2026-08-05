// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const TIPS = ['## First tip\n\nUse **KISS Sorcar** like a pro.'];

function makeWebview({show}) {
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(`window.__TIPS__ = ${JSON.stringify({tips: TIPS, show})};`);
  win.eval(fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function panels(win) {
  return Array.from(win.document.querySelectorAll('kiss-tips-panel'));
}

function panelBodyText(panel) {
  const body = panel.shadowRoot && panel.shadowRoot.querySelector('.tips-body');
  return body ? body.textContent.trim() : '';
}

function testResultSummaryCannotSpawnBlankPanel() {
  const {win} = makeWebview({show: true});

  assert.strictEqual(
    panels(win).length,
    1,
    'fresh install: exactly one auto-shown tips panel',
  );
  assert.ok(
    panelBodyText(panels(win)[0]).includes('Use KISS Sorcar like a pro.'),
    'the auto-shown panel renders the tip contents',
  );

  send(win, {
    type: 'result',
    summary: 'Read media/tips.js fully: <kiss-tips-panel>; auto-show wired.',
    total_tokens: 10,
    cost: '$0.01',
  });

  const after = panels(win);
  assert.strictEqual(
    after.length,
    1,
    'BUG: a result summary containing a raw <kiss-tips-panel> tag ' +
      'mounted a second (blank) Tips window',
  );
  assert.ok(
    panelBodyText(after[0]).includes('Use KISS Sorcar like a pro.'),
    'the surviving panel must be the real tips window, not the blank one',
  );
  win.close();
  console.log('  ok - result summary cannot spawn a blank tips panel');
}

function testPromptTextCannotSpawnBlankPanel() {
  const {win} = makeWebview({show: false});

  send(win, {
    type: 'prompt',
    text: 'why does <kiss-tips-panel> show up blank after a rebuild?',
  });

  assert.strictEqual(
    panels(win).length,
    0,
    'BUG: a prompt containing a raw <kiss-tips-panel> tag mounted a ' +
      'blank Tips window',
  );
  win.close();
  console.log('  ok - prompt text cannot spawn a blank tips panel');
}

function testLegacyCodeSpanMentionStaysVisibleText() {
  const {win} = makeWebview({show: false});

  // Persisted result events from before the HTML summary migration contain
  // Markdown and are replayed unchanged from history.
  send(win, {
    type: 'result',
    summary: 'The Tips window is the `<kiss-tips-panel>` web component.',
    total_tokens: 10,
    cost: '$0.01',
  });

  assert.strictEqual(panels(win).length, 0, 'no panel from a code span');
  const output = win.document.getElementById('output');
  assert.ok(
    output.textContent.includes('<kiss-tips-panel>'),
    'a legacy code-span mention must stay visible as literal text',
  );
  assert.ok(output.querySelector('code'), 'legacy Markdown code span renders');
  win.close();
  console.log('  ok - legacy code-span mention stays visible literal text');
}

function testHtmlCodeSpanMentionStaysVisibleText() {
  const {win} = makeWebview({show: false});

  send(win, {
    type: 'result',
    summary:
      '<p>The Tips window is the ' +
      '<code>&lt;kiss-tips-panel&gt;</code> web component.</p>',
    total_tokens: 10,
    cost: '$0.01',
  });

  assert.strictEqual(panels(win).length, 0, 'no panel from escaped HTML');
  const output = win.document.getElementById('output');
  assert.ok(output.textContent.includes('<kiss-tips-panel>'));
  assert.ok(output.querySelector('code'), 'HTML code element survives');
  win.close();
  console.log('  ok - HTML code-span mention stays visible literal text');
}

function testLegacyKnownTagCodeSpanStaysLiteral() {
  const {win} = makeWebview({show: false});

  send(win, {
    type: 'result',
    summary: 'Legacy `<div>` code and **bold** history.',
  });

  const output = win.document.getElementById('output');
  const code = output.querySelector('code');
  assert.ok(code, 'known HTML tag inside legacy code span must render as code');
  assert.strictEqual(code.textContent, '<div>');
  assert.ok(output.querySelector('strong'), 'legacy Markdown remains formatted');
  win.close();
  console.log('  ok - known HTML tag in legacy code span stays literal');
}

function testLegacyEncodedJavascriptLinkIsStripped() {
  const {win} = makeWebview({show: false});

  send(win, {
    type: 'result',
    summary: '[click](java&#x09;script:window.__pwned=1)',
  });

  const link = win.document.querySelector('#output a');
  assert.ok(link, 'legacy Markdown link renders');
  assert.ok(!link.hasAttribute('href'), 'encoded javascript URL is stripped');
  assert.strictEqual(win.__pwned, undefined, 'unsafe URL must not execute');
  win.close();
  console.log('  ok - encoded javascript URL is stripped');
}

function testParsedPanelSelfRemovesButProgrammaticPanelSurvives() {
  const {win} = makeWebview({show: false});

  const div = win.document.createElement('div');
  div.innerHTML = '<kiss-tips-panel></kiss-tips-panel>';
  win.document.body.appendChild(div);
  assert.strictEqual(
    panels(win).length,
    0,
    'BUG: a <kiss-tips-panel> upgraded from parsed HTML (no tips ever ' +
      'assigned) must self-remove instead of covering the chat as a ' +
      'blank overlay',
  );

  const empty = win.__kissShowTipsPanel([]);
  assert.strictEqual(panels(win).length, 1, 'empty panel still mounts');
  empty.shadowRoot.querySelector('.tips-close').click();
  assert.strictEqual(panels(win).length, 0, 'empty panel closes');

  win.__kissShowTipsPanel(TIPS);
  assert.strictEqual(panels(win).length, 1, 'real panel still mounts');
  assert.ok(panelBodyText(panels(win)[0]).includes('Use KISS Sorcar'));
  win.close();
  console.log('  ok - parsed panel self-removes; programmatic panels work');
}

testResultSummaryCannotSpawnBlankPanel();
testPromptTextCannotSpawnBlankPanel();
testLegacyCodeSpanMentionStaysVisibleText();
testHtmlCodeSpanMentionStaysVisibleText();
testLegacyKnownTagCodeSpanStaysLiteral();
testLegacyEncodedJavascriptLinkIsStripped();
testParsedPanelSelfRemovesButProgrammaticPanelSurvives();
console.log('tipsNoBlankPanelFromTranscript: all tests passed');
