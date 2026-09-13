// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM integration test: images a tool call generated — embedded by
// the server as base64 `images` payloads on the tool_result event —
// are rendered inline in the corresponding event panel (the tool-call
// panel when one exists, the transcript otherwise), with a path
// caption and click-to-zoom.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// A real, valid 1x1 transparent PNG.
const PNG_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg' +
  'YGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==';

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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};

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

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function startTask(wv, chatId) {
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  send(wv.win, {type: 'clear', chat_id: chatId, tabId: ready.tabId});
  send(wv.win, {
    type: 'status',
    running: true,
    tabId: ready.tabId,
    startTs: Date.now(),
  });
  return ready.tabId;
}

function testImagesRenderInToolPanel() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = startTask(wv, 'chat-img-1');
  const output = win.document.getElementById('output');

  send(win, {
    type: 'tool_call',
    name: 'screenshot',
    path: 'shot.png',
    tabId: TAB,
  });
  send(win, {
    type: 'tool_result',
    content: 'Screenshot saved to /w/shot.png',
    is_error: false,
    tool_name: 'screenshot',
    images: [{path: '/w/shot.png', mime: 'image/png', b64: PNG_B64}],
    tabId: TAB,
  });

  const tc = output.querySelector('.ev.tc');
  assert.ok(tc, 'tool_call panel must exist');
  const wrap = tc.querySelector('.tr-images');
  assert.ok(
    wrap,
    'BUG: the generated image must be rendered inside the tool-call ' +
      'event panel (.tr-images missing)',
  );
  const img = wrap.querySelector('img.tr-img');
  assert.ok(img, 'an <img> must be rendered for the embedded image');
  assert.strictEqual(
    img.getAttribute('src'),
    'data:image/png;base64,' + PNG_B64,
    'the image must be rendered from its embedded base64 payload',
  );
  assert.strictEqual(img.getAttribute('alt'), '/w/shot.png');
  assert.strictEqual(img.getAttribute('title'), '/w/shot.png');
  const cap = wrap.querySelector('.tr-img-cap');
  assert.ok(cap, 'the image must carry its path as a caption');
  assert.strictEqual(cap.textContent, '/w/shot.png');

  // The textual result panel must still be rendered alongside, and
  // the images must come AFTER it inside the same tool panel.
  const op = tc.querySelector('.bash-panel');
  assert.ok(op, 'the textual output panel must still render');
  assert.ok(
    op.compareDocumentPosition(wrap) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    'images must be appended after the textual output panel',
  );

  // Click-to-zoom toggles the full-size class both ways.
  img.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    img.classList.contains('tr-img-full'),
    'clicking the image must expand it to natural size',
  );
  img.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !img.classList.contains('tr-img-full'),
    'clicking again must restore the capped preview',
  );
  console.log('ok: images render inside the tool-call event panel');
}

function testImagesRenderOnStreamedBashPath() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = startTask(wv, 'chat-img-2');
  const output = win.document.getElementById('output');

  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'python plot.py',
    description: 'plot',
    tabId: TAB,
  });
  // Streamed bash output: the server suppresses the duplicate result
  // text (content '') but still embeds the generated image.
  send(win, {type: 'system_output', text: 'Saved tmp/plot.png\n', tabId: TAB});
  send(win, {
    type: 'tool_result',
    content: '',
    is_error: false,
    tool_name: 'Bash',
    images: [{path: 'tmp/plot.png', mime: 'image/png', b64: PNG_B64}],
    tabId: TAB,
  });

  const tc = output.querySelector('.ev.tc');
  assert.ok(tc, 'bash tool_call panel must exist');
  const img = tc.querySelector('.tr-images img.tr-img');
  assert.ok(
    img,
    'BUG: images must render even when bash output was streamed ' +
      '(the early-break path in the tool_result case)',
  );
  assert.strictEqual(
    img.getAttribute('src'),
    'data:image/png;base64,' + PNG_B64,
  );
  console.log('ok: images render on the streamed-bash early-break path');
}

function testNoImagesNoPanelAndBadEntriesSkipped() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = startTask(wv, 'chat-img-3');
  const output = win.document.getElementById('output');

  // 1. A result without images must not create a .tr-images wrapper.
  send(win, {type: 'tool_call', name: 'Read', path: 'a.txt', tabId: TAB});
  send(win, {
    type: 'tool_result',
    content: 'file contents',
    is_error: false,
    tool_name: 'Read',
    tabId: TAB,
  });
  assert.strictEqual(
    output.querySelectorAll('.tr-images').length,
    0,
    'no .tr-images wrapper may appear for an image-less result',
  );

  // 2. Entries missing b64/mime are skipped; an all-bad list renders
  //    nothing (the empty wrapper is not appended).
  send(win, {type: 'tool_call', name: 'screenshot', tabId: TAB});
  send(win, {
    type: 'tool_result',
    content: 'x',
    is_error: false,
    tool_name: 'screenshot',
    images: [{path: 'a.png'}, {path: 'b.png', mime: 'image/png'}, null],
    tabId: TAB,
  });
  assert.strictEqual(
    output.querySelectorAll('.tr-images').length,
    0,
    'entries without a b64 payload must be skipped entirely',
  );

  // 3. An image without a path renders with the fallback alt text and
  //    no caption.
  send(win, {type: 'tool_call', name: 'screenshot', tabId: TAB});
  send(win, {
    type: 'tool_result',
    content: 'y',
    is_error: false,
    tool_name: 'screenshot',
    images: [{mime: 'image/png', b64: PNG_B64}],
    tabId: TAB,
  });
  const wraps = output.querySelectorAll('.tr-images');
  assert.strictEqual(wraps.length, 1, 'the valid image must render');
  const img = wraps[0].querySelector('img.tr-img');
  assert.strictEqual(img.getAttribute('alt'), 'tool result image');
  assert.strictEqual(
    wraps[0].querySelector('.tr-img-cap'),
    null,
    'no caption without a path',
  );
  console.log('ok: image-less results and bad entries render no images');
}

// Exported share pages serialize the live DOM's outerHTML, which drops
// per-element listeners; share.js must re-wire click-to-zoom through
// its delegated document click handler.
function testSharePageClickToZoom() {
  const shareJs = fs.readFileSync(path.join(MEDIA, 'share.js'), 'utf8');
  const dom = new JSDOM(
    '<!DOCTYPE html><html><head>' +
      '<style id="hljs-style-dark">.hljs{color:#fff}</style>' +
      '<style id="hljs-style-light" media="not all">.hljs{color:#000}' +
      '</style></head><body>' +
      '<button id="share-theme-btn" type="button"></button>' +
      '<div id="app"><div id="output">' +
      '<div class="ev tc"><div class="tr-images"><div class="tr-img-box">' +
      '<img class="tr-img" src="data:image/png;base64,' +
      PNG_B64 +
      '" alt="/w/shot.png">' +
      '<div class="tr-img-cap">/w/shot.png</div>' +
      '</div></div></div>' +
      '</div></div></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true, url: 'https://x/'},
  );
  const win = dom.window;
  win.eval(shareJs + '\n//# sourceURL=share.js');
  const img = win.document.querySelector('img.tr-img');
  const clickOpts = {bubbles: true, cancelable: true};
  img.dispatchEvent(new win.MouseEvent('click', clickOpts));
  assert.ok(
    img.classList.contains('tr-img-full'),
    'BUG: clicking an image on the exported share page must zoom it ' +
      '(delegated .tr-img handler missing in share.js)',
  );
  img.dispatchEvent(new win.MouseEvent('click', clickOpts));
  assert.ok(
    !img.classList.contains('tr-img-full'),
    'clicking again on the share page must restore the preview size',
  );
  // A click on the caption (not the image) must not toggle anything.
  const cap = win.document.querySelector('.tr-img-cap');
  cap.dispatchEvent(new win.MouseEvent('click', clickOpts));
  assert.ok(
    !img.classList.contains('tr-img-full'),
    'caption clicks must not zoom the image',
  );
  console.log('ok: exported share page keeps click-to-zoom');
}

function main() {
  testImagesRenderInToolPanel();
  testImagesRenderOnStreamedBashPath();
  testNoImagesNoPanelAndBadEntriesSkipped();
  testSharePageClickToZoom();
  console.log('All tests passed');
  // The webview keeps ticking panel-time intervals; exit explicitly
  // like the other tests so node does not wait on them forever.
  process.exit(0);
}

try {
  main();
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
