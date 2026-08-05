// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end jsdom tests: opening an .html file (or a generated report) in
// a Sorcar content tab must offer a contextual menu on right-click with at
// least Copy and Select All.
//
// The bug: the content tab renders the file inside
// `<iframe sandbox="allow-scripts">`, and a VS Code webview suppresses
// Chromium's native context menu, so right-clicking used to do nothing at
// all — no Copy, no Select All, no way to get the text out of the tab.
// The iframe's origin is opaque (no `allow-same-origin`), so the parent
// cannot reach into it; the menu must be shipped inside the srcdoc.
//
// These tests drive the real webview (media/chat.html + media/main.js +
// media/contentContextMenu.js) exactly as the extension and the remote web
// app do, render the produced srcdoc into a second jsdom document — which
// is what the browser does with the sandboxed iframe — and then right-click
// inside it.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const MENU_SELECTOR = '#sorcar-content-context-menu';

function makeWebview(opts) {
  opts = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (opts.remote) html = html.replace('<body', '<body class="remote-chat"');

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  // media/chat.html loads contentContextMenu.js before main.js; omitting it
  // (opts.withoutMenuModule) reproduces the pre-fix webview.
  if (!opts.withoutMenuModule) {
    win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  }
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=ctxmenu-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function openHtmlFile(win, name, content) {
  send(win, {type: 'fileContent', name: name, path: name, content: content});
}

function activeSrcdoc(win) {
  const frames = Array.from(
    win.document.querySelectorAll(
      '#content-tab-area .content-tab-view iframe.content-html-frame',
    ),
  ).filter(f => f.closest('.content-tab-view').style.display !== 'none');
  assert.strictEqual(frames.length, 1, 'expected exactly one visible frame');
  return frames[0].getAttribute('srcdoc') || '';
}

// The browser parses the srcdoc into its own document; jsdom reproduces that
// faithfully, including running the bootstrap <script> the fix injects.
//
// The bootstrap is a Function.prototype.toString re-serialisation of
// media/contentContextMenu.js, so V8 attributes its execution to the iframe
// rather than to the module file.  Re-evaluating the module text over the
// same document with a //# sourceURL pragma keeps the coverage gate honest
// while running the exact same code — and asserting that the two agree is
// itself a regression test against the two copies drifting apart.
// Exactly what the browser does with the iframe: nothing but the srcdoc.
function renderPlain(srcdoc) {
  return new JSDOM(srcdoc, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
  }).window;
}

function renderSandbox(srcdoc) {
  const win = renderPlain(srcdoc);
  assert.ok(
    win.document.querySelector('script[data-sorcar-ctx]'),
    'the sandbox document must carry the injected bootstrap',
  );
  assert.ok(
    win.__sorcarContentContextMenu,
    'the bootstrap must publish its handle so it can be torn down',
  );
  win.__sorcarContentContextMenu.dispose();
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8') +
      '\n//# sourceURL=contentContextMenu.js',
  );
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  return win;
}

function rightClick(win, target, x, y) {
  const ev = new win.MouseEvent('contextmenu', {
    bubbles: true,
    cancelable: true,
    clientX: x || 40,
    clientY: y || 60,
  });
  target.dispatchEvent(ev);
  return ev;
}

function menuOf(win) {
  return win.document.querySelector(MENU_SELECTOR);
}

function menuActions(win) {
  const menu = menuOf(win);
  if (!menu) return [];
  return Array.from(menu.querySelectorAll('.sorcar-ctx-item')).map(
    el => el.dataset.action,
  );
}

function clickAction(win, action) {
  const menu = menuOf(win);
  assert.ok(menu, 'menu must be open to click ' + action);
  const el = menu.querySelector('[data-action="' + action + '"]');
  assert.ok(el, 'no menu item ' + action + ' in ' + menuActions(win));
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true, cancelable: true}));
}

// execCommand('copy') is not implemented by jsdom.  The sandboxed iframe has
// no async Clipboard API either, so the production path is always the hidden
// textarea; its live value at execCommand time is what a browser would copy.
function captureClipboard(win) {
  const box = {text: null};
  win.document.execCommand = function (cmd) {
    if (cmd !== 'copy') return false;
    const areas = win.document.querySelectorAll('textarea[readonly]');
    box.text = areas.length ? areas[areas.length - 1].value : '';
    return true;
  };
  return box;
}

const PAGE = [
  '<!DOCTYPE html>',
  '<html><head><title>Report</title>',
  '<style>body{color:#333}</style>',
  '</head><body>',
  '<h1>Quarterly Report</h1>',
  '<p id="p">Revenue grew by 12%.</p>',
  '<p><a id="lnk" href="https://example.com/x">details</a></p>',
  '<p><img id="pic" src="https://example.com/chart.png" alt="chart"></p>',
  '<textarea id="ta">draft</textarea>',
  '<script>window.__PAGE_SCRIPT__ = 1;</' + 'script>',
  '</body></html>',
].join('\n');

// ---------------------------------------------------------------- the bug

function testBugWithoutMenuModuleNoContextMenu() {
  const {win} = makeWebview({withoutMenuModule: true});
  openHtmlFile(win, 'report.html', PAGE);
  const srcdoc = activeSrcdoc(win);
  assert.strictEqual(
    srcdoc,
    PAGE,
    'without the fix the srcdoc is the raw file, unchanged',
  );
  const sandbox = renderPlain(srcdoc);
  const ev = rightClick(sandbox, sandbox.document.getElementById('p'));
  assert.strictEqual(
    menuOf(sandbox),
    null,
    'BUG REPRODUCED: right-clicking an .html tab shows no contextual menu',
  );
  assert.strictEqual(
    ev.defaultPrevented,
    false,
    'and nothing intercepts the contextmenu event',
  );
  console.log('  ok - bug reproduced without the context-menu module');
}

// ------------------------------------------------------------- the fix

function testHtmlTabShipsMenuIntoSandbox() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const srcdoc = activeSrcdoc(win);
  assert.ok(
    srcdoc.indexOf('data-sorcar-ctx') > 0,
    'the fixed srcdoc must carry the context-menu bootstrap',
  );
  assert.ok(
    srcdoc.indexOf('<h1>Quarterly Report</h1>') > 0,
    'the original document must be preserved verbatim',
  );
  assert.ok(
    srcdoc.lastIndexOf('data-sorcar-ctx') < srcdoc.lastIndexOf('</body>'),
    'the bootstrap must be injected before </body>',
  );
  console.log('  ok - html tab srcdoc carries the menu bootstrap');
}

function testRightClickOpensCopyAndSelectAll() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  const ev = rightClick(sandbox, sandbox.document.getElementById('p'));
  assert.strictEqual(
    ev.defaultPrevented,
    true,
    'the native menu must be suppressed in favour of ours',
  );
  const menu = menuOf(sandbox);
  assert.ok(menu, 'right-click must open the contextual menu');
  const actions = menuActions(sandbox);
  assert.ok(actions.indexOf('copy') >= 0, 'Copy must be offered: ' + actions);
  assert.ok(
    actions.indexOf('select-all') >= 0,
    'Select All must be offered: ' + actions,
  );
  assert.strictEqual(menu.getAttribute('role'), 'menu');
  console.log('  ok - right-click offers Copy and Select All');
}

function testSelectAllThenCopyGrabsWholeDocument() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  const clip = captureClipboard(sandbox);
  rightClick(sandbox, sandbox.document.getElementById('p'));
  clickAction(sandbox, 'select-all');
  assert.strictEqual(menuOf(sandbox), null, 'the menu closes after activation');
  rightClick(sandbox, sandbox.document.getElementById('p'));
  clickAction(sandbox, 'copy');
  assert.ok(clip.text, 'Copy after Select All must place text on the clipboard');
  assert.ok(
    clip.text.indexOf('Quarterly Report') >= 0 &&
      clip.text.indexOf('Revenue grew by 12%.') >= 0,
    'the whole document text must be copied, got: ' + clip.text,
  );
  assert.strictEqual(
    clip.text.indexOf('__PAGE_SCRIPT__'),
    -1,
    'inline <script> source must never reach the clipboard: ' + clip.text,
  );
  assert.strictEqual(
    clip.text.indexOf('sorcar-content-context-menu'),
    -1,
    "the menu's own markup must never reach the clipboard: " + clip.text,
  );
  console.log('  ok - Select All then Copy yields the document text only');
}

function testCopyDisabledWithoutSelection() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  rightClick(sandbox, sandbox.document.getElementById('p'));
  const item = menuOf(sandbox).querySelector('[data-action="copy"]');
  assert.ok(
    item.classList.contains('disabled'),
    'Copy must be disabled while nothing is selected',
  );
  const clip = captureClipboard(sandbox);
  item.dispatchEvent(new sandbox.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(clip.text, null, 'a disabled Copy must do nothing');
  console.log('  ok - Copy is disabled without a selection');
}

function testCopySelectionOnly() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  const doc = sandbox.document;
  const range = doc.createRange();
  range.selectNodeContents(doc.getElementById('p'));
  const sel = sandbox.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
  const clip = captureClipboard(sandbox);
  rightClick(sandbox, doc.getElementById('p'));
  const item = menuOf(sandbox).querySelector('[data-action="copy"]');
  assert.ok(!item.classList.contains('disabled'), 'Copy must be enabled');
  clickAction(sandbox, 'copy');
  assert.strictEqual(clip.text, 'Revenue grew by 12%.');
  console.log('  ok - Copy copies just the selection');
}

function testLinkAndImageAndEditableItems() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  const doc = sandbox.document;

  rightClick(sandbox, doc.getElementById('lnk'));
  assert.deepStrictEqual(menuActions(sandbox), [
    'copy',
    'copy-link',
    'select-all',
  ]);
  const clip = captureClipboard(sandbox);
  clickAction(sandbox, 'copy-link');
  assert.strictEqual(clip.text, 'https://example.com/x');

  rightClick(sandbox, doc.getElementById('pic'));
  assert.deepStrictEqual(menuActions(sandbox), [
    'copy',
    'copy-image-link',
    'select-all',
  ]);
  clickAction(sandbox, 'copy-image-link');
  assert.strictEqual(clip.text, 'https://example.com/chart.png');

  rightClick(sandbox, doc.getElementById('ta'));
  assert.deepStrictEqual(menuActions(sandbox), ['copy', 'paste', 'select-all']);
  console.log('  ok - link, image and editable targets add their own items');
}

function testPasteIntoEditable() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  const ta = sandbox.document.getElementById('ta');
  sandbox.navigator.clipboard = {
    readText: () => Promise.resolve(' pasted'),
    writeText: () => Promise.resolve(),
  };
  ta.selectionStart = ta.value.length;
  ta.selectionEnd = ta.value.length;
  rightClick(sandbox, ta);
  clickAction(sandbox, 'paste');
  return new Promise(resolve => {
    setTimeout(() => {
      assert.strictEqual(ta.value, 'draft pasted');
      console.log('  ok - Paste inserts the clipboard text into a field');
      resolve();
    }, 0);
  });
}

function testMenuClosesOnEscapeOutsideClickAndBlur() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  const doc = sandbox.document;

  rightClick(sandbox, doc.getElementById('p'));
  assert.ok(menuOf(sandbox), 'menu open');
  doc.dispatchEvent(new sandbox.KeyboardEvent('keydown', {key: 'Escape'}));
  assert.strictEqual(menuOf(sandbox), null, 'Escape closes the menu');

  rightClick(sandbox, doc.getElementById('p'));
  doc.dispatchEvent(new sandbox.KeyboardEvent('keydown', {key: 'a'}));
  assert.ok(menuOf(sandbox), 'other keys leave the menu open');
  doc
    .getElementById('h' in doc ? 'p' : 'p')
    .dispatchEvent(new sandbox.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(menuOf(sandbox), null, 'an outside click closes the menu');

  rightClick(sandbox, doc.getElementById('p'));
  sandbox.dispatchEvent(new sandbox.Event('blur'));
  assert.strictEqual(menuOf(sandbox), null, 'losing focus closes the menu');

  rightClick(sandbox, doc.getElementById('p'));
  sandbox.dispatchEvent(new sandbox.Event('resize'));
  assert.strictEqual(menuOf(sandbox), null, 'resizing closes the menu');
  console.log('  ok - Escape, outside click, blur and resize close the menu');
}

function testClickInsideMenuKeepsItOpen() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  rightClick(sandbox, sandbox.document.getElementById('p'));
  const menu = menuOf(sandbox);
  menu.dispatchEvent(new sandbox.MouseEvent('click', {bubbles: true}));
  assert.ok(menuOf(sandbox), 'clicking the menu chrome must not close it');
  console.log('  ok - clicking menu chrome keeps the menu open');
}

function testMenuIsClampedInsideTheViewport() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  rightClick(sandbox, sandbox.document.getElementById('p'), 99999, 99999);
  const menu = menuOf(sandbox);
  assert.ok(
    parseInt(menu.style.left, 10) < sandbox.innerWidth,
    'the menu must stay inside the viewport horizontally',
  );
  assert.ok(
    parseInt(menu.style.top, 10) < sandbox.innerHeight,
    'the menu must stay inside the viewport vertically',
  );
  console.log('  ok - the menu is clamped inside the viewport');
}

function testSecondRightClickReplacesTheMenu() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  rightClick(sandbox, sandbox.document.getElementById('p'));
  rightClick(sandbox, sandbox.document.getElementById('lnk'));
  assert.strictEqual(
    sandbox.document.querySelectorAll(MENU_SELECTOR).length,
    1,
    'only one menu may exist at a time',
  );
  assert.deepStrictEqual(menuActions(sandbox), [
    'copy',
    'copy-link',
    'select-all',
  ]);
  console.log('  ok - a second right-click replaces the first menu');
}

// ------------------------------------------------- parent (chat) document

// The parent-document menu belongs to the content view only.  A .txt / .md
// file opens as a Monaco (or <pre>) surface inside #content-tab-area, and
// that is the one place in the chat webview where the native menu is gone
// and ours has to take over.
function testParentDocumentMenuOnlyInTheContentView() {
  const {win} = makeWebview();
  openHtmlFile(win, 'notes.txt', 'plain text');
  const holder = win.document.querySelector(
    '#content-tab-area .content-tab-view',
  );
  assert.ok(holder, 'the .txt tab must render a content view');
  const ev = rightClick(win, holder);
  assert.strictEqual(ev.defaultPrevented, true);
  assert.deepStrictEqual(menuActions(win), ['copy', 'select-all']);
  console.log('  ok - the content view has a Copy / Select All menu');
}

function testTabStripKeepsItsOwnMenu() {
  const {win} = makeWebview();
  openHtmlFile(win, 'report.html', PAGE);
  const tab = win.document.querySelector('#tab-list .chat-tab');
  assert.ok(tab, 'a tab must exist');
  rightClick(win, tab);
  assert.strictEqual(
    menuOf(win),
    null,
    'the content menu must not hijack the tab strip',
  );
  const strip = win.document.getElementById('tab-context-menu');
  assert.ok(
    strip && strip.classList.contains('open'),
    'the tab strip menu must still open',
  );
  console.log('  ok - the tab strip keeps its own Close/Close Others menu');
}

function testRemoteWebAppAlsoGetsTheMenu() {
  const {win} = makeWebview({remote: true});
  openHtmlFile(win, 'report.html', PAGE);
  const sandbox = renderSandbox(activeSrcdoc(win));
  rightClick(sandbox, sandbox.document.getElementById('p'));
  assert.ok(menuOf(sandbox), 'the remote web app shares media/main.js');
  console.log('  ok - the remote web app gets the same menu');
}

// -------------------------------------------------------- edge cases

function testHtmFragmentWithoutBodyTag() {
  const {win} = makeWebview();
  openHtmlFile(win, 'snippet.htm', '<h2>bare fragment</h2>');
  const srcdoc = activeSrcdoc(win);
  assert.ok(
    srcdoc.indexOf('<h2>bare fragment</h2>') === 0,
    'a body-less fragment must keep its content first',
  );
  const sandbox = renderSandbox(srcdoc);
  rightClick(sandbox, sandbox.document.querySelector('h2'));
  assert.ok(menuOf(sandbox), 'a body-less .htm file still gets the menu');
  console.log('  ok - a fragment without <body> still gets the menu');
}

function testNonHtmlTabIsUntouched() {
  const {win} = makeWebview();
  openHtmlFile(win, 'notes.txt', 'plain text');
  const frames = win.document.querySelectorAll('iframe.content-html-frame');
  assert.strictEqual(frames.length, 0, 'a .txt file must not become an iframe');
  console.log('  ok - non-HTML tabs are unaffected');
}

function testSelectAllSkipsHiddenSourceNodes() {
  const menuApi = require(path.join(MEDIA, 'contentContextMenu.js'));
  const dom = new JSDOM(
    '<body>alpha<style>.x{}</style>beta<script>var q=1;</' +
      'script>gamma</body>',
    {pretendToBeVisual: true},
  );
  const text = menuApi.selectAll(dom.window.document);
  assert.strictEqual(text, 'alpha\nbeta\ngamma');
  console.log('  ok - selectAll skips <script>/<style> text');
}

function testCopyTextUsesAsyncClipboardWhenAvailable() {
  const dom = new JSDOM('<body>x</body>', {pretendToBeVisual: true});
  const menuApi = require(path.join(MEDIA, 'contentContextMenu.js'));
  const seen = [];
  dom.window.navigator.clipboard = {
    writeText: t => {
      seen.push(t);
      return Promise.resolve();
    },
  };
  return menuApi
    .copyText(dom.window.document, 'hello')
    .then(ok => {
      assert.strictEqual(ok, true);
      assert.deepStrictEqual(seen, ['hello']);
      return menuApi.copyText(dom.window.document, '');
    })
    .then(ok => {
      assert.strictEqual(ok, false, 'copying nothing is a no-op');
      console.log('  ok - copyText prefers the async Clipboard API');
    });
}

function testCopyTextFallsBackWhenClipboardRejects() {
  const dom = new JSDOM('<body>x</body>', {pretendToBeVisual: true});
  const menuApi = require(path.join(MEDIA, 'contentContextMenu.js'));
  dom.window.navigator.clipboard = {
    writeText: () => Promise.reject(new Error('denied')),
  };
  const clip = captureClipboard(dom.window);
  return menuApi.copyText(dom.window.document, 'fallback').then(ok => {
    assert.strictEqual(ok, true);
    assert.strictEqual(clip.text, 'fallback');
    console.log('  ok - copyText falls back to execCommand');
  });
}

function testWindowsShortcutHints() {
  const dom = new JSDOM('<body><p id="p">hi</p></body>', {
    pretendToBeVisual: true,
  });
  const menuApi = require(path.join(MEDIA, 'contentContextMenu.js'));
  Object.defineProperty(dom.window.navigator, 'platform', {
    value: 'Win32',
    configurable: true,
  });
  const model = menuApi.buildMenuModel(
    dom.window.document,
    dom.window.document.getElementById('p'),
  );
  assert.strictEqual(model[0].key, 'Ctrl+C');
  assert.strictEqual(model[model.length - 1].key, 'Ctrl+A');
  console.log('  ok - shortcut hints are platform aware');
}

// ------------------------------------------- direct module-level paths

function freshMenuDoc(markup) {
  const win = new JSDOM(markup || '<body><p id="p">text</p></body>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
  }).window;
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8') +
      '\n//# sourceURL=contentContextMenu.js',
  );
  return win;
}

function testMousedownOnMenuKeepsTheSelection() {
  const win = freshMenuDoc();
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  rightClick(win, win.document.getElementById('p'));
  const item = menuOf(win).querySelector('[data-action="select-all"]');
  const ev = new win.MouseEvent('mousedown', {bubbles: true, cancelable: true});
  item.dispatchEvent(ev);
  assert.strictEqual(
    ev.defaultPrevented,
    true,
    'pressing a menu item must not blur the page and drop the selection',
  );
  console.log('  ok - mousedown on a menu item preserves the selection');
}

function testShouldOpenCanVetoTheMenu() {
  const win = freshMenuDoc();
  win.ContentContextMenu.installContentContextMenu(win.document, {
    shouldOpen: () => false,
  });
  const ev = rightClick(win, win.document.getElementById('p'));
  assert.strictEqual(menuOf(win), null, 'a veto must suppress the menu');
  assert.strictEqual(
    ev.defaultPrevented,
    false,
    'a vetoed right-click must fall through to the host',
  );
  console.log('  ok - shouldOpen can veto the menu');
}

function testDisposeRemovesEveryListener() {
  const win = freshMenuDoc();
  const handle = win.ContentContextMenu.installContentContextMenu(
    win.document,
    {},
  );
  rightClick(win, win.document.getElementById('p'));
  assert.ok(menuOf(win), 'menu open before dispose');
  handle.dispose();
  assert.strictEqual(menuOf(win), null, 'dispose closes the open menu');
  rightClick(win, win.document.getElementById('p'));
  assert.strictEqual(
    menuOf(win),
    null,
    'dispose must unbind the contextmenu listener',
  );
  console.log('  ok - dispose tears the menu down completely');
}

function testBootstrapHtmlIsSelfContained() {
  const menuApi = require(path.join(MEDIA, 'contentContextMenu.js'));
  const boot = menuApi.contentContextMenuBootstrapHtml();
  assert.ok(boot.indexOf('<script data-sorcar-ctx>') === 1);
  assert.ok(
    boot.trim().endsWith('</script>'),
    'the bootstrap must close its own script tag',
  );
  assert.strictEqual(
    boot.indexOf('</script>'),
    boot.length - '</script>\n'.length,
    'the serialised source must not contain a stray </script>',
  );
  assert.ok(
    boot.indexOf('function installContentContextMenu') > 0,
    'the bootstrap must carry the real implementation',
  );
  console.log('  ok - the bootstrap markup is self-contained');
}

function testCopyReportsFailureWhenExecCommandThrows() {
  const win = freshMenuDoc();
  win.document.execCommand = function () {
    throw new Error('blocked by policy');
  };
  return win.ContentContextMenu.copyText(win.document, 'x').then(ok => {
    assert.strictEqual(ok, false, 'a blocked copy must report failure');
    assert.strictEqual(
      win.document.querySelectorAll('textarea').length,
      0,
      'the scratch textarea must be removed even when the copy fails',
    );
    console.log('  ok - a blocked execCommand reports failure and cleans up');
  });
}

function testCopyFallsBackWhenClipboardThrowsSynchronously() {
  const win = freshMenuDoc();
  win.navigator.clipboard = {
    writeText: () => {
      throw new Error('not allowed');
    },
  };
  const clip = captureClipboard(win);
  return win.ContentContextMenu.copyText(win.document, 'sync').then(ok => {
    assert.strictEqual(ok, true);
    assert.strictEqual(clip.text, 'sync');
    console.log('  ok - a synchronously throwing clipboard falls back');
  });
}

function testPasteIsANoOpWithoutClipboardRead() {
  const win = freshMenuDoc('<body><textarea id="t">a</textarea></body>');
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const ta = win.document.getElementById('t');
  delete win.navigator.clipboard;
  rightClick(win, ta);
  clickAction(win, 'paste');
  return new Promise(resolve => {
    setTimeout(() => {
      assert.strictEqual(ta.value, 'a', 'no clipboard read means no change');
      console.log('  ok - Paste is a no-op without clipboard read access');
      resolve();
    }, 0);
  });
}

// REGRESSION (defect 6d): a contenteditable has no `.value`, so the old
// fallback wrote a synthetic property and the user saw nothing appear.
// Paste must change the text the page actually renders.
function testPasteIntoContentEditableChangesRenderedText() {
  const win = freshMenuDoc('<body><div id="d" contenteditable>head</div></body>');
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const div = win.document.getElementById('d');
  // jsdom reports contentEditable through the attribute only.
  Object.defineProperty(div, 'isContentEditable', {value: true});
  win.navigator.clipboard = {readText: () => Promise.resolve('-tail')};
  rightClick(win, div);
  clickAction(win, 'paste');
  return new Promise(resolve => {
    setTimeout(() => {
      assert.strictEqual(
        div.textContent,
        'head-tail',
        'the pasted text must be visible in the contenteditable',
      );
      assert.strictEqual(
        div.value,
        undefined,
        'a contenteditable must never grow a synthetic .value',
      );
      console.log('  ok - Paste inserts real text into a contenteditable');
      resolve();
    }, 0);
  });
}

// The caret sits inside the contenteditable, so Paste must insert exactly
// there — replacing whatever the caret range covers — not append at the end.
function testPasteIntoContentEditableHonoursTheCaret() {
  const win = freshMenuDoc(
    '<body><div id="d" contenteditable>head TAIL</div></body>',
  );
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const doc = win.document;
  const div = doc.getElementById('d');
  Object.defineProperty(div, 'isContentEditable', {value: true});
  // Select the word TAIL: the paste replaces it.
  const range = doc.createRange();
  range.setStart(div.firstChild, 5);
  range.setEnd(div.firstChild, 9);
  const sel = win.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
  win.navigator.clipboard = {readText: () => Promise.resolve('body')};
  rightClick(win, div);
  clickAction(win, 'paste');
  return new Promise(resolve => {
    setTimeout(() => {
      assert.strictEqual(
        div.textContent,
        'head body',
        'the paste must land at the caret, replacing the selected word',
      );
      assert.strictEqual(
        win.getSelection().getRangeAt(0).collapsed,
        true,
        'the caret must end up collapsed after the inserted text',
      );
      console.log('  ok - Paste honours the caret in a contenteditable');
      resolve();
    }, 0);
  });
}

// A text-like control without setRangeText (older engines) must still
// receive the clipboard text.
function testPasteFallsBackWhenSetRangeTextIsMissing() {
  const win = freshMenuDoc('<body><input id="i" type="text" value="a"></body>');
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const input = win.document.getElementById('i');
  input.setRangeText = undefined;
  win.navigator.clipboard = {readText: () => Promise.resolve('b')};
  rightClick(win, input);
  clickAction(win, 'paste');
  return new Promise(resolve => {
    setTimeout(() => {
      assert.strictEqual(input.value, 'ab');
      console.log('  ok - Paste falls back when setRangeText is missing');
      resolve();
    }, 0);
  });
}

function testCopyAfterSelectAllKeepsEveryRange() {
  const win = freshMenuDoc(
    '<body>alpha<script>var z=1;</' + 'script>omega</body>',
  );
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const clip = captureClipboard(win);
  const body = win.document.body;
  rightClick(win, body);
  clickAction(win, 'select-all');
  rightClick(win, body);
  clickAction(win, 'copy');
  assert.strictEqual(
    clip.text,
    'alpha\nomega',
    'Copy must reuse the full multi-range Select All text',
  );
  console.log('  ok - Copy after Select All keeps every range');
}

// ------------------------------------------- regressions (review round 2)

// REGRESSION (defect 1): a VS Code webview serves `script-src 'nonce-…'`,
// and an about:srcdoc iframe INHERITS that policy — `allow-scripts` does
// not lift it.  A bootstrap without the nonce is silently blocked, so the
// shipped menu never installs in the real product.
function testBootstrapCarriesTheWebviewNonce() {
  const NONCE = 'aB3"<x&y';
  const win = new JSDOM('<body></body>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
  }).window;
  const src = fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8');
  const tag = win.document.createElement('script');
  tag.setAttribute('nonce', NONCE);
  // The sourceURL pragma keeps this nonce-bearing instance visible to the
  // coverage gate, which measures the module by URL.
  tag.textContent = src + '\n//# sourceURL=contentContextMenu.js';
  win.document.body.appendChild(tag);

  assert.strictEqual(
    win.ContentContextMenu.SCRIPT_NONCE,
    NONCE,
    'the module must capture the nonce of the script tag that loaded it',
  );
  const boot = win.ContentContextMenu.contentContextMenuBootstrapHtml();
  assert.ok(
    boot.indexOf('<script nonce="aB3&quot;&lt;x&amp;y" data-sorcar-ctx>') === 1,
    'the bootstrap must carry the HTML-escaped nonce, got: ' +
      boot.slice(0, 80),
  );
  assert.strictEqual(
    boot.indexOf(NONCE),
    -1,
    'the raw nonce must never be emitted unescaped',
  );

  // Without a nonce (remote web app, tests) no bogus attribute is emitted.
  const plain = freshMenuDoc('<body></body>');
  assert.strictEqual(
    plain.ContentContextMenu.SCRIPT_NONCE,
    '',
    'no nonce means no captured value',
  );
  const plainBoot = plain.ContentContextMenu.contentContextMenuBootstrapHtml();
  assert.ok(
    plainBoot.indexOf('<script data-sorcar-ctx>') === 1,
    'with no nonce the attribute must be omitted entirely: ' +
      plainBoot.slice(0, 60),
  );
  assert.strictEqual(
    plainBoot.indexOf('nonce'),
    -1,
    'no empty nonce="" may be emitted',
  );
  console.log('  ok - the bootstrap carries the webview CSP nonce');
}

// REGRESSION (defect 2): `html.toLowerCase().lastIndexOf('</body>')` indexed
// a lower-cased COPY and sliced the ORIGINAL.  U+0130 lower-cases to two
// UTF-16 units, so every such character shifted the splice point right and
// the injection landed inside the markup, corrupting the document.
function testBodySpliceSurvivesUnicodeLowercasing() {
  const {win} = makeWebview();
  const page =
    '<!DOCTYPE html><html><body><p id="p">\u0130\u0130\u0130stanbul</p>' +
    '<div id="tail">tail marker</div></BODY></html>';
  openHtmlFile(win, 'unicode.html', page);
  const srcdoc = activeSrcdoc(win);
  assert.ok(
    srcdoc.indexOf('<div id="tail">tail marker</div>') > 0,
    'the markup before </body> must survive intact, got: ' + srcdoc.slice(0, 200),
  );
  assert.ok(
    srcdoc.indexOf('<p id="p">\u0130\u0130\u0130stanbul</p>') > 0,
    'the Unicode text must be untouched',
  );
  assert.ok(
    srcdoc.indexOf('data-sorcar-ctx>') <
      srcdoc.toLowerCase().lastIndexOf('</body>'),
    'the bootstrap must be injected before the closing tag',
  );
  // The document still parses and the menu still installs, which it cannot
  // do when the splice lands mid-tag.
  const sandbox = renderSandbox(srcdoc);
  assert.strictEqual(
    sandbox.document.getElementById('tail').textContent,
    'tail marker',
    'the spliced document must still parse correctly',
  );
  rightClick(sandbox, sandbox.document.getElementById('p'));
  assert.ok(menuOf(sandbox), 'the menu must install in the spliced document');
  console.log('  ok - the </body> splice survives Unicode lowercasing');
}

// REGRESSION (defect 3): the hidden-node skip only looked at DIRECT children
// of <body>, so a <script> nested in a <div> leaked its source into both the
// Select All range and the clipboard.
function testSelectAllSkipsNestedScriptAndStyle() {
  const menuApi = require(path.join(MEDIA, 'contentContextMenu.js'));
  const markup =
    '<body><div>alpha<script>var SECRET_TOKEN=1;</' +
    'script>omega</div>' +
    '<section><p>beta</p><style>.q{color:red}</style></section></body>';
  const dom = new JSDOM(markup, {pretendToBeVisual: true});
  const text = menuApi.selectAll(dom.window.document);
  assert.strictEqual(
    text.indexOf('SECRET_TOKEN'),
    -1,
    'a nested <script> must never be selected, got: ' + text,
  );
  assert.strictEqual(
    text.indexOf('color:red'),
    -1,
    'a nested <style> must never be selected, got: ' + text,
  );
  assert.strictEqual(text, 'alpha\nomegabeta');

  // ...and the same through the real menu, all the way to the clipboard.
  const win = freshMenuDoc(markup);
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const clip = captureClipboard(win);
  rightClick(win, win.document.querySelector('p'));
  clickAction(win, 'select-all');
  rightClick(win, win.document.querySelector('p'));
  clickAction(win, 'copy');
  assert.strictEqual(
    clip.text.indexOf('SECRET_TOKEN'),
    -1,
    'nested script source must never reach the clipboard: ' + clip.text,
  );
  assert.strictEqual(clip.text, 'alpha\nomegabeta');
  console.log('  ok - Select All skips nested <script>/<style> too');
}

// REGRESSION (defect 4): the Select All text was kept whenever the live
// selection was a PREFIX of it, so narrowing the selection to the first
// paragraph and choosing Copy still copied the whole document.
function testCopyAfterNarrowingTheSelection() {
  const win = freshMenuDoc(
    '<body><p id="a">alpha</p><p id="b">beta</p><p id="c">gamma</p></body>',
  );
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const doc = win.document;
  const clip = captureClipboard(win);

  rightClick(win, doc.getElementById('a'));
  clickAction(win, 'select-all');

  // The user now drags out a smaller selection — exactly 'alpha', which is
  // a prefix of the Select All text.
  const range = doc.createRange();
  range.selectNodeContents(doc.getElementById('a'));
  const sel = win.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);

  rightClick(win, doc.getElementById('a'));
  clickAction(win, 'copy');
  assert.strictEqual(
    clip.text,
    'alpha',
    'Copy must honour the narrowed selection, not the stale Select All',
  );
  console.log('  ok - narrowing the selection invalidates the Select All');
}

// REGRESSION (defect 5): the parent install replaced the native menu across
// the entire chat UI — composer, settings, panels — losing native editing
// affordances everywhere.
function testChatUiKeepsItsNativeMenus() {
  const {win} = makeWebview();
  const doc = win.document;

  const composer = doc.getElementById('task-input');
  assert.ok(composer, 'the composer textarea must exist');
  const ev1 = rightClick(win, composer);
  assert.strictEqual(
    menuOf(win),
    null,
    'the content menu must not hijack the task composer',
  );
  assert.strictEqual(
    ev1.defaultPrevented,
    false,
    'the composer must keep the native editing menu',
  );

  const out = doc.getElementById('output');
  assert.ok(out, 'the chat transcript must exist');
  rightClick(win, out);
  assert.strictEqual(
    menuOf(win),
    null,
    'the content menu must not hijack the chat transcript',
  );

  // A handler that already claimed the event is never overridden.
  openHtmlFile(win, 'notes.txt', 'plain text');
  const view = doc.querySelector('#content-tab-area .content-tab-view');
  const claimed = new win.MouseEvent('contextmenu', {
    bubbles: true,
    cancelable: true,
  });
  claimed.preventDefault();
  view.dispatchEvent(claimed);
  assert.strictEqual(
    menuOf(win),
    null,
    'an already-handled right-click must not open the content menu',
  );
  console.log('  ok - the chat UI keeps its native and own menus');
}

// REGRESSION (defect 6a): a field keeps its selection in its own value, not
// in the document Selection, so Copy used to be dead over a highlighted
// <input>/<textarea>.
function testCopyReadsTheFieldSelection() {
  const win = freshMenuDoc(
    '<body><textarea id="t">hello world</textarea></body>',
  );
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const ta = win.document.getElementById('t');
  ta.selectionStart = 6;
  ta.selectionEnd = 11;
  const clip = captureClipboard(win);
  rightClick(win, ta);
  const item = menuOf(win).querySelector('[data-action="copy"]');
  assert.ok(
    !item.classList.contains('disabled'),
    'Copy must be enabled for text selected inside a field',
  );
  clickAction(win, 'copy');
  assert.strictEqual(clip.text, 'world');
  console.log('  ok - Copy reads the selection inside a field');
}

// REGRESSION (defect 6b): Select All over a field must select that field,
// not the whole document.
function testSelectAllInsideAFieldSelectsTheField() {
  const win = freshMenuDoc(
    '<body><p>outside text</p><textarea id="t">field text</textarea></body>',
  );
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const ta = win.document.getElementById('t');
  rightClick(win, ta);
  clickAction(win, 'select-all');
  assert.strictEqual(ta.selectionStart, 0);
  assert.strictEqual(ta.selectionEnd, 'field text'.length);
  const clip = captureClipboard(win);
  rightClick(win, ta);
  clickAction(win, 'copy');
  assert.strictEqual(
    clip.text,
    'field text',
    'Select All in a field then Copy must yield the field contents only',
  );
  console.log('  ok - Select All over a field selects just that field');
}

// REGRESSION (defect 6c): Paste was offered for controls that cannot accept
// typed text at all.
function testPasteIsOnlyOfferedForWritableControls() {
  const win = freshMenuDoc(
    '<body>' +
      '<input id="ro" type="text" readonly value="x">' +
      '<input id="dis" type="text" disabled value="x">' +
      '<input id="chk" type="checkbox">' +
      '<input id="file" type="file">' +
      '<input id="ok" type="text" value="x">' +
      '<input id="search" type="search">' +
      '<div id="ce">rich</div>' +
      '</body>',
  );
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const doc = win.document;
  const ce = doc.getElementById('ce');
  Object.defineProperty(ce, 'isContentEditable', {value: true});

  const offered = id => {
    rightClick(win, doc.getElementById(id));
    return menuActions(win).indexOf('paste') >= 0;
  };
  assert.strictEqual(offered('ro'), false, 'readonly input');
  assert.strictEqual(offered('dis'), false, 'disabled input');
  assert.strictEqual(offered('chk'), false, 'checkbox');
  assert.strictEqual(offered('file'), false, 'file input');
  assert.strictEqual(offered('ok'), true, 'plain text input');
  assert.strictEqual(offered('search'), true, 'search input');
  assert.strictEqual(offered('ce'), true, 'contenteditable');
  console.log('  ok - Paste is offered only for writable controls');
}

// REGRESSION (defect 6e): readText() throws synchronously when the document
// may not read the clipboard; the exception used to escape the handler.
function testPasteSurvivesASynchronousClipboardThrow() {
  const win = freshMenuDoc('<body><textarea id="t">a</textarea></body>');
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const ta = win.document.getElementById('t');
  win.navigator.clipboard = {
    readText: () => {
      throw new Error('clipboard read blocked');
    },
  };
  rightClick(win, ta);
  clickAction(win, 'paste');
  return new Promise(resolve => {
    setTimeout(() => {
      assert.strictEqual(ta.value, 'a', 'a blocked read must change nothing');
      console.log('  ok - a synchronously throwing readText is contained');
      resolve();
    }, 0);
  });
}

// REGRESSION (minor): the raw href attribute is useless when relative, and
// an empty address must not be offered as a live menu item.
function testLinkAndImageAddressesAreResolved() {
  const win = new JSDOM(
    '<body><a id="rel" href="sub/page.html">rel</a>' +
      '<a id="empty" href="">empty</a>' +
      '<img id="pic" src="img/chart.png">' +
      '<img id="blank" alt="no src"></body>',
    {runScripts: 'dangerously', pretendToBeVisual: true, url: 'https://host/d/'},
  ).window;
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8') +
      '\n//# sourceURL=contentContextMenu.js',
  );
  win.ContentContextMenu.installContentContextMenu(win.document, {});
  const doc = win.document;
  const clip = captureClipboard(win);

  rightClick(win, doc.getElementById('rel'));
  clickAction(win, 'copy-link');
  assert.strictEqual(
    clip.text,
    'https://host/d/sub/page.html',
    'a relative link must be copied resolved',
  );

  rightClick(win, doc.getElementById('pic'));
  clickAction(win, 'copy-image-link');
  assert.strictEqual(clip.text, 'https://host/d/img/chart.png');

  rightClick(win, doc.getElementById('blank'));
  const imgItem = menuOf(win).querySelector('[data-action="copy-image-link"]');
  assert.ok(
    imgItem.classList.contains('disabled'),
    'an image without a source must offer a disabled item',
  );
  console.log('  ok - link and image addresses are resolved, empty disabled');
}

async function main() {
  testBugWithoutMenuModuleNoContextMenu();
  testHtmlTabShipsMenuIntoSandbox();
  testRightClickOpensCopyAndSelectAll();
  testSelectAllThenCopyGrabsWholeDocument();
  testCopyDisabledWithoutSelection();
  testCopySelectionOnly();
  testLinkAndImageAndEditableItems();
  await testPasteIntoEditable();
  testMenuClosesOnEscapeOutsideClickAndBlur();
  testClickInsideMenuKeepsItOpen();
  testMenuIsClampedInsideTheViewport();
  testSecondRightClickReplacesTheMenu();
  testParentDocumentMenuOnlyInTheContentView();
  testTabStripKeepsItsOwnMenu();
  testRemoteWebAppAlsoGetsTheMenu();
  testHtmFragmentWithoutBodyTag();
  testNonHtmlTabIsUntouched();
  testSelectAllSkipsHiddenSourceNodes();
  await testCopyTextUsesAsyncClipboardWhenAvailable();
  await testCopyTextFallsBackWhenClipboardRejects();
  testWindowsShortcutHints();
  testMousedownOnMenuKeepsTheSelection();
  testShouldOpenCanVetoTheMenu();
  testDisposeRemovesEveryListener();
  testBootstrapHtmlIsSelfContained();
  await testCopyReportsFailureWhenExecCommandThrows();
  await testCopyFallsBackWhenClipboardThrowsSynchronously();
  await testPasteIsANoOpWithoutClipboardRead();
  await testPasteIntoContentEditableChangesRenderedText();
  await testPasteIntoContentEditableHonoursTheCaret();
  await testPasteFallsBackWhenSetRangeTextIsMissing();
  testCopyAfterSelectAllKeepsEveryRange();
  testBootstrapCarriesTheWebviewNonce();
  testBodySpliceSurvivesUnicodeLowercasing();
  testSelectAllSkipsNestedScriptAndStyle();
  testCopyAfterNarrowingTheSelection();
  testChatUiKeepsItsNativeMenus();
  testCopyReadsTheFieldSelection();
  testSelectAllInsideAFieldSelectsTheField();
  testPasteIsOnlyOfferedForWritableControls();
  await testPasteSurvivesASynchronousClipboardThrow();
  testLinkAndImageAddressesAreResolved();
  console.log('htmlTabContextMenu.test.js: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
