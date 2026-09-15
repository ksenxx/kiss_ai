// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end (jsdom) test of directory-listing content tabs: a clicked
// directory link makes the remote web server reply with a fileContent
// message carrying isDirectory:true and a plain-text listing.  The
// client must render that listing as TEXT regardless of the directory's
// NAME — a directory named site.html (or notes.md) must not have its
// listing interpreted as HTML (or markdown) and injected into the
// sandboxed content iframe.

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
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  // Monaco loads from a CDN, which jsdom never fetches (the loader
  // would only give up via its 10s timeout).  A window-level `monaco`
  // that records editor.create calls stands in for the unreachable CDN
  // build: ensureMonaco() resolves with it immediately, so the test can
  // assert the exact text and language handed to the editor.
  const created = [];
  win.monaco = {
    editor: {
      create: (holder, opts) => {
        created.push({holder, value: opts.value, language: opts.language});
        return {dispose: () => {}, layout: () => {}};
      },
    },
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, created};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

async function waitFor(predicate, message, timeoutMs = 2000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 10));
  }
  throw new Error(message || 'waitFor timed out');
}

const LISTING = '/ws/site.html:\n\n/ws/site.html/<img src=x>.txt\n';

async function testDirectoryListingRendersAsText(name) {
  const {win, created} = makeWebview();
  send(win, {
    type: 'fileContent',
    path: '/ws/' + name,
    name,
    isDirectory: true,
    content: LISTING,
  });
  assert.strictEqual(
    win.document.querySelectorAll('.content-html-frame').length,
    0,
    name + ': a directory listing must never render in the HTML iframe',
  );
  assert.strictEqual(
    win.document.querySelectorAll('.content-monaco-holder').length,
    1,
    name + ': the listing must render in a text (code) holder',
  );
  const editor = await waitFor(
    () => created[0],
    name + ': the listing must be handed to the text editor',
  );
  assert.strictEqual(editor.value, LISTING, name + ': raw listing text');
  assert.strictEqual(
    editor.language,
    'plaintext',
    name + ': listing must be plaintext regardless of the directory name',
  );
  win.close();
  console.log('  ok - ' + name + ' directory listing renders as text');
}

async function testPlainFileStillRendersHtml() {
  const {win, created} = makeWebview();
  send(win, {
    type: 'fileContent',
    path: '/ws/report.html',
    name: 'report.html',
    content: '<p>report</p>',
  });
  assert.strictEqual(
    win.document.querySelectorAll('.content-html-frame').length,
    1,
    'a real .html FILE must still render in the sandboxed iframe',
  );
  assert.strictEqual(created.length, 0, 'no text editor for an html file');
  win.close();
  console.log('  ok - a real .html file still renders as HTML');
}

(async () => {
  try {
    await testDirectoryListingRendersAsText('site.html');
    await testDirectoryListingRendersAsText('notes.md');
    await testPlainFileStillRendersHtml();
    console.log('directoryListingContentTab.test.js: all tests passed');
  } catch (err) {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exitCode = 1;
  }
})();
