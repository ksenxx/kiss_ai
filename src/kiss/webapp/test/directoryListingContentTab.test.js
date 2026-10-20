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
const {
  makePage: buildPage,
  send,
  stubMonaco,
  waitFor,
} = require('./simplify2_harness');

// Monaco loads from a CDN, which jsdom never fetches (the loader would
// only give up via its 10s timeout).  The harness's recording `monaco`
// stands in for the unreachable CDN build: ensureMonaco() resolves with
// it immediately, so the test can assert the exact text and language
// handed to the editor.
function makePage() {
  let created;
  const {win} = buildPage({
    beforeScripts: w => {
      created = stubMonaco(w);
    },
  });
  return {win, created};
}

const LISTING = '/ws/site.html:\n\n/ws/site.html/<img src=x>.txt\n';

async function testDirectoryListingRendersAsText(name) {
  const {win, created} = makePage();
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
  const {win, created} = makePage();
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
