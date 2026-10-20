// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// pdfView.js: when the pdf.js module import() fails, the worker download
// started alongside it must be aborted.  It used to keep running with
// nobody waiting for it (about 1 MB from the CDN), and the retry 500 ms
// later started a second download next to the orphaned first one.
//
// jsdom evaluates the viewer in a vm context whose import() of an https:
// URL rejects at once (Node's loader refuses the scheme), which is the
// "import failed before the worker arrived" case.  The page has no fetch
// of its own; the one the test supplies records each request's
// AbortSignal and never answers, like a CDN connection that stalls.  The
// rejection is a plain Error, not the TypeError a browser raises for a
// network failure, so the single-retry path is not exercised here.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {waitFor} = require('./simplify2_harness');

const MEDIA = path.join(__dirname, '..', 'media');

function makePage() {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  const fetches = [];
  win.fetch = (url, init) => {
    fetches.push({url, signal: init.signal});
    return new Promise(() => {});
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'pdfView.js'), 'utf8'));
  return {win, fetches};
}

async function main() {
  const {win, fetches} = makePage();
  const holder = win.document.createElement('div');
  win.document.body.appendChild(holder);
  const viewer = win.mountPdfViewer(holder, new win.Uint8Array(0), {
    name: 'report.pdf',
  });
  const noteOf = () => holder.querySelector('.content-binary-note');
  await waitFor(() => noteOf() !== null, 'the viewer never gave up', 5000);
  assert.ok(
    noteOf().textContent.startsWith('Cannot display report.pdf: '),
    noteOf().textContent,
  );
  assert.deepStrictEqual(
    fetches.map(f => f.url.split('/').pop()),
    ['pdf.worker.min.mjs'],
    'the worker download started alongside the module import',
  );
  assert.strictEqual(
    fetches[0].signal.aborted,
    true,
    'a worker download whose import failed is aborted, not orphaned',
  );
  viewer.dispose();
  console.log('pdfViewImportFailureAbortsWorkerFetch.test.js passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
