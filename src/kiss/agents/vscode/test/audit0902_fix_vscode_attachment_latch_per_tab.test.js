// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) test for review-vscode.md #10: sendMessage() parked a
// submit behind the module-global `awaitingAttachments` latch while an
// attachment was still converting.  Attachments -- and their conversion
// promises -- are per tab (saveCurrentTab stores `tab.attachments`), so the
// latch set by tab A's pending HEIC made tab B's Enter return silently:
// B's prompt (and its own, already-decoded photo) were never sent.
//
// Tab A pastes a HEIC whose decode is parked on a createImageBitmap the
// test controls (JSDOM has no image decoder, so the test supplies the
// browser API the page would normally get), types a prompt and presses
// Enter.  Tab B pastes a PNG, types a prompt and presses Enter twice: the
// first Enter must submit once its own attachment has landed and the
// second must be deduplicated within B only.  Finally A's decode fails:
// A's parked submit is dropped (its photo is gone), A's latch clears, and a
// later Enter in A sends.

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

  // The HEIC decoder the page asks the browser for: every call is parked
  // until the test settles it.
  const decodes = [];
  win.createImageBitmap = function () {
    return new Promise((resolve, reject) => {
      decodes.push({resolve, reject});
    });
  };

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=audit0902-attachlatch-main.js',
  );
  return {win, posted, decodes};
}

function paste(win, file) {
  const inp = win.document.getElementById('task-input');
  const ev = new win.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(ev, 'clipboardData', {
    value: {items: [{kind: 'file', getAsFile: () => file}]},
  });
  inp.dispatchEvent(ev);
}

function typeAndEnter(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

// encodeAsJpeg retries createImageBitmap once without the orientation
// option, so a failed decode is two parked calls.
async function failDecode(decodes, first, reason) {
  decodes[first].reject(new Error(reason));
  await settle();
  assert.strictEqual(decodes.length, first + 2, 'the decoder was retried');
  decodes[first + 1].reject(new Error(reason));
  await settle();
}

function submits(posted) {
  return posted.filter(m => m.type === 'submit');
}

function settle() {
  // Several macrotask turns: FileReader completion + the awaited chain.
  return new Promise(resolve => setTimeout(resolve, 50));
}

async function main() {
  const {win, posted, decodes} = makeWebview();
  win._testApi.endLaunch();
  const tabA = win._testApi.getActiveTabId();

  // Tab A: a camera HEIC, parked in the decoder.
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0001.HEIC', {
    type: 'image/heic',
  });
  paste(win, heic);
  await settle();
  assert.strictEqual(decodes.length, 1, 'the HEIC reached the decoder');
  typeAndEnter(win, 'prompt for A');
  assert.strictEqual(submits(posted).length, 0, 'A waits for its photo');

  // Tab B: a PNG (base64 via FileReader, no decode needed), Enter twice.
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabB, tabA);
  const png = new win.File([new Uint8Array([137, 80, 78, 71])], 'shot.png', {
    type: 'image/png',
  });
  paste(win, png);
  typeAndEnter(win, 'prompt for B');
  typeAndEnter(win, 'prompt for B');
  await settle();
  const fromB = submits(posted).filter(m => m.tabId === tabB);
  assert.strictEqual(
    fromB.length,
    1,
    `tab B's Enter was ${fromB.length === 0 ? 'silently dropped while tab A waited for its HEIC' : 'submitted more than once'} (${fromB.length} submits)`,
  );
  assert.strictEqual(fromB[0].prompt, 'prompt for B');
  assert.strictEqual(fromB[0].attachments.length, 1, 'B sent its photo');
  assert.strictEqual(fromB[0].attachments[0].name, 'shot.png');
  assert.strictEqual(submits(posted).length, 1, 'A still waits');

  // A's decode fails: its parked submit is dropped (the photo is gone,
  // sending without it is the silent loss to avoid), never sent for B.
  await failDecode(decodes, 0, 'undecodable HEIC');
  assert.strictEqual(submits(posted).length, 1, 'no submit from A yet');

  // Back on A: the latch must be clear, so a plain Enter sends.
  const tabEl = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabA)}]`,
  );
  assert.ok(tabEl, 'tab A is still there');
  tabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);
  const chips = win.document.getElementById('file-chips');
  assert.match(
    chips.textContent,
    /undecodable HEIC/,
    'A shows why its photo was dropped',
  );
  typeAndEnter(win, 'prompt for A again');
  await settle();
  const fromA = submits(posted).filter(m => m.tabId === tabA);
  assert.strictEqual(
    fromA.length,
    1,
    `A could not send after the wait (${fromA.length})`,
  );
  assert.strictEqual(fromA[0].prompt, 'prompt for A again');
  assert.strictEqual(fromA[0].attachments.length, 0);

  // Switching tabs during the wait abandons the submit (it no longer
  // matches what the user sees) but keeps the tab's attachment for later.
  win._testApi.createNewTab();
  const tabC = win._testApi.getActiveTabId();
  const png3 = new win.File([new Uint8Array([137, 80, 78, 71, 1])], 'c.png', {
    type: 'image/png',
  });
  paste(win, png3);
  typeAndEnter(win, 'prompt for C');
  tabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);
  await settle();
  assert.strictEqual(
    submits(posted).filter(m => m.tabId === tabC).length,
    0,
    'a submit whose tab was switched away from must not be sent',
  );
  win.document
    .querySelector(`.chat-tab[data-tab-id=${JSON.stringify(tabC)}]`)
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(win._testApi.getActiveTabId(), tabC);
  typeAndEnter(win, 'prompt for C again');
  await settle();
  const fromC = submits(posted).filter(m => m.tabId === tabC);
  assert.strictEqual(fromC.length, 1, 'C sends once its user is back');
  assert.strictEqual(fromC[0].attachments.length, 1, 'C kept its photo');
  assert.strictEqual(fromC[0].attachments[0].name, 'c.png');
  tabEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);

  // Same-tab deduplication still holds with a pending attachment: a burst
  // of Enters while a photo is still being read submits exactly once.
  const png2 = new win.File([new Uint8Array([137, 80, 78, 71, 13])], 'b.png', {
    type: 'image/png',
  });
  paste(win, png2);
  typeAndEnter(win, 'burst');
  typeAndEnter(win, 'burst');
  typeAndEnter(win, 'burst');
  await settle();
  const bursts = submits(posted).filter(m => m.prompt === 'burst');
  assert.strictEqual(
    bursts.length,
    1,
    `a burst of Enters in one tab submitted ${bursts.length} times`,
  );
  assert.strictEqual(bursts[0].tabId, tabA);
  assert.strictEqual(bursts[0].attachments.length, 1);
  assert.strictEqual(bursts[0].attachments[0].name, 'b.png');
  win.close();
  console.log(
    'audit0902_fix_vscode_attachment_latch_per_tab: all tests passed',
  );
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
