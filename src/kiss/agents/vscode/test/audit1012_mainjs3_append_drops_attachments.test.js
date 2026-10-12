// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-10-12 (mainjs-3): a photo attached while the tab's task is
// running is silently lost.  sendMessage()'s append-to-running-task
// branch posts `appendUserMessage({prompt, tabId})` -- the daemon command
// has no attachments field (server/commands.py _cmd_append_user_message,
// src/types.ts) -- and then resetComposerAfterSend() empties the
// attachment list.  The ask-answer branch of the same function keeps
// the attachments for the next prompt ("photos are not part of it");
// the append branch must do the same instead of discarding them with
// no notice.
//
// Expected: after the append the prompt text is gone but the attachment
// chip is still in the composer.

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
      '\n//# sourceURL=audit1012-mainjs3-append-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
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

function settle() {
  return new Promise(resolve => setTimeout(resolve, 50));
}

async function main() {
  const {win, posted} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId});
  const png = new win.File([new Uint8Array([137, 80, 78, 71])], 'shot.png', {
    type: 'image/png',
  });
  paste(win, png);
  await settle();
  const chips = win.document.getElementById('file-chips');
  assert.strictEqual(
    chips.querySelectorAll('.file-chip:not(.pending):not(.error)').length,
    1,
    'precondition: the photo is attached and ready',
  );
  typeAndEnter(win, 'and look at this photo');
  await settle();
  assert.ok(
    !posted.some(m => m && m.type === 'submit'),
    'no second task is submitted while one is running',
  );
  // appendUserMessage cannot carry the photo.  Whatever the fix does
  // with the text, the photo must not vanish: it is still attached, and
  // the user is told why the message did not take it.
  assert.strictEqual(
    chips.querySelectorAll('.file-chip').length,
    1,
    'the attachment the append could not carry is still in the composer',
  );
  const toast = Array.from(
    win.document.querySelectorAll('.kiss-notification'),
  ).map(el => el.textContent);
  assert.ok(
    toast.some(t => /attach/i.test(t)),
    'a toast says the attachment was not sent; got ' + JSON.stringify(toast),
  );
  // Nothing was posted with the photo silently stripped: either the
  // append was refused (prompt kept) or it went out as text with the
  // photo kept -- never a lost photo.
  const append = posted.find(m => m && m.type === 'appendUserMessage');
  if (!append) {
    assert.strictEqual(
      win.document.getElementById('task-input').value,
      'and look at this photo',
      'a refused append keeps the prompt in the composer',
    );
  }
  win.close();
  console.log(
    'audit1012_mainjs3_append_drops_attachments.test.js: all passed.',
  );
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
