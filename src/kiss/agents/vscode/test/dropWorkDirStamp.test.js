// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E test (webview harness): dropping files onto the chat input must
// stamp the active tab's work dir on the `resolveDroppedPaths` command.
//
// Without the stamp, the extension host falls back to the window's
// workspace folder / host cwd, which in a no-folder window (Dock-
// launched, cwd '/') is NOT where the tab's tasks run — dropped files
// were relativized against the wrong root and inserted as unusable
// `./Users/...` paths.

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function drop(win, uriList) {
  const container = win.document.getElementById('input-container');
  assert.ok(container, '#input-container must exist');
  const ev = new win.Event('drop', {bubbles: true, cancelable: true});
  ev.dataTransfer = {
    getData: type => (type === 'text/uri-list' ? uriList : ''),
    files: [],
  };
  container.dispatchEvent(ev);
}

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

// A drop in a tab with no explicit workDir must stamp the config-level
// work dir (the same fallback every other tab-scoped command uses).
function testDropStampsConfigWorkDir() {
  const {win, posted} = makeWebview();
  send(win, {type: 'configData', config: {work_dir: '/x/y'}});
  drop(win, 'file:///x/y/src/a.ts\n# comment line\n');
  const cmd = lastMsg(posted, 'resolveDroppedPaths');
  assert.ok(cmd, 'drop must send resolveDroppedPaths');
  // Element-wise: the array was built in the JSDOM realm, so its
  // prototype differs from this realm's Array and deepStrictEqual
  // would fail on the reference-identity of the prototypes.
  assert.strictEqual(cmd.uris.length, 1);
  assert.strictEqual(cmd.uris[0], 'file:///x/y/src/a.ts');
  assert.strictEqual(
    cmd.workDir,
    '/x/y',
    'resolveDroppedPaths must carry the tab work dir fallback',
  );
  win.close();
  console.log('ok - drop stamps the config work dir on resolveDroppedPaths');
}

testDropStampsConfigWorkDir();
console.log('all dropWorkDirStamp tests passed');
