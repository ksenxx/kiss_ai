// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the settings panel's "Open chats as
// editor tabs" toggle OUTSIDE editor-tabs mode:
//  - in the VS Code SIDEBAR webview (no body class) it is visible,
//    unchecked, posts `setEditorTabsMode {enabled:true}` on switch-on,
//    and the internal tab bar keeps its '+' / settings buttons;
//  - in the REMOTE web app (body.remote-chat) it stays hidden —
//    browser tabs already are that surface's chat tabs, and there is
//    no VS Code configuration to flip.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs);
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
      '\n//# sourceURL=editor-tabs-toggle-main.js',
  );

  return {win, posted};
}

function testSidebarWebview() {
  const {win, posted} = makeWebview('');
  const label = win.document.getElementById('cfg-editor-tabs-mode-label');
  const box = win.document.getElementById('cfg-editor-tabs-mode');
  assert.strictEqual(label.style.display, '', 'toggle visible in sidebar');
  assert.strictEqual(box.checked, false, 'mode off: unchecked');

  box.checked = true;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
  const msgs = posted.filter(m => m.type === 'setEditorTabsMode');
  assert.strictEqual(msgs.length, 1);
  assert.strictEqual(msgs[0].enabled, true);

  // Sidebar mode keeps its internal tab bar and the footer's new-chat
  // and settings controls (the latter inside the "..." menu).
  assert.strictEqual(win.document.getElementById('tab-bar').style.display, '');
  assert.ok(win.document.getElementById('new-chat-btn'));
  assert.ok(win.document.getElementById('settings-btn'));

  // No stray panel-title reports outside editor-tabs mode.
  assert.strictEqual(
    posted.filter(m => m.type === 'panelTitle').length,
    0,
    'panelTitle is editor-tabs-mode only',
  );
}

function testRemoteWebapp() {
  const {win} = makeWebview(' class="remote-chat"');
  const label = win.document.getElementById('cfg-editor-tabs-mode-label');
  assert.strictEqual(
    label.style.display,
    'none',
    'toggle hidden in the remote web app',
  );
}

function main() {
  testSidebarWebview();
  testRemoteWebapp();
  console.log('editorTabsSettingsToggle: all tests passed');
}

main();
