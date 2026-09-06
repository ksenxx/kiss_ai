// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end webview tests: the top status bar (the one showing
// tokens, cost and steps) shows the server's machine name in its
// middle.  The daemon stamps `machine` into every configData reply
// (src/kiss/server/commands.py _cmd_get_config), which reaches
// media/main.js in both the VS Code webview and the remote webapp.

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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage() {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function testMachineNameLandsInTheStatusBar() {
  const win = makeWebview();
  const el = win.document.getElementById('status-machine');
  assert.ok(el, '#status-machine must exist in the status bar');
  assert.strictEqual(
    el.closest('#tab-status-bar'),
    win.document.getElementById('tab-status-bar'),
    'the machine name lives in the tokens/cost/steps bar',
  );
  assert.strictEqual(el.textContent, '', 'empty until the daemon reports');
  send(win, {type: 'configData', config: {}, machine: 'build-box-7'});
  assert.strictEqual(el.textContent, 'build-box-7');
  win.close();
}

function testConfigDataWithoutMachineLeavesTheNameAlone() {
  const win = makeWebview();
  const el = win.document.getElementById('status-machine');
  send(win, {type: 'configData', config: {}, machine: 'build-box-7'});
  // A later reply without the field (an older daemon) must not blank
  // the name the user is looking at.
  send(win, {type: 'configData', config: {}});
  assert.strictEqual(el.textContent, 'build-box-7');
  win.close();
}

function testMachineNameSitsMidBarWithoutOverlapRisk() {
  // chat.html links main.css; jsdom does not fetch links, so the real
  // stylesheet is inlined and the layout intent is verified through
  // the computed style — the same cascade a browser resolves.  The
  // name must be a normal flex item (its auto margin pairing with
  // #status-tokens's splits the free space evenly around it), NOT an
  // absolutely-positioned overlay that would paint over the Tokens /
  // Cost metrics on a narrow pane; on overflow it gives way first
  // (huge flex-shrink, min-width 0) and ellipsizes.  A real-browser
  // geometry check lives in
  // src/kiss/tests/agents/vscode/test_status_machine_layout.py.
  const win = makeWebview();
  const style = win.document.createElement('style');
  style.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(style);
  const el = win.document.getElementById('status-machine');
  const cs = win.getComputedStyle(el);
  assert.strictEqual(cs.position, 'static', 'a flex item, not an overlay');
  assert.strictEqual(cs.minWidth, '0px', 'allowed to shrink');
  assert.strictEqual(cs.textOverflow, 'ellipsis');
  // The flanking side groups: equal flex shares, floored at their own
  // content so the machine name shrinks first.
  const left = el.previousElementSibling;
  const right = el.nextElementSibling;
  assert.ok(
    left.classList.contains('status-side') &&
      left.contains(win.document.getElementById('status-text')),
    'the status text flanks the name on the left',
  );
  assert.ok(
    right.classList.contains('status-side') &&
      right.contains(win.document.getElementById('status-tokens')) &&
      right.contains(win.document.getElementById('status-budget')) &&
      right.contains(win.document.getElementById('status-steps')),
    'the tokens/cost/steps metrics flank the name on the right',
  );
  for (const side of [left, right]) {
    const scs = win.getComputedStyle(side);
    // jsdom's cssstyle cannot resolve the `flex: 1 1 0` shorthand, so
    // the equal-share geometry (and no-overlap at 300-1000px) is
    // asserted in a real browser by test_status_machine_layout.py;
    // here only the longhand floor is verifiable.
    assert.strictEqual(scs.minWidth, 'max-content', 'sides never squashed');
  }
  win.close();
}

function main() {
  testMachineNameLandsInTheStatusBar();
  console.log('  ok - configData machine name shows in the status bar');
  testConfigDataWithoutMachineLeavesTheNameAlone();
  console.log('  ok - a reply without machine keeps the shown name');
  testMachineNameSitsMidBarWithoutOverlapRisk();
  console.log('  ok - the machine name sits mid-bar with no overlap risk');
  console.log('statusBarMachineName.test.js: all tests passed');
}

main();
